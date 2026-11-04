package assistant

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unicode"

	"m3tal-ai/internal/lm"
)

// Generator is optional local text generation used for drafts and plans.
type Generator interface {
	Generate(prompt string, maxTokens int, temperature float64, seed int64) (string, error)
}

// Assistant combines a small intent model, local documentation retrieval, and
// M3TAL Core tools.
type Assistant struct {
	mu            sync.Mutex
	classifier    *classifier
	knowledge     *KnowledgeBase
	core          *CoreAPI
	languageModel Generator
	jsonOutput    bool
	lastTarget    *namedTarget
}

type namedTarget struct {
	kind, name string
}

// SetLanguageModel enables local model answers and constrained plan proposals.
func (assistant *Assistant) SetLanguageModel(model Generator) { assistant.languageModel = model }

// HasLanguageModel returns true if a local model is loaded.
func (assistant *Assistant) HasLanguageModel() bool { return assistant.languageModel != nil }

// SetJSONOutput prints Core payloads as indented JSON instead of labeled fields.
func (assistant *Assistant) SetJSONOutput(enabled bool) { assistant.jsonOutput = enabled }

// Knowledge returns the indexed documentation set.
func (assistant *Assistant) Knowledge() *KnowledgeBase { return assistant.knowledge }

type mutationSpec struct{ targetType, action string }

var mutationPolicy = map[intent]mutationSpec{
	intentStartStack: {"stack", "start"}, intentStopStack: {"stack", "stop"}, intentRestart: {"stack", "restart"},
	intentStartContainer: {"container", "start"}, intentStopContainer: {"container", "stop"}, intentRestartContainer: {"container", "restart"},
	intentStartService: {"service", "start"}, intentStopService: {"service", "stop"}, intentRestartService: {"service", "restart"},
}

// New creates an assistant and trains its small intent classifier locally.
func New(knowledgePaths []string, core *CoreAPI) (*Assistant, error) {
	classifier, err := newClassifier()
	if err != nil {
		return nil, fmt.Errorf("train intent model: %w", err)
	}
	knowledge, err := LoadKnowledgeBase(knowledgePaths)
	if err != nil {
		return nil, err
	}
	if core == nil {
		core = NewCoreAPI("", "")
	}
	return &Assistant{classifier: classifier, knowledge: knowledge, core: core}, nil
}

// PingCore checks that M3TAL Core answers a health request.
func (assistant *Assistant) PingCore(ctx context.Context) error {
	_, err := assistant.core.get(ctx, "/api/v2/system/health")
	if err != nil {
		return fmt.Errorf("M3TAL Core is unreachable at %s: %w", assistant.core.BaseURL, err)
	}
	return nil
}

// Ask answers a question or operates M3TAL Core. confirm is called only after
// a mutating target has been found in Core's live allowlist. Read-only requests
// do not call it.
func (assistant *Assistant) Ask(ctx context.Context, query string, confirm func(action string) bool, output io.Writer) error {
	if strings.TrimSpace(query) == "" {
		return fmt.Errorf("enter a question or M3TAL request")
	}
	if output == nil {
		output = io.Discard
	}
	if assistant.languageModel != nil && looksMultiStep(query) {
		plan, err := assistant.proposePlan(query)
		if err == nil {
			return assistant.executePlan(ctx, plan, confirm, output)
		}
		fmt.Fprintln(output, "The local model did not produce a valid tool plan; using standard routing.")
	}
	selected, _, err := assistant.classifier.classify(query)
	if err != nil {
		return err
	}
	switch selected {
	case intentStats:
		return assistant.printCoreData(ctx, "/api/v2/system/stats", "M3TAL system stats", output)
	case intentHealth:
		return assistant.printCoreData(ctx, "/api/v2/system/health", "M3TAL health", output)
	case intentContainers:
		return assistant.printCoreContainers(ctx, query, output)
	case intentLogs:
		name, err := requireName(query, "for")
		if err != nil {
			name, err = requireName(query, "container")
		}
		name, err = assistant.resolveTarget(ctx, query, "container", name, err)
		if err != nil {
			return fmt.Errorf("tell me which container, for example: show logs for jellyfin")
		}
		if err := assistant.printCoreData(ctx, "/api/v2/containers/"+url.PathEscape(name)+"/logs?tail=100", "Recent logs for "+name, output); err != nil {
			return err
		}
		assistant.setLastTarget("container", name)
		return nil
	case intentUnknown:
		fmt.Fprintln(output, "I'm not sure I understood. Did you mean system stats, health, running containers, logs for a named container, a stack/service action, or a M3TAL documentation question?")
		return nil
	default:
		if spec, ok := mutationPolicy[selected]; ok {
			return assistant.controlTarget(ctx, query, spec, confirm, output)
		}
		return assistant.printKnowledge(query, output)
	}
}

func (assistant *Assistant) printCoreContainers(ctx context.Context, query string, output io.Writer) error {
	data, err := assistant.core.get(ctx, "/api/v2/services")
	if err != nil {
		return err
	}
	target := extractContainerTarget(query)
	if target != "" {
		filtered := filterServicesForContainer(data, target)
		if len(filtered) > 0 {
			assistant.setLastTarget("container", target)
			fmt.Fprint(output, formatCorePayload("M3TAL container: "+target, filtered, assistant.jsonOutput))
			return nil
		}
	}
	fmt.Fprint(output, formatCorePayload("M3TAL containers and services", data, assistant.jsonOutput))
	return nil
}

func extractContainerTarget(query string) string {
	stopWords := map[string]bool{
		"is": true, "are": true, "running": true, "container": true, "containers": true,
		"docker": true, "service": true, "services": true, "status": true, "check": true,
		"of": true, "for": true, "the": true, "a": true, "an": true, "what": true,
		"which": true, "show": true, "list": true, "get": true, "all": true, "me": true,
		"if": true, "and": true, "or": true, "please": true, "active": true, "stopped": true,
	}
	for _, word := range strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '-' && r != '_'
	}) {
		if !stopWords[word] && len(word) > 1 {
			return word
		}
	}
	return ""
}

func filterServicesForContainer(data json.RawMessage, target string) json.RawMessage {
	var entries []map[string]any
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil
	}
	lowerTarget := strings.ToLower(target)
	var matched []map[string]any
	for _, entry := range entries {
		match := false
		if name, ok := entry["name"].(string); ok && strings.EqualFold(name, lowerTarget) {
			match = true
		}
		if names, ok := entry["names"].([]any); ok {
			for _, n := range names {
				clean := strings.ToLower(strings.TrimPrefix(fmt.Sprint(n), "/"))
				if clean == lowerTarget {
					match = true
					break
				}
			}
		}
		if match {
			matched = append(matched, entry)
		}
	}
	if len(matched) == 0 {
		return nil
	}
	encoded, err := json.Marshal(matched)
	if err != nil {
		return nil
	}
	return encoded
}

func (assistant *Assistant) printCoreData(ctx context.Context, path, heading string, output io.Writer) error {
	data, err := assistant.core.get(ctx, path)
	if err != nil {
		return err
	}
	fmt.Fprint(output, formatCorePayload(heading, data, assistant.jsonOutput))
	return nil
}

func (assistant *Assistant) controlTarget(ctx context.Context, query string, spec mutationSpec, confirm func(string) bool, output io.Writer) error {
	name, err := requireName(query, spec.targetType)
	name, err = assistant.resolveTarget(ctx, query, spec.targetType, name, err)
	if err != nil {
		return fmt.Errorf("say which %s to %s, for example: %s %s media", spec.targetType, spec.action, spec.action, spec.targetType)
	}
	if err := assistant.validateTarget(ctx, spec.targetType, name); err != nil {
		return err
	}
	fmt.Fprintf(output, "Preview: %s %s %q via %s\n", spec.action, spec.targetType, name, mutationEndpoint(spec, name))
	phrase := fmt.Sprintf("confirm %s %s %s", spec.action, spec.targetType, name)
	if confirm == nil || !confirm(phrase) {
		fmt.Fprintf(output, "Canceled: %s %s %s.\n", spec.action, spec.targetType, name)
		return nil
	}
	if err := assistant.validateTarget(ctx, spec.targetType, name); err != nil {
		return fmt.Errorf("target changed after confirmation: %w", err)
	}
	result, err := assistant.mutate(ctx, spec.targetType, name, spec.action)
	if err != nil {
		return err
	}
	assistant.setLastTarget(spec.targetType, name)
	fmt.Fprintf(output, "%s %s %s completed.\n", spec.targetType, spec.action, name)
	if len(result) > 0 {
		fmt.Fprint(output, formatCorePayload("Result", result, assistant.jsonOutput))
	}
	return nil
}

func (assistant *Assistant) setLastTarget(kind, name string) {
	assistant.mu.Lock()
	defer assistant.mu.Unlock()
	assistant.lastTarget = &namedTarget{kind, name}
}

func (assistant *Assistant) clearLastTarget() {
	assistant.mu.Lock()
	defer assistant.mu.Unlock()
	assistant.lastTarget = nil
}

func (assistant *Assistant) getLastTarget(kind string) (string, bool) {
	assistant.mu.Lock()
	defer assistant.mu.Unlock()
	if assistant.lastTarget != nil && assistant.lastTarget.kind == kind {
		return assistant.lastTarget.name, true
	}
	return "", false
}

func (assistant *Assistant) resolveTarget(ctx context.Context, query, kind, name string, err error) (string, error) {
	if err == nil && (name == "it" || name == "that" || name == "this") {
		err = fmt.Errorf("pronoun target")
		name = ""
	}
	if err == nil {
		return name, nil
	}
	if candidate, ok := assistant.getLastTarget(kind); ok {
		if validateErr := assistant.validateTarget(ctx, kind, candidate); validateErr == nil {
			return candidate, nil
		}
		assistant.clearLastTarget()
	}
	_ = query
	return "", err
}

// ToolStep is a typed step proposed by the local model. It is always validated
// by the assistant before any API request is made.
type ToolStep struct {
	Tool   string `json:"tool"`
	Target string `json:"target,omitempty"`
}

type toolPlan struct {
	Steps []ToolStep `json:"steps"`
}

func looksMultiStep(query string) bool {
	lower := strings.ToLower(query)
	if !strings.Contains(lower, " and ") && !strings.Contains(lower, " then ") && !strings.Contains(lower, " after that ") {
		return false
	}
	for _, cue := range []string{"stats", "health", "container", "logs", "stack", "service", "start", "stop", "restart"} {
		if strings.Contains(lower, cue) {
			return true
		}
	}
	return false
}

var (
	stepKeyRegex               = regexp.MustCompile(`"(?:step)+s*":`)
	trailingDanglingQuoteRegex = regexp.MustCompile(`,\s*"+\s*([}\]])`)
)

func (assistant *Assistant) proposePlan(query string) (toolPlan, error) {
	prompt := "Plan the user's request using only these tool names: stats, health, containers, logs, start_stack, stop_stack, restart_stack, start_container, stop_container, restart_container, start_service, stop_service, restart_service. Return JSON with a steps array; each item has tool and optional target fields. Do not include prose. User request: " + query + "\nPLAN:"
	generated, err := assistant.languageModel.Generate(prompt, 180, 0.05, 43)
	if err != nil {
		return toolPlan{}, err
	}
	return parsePlanJSON(generated)
}

func extractFirstJSONObject(s string) (string, error) {
	s = trailingDanglingQuoteRegex.ReplaceAllString(s, "$1")
	start := strings.Index(s, "{")
	if start < 0 {
		return "", fmt.Errorf("model did not return a JSON plan")
	}
	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if !inString {
			if c == '{' {
				depth++
			} else if c == '}' {
				depth--
				if depth == 0 {
					return s[start : i+1], nil
				}
			}
		}
	}
	return "", fmt.Errorf("model did not return a complete JSON plan")
}

func parsePlanJSON(generated string) (toolPlan, error) {
	rawJSON, err := extractFirstJSONObject(generated)
	if err == nil {
		cleaned := stepKeyRegex.ReplaceAllString(rawJSON, `"steps":`)
		decoder := json.NewDecoder(strings.NewReader(cleaned))
		decoder.DisallowUnknownFields()
		var plan toolPlan
		if decodeErr := decoder.Decode(&plan); decodeErr == nil {
			var trailing any
			if trailingErr := decoder.Decode(&trailing); trailingErr == io.EOF {
				if validatePlan(plan) == nil {
					return plan, nil
				}
			}
		}
	}
	recovered := recoverToolSteps(generated)
	if len(recovered) > 0 {
		plan := toolPlan{Steps: recovered}
		if valErr := validatePlan(plan); valErr == nil {
			return plan, nil
		}
	}
	if err != nil {
		return toolPlan{}, err
	}
	return toolPlan{}, fmt.Errorf("model did not return a valid tool plan")
}

func validatePlan(plan toolPlan) error {
	if len(plan.Steps) == 0 || len(plan.Steps) > 5 {
		return fmt.Errorf("plan must have one to five steps")
	}
	for _, step := range plan.Steps {
		if !knownTool(step.Tool) {
			return fmt.Errorf("unknown tool %q", step.Tool)
		}
		if toolRequiresTarget(step.Tool) && strings.TrimSpace(step.Target) == "" {
			return fmt.Errorf("tool %q requires a target", step.Tool)
		}
	}
	return nil
}

var stepObjectRegex = regexp.MustCompile(`\{[^{}]*"tool"[^{}]*\}`)

func recoverToolSteps(raw string) []ToolStep {
	matches := stepObjectRegex.FindAllString(raw, -1)
	if len(matches) == 0 {
		return nil
	}
	var steps []ToolStep
	for _, m := range matches {
		var step ToolStep
		if err := json.Unmarshal([]byte(m), &step); err == nil && knownTool(step.Tool) {
			if !toolRequiresTarget(step.Tool) || strings.TrimSpace(step.Target) != "" {
				steps = append(steps, step)
			}
		}
	}
	return steps
}

func knownTool(tool string) bool {
	if tool == "stats" || tool == "health" || tool == "containers" || tool == "logs" {
		return true
	}
	_, ok := mutationToolSpec(tool)
	return ok
}

func toolRequiresTarget(tool string) bool { return tool == "logs" || strings.Contains(tool, "_") }

func mutationToolSpec(tool string) (mutationSpec, bool) {
	switch tool {
	case "start_stack":
		return mutationSpec{"stack", "start"}, true
	case "stop_stack":
		return mutationSpec{"stack", "stop"}, true
	case "restart_stack":
		return mutationSpec{"stack", "restart"}, true
	case "start_container":
		return mutationSpec{"container", "start"}, true
	case "stop_container":
		return mutationSpec{"container", "stop"}, true
	case "restart_container":
		return mutationSpec{"container", "restart"}, true
	case "start_service":
		return mutationSpec{"service", "start"}, true
	case "stop_service":
		return mutationSpec{"service", "stop"}, true
	case "restart_service":
		return mutationSpec{"service", "restart"}, true
	default:
		return mutationSpec{}, false
	}
}

func (assistant *Assistant) executePlan(ctx context.Context, plan toolPlan, confirm func(string) bool, output io.Writer) error {
	for _, step := range plan.Steps {
		if spec, ok := mutationToolSpec(step.Tool); ok {
			if err := assistant.validateTarget(ctx, spec.targetType, step.Target); err != nil {
				return err
			}
		}
	}
	for _, step := range plan.Steps {
		switch step.Tool {
		case "stats":
			if err := assistant.printCoreData(ctx, "/api/v2/system/stats", "M3TAL system stats", output); err != nil {
				return err
			}
		case "health":
			if err := assistant.printCoreData(ctx, "/api/v2/system/health", "M3TAL health", output); err != nil {
				return err
			}
		case "containers":
			if step.Target != "" {
				if err := assistant.printCoreContainers(ctx, step.Target, output); err != nil {
					return err
				}
			} else {
				if err := assistant.printCoreData(ctx, "/api/v2/services", "M3TAL containers and services", output); err != nil {
					return err
				}
			}
		case "logs":
			if err := assistant.printCoreData(ctx, "/api/v2/containers/"+url.PathEscape(step.Target)+"/logs?tail=100", "Recent logs for "+step.Target, output); err != nil {
				return err
			}
			assistant.setLastTarget("container", step.Target)
		default:
			spec, ok := mutationToolSpec(step.Tool)
			if !ok {
				return fmt.Errorf("rejected unknown tool %q", step.Tool)
			}
			fmt.Fprintf(output, "Preview: %s %s %q via %s\n", spec.action, spec.targetType, step.Target, mutationEndpoint(spec, step.Target))
			phrase := fmt.Sprintf("confirm %s %s %s", spec.action, spec.targetType, step.Target)
			if confirm == nil || !confirm(phrase) {
				fmt.Fprintf(output, "Canceled: %s %s %s. Remaining plan steps were not run.\n", spec.action, spec.targetType, step.Target)
				return nil
			}
			if err := assistant.validateTarget(ctx, spec.targetType, step.Target); err != nil {
				return fmt.Errorf("target changed after confirmation: %w", err)
			}
			result, err := assistant.mutate(ctx, spec.targetType, step.Target, spec.action)
			if err != nil {
				return err
			}
			assistant.setLastTarget(spec.targetType, step.Target)
			fmt.Fprintf(output, "%s %s %s completed.\n", spec.targetType, spec.action, step.Target)
			if len(result) > 0 {
				fmt.Fprint(output, formatCorePayload("Result", result, assistant.jsonOutput))
			}
		}
	}
	return nil
}

func (assistant *Assistant) validateTarget(ctx context.Context, kind, name string) error {
	path := map[string]string{"stack": "/api/v2/stacks", "container": "/api/v2/services", "service": "/api/v2/system/services"}[kind]
	data, err := assistant.core.get(ctx, path)
	if err != nil {
		return fmt.Errorf("cannot validate %s allowlist: %w", kind, err)
	}
	found := false
	var candidates []string
	switch kind {
	case "stack":
		var entries []struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(data, &entries); err != nil {
			return fmt.Errorf("decode stack allowlist: %w", err)
		}
		for _, entry := range entries {
			if entry.Name == name {
				found = true
			}
			if strings.Contains(strings.ToLower(entry.Name), strings.ToLower(name)) || strings.Contains(strings.ToLower(name), strings.ToLower(entry.Name)) {
				candidates = append(candidates, entry.Name)
			}
		}
	case "container":
		var entries []struct {
			Names []string `json:"names"`
		}
		if err := json.Unmarshal(data, &entries); err != nil {
			return fmt.Errorf("decode container allowlist: %w", err)
		}
		for _, entry := range entries {
			for _, candidate := range entry.Names {
				clean := strings.TrimPrefix(candidate, "/")
				if clean == name {
					found = true
				}
				if strings.Contains(strings.ToLower(clean), strings.ToLower(name)) || strings.Contains(strings.ToLower(name), strings.ToLower(clean)) {
					candidates = append(candidates, clean)
				}
			}
		}
	case "service":
		var entries []struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(data, &entries); err != nil {
			return fmt.Errorf("decode service allowlist: %w", err)
		}
		for _, entry := range entries {
			if entry.Name == name {
				found = true
			}
			if strings.Contains(strings.ToLower(entry.Name), strings.ToLower(name)) || strings.Contains(strings.ToLower(name), strings.ToLower(entry.Name)) {
				candidates = append(candidates, entry.Name)
			}
		}
	}
	if !found {
		if len(candidates) > 0 {
			return fmt.Errorf("%s %q not found; did you mean one of: %s?", kind, name, strings.Join(candidates, ", "))
		}
		return fmt.Errorf("%s %q is not present in M3TAL Core's current allowlist", kind, name)
	}
	return nil
}

func (assistant *Assistant) mutate(ctx context.Context, kind, name, action string) (json.RawMessage, error) {
	if kind == "stack" {
		if action == "restart" {
			if _, err := assistant.core.stackAction(ctx, name, "down"); err != nil {
				return nil, fmt.Errorf("restart %s failed during stop: %w", name, err)
			}
			result, err := assistant.core.stackAction(ctx, name, "up")
			if err != nil {
				return nil, fmt.Errorf("restart %s stopped the stack but failed during start: %w", name, err)
			}
			return result, nil
		}
		if action == "start" {
			action = "up"
		} else {
			action = "down"
		}
		return assistant.core.stackAction(ctx, name, action)
	}
	path := "/api/v2/containers/"
	if kind == "service" {
		path = "/api/v2/system/services/"
	}
	data, err := assistant.core.post(ctx, path+url.PathEscape(name)+"/"+action)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (assistant *Assistant) printKnowledge(query string, output io.Writer) error {
	// Retrieved document text is untrusted reference data: it is printed for the
	// user and is never interpreted as a command, policy, or tool instruction.
	if isHubAppQuestion(query) {
		apps := assistant.knowledge.Apps()
		if len(apps) > 0 {
			fmt.Fprintf(output, "M3TAL-Hub registers %d applications:\n", len(apps))
			for _, app := range apps {
				fmt.Fprintf(output, "\n- %s", app.Name)
				if app.Status != "" {
					fmt.Fprintf(output, " [%s]", app.Status)
				}
				if app.Description != "" {
					fmt.Fprintf(output, " — %s", app.Description)
				}
				fmt.Fprintln(output)
			}
			return nil
		}
	}
	results := assistant.knowledge.Search(query, 3)
	if len(results) == 0 {
		fmt.Fprintln(output, "I couldn't find a matching M3TAL document. Add docs with --docs or set M3TAL_DOCS_PATHS.")
		return nil
	}
	if assistant.languageModel != nil {
		draft, err := assistant.draftKnowledgeAnswer(query, results)
		if err != nil {
			return err
		}
		if draftStatesUncertainty(draft) {
			fmt.Fprintf(output, "Local model found no answer in retrieved sources:\n%s\n\n", strings.TrimSpace(draft))
			printKnowledgeExcerpts(results, output)
			return nil
		}
		if draftCitesSources(draft, results) {
			printGroundedAnswer(draft, results, output)
			return nil
		}
		fmt.Fprintln(output, "The local model draft was not grounded in retrieved sources; showing excerpts instead.")
	}
	printKnowledgeExcerpts(results, output)
	return nil
}

func (assistant *Assistant) draftKnowledgeAnswer(query string, results []KnowledgeResult) (string, error) {
	var prompt strings.Builder
	prompt.WriteString("Question: ")
	prompt.WriteString(query)
	prompt.WriteString("\nReference notes:\n")
	for i, result := range results {
		title := result.Title
		if title == "" {
			title = filepath.Base(result.Source)
		}
		excerpt := strings.TrimSpace(result.Excerpt)
		if len(excerpt) > 400 {
			excerpt = excerpt[:400] + "..."
		}
		fmt.Fprintf(&prompt, "[%d] %s (%s)\n%s\n", i+1, title, result.Source, excerpt)
	}
	prompt.WriteString("Answer using the reference notes and cite sources with [1], [2], or title. If they do not contain the answer, say so.\nAnswer:")
	answer, err := assistant.languageModel.Generate(prompt.String(), 120, 0.2, 31)
	if err != nil {
		return "", fmt.Errorf("generate local answer: %w", err)
	}
	return strings.TrimSpace(strings.TrimPrefix(answer, prompt.String())), nil
}

func printKnowledgeExcerpts(results []KnowledgeResult, output io.Writer) {
	fmt.Fprintln(output, "M3TAL knowledge matches:")
	for _, result := range results {
		fmt.Fprintf(output, "\n%s: ", result.Kind)
		if result.Title != "" {
			fmt.Fprintf(output, "%s — %s\n", result.Title, result.Source)
		} else {
			fmt.Fprintf(output, "%s\n", result.Source)
		}
		if result.Freshness.Label != "" {
			fmt.Fprintf(output, "Freshness check: %s — %s\n", result.Freshness.Label, result.Freshness.Detail)
		}
		fmt.Fprintln(output, result.Excerpt)
	}
}

func isHubAppQuestion(query string) bool {
	query = strings.ToLower(query)
	hasApps := strings.Contains(query, "app") || strings.Contains(query, "project")
	hasEcosystem := strings.Contains(query, "hub") || strings.Contains(query, "ecosystem") || strings.Contains(query, "m3tal")
	tasksForList := strings.Contains(query, "what") || strings.Contains(query, "which") || strings.Contains(query, "list") || strings.Contains(query, "show")
	return hasApps && hasEcosystem && tasksForList
}

// Ensure the real language model satisfies Generator.
var _ Generator = (*lm.Model)(nil)
