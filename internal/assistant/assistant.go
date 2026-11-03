package assistant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strings"
)

// Assistant combines a small intent model, local documentation retrieval, and
// M3TAL Core tools.
type Assistant struct {
	classifier *classifier
	knowledge  *KnowledgeBase
	core       *CoreAPI
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

// Ask answers a question or operates M3TAL Core. confirm is called before any
// stack start, stop, or restart request; read-only requests do not call it.
func (assistant *Assistant) Ask(ctx context.Context, query string, confirm func(action string) bool, output io.Writer) error {
	if strings.TrimSpace(query) == "" {
		return fmt.Errorf("enter a question or M3TAL request")
	}
	if output == nil {
		output = io.Discard
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
		return assistant.printCoreData(ctx, "/api/v2/services", "M3TAL containers and services", output)
	case intentLogs:
		name, err := requireName(query, "for")
		if err != nil {
			name, err = requireName(query, "container")
		}
		if err != nil {
			return fmt.Errorf("tell me which container, for example: show logs for jellyfin")
		}
		return assistant.printCoreData(ctx, "/api/v2/containers/"+url.PathEscape(name)+"/logs?tail=100", "Recent logs for "+name, output)
	case intentStartStack, intentStopStack, intentRestart:
		return assistant.controlStack(ctx, query, selected, confirm, output)
	default:
		return assistant.printKnowledge(query, output)
	}
}

func (assistant *Assistant) printCoreData(ctx context.Context, path, heading string, output io.Writer) error {
	data, err := assistant.core.get(ctx, path)
	if err != nil {
		return err
	}
	var pretty bytes.Buffer
	if err := json.Indent(&pretty, data, "", "  "); err != nil {
		fmt.Fprintf(output, "%s\n%s\n", heading, string(data))
		return nil
	}
	fmt.Fprintf(output, "%s\n%s\n", heading, pretty.String())
	return nil
}

func (assistant *Assistant) controlStack(ctx context.Context, query string, selected intent, confirm func(string) bool, output io.Writer) error {
	name, err := requireName(query, "stack")
	if err != nil {
		return err
	}
	action := strings.TrimSuffix(string(selected), "_stack")
	if confirm == nil || !confirm(action+" stack "+name) {
		fmt.Fprintf(output, "Canceled: %s stack %s.\n", strings.TrimSuffix(action, "_stack"), name)
		return nil
	}
	var result json.RawMessage
	switch selected {
	case intentStartStack:
		result, err = assistant.core.stackAction(ctx, name, "up")
	case intentStopStack:
		result, err = assistant.core.stackAction(ctx, name, "down")
	case intentRestart:
		if _, err = assistant.core.stackAction(ctx, name, "down"); err == nil {
			result, err = assistant.core.stackAction(ctx, name, "up")
		}
	}
	if err != nil {
		return err
	}
	var pretty bytes.Buffer
	if json.Indent(&pretty, result, "", "  ") == nil && len(result) > 0 {
		fmt.Fprintf(output, "Stack %s %s completed.\n%s\n", action, name, pretty.String())
	} else {
		fmt.Fprintf(output, "Stack %s %s completed.\n", action, name)
	}
	return nil
}

func (assistant *Assistant) printKnowledge(query string, output io.Writer) error {
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
	return nil
}

func isHubAppQuestion(query string) bool {
	query = strings.ToLower(query)
	hasApps := strings.Contains(query, "app") || strings.Contains(query, "project")
	hasEcosystem := strings.Contains(query, "hub") || strings.Contains(query, "ecosystem") || strings.Contains(query, "m3tal")
	tasksForList := strings.Contains(query, "what") || strings.Contains(query, "which") || strings.Contains(query, "list") || strings.Contains(query, "show")
	return hasApps && hasEcosystem && tasksForList
}
