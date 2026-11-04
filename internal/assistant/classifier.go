package assistant

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"m3tal-ai/internal/nn"
	"m3tal-ai/internal/tensor"
)

type intent string

const (
	intentDocs             intent = "docs"
	intentStats            intent = "stats"
	intentHealth           intent = "health"
	intentContainers       intent = "containers"
	intentLogs             intent = "logs"
	intentStartStack       intent = "start_stack"
	intentStopStack        intent = "stop_stack"
	intentRestart          intent = "restart_stack"
	intentStartContainer   intent = "start_container"
	intentStopContainer    intent = "stop_container"
	intentRestartContainer intent = "restart_container"
	intentStartService     intent = "start_service"
	intentStopService      intent = "stop_service"
	intentRestartService   intent = "restart_service"
	intentUnknown          intent = "unknown"
)

type example struct {
	phrase string
	intent intent
}

var trainingExamples = []example{
	{"show system stats", intentStats}, {"how is the server doing", intentStats}, {"show cpu and memory usage", intentStats},
	{"check host resource usage", intentStats}, {"what is the system uptime", intentStats}, {"show disk usage", intentStats},
	{"how loaded is the host", intentStats}, {"get m3tal system metrics", intentStats}, {"show hardware telemetry", intentStats},
	{"check system health", intentHealth}, {"is m3tal healthy", intentHealth}, {"show component health status", intentHealth},
	{"are all system services healthy", intentHealth}, {"check the api daemon health", intentHealth}, {"run a health check", intentHealth},
	{"show me system status", intentHealth}, {"is the server okay", intentHealth}, {"which components are unhealthy", intentHealth},
	{"list running containers", intentContainers}, {"show docker containers", intentContainers}, {"what containers are running", intentContainers},
	{"list m3tal services", intentContainers}, {"show container status", intentContainers}, {"what is running in docker", intentContainers},
	{"get the container list", intentContainers}, {"show all active services", intentContainers}, {"which containers are stopped", intentContainers},
	{"is jellyfin running", intentContainers}, {"status of jellyfin", intentContainers}, {"is traefik running", intentContainers},
	{"check container jellyfin", intentContainers}, {"is container jellyfin running", intentContainers}, {"check if jellyfin is running", intentContainers},
	{"show container logs", intentLogs}, {"get logs for jellyfin", intentLogs}, {"view docker logs", intentLogs},
	{"what do the service logs say", intentLogs}, {"show me logs from a container", intentLogs}, {"read the logs for traefik", intentLogs},
	{"why is this container failing check its logs", intentLogs}, {"fetch recent logs", intentLogs}, {"container log output", intentLogs},
	{"how do i install m3tal", intentDocs}, {"where is the m3tal config file", intentDocs}, {"explain the traefik setup", intentDocs},
	{"how does m3tal core work", intentDocs}, {"what apps are in m3tal hub", intentDocs}, {"tell me about monster lab", intentDocs},
	{"how do i configure the api token", intentDocs}, {"where can i find the documentation", intentDocs}, {"help me set up m3tal", intentDocs},
	{"how do i expose an app through the proxy", intentDocs}, {"what does the doctor command do", intentDocs}, {"how are apps deployed to the hub", intentDocs},
	{"explain docker in m3tal", intentDocs}, {"where are docker compose files stored", intentDocs}, {"how does docker work", intentDocs},
	{"tell me about docker stacks and containers", intentDocs}, {"how does the m3tal-docker plugin work", intentDocs},
	{"start the media stack", intentStartStack}, {"bring up the traefik stack", intentStartStack}, {"deploy stack jellyfin", intentStartStack},
	{"start stack downloads", intentStartStack}, {"turn on the homelab stack", intentStartStack}, {"launch stack media", intentStartStack},
	{"stop the media stack", intentStopStack}, {"bring down stack traefik", intentStopStack}, {"shut down stack jellyfin", intentStopStack},
	{"stop stack downloads", intentStopStack}, {"turn off the homelab stack", intentStopStack}, {"take down stack media", intentStopStack},
	{"restart the media stack", intentRestart}, {"restart stack traefik", intentRestart}, {"reboot stack jellyfin", intentRestart},
	{"restart stack downloads", intentRestart}, {"cycle the homelab stack", intentRestart}, {"restart stack media", intentRestart},
	{"start container jellyfin", intentStartContainer}, {"start the jellyfin container", intentStartContainer}, {"start container traefik", intentStartContainer},
	{"stop container jellyfin", intentStopContainer}, {"stop the jellyfin container", intentStopContainer}, {"stop container traefik", intentStopContainer},
	{"restart container jellyfin", intentRestartContainer}, {"restart the jellyfin container", intentRestartContainer}, {"restart container traefik", intentRestartContainer},
	{"start system service m3tal-api", intentStartService}, {"start service docker", intentStartService}, {"start the m3tal-api service", intentStartService},
	{"stop system service m3tal-api", intentStopService}, {"stop service docker", intentStopService}, {"stop the m3tal-api service", intentStopService},
	{"restart system service m3tal-api", intentRestartService}, {"restart service docker", intentRestartService}, {"restart the m3tal-api service", intentRestartService},
}

type classifier struct {
	model      *nn.Sequential
	vocabulary map[string]int
	intents    []intent
}

func newClassifier() (*classifier, error) {
	vocabulary := make(map[string]struct{})
	intentSet := make(map[intent]struct{})
	for _, sample := range trainingExamples {
		for _, feature := range features(sample.phrase) {
			vocabulary[feature] = struct{}{}
		}
		intentSet[sample.intent] = struct{}{}
	}
	words := make([]string, 0, len(vocabulary))
	for word := range vocabulary {
		words = append(words, word)
	}
	sort.Strings(words)
	wordIndex := make(map[string]int, len(words))
	for index, word := range words {
		wordIndex[word] = index
	}
	classes := make([]intent, 0, len(intentSet))
	for value := range intentSet {
		classes = append(classes, value)
	}
	sort.Slice(classes, func(i, j int) bool { return classes[i] < classes[j] })
	classIndex := make(map[intent]int, len(classes))
	for index, value := range classes {
		classIndex[value] = index
	}
	if cached, ok := loadClassifierCache(wordIndex, classes); ok {
		return cached, nil
	}

	inputs := make([]float64, len(trainingExamples)*len(words))
	targets := make([]float64, len(trainingExamples)*len(classes))
	for row, sample := range trainingExamples {
		for _, feature := range features(sample.phrase) {
			if index, found := wordIndex[feature]; found {
				inputs[row*len(words)+index] = 1
			}
		}
		targets[row*len(classes)+classIndex[sample.intent]] = 1
	}
	inputMatrix, err := tensor.NewMatrix(len(trainingExamples), len(words), inputs)
	if err != nil {
		return nil, err
	}
	targetMatrix, err := tensor.NewMatrix(len(trainingExamples), len(classes), targets)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(17))
	hidden, err := nn.NewDense(len(words), 16, rng)
	if err != nil {
		return nil, err
	}
	output, err := nn.NewDense(16, len(classes), rng)
	if err != nil {
		return nil, err
	}
	model := nn.NewSequential(hidden, &nn.ReLULayer{}, output)
	_, err = (nn.Trainer{Epochs: 300, LearningRate: 0.08, Loss: nn.SoftmaxCrossEntropy{}}).Fit(model, inputMatrix, targetMatrix)
	if err != nil {
		return nil, err
	}
	trained := &classifier{model: model, vocabulary: wordIndex, intents: classes}
	saveClassifierCache(trained)
	return trained, nil
}

type classifierCache struct {
	Signature  string    `json:"signature"`
	Vocabulary []string  `json:"vocabulary"`
	Intents    []string  `json:"intents"`
	Values     []float64 `json:"values"`
}

func classifierSignature() string {
	hash := sha256.New()
	for _, sample := range trainingExamples {
		fmt.Fprintf(hash, "%s\t%s\n", sample.phrase, sample.intent)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func classifierCachePath() string {
	if override := strings.TrimSpace(os.Getenv("M3TAL_CLASSIFIER_CACHE")); override != "" {
		if strings.EqualFold(override, "off") {
			return ""
		}
		return override
	}
	dir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "m3tal-ai", "intent-classifier.json")
}

func loadClassifierCache(vocabulary map[string]int, intents []intent) (*classifier, bool) {
	path := classifierCachePath()
	if path == "" {
		return nil, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var state classifierCache
	if err := json.Unmarshal(data, &state); err != nil || state.Signature != classifierSignature() {
		return nil, false
	}
	if len(state.Vocabulary) != len(vocabulary) || len(state.Intents) != len(intents) {
		return nil, false
	}
	for index, word := range state.Vocabulary {
		if vocabulary[word] != index {
			return nil, false
		}
	}
	for index, name := range state.Intents {
		if intents[index] != intent(name) {
			return nil, false
		}
	}
	rng := rand.New(rand.NewSource(17))
	hidden, err := nn.NewDense(len(state.Vocabulary), 16, rng)
	if err != nil {
		return nil, false
	}
	output, err := nn.NewDense(16, len(state.Intents), rng)
	if err != nil {
		return nil, false
	}
	model := nn.NewSequential(hidden, &nn.ReLULayer{}, output)
	offset := 0
	for _, parameter := range model.Parameters() {
		end := offset + len(parameter.Values())
		if end > len(state.Values) {
			return nil, false
		}
		if err := parameter.Load(state.Values[offset:end]); err != nil {
			return nil, false
		}
		offset = end
	}
	if offset != len(state.Values) {
		return nil, false
	}
	return &classifier{model: model, vocabulary: vocabulary, intents: intents}, true
}

func saveClassifierCache(model *classifier) {
	path := classifierCachePath()
	if path == "" {
		return
	}
	state := classifierCache{Signature: classifierSignature()}
	words := make([]string, len(model.vocabulary))
	for word, index := range model.vocabulary {
		words[index] = word
	}
	state.Vocabulary = words
	for _, value := range model.intents {
		state.Intents = append(state.Intents, string(value))
	}
	for _, parameter := range model.model.Parameters() {
		state.Values = append(state.Values, parameter.Values()...)
	}
	data, err := json.Marshal(state)
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return
	}
	_ = os.WriteFile(path, data, 0o600)
}

func (c *classifier) classify(query string) (intent, float64, error) {
	if selected, decisive := deterministicIntent(query); decisive {
		return selected, 1, nil
	}
	values := make([]float64, len(c.vocabulary))
	known := 0
	for _, feature := range features(query) {
		if index, found := c.vocabulary[feature]; found {
			values[index] = 1
			known++
		}
	}
	if known == 0 {
		return intentUnknown, 0, nil
	}
	input, err := tensor.NewMatrix(1, len(values), values)
	if err != nil {
		return "", 0, err
	}
	logits, err := c.model.Forward(input)
	if err != nil {
		return "", 0, err
	}
	maximum := math.Inf(-1)
	for col := 0; col < logits.Cols(); col++ {
		if logits.At(0, col) > maximum {
			maximum = logits.At(0, col)
		}
	}
	var sum float64
	for col := 0; col < logits.Cols(); col++ {
		sum += math.Exp(logits.At(0, col) - maximum)
	}
	bestIndex := 0
	bestProbability := 0.0
	for col := 0; col < logits.Cols(); col++ {
		probability := math.Exp(logits.At(0, col)-maximum) / sum
		if probability > bestProbability {
			bestIndex = col
			bestProbability = probability
		}
	}
	if bestProbability < 0.55 {
		return intentUnknown, bestProbability, nil
	}
	return c.intents[bestIndex], bestProbability, nil
}

func deterministicIntent(query string) (intent, bool) {
	tokens := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
	actions, targets := map[string]bool{}, map[string]bool{}
	readOnlyCue := false
	negated := mutationNeedsClarification(query)
	for _, word := range tokens {
		if word == "not" || word == "never" || word == "dont" || word == "cannot" || word == "cant" || word == "shouldnt" || word == "avoid" {
			negated = true
		}
		if word == "start" || word == "stop" || word == "restart" {
			actions[word] = true
		}
		if word == "stack" {
			targets["stack"] = true
		}
		if word == "container" || word == "containers" {
			targets["container"] = true
		}
		if word == "service" || word == "services" || word == "systemd" {
			targets["service"] = true
		}
		if word == "logs" || word == "log" || word == "stats" || word == "metrics" || word == "health" || word == "healthy" || word == "doctor" {
			readOnlyCue = true
		}
	}
	if negated && len(actions) > 0 && len(targets) > 0 {
		return intentUnknown, true
	}
	if len(actions) > 1 || len(targets) > 1 || len(actions) == 1 && len(targets) == 1 && readOnlyCue {
		return intentUnknown, true
	}
	if selected, decisive := deterministicMutationIntent(query); decisive {
		return selected, true
	}
	words := make(map[string]bool)
	for _, word := range strings.FieldsFunc(strings.ToLower(query), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) }) {
		words[word] = true
	}
	matches := make(map[intent]bool)
	if words["log"] || words["logs"] {
		matches[intentLogs] = true
	}
	if (words["container"] || words["containers"]) && (words["list"] || words["show"] || words["running"] || words["status"] || words["which"]) ||
		(words["running"] && (words["is"] || words["check"] || words["what"])) ||
		(words["status"] && (words["of"] || words["container"] || words["containers"] || words["service"] || words["services"]) && !words["system"]) {
		matches[intentContainers] = true
	}
	if words["health"] || words["healthy"] || words["unhealthy"] || words["doctor"] || (words["system"] && words["status"]) {
		matches[intentHealth] = true
	}
	if words["stats"] || words["metrics"] || words["cpu"] || words["memory"] || words["disk"] || words["uptime"] || words["telemetry"] {
		matches[intentStats] = true
	}
	if words["docs"] || words["documentation"] || words["install"] || words["installation"] || words["configure"] || words["configuration"] || words["explain"] || words["guide"] || words["manual"] || (words["where"] && (words["file"] || words["files"] || words["config"] || words["stored"] || words["docs"])) || ((words["how"] || words["what"]) && (words["work"] || words["works"] || words["architected"] || words["architecture"])) {
		matches[intentDocs] = true
	}
	if len(matches) != 1 {
		return "", false
	}
	for selected := range matches {
		return selected, true
	}
	return "", false
}

func deterministicMutationIntent(query string) (intent, bool) {
	words := strings.FieldsFunc(strings.ToLower(query), func(character rune) bool {
		return !unicode.IsLetter(character) && !unicode.IsNumber(character) && character != '_' && character != '-'
	})
	actions := map[string]string{}
	targets := map[string]bool{}
	for _, word := range words {
		if word == "start" || word == "stop" || word == "restart" {
			actions[word] = word
		}
		if word == "stack" {
			targets["stack"] = true
		}
		if word == "container" || word == "containers" {
			targets["container"] = true
		}
		if word == "service" || word == "services" || word == "systemd" {
			targets["service"] = true
		}
	}
	if len(actions) > 0 && len(targets) > 0 && mutationNeedsClarification(query) {
		return intentUnknown, true
	}
	if len(actions) != 1 || len(targets) != 1 {
		return "", false
	}
	action := ""
	for candidate := range actions {
		action = candidate
	}
	target := ""
	for candidate := range targets {
		target = candidate
	}
	for selected, spec := range mutationPolicy {
		if spec.action == action && spec.targetType == target {
			return selected, true
		}
	}
	return "", false
}

func mutationNeedsClarification(query string) bool {
	lower := strings.ToLower(strings.TrimSpace(query))
	for _, phrase := range []string{"don't", "do not", "never", "cannot", "can't", "shouldn't", "should not", "avoid", "why ", "should i ", "could i ", "can i ", "how to "} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return strings.Contains(lower, "?")
}

func features(text string) []string {
	words := strings.FieldsFunc(strings.ToLower(text), func(character rune) bool {
		return !unicode.IsLetter(character) && !unicode.IsNumber(character) && character != '_' && character != '-'
	})
	result := append([]string(nil), words...)
	for index := 0; index+1 < len(words); index++ {
		result = append(result, words[index]+" "+words[index+1])
	}
	return result
}

func (i intent) String() string { return string(i) }

func requireName(query, marker string) (string, error) {
	words := strings.Fields(strings.ToLower(query))
	clean := func(word string) string { return strings.Trim(word, " \t\n\r.,!?;:()[]{}\"'") }
	for i := range words {
		words[i] = clean(words[i])
	}
	for index, word := range words {
		if word == marker && index+1 < len(words) {
			name := words[index+1]
			if name == "the" || name == "a" || name == "an" {
				if index+2 < len(words) {
					return words[index+2], nil
				}
				break
			}
			return name, nil
		}
		if word == marker && index > 0 {
			name := words[index-1]
			if name != "the" && name != "a" && name != "an" {
				return name, nil
			}
		}
	}
	return "", fmt.Errorf("please include a stack name, for example: restart stack media")
}
