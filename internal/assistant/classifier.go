package assistant

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"unicode"

	"m3tal-ai/internal/nn"
	"m3tal-ai/internal/tensor"
)

type intent string

const (
	intentDocs       intent = "docs"
	intentStats      intent = "stats"
	intentHealth     intent = "health"
	intentContainers intent = "containers"
	intentLogs       intent = "logs"
	intentStartStack intent = "start_stack"
	intentStopStack  intent = "stop_stack"
	intentRestart    intent = "restart_stack"
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
	{"show container logs", intentLogs}, {"get logs for jellyfin", intentLogs}, {"view docker logs", intentLogs},
	{"what do the service logs say", intentLogs}, {"show me logs from a container", intentLogs}, {"read the logs for traefik", intentLogs},
	{"why is this container failing check its logs", intentLogs}, {"fetch recent logs", intentLogs}, {"container log output", intentLogs},
	{"how do i install m3tal", intentDocs}, {"where is the m3tal config file", intentDocs}, {"explain the traefik setup", intentDocs},
	{"how does m3tal core work", intentDocs}, {"what apps are in m3tal hub", intentDocs}, {"tell me about monster lab", intentDocs},
	{"how do i configure the api token", intentDocs}, {"where can i find the documentation", intentDocs}, {"help me set up m3tal", intentDocs},
	{"how do i expose an app through the proxy", intentDocs}, {"what does the doctor command do", intentDocs}, {"how are apps deployed to the hub", intentDocs},
	{"start the media stack", intentStartStack}, {"bring up the traefik stack", intentStartStack}, {"deploy stack jellyfin", intentStartStack},
	{"start stack downloads", intentStartStack}, {"turn on the homelab stack", intentStartStack}, {"launch stack media", intentStartStack},
	{"stop the media stack", intentStopStack}, {"bring down stack traefik", intentStopStack}, {"shut down stack jellyfin", intentStopStack},
	{"stop stack downloads", intentStopStack}, {"turn off the homelab stack", intentStopStack}, {"take down stack media", intentStopStack},
	{"restart the media stack", intentRestart}, {"restart stack traefik", intentRestart}, {"reboot stack jellyfin", intentRestart},
	{"restart stack downloads", intentRestart}, {"cycle the homelab stack", intentRestart}, {"restart stack media", intentRestart},
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
	return &classifier{model: model, vocabulary: wordIndex, intents: classes}, nil
}

func (c *classifier) classify(query string) (intent, float64, error) {
	values := make([]float64, len(c.vocabulary))
	known := 0
	for _, feature := range features(query) {
		if index, found := c.vocabulary[feature]; found {
			values[index] = 1
			known++
		}
	}
	if known == 0 {
		return intentDocs, 0, nil
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
	if bestProbability < 0.42 {
		return intentDocs, bestProbability, nil
	}
	return c.intents[bestIndex], bestProbability, nil
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
	words := strings.FieldsFunc(strings.ToLower(query), func(character rune) bool {
		return !unicode.IsLetter(character) && !unicode.IsNumber(character) && character != '-' && character != '_'
	})
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
		if word == marker && marker == "stack" && index > 0 {
			name := words[index-1]
			if name != "the" && name != "a" && name != "an" {
				return name, nil
			}
		}
	}
	return "", fmt.Errorf("please include a stack name, for example: restart stack media")
}
