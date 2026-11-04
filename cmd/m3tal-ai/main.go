package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"m3tal-ai/internal/assistant"
	"m3tal-ai/internal/lm"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "m3tal-ai:", err)
		os.Exit(1)
	}
}

func run(arguments []string) error {
	flags := flag.NewFlagSet("m3tal-ai", flag.ContinueOnError)
	docsFlag := flags.String("docs", "", "additional documentation file or directory (path-list separated)")
	apiURL := flags.String("api", os.Getenv("M3TAL_API_URL"), "M3TAL Core API URL (default http://localhost:5050)")
	modelPath := flags.String("model", os.Getenv("M3TAL_MODEL_PATH"), "optional local language-model checkpoint")
	interactive := flags.Bool("interactive", false, "run an interactive assistant session")
	jsonOutput := flags.Bool("json", false, "print Core payloads as JSON")
	showVersion := flags.Bool("version", false, "print version and exit")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *showVersion {
		fmt.Printf("m3tal-ai %s\n", assistant.Version)
		return nil
	}
	query := strings.TrimSpace(strings.Join(flags.Args(), " "))
	if query == "" && !*interactive {
		return fmt.Errorf("usage: go run ./cmd/m3tal-ai [--docs PATH] [--api URL] [--model CHECKPOINT] [--interactive] [--json] [--version] \"question or M3TAL request\"")
	}

	docPaths := defaultDocsPaths()
	if environment := os.Getenv("M3TAL_DOCS_PATHS"); environment != "" {
		docPaths = filepath.SplitList(environment)
	}
	if *docsFlag != "" {
		docPaths = append(docPaths, filepath.SplitList(*docsFlag)...)
	}

	core := assistant.NewCoreAPI(*apiURL, os.Getenv("M3TAL_API_TOKEN"))
	agent, err := assistant.New(docPaths, core)
	if err != nil {
		return err
	}
	agent.SetJSONOutput(*jsonOutput)
	stats := agent.Knowledge().Stats()
	fmt.Fprintln(os.Stderr, agent.Knowledge().Report())
	if len(stats.MissingRoots) > 0 {
		fmt.Fprintf(os.Stderr, "warning: %d documentation root(s) not found\n", len(stats.MissingRoots))
	}
	if stats.Files == 0 && stats.Documents == 0 {
		fmt.Fprintln(os.Stderr, "warning: knowledge base is empty; add docs with --docs or set M3TAL_DOCS_PATHS")
	}
	if *modelPath != "" {
		if _, err := os.Stat(*modelPath); err != nil {
			return fmt.Errorf("model checkpoint not readable: %w", err)
		}
		model, err := lm.LoadCheckpoint(*modelPath)
		if err != nil {
			return fmt.Errorf("load local model: %w", err)
		}
		agent.SetLanguageModel(model)
		fmt.Fprintln(os.Stderr, "loaded local language-model checkpoint")
	}
	if err := agent.PingCore(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "warning: %v\n", err)
	}
	reader := bufio.NewReader(os.Stdin)
	confirm := func(action string) bool {
		fmt.Fprintf(os.Stderr, "Type exactly to confirm: %s\n> ", action)
		answer, _ := reader.ReadString('\n')
		return confirmationMatches(action, answer)
	}
	if !*interactive {
		return agent.Ask(context.Background(), query, confirm, os.Stdout)
	}
	for {
		fmt.Fprint(os.Stdout, "m3tal> ")
		line, readErr := reader.ReadString('\n')
		line = strings.TrimSpace(line)
		if readErr != nil && line == "" {
			return nil
		}
		if line == "exit" || line == "quit" {
			return nil
		}
		if line == "" {
			continue
		}
		if err := agent.Ask(context.Background(), line, confirm, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, "m3tal-ai:", err)
		}
		if readErr != nil {
			return nil
		}
	}
}

func confirmationMatches(expected, actual string) bool {
	return strings.TrimSpace(actual) == expected
}

func defaultDocsPaths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	appsRoot := filepath.Join(home, "apps")
	paths := []string{
		filepath.Join(appsRoot, "M3tal-core"),
		filepath.Join(appsRoot, "M3tal-Hub"),
		filepath.Join(appsRoot, "m3tal-plugins"),
	}
	manifest := filepath.Join(appsRoot, "M3tal-Hub", "apps", "manifest.yml")
	contents, err := os.ReadFile(manifest)
	if err != nil {
		return paths
	}
	folders := make(map[string]string)
	entries, _ := os.ReadDir(appsRoot)
	for _, entry := range entries {
		if entry.IsDir() {
			folders[strings.ToLower(entry.Name())] = filepath.Join(appsRoot, entry.Name())
		}
	}
	for _, line := range strings.Split(string(contents), "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "repo:") {
			continue
		}
		repository := strings.Trim(strings.TrimSpace(strings.TrimPrefix(trimmed, "repo:")), "\"'")
		if path, found := folders[strings.ToLower(repository)]; found {
			paths = append(paths, path)
		}
	}
	return paths
}
