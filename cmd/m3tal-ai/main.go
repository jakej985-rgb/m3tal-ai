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
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	query := strings.TrimSpace(strings.Join(flags.Args(), " "))
	if query == "" {
		return fmt.Errorf("usage: go run ./cmd/m3tal-ai [--docs PATH] [--api URL] \"question or M3TAL request\"")
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
	confirm := func(action string) bool {
		fmt.Fprintf(os.Stderr, "Confirm M3TAL Core action '%s'? [y/N] ", action)
		answer, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		return strings.EqualFold(strings.TrimSpace(answer), "y") || strings.EqualFold(strings.TrimSpace(answer), "yes")
	}
	return agent.Ask(context.Background(), query, confirm, os.Stdout)
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
