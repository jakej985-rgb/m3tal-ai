package lm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// InstructionExample is a prompt-completion pair for instruction tuning.
type InstructionExample struct {
	Prompt     string
	Completion string
	Category   string // "plan" or "rag"
}

// PlanPrompt formats a user query into the tool planning prompt used by Assistant.
func PlanPrompt(query string) string {
	return "Plan the user's request using only these tool names: stats, health, containers, logs, start_stack, stop_stack, restart_stack, start_container, stop_container, restart_container, start_service, stop_service, restart_service. Return JSON with a steps array; each item has tool and optional target fields. Do not include prose. User request: " + query + "\nPLAN:"
}

// RAGNote represents a retrieved document passage for grounding.
type RAGNote struct {
	Title   string
	Source  string
	Excerpt string
}

// RAGPrompt formats reference notes into the grounded Q&A prompt used by Assistant.
func RAGPrompt(query string, notes []RAGNote) string {
	var b strings.Builder
	b.WriteString("Question: ")
	b.WriteString(query)
	b.WriteString("\nReference notes:\n")
	for i, note := range notes {
		title := note.Title
		if title == "" {
			title = filepath.Base(note.Source)
		}
		fmt.Fprintf(&b, "[%d] %s (%s)\n%s\n", i+1, title, note.Source, note.Excerpt)
	}
	b.WriteString("Answer using the reference notes and cite sources with [1], [2], or title. If they do not contain the answer, say so.\nAnswer:")
	return b.String()
}

// DefaultInstructionSet returns self-owned instruction examples for tool planning and grounded Q&A.
func DefaultInstructionSet() []InstructionExample {
	return []InstructionExample{
		// Multi-step read-only operations
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show stats and health"),
			Completion: `{"steps":[{"tool":"stats"},{"tool":"health"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show stats then check containers"),
			Completion: `{"steps":[{"tool":"stats"},{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show stats and health and containers"),
			Completion: `{"steps":[{"tool":"stats"},{"tool":"health"},{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("check health then show stats"),
			Completion: `{"steps":[{"tool":"health"},{"tool":"stats"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("list containers and check system stats"),
			Completion: `{"steps":[{"tool":"containers"},{"tool":"stats"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show health and list containers"),
			Completion: `{"steps":[{"tool":"health"},{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("is jellyfin running"),
			Completion: `{"steps":[{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("status of jellyfin"),
			Completion: `{"steps":[{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("check container status"),
			Completion: `{"steps":[{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("what containers are running"),
			Completion: `{"steps":[{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show containers then stats"),
			Completion: `{"steps":[{"tool":"containers"},{"tool":"stats"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("check containers and health"),
			Completion: `{"steps":[{"tool":"containers"},{"tool":"health"}]}`,
		},

		// Multi-step operations with logs & targets
		{
			Category:   "plan",
			Prompt:     PlanPrompt("restart stack media and show logs"),
			Completion: `{"steps":[{"tool":"restart_stack","target":"media"},{"tool":"logs","target":"media"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("start container jellyfin and stop container traefik"),
			Completion: `{"steps":[{"tool":"start_container","target":"jellyfin"},{"tool":"stop_container","target":"traefik"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show logs for jellyfin and check stats"),
			Completion: `{"steps":[{"tool":"logs","target":"jellyfin"},{"tool":"stats"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("restart container traefik and show health"),
			Completion: `{"steps":[{"tool":"restart_container","target":"traefik"},{"tool":"health"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("stop stack media then start stack media"),
			Completion: `{"steps":[{"tool":"stop_stack","target":"media"},{"tool":"start_stack","target":"media"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("restart service core and check health"),
			Completion: `{"steps":[{"tool":"restart_service","target":"core"},{"tool":"health"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("stop container jellyfin then show containers"),
			Completion: `{"steps":[{"tool":"stop_container","target":"jellyfin"},{"tool":"containers"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("start stack monitoring after that show health"),
			Completion: `{"steps":[{"tool":"start_stack","target":"monitoring"},{"tool":"health"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("show logs for core and restart service core"),
			Completion: `{"steps":[{"tool":"logs","target":"core"},{"tool":"restart_service","target":"core"}]}`,
		},
		{
			Category:   "plan",
			Prompt:     PlanPrompt("stop service proxy then start service proxy"),
			Completion: `{"steps":[{"tool":"stop_service","target":"proxy"},{"tool":"start_service","target":"proxy"}]}`,
		},

		// Grounded Q&A positive examples citing [1]
		{
			Category: "rag",
			Prompt: RAGPrompt("how do I install m3tal", []RAGNote{
				{
					Title:   "Install",
					Source:  "/apps/M3tal-Hub/scripts/build_hub.py",
					Excerpt: "Run the install command to build and install M3TAL Hub.",
				},
			}),
			Completion: "According to [1], run the install command in build_hub.py to set up M3TAL Hub.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("where is the m3tal config file", []RAGNote{
				{
					Title:   "Configuration",
					Source:  "/apps/M3tal-core/docs/ENVIRONMENT_VARIABLES.md",
					Excerpt: "M3TAL environment variables and default configuration paths are documented here.",
				},
			}),
			Completion: "According to [1] (Configuration), configuration variables are defined in ENVIRONMENT_VARIABLES.md.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("explain the traefik setup", []RAGNote{
				{
					Title:   "Traefik",
					Source:  "/apps/M3tal-core/docs/TRAEFIK.md",
					Excerpt: "Traefik is deployed as the primary reverse proxy and SSL terminator for M3TAL services.",
				},
			}),
			Completion: "As explained in [1], Traefik serves as the primary reverse proxy and SSL terminator.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("how does m3tal core work", []RAGNote{
				{
					Title:   "Architecture",
					Source:  "/apps/M3tal-core/docs/ARCHITECTURE.md",
					Excerpt: "M3TAL Core provides the central daemon, REST API, container supervision, and state management.",
				},
			}),
			Completion: "According to [1], M3TAL Core provides the central daemon, REST API, and container supervision.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("how do I configure the api token", []RAGNote{
				{
					Title:   "API",
					Source:  "/apps/M3tal-core/docs/API.md",
					Excerpt: "Authenticate requests by setting the M3TAL_API_TOKEN environment variable or passing the token header.",
				},
			}),
			Completion: "According to [1], configure the API token by setting the M3TAL_API_TOKEN environment variable.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("what does the doctor command do", []RAGNote{
				{
					Title:   "Doctor",
					Source:  "/apps/M3tal-core/docs/DOCTOR.md",
					Excerpt: "The doctor command performs preflight checks, validates Docker networking, and reports configuration warnings.",
				},
			}),
			Completion: "According to [1], the doctor command runs preflight diagnostics and validates system configuration.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("explain docker in m3tal", []RAGNote{
				{
					Title:   "Docker",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "M3TAL uses Docker Engine and Docker Compose V2 as its foundational container runtime with stacks in /docker.",
				},
			}),
			Completion: "According to [1], M3TAL uses Docker Engine and Docker Compose V2 with stacks stored in /docker.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("where are docker compose files stored", []RAGNote{
				{
					Title:   "Stacks",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "M3TAL organizes container workloads in declarative Compose stacks located in /docker (symlinked from /opt/m3tal/stack).",
				},
			}),
			Completion: "According to [1], Docker compose files are stored in /docker (symlinked from /opt/m3tal/stack).",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("what containers run by default in m3tal", []RAGNote{
				{
					Title:   "Fleet",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "Default container fleet includes traefik for ingress routing, cloudflared for tunnels, ollama for AI inference, and jellyfin for media streaming.",
				},
			}),
			Completion: "According to [1], default containers include traefik, cloudflared, ollama, and jellyfin.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("how does the m3tal-docker plugin work", []RAGNote{
				{
					Title:   "Plugin",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "The m3tal-docker plugin interacts with unix:///var/run/docker.sock over JSON-RPC 2.0 stdio to collect container metrics and manage lifecycle actions.",
				},
			}),
			Completion: "As explained in [1], the m3tal-docker plugin connects to unix:///var/run/docker.sock to monitor metrics and manage containers via JSON-RPC 2.0.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("what docker version is supported", []RAGNote{
				{
					Title:   "Docker",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "M3TAL requires Docker Engine 24.0 or newer and Docker Compose V2 for container lifecycle orchestration.",
				},
			}),
			Completion: "According to [1], M3TAL requires Docker Engine 24.0+ and Docker Compose V2.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("what is the container restart policy", []RAGNote{
				{
					Title:   "Policies",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "Production service definitions in M3TAL specify restart: unless-stopped to survive system reboots.",
				},
			}),
			Completion: "According to [1], production service definitions use restart: unless-stopped to survive reboots.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("how do I inspect running docker containers", []RAGNote{
				{
					Title:   "Inspection",
					Source:  "/apps/M3tal-core/docs/DOCKER.md",
					Excerpt: "Inspect running containers using docker ps or query the M3TAL Core /api/v2/services endpoint.",
				},
			}),
			Completion: "According to [1], inspect containers using docker ps or the /api/v2/services endpoint.",
		},

		// Grounded Q&A abstention / negative examples
		{
			Category: "rag",
			Prompt: RAGPrompt("what is the weather today", []RAGNote{
				{
					Title:   "Install",
					Source:  "/apps/M3tal-Hub/scripts/build_hub.py",
					Excerpt: "Run the install command to build and install M3TAL Hub.",
				},
			}),
			Completion: "I don't know based on the provided reference notes.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("how do I deploy kubernetes on bare metal", []RAGNote{
				{
					Title:   "Configuration",
					Source:  "/apps/M3tal-core/docs/ENVIRONMENT_VARIABLES.md",
					Excerpt: "M3TAL environment variables and default configuration paths are documented here.",
				},
			}),
			Completion: "I don't know based on the provided reference notes.",
		},
		{
			Category: "rag",
			Prompt: RAGPrompt("what database does postgresql replace", []RAGNote{
				{
					Title:   "Traefik",
					Source:  "/apps/M3tal-core/docs/TRAEFIK.md",
					Excerpt: "Traefik is deployed as the primary reverse proxy and SSL terminator for M3TAL services.",
				},
			}),
			Completion: "I don't know based on the provided reference notes.",
		},
	}
}

// FormatInstructionCorpus formats instruction pairs into plain text training passages.
func FormatInstructionCorpus(examples []InstructionExample) []string {
	var passages []string
	for _, ex := range examples {
		passage := strings.TrimSpace(ex.Prompt) + " " + strings.TrimSpace(ex.Completion)
		passages = append(passages, passage)
	}
	return passages
}

// WriteInstructionCorpusFile writes instruction examples formatted for training to a file.
func WriteInstructionCorpusFile(path string, examples []InstructionExample) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create instruction directory %q: %w", dir, err)
	}
	var b strings.Builder
	for _, ex := range examples {
		b.WriteString(strings.TrimSpace(ex.Prompt))
		b.WriteString(" ")
		b.WriteString(strings.TrimSpace(ex.Completion))
		b.WriteString("\n\n")
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("write instruction corpus %q: %w", path, err)
	}
	return nil
}

// ValidatePlanJSON verifies that an instruction's plan completion conforms to schema.
func ValidatePlanJSON(completion string) error {
	start := strings.Index(completion, "{")
	end := strings.LastIndex(completion, "}")
	if start < 0 || end <= start {
		return fmt.Errorf("missing JSON object")
	}
	var raw struct {
		Steps []struct {
			Tool   string `json:"tool"`
			Target string `json:"target,omitempty"`
		} `json:"steps"`
	}
	decoder := json.NewDecoder(strings.NewReader(completion[start : end+1]))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&raw); err != nil {
		return fmt.Errorf("invalid plan JSON: %w", err)
	}
	if len(raw.Steps) == 0 {
		return fmt.Errorf("plan has 0 steps")
	}
	return nil
}

