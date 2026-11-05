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
	Category   string // "plan", "cot_plan", or "rag"
	Thought    string // optional scratchpad reasoning thought for CoT
}

// PlanPrompt formats a user query into the tool planning prompt with scratchpad reasoning.
func PlanPrompt(query string) string {
	return "Plan the user's request using only these tool names: stats, health, containers, logs, start_stack, stop_stack, restart_stack, start_container, stop_container, restart_container, start_service, stop_service, restart_service. Return JSON with a steps array; each item has tool and optional target fields. Do not include prose. You may first write a brief Thought before the JSON. User request: " + query + "\nPLAN:"
}

// CoTPlanPrompt is an alias for PlanPrompt.
func CoTPlanPrompt(query string) string {
	return PlanPrompt(query)
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

		// Chain-of-Thought (CoT) planning examples with intermediate scratchpad reasoning
		{
			Category:   "plan",
			Thought:    "The user wants both system stats and health metrics. First collect system stats, then inspect component health.",
			Prompt:     CoTPlanPrompt("show stats and health"),
			Completion: "Thought: The user wants both system stats and health metrics. First collect system stats, then inspect component health.\n{\"steps\":[{\"tool\":\"stats\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants to inspect system resource statistics followed by the container list.",
			Prompt:     CoTPlanPrompt("show stats then check containers"),
			Completion: "Thought: The user wants to inspect system resource statistics followed by the container list.\n{\"steps\":[{\"tool\":\"stats\"},{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user requests an overview of stats, health status, and active containers.",
			Prompt:     CoTPlanPrompt("show stats and health and containers"),
			Completion: "Thought: The user requests an overview of stats, health status, and active containers.\n{\"steps\":[{\"tool\":\"stats\"},{\"tool\":\"health\"},{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants to evaluate service health first and then review system resource usage.",
			Prompt:     CoTPlanPrompt("check health then show stats"),
			Completion: "Thought: The user wants to evaluate service health first and then review system resource usage.\n{\"steps\":[{\"tool\":\"health\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants container inventory first, followed by host resource statistics.",
			Prompt:     CoTPlanPrompt("list containers and check system stats"),
			Completion: "Thought: The user wants container inventory first, followed by host resource statistics.\n{\"steps\":[{\"tool\":\"containers\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants a health check followed by the container listing.",
			Prompt:     CoTPlanPrompt("show health and list containers"),
			Completion: "Thought: The user wants a health check followed by the container listing.\n{\"steps\":[{\"tool\":\"health\"},{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "To determine whether jellyfin is running, inspect the active containers.",
			Prompt:     CoTPlanPrompt("is jellyfin running"),
			Completion: "Thought: To determine whether jellyfin is running, inspect the active containers.\n{\"steps\":[{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user is querying the status of the jellyfin container, so list containers.",
			Prompt:     CoTPlanPrompt("status of jellyfin"),
			Completion: "Thought: The user is querying the status of the jellyfin container, so list containers.\n{\"steps\":[{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants an overview of container states across the system.",
			Prompt:     CoTPlanPrompt("check container status"),
			Completion: "Thought: The user wants an overview of container states across the system.\n{\"steps\":[{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Query the container runtime to list all running containers.",
			Prompt:     CoTPlanPrompt("what containers are running"),
			Completion: "Thought: Query the container runtime to list all running containers.\n{\"steps\":[{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants to see running containers first, then host performance statistics.",
			Prompt:     CoTPlanPrompt("show containers then stats"),
			Completion: "Thought: The user wants to see running containers first, then host performance statistics.\n{\"steps\":[{\"tool\":\"containers\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "First inspect container status, then check overall system health.",
			Prompt:     CoTPlanPrompt("check containers and health"),
			Completion: "Thought: First inspect container status, then check overall system health.\n{\"steps\":[{\"tool\":\"containers\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants to restart the media stack and immediately tail its logs.",
			Prompt:     CoTPlanPrompt("restart stack media and show logs"),
			Completion: "Thought: The user wants to restart the media stack and immediately tail its logs.\n{\"steps\":[{\"tool\":\"restart_stack\",\"target\":\"media\"},{\"tool\":\"logs\",\"target\":\"media\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user requests starting jellyfin and stopping traefik sequentially.",
			Prompt:     CoTPlanPrompt("start container jellyfin and stop container traefik"),
			Completion: "Thought: The user requests starting jellyfin and stopping traefik sequentially.\n{\"steps\":[{\"tool\":\"start_container\",\"target\":\"jellyfin\"},{\"tool\":\"stop_container\",\"target\":\"traefik\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants to inspect jellyfin logs and then check system resource consumption.",
			Prompt:     CoTPlanPrompt("show logs for jellyfin and check stats"),
			Completion: "Thought: The user wants to inspect jellyfin logs and then check system resource consumption.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"jellyfin\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart the traefik reverse proxy container and then verify system health.",
			Prompt:     CoTPlanPrompt("restart container traefik and show health"),
			Completion: "Thought: Restart the traefik reverse proxy container and then verify system health.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"traefik\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants to manually cycle the media stack by stopping and restarting it.",
			Prompt:     CoTPlanPrompt("stop stack media then start stack media"),
			Completion: "Thought: The user wants to manually cycle the media stack by stopping and restarting it.\n{\"steps\":[{\"tool\":\"stop_stack\",\"target\":\"media\"},{\"tool\":\"start_stack\",\"target\":\"media\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart the core system service, then verify system health to confirm recovery.",
			Prompt:     CoTPlanPrompt("restart service core and check health"),
			Completion: "Thought: Restart the core system service, then verify system health to confirm recovery.\n{\"steps\":[{\"tool\":\"restart_service\",\"target\":\"core\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Stop the jellyfin container and verify the updated container list.",
			Prompt:     CoTPlanPrompt("stop container jellyfin then show containers"),
			Completion: "Thought: Stop the jellyfin container and verify the updated container list.\n{\"steps\":[{\"tool\":\"stop_container\",\"target\":\"jellyfin\"},{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Start the monitoring stack, then check overall health to ensure monitoring agents are up.",
			Prompt:     CoTPlanPrompt("start stack monitoring after that show health"),
			Completion: "Thought: Start the monitoring stack, then check overall health to ensure monitoring agents are up.\n{\"steps\":[{\"tool\":\"start_stack\",\"target\":\"monitoring\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Examine core service logs before issuing a restart to preserve diagnosis context.",
			Prompt:     CoTPlanPrompt("show logs for core and restart service core"),
			Completion: "Thought: Examine core service logs before issuing a restart to preserve diagnosis context.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"core\"},{\"tool\":\"restart_service\",\"target\":\"core\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Stop the proxy service and then restart it to reload configuration.",
			Prompt:     CoTPlanPrompt("stop service proxy then start service proxy"),
			Completion: "Thought: Stop the proxy service and then restart it to reload configuration.\n{\"steps\":[{\"tool\":\"stop_service\",\"target\":\"proxy\"},{\"tool\":\"start_service\",\"target\":\"proxy\"}]}",
		},

		// Compound multi-step and diagnostic CoT planning examples
		{
			Category:   "plan",
			Thought:    "To diagnose jellyfin, check running containers, inspect jellyfin logs, and verify overall health.",
			Prompt:     CoTPlanPrompt("diagnose jellyfin container"),
			Completion: "Thought: To diagnose jellyfin, check running containers, inspect jellyfin logs, and verify overall health.\n{\"steps\":[{\"tool\":\"containers\"},{\"tool\":\"logs\",\"target\":\"jellyfin\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "First check container status to see if jellyfin is running, restart jellyfin, and verify health.",
			Prompt:     CoTPlanPrompt("verify and restart jellyfin"),
			Completion: "Thought: First check container status to see if jellyfin is running, restart jellyfin, and verify health.\n{\"steps\":[{\"tool\":\"containers\"},{\"tool\":\"restart_container\",\"target\":\"jellyfin\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart the media stack and check health to confirm all media services recovered.",
			Prompt:     CoTPlanPrompt("cycle media stack and verify health"),
			Completion: "Thought: Restart the media stack and check health to confirm all media services recovered.\n{\"steps\":[{\"tool\":\"restart_stack\",\"target\":\"media\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "To troubleshoot traefik, check containers, pull traefik logs, and review system stats.",
			Prompt:     CoTPlanPrompt("troubleshoot traefik proxy issues"),
			Completion: "Thought: To troubleshoot traefik, check containers, pull traefik logs, and review system stats.\n{\"steps\":[{\"tool\":\"containers\"},{\"tool\":\"logs\",\"target\":\"traefik\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Perform a full system audit by checking system stats, service health, and all running containers.",
			Prompt:     CoTPlanPrompt("full system health audit"),
			Completion: "Thought: Perform a full system audit by checking system stats, service health, and all running containers.\n{\"steps\":[{\"tool\":\"stats\"},{\"tool\":\"health\"},{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart the traefik container, tail its startup logs, and verify system health.",
			Prompt:     CoTPlanPrompt("restart traefik and check logs and health"),
			Completion: "Thought: Restart the traefik container, tail its startup logs, and verify system health.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"traefik\"},{\"tool\":\"logs\",\"target\":\"traefik\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Inspect media logs before restarting the media stack, then check health after restart.",
			Prompt:     CoTPlanPrompt("safe restart of media stack with log inspection"),
			Completion: "Thought: Inspect media logs before restarting the media stack, then check health after restart.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"media\"},{\"tool\":\"restart_stack\",\"target\":\"media\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Retrieve logs for ollama inference container and verify system resource usage.",
			Prompt:     CoTPlanPrompt("check ollama logs and system stats"),
			Completion: "Thought: Retrieve logs for ollama inference container and verify system resource usage.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"ollama\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart core service, verify containers are still operating, and confirm system health.",
			Prompt:     CoTPlanPrompt("restart core service and verify containers and health"),
			Completion: "Thought: Restart core service, verify containers are still operating, and confirm system health.\n{\"steps\":[{\"tool\":\"restart_service\",\"target\":\"core\"},{\"tool\":\"containers\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Start monitoring stack, confirm overall health, and review current system statistics.",
			Prompt:     CoTPlanPrompt("bring up monitoring and check health and stats"),
			Completion: "Thought: Start monitoring stack, confirm overall health, and review current system statistics.\n{\"steps\":[{\"tool\":\"start_stack\",\"target\":\"monitoring\"},{\"tool\":\"health\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Stop the failing jellyfin container and fetch its exit logs for analysis.",
			Prompt:     CoTPlanPrompt("stop broken container jellyfin and inspect logs"),
			Completion: "Thought: Stop the failing jellyfin container and fetch its exit logs for analysis.\n{\"steps\":[{\"tool\":\"stop_container\",\"target\":\"jellyfin\"},{\"tool\":\"logs\",\"target\":\"jellyfin\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart cloudflared tunnel container and verify system health.",
			Prompt:     CoTPlanPrompt("restart cloudflared tunnel and verify health"),
			Completion: "Thought: Restart cloudflared tunnel container and verify system health.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"cloudflared\"},{\"tool\":\"health\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Restart proxy service and review its startup logs to ensure routing is clean.",
			Prompt:     CoTPlanPrompt("reboot proxy service and check logs"),
			Completion: "Thought: Restart proxy service and review its startup logs to ensure routing is clean.\n{\"steps\":[{\"tool\":\"restart_service\",\"target\":\"proxy\"},{\"tool\":\"logs\",\"target\":\"proxy\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "First check system health, then check traefik container logs.",
			Prompt:     CoTPlanPrompt("check if traefik is healthy and review its logs"),
			Completion: "Thought: First check system health, then check traefik container logs.\n{\"steps\":[{\"tool\":\"health\"},{\"tool\":\"logs\",\"target\":\"traefik\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Stop the media stack, verify container teardown, and check system resource release.",
			Prompt:     CoTPlanPrompt("stop media stack check containers and stats"),
			Completion: "Thought: Stop the media stack, verify container teardown, and check system resource release.\n{\"steps\":[{\"tool\":\"stop_stack\",\"target\":\"media\"},{\"tool\":\"containers\"},{\"tool\":\"stats\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Start the media stack, check container presence, and view media stack logs.",
			Prompt:     CoTPlanPrompt("start media stack and check containers and logs"),
			Completion: "Thought: Start the media stack, check container presence, and view media stack logs.\n{\"steps\":[{\"tool\":\"start_stack\",\"target\":\"media\"},{\"tool\":\"containers\"},{\"tool\":\"logs\",\"target\":\"media\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "To diagnose high resource usage, inspect system statistics and list running containers.",
			Prompt:     CoTPlanPrompt("diagnose high resource usage"),
			Completion: "Thought: To diagnose high resource usage, inspect system statistics and list running containers.\n{\"steps\":[{\"tool\":\"stats\"},{\"tool\":\"containers\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Fetch core logs to inspect crash reason, restart core service, and verify health.",
			Prompt:     CoTPlanPrompt("recover crashed core service"),
			Completion: "Thought: Fetch core logs to inspect crash reason, restart core service, and verify health.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"core\"},{\"tool\":\"restart_service\",\"target\":\"core\"},{\"tool\":\"health\"}]}",
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
		// Entity disambiguation examples: bare action + name, no entity-type keyword.
		// The model must resolve the entity type from context (container, service, stack).
		{
			Category:   "plan",
			Thought:    "Jellyfin is a media streaming container workload managed by Docker Compose, so restart_container is the correct tool.",
			Prompt:     CoTPlanPrompt("restart jellyfin"),
			Completion: "Thought: Jellyfin is a media streaming container workload managed by Docker Compose, so restart_container is the correct tool.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"jellyfin\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Jellyfin is a Docker container; stop_container is the right tool.",
			Prompt:     CoTPlanPrompt("stop jellyfin"),
			Completion: "Thought: Jellyfin is a Docker container; stop_container is the right tool.\n{\"steps\":[{\"tool\":\"stop_container\",\"target\":\"jellyfin\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Jellyfin is a Docker container; start_container is the right tool.",
			Prompt:     CoTPlanPrompt("start jellyfin"),
			Completion: "Thought: Jellyfin is a Docker container; start_container is the right tool.\n{\"steps\":[{\"tool\":\"start_container\",\"target\":\"jellyfin\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Traefik is the reverse proxy container; restart_container applies.",
			Prompt:     CoTPlanPrompt("restart traefik"),
			Completion: "Thought: Traefik is the reverse proxy container; restart_container applies.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"traefik\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Cloudflared runs as a tunnel container; restart_container is correct.",
			Prompt:     CoTPlanPrompt("restart cloudflared"),
			Completion: "Thought: Cloudflared runs as a tunnel container; restart_container is correct.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"cloudflared\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Docker is an OS-level system service, not a container; restart_service is the correct tool.",
			Prompt:     CoTPlanPrompt("restart docker"),
			Completion: "Thought: Docker is an OS-level system service, not a container; restart_service is the correct tool.\n{\"steps\":[{\"tool\":\"restart_service\",\"target\":\"docker\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Media is a Docker Compose stack containing multiple services; restart_stack is the correct tool.",
			Prompt:     CoTPlanPrompt("restart media"),
			Completion: "Thought: Media is a Docker Compose stack containing multiple services; restart_stack is the correct tool.\n{\"steps\":[{\"tool\":\"restart_stack\",\"target\":\"media\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants recent logs from the jellyfin container.",
			Prompt:     CoTPlanPrompt("logs for jellyfin"),
			Completion: "Thought: The user wants recent logs from the jellyfin container.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"jellyfin\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants logs from the cloudflared tunnel container.",
			Prompt:     CoTPlanPrompt("logs cloudflared"),
			Completion: "Thought: The user wants logs from the cloudflared tunnel container.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"cloudflared\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "Ollama is an AI inference container; restart_container is correct.",
			Prompt:     CoTPlanPrompt("restart ollama"),
			Completion: "Thought: Ollama is an AI inference container; restart_container is correct.\n{\"steps\":[{\"tool\":\"restart_container\",\"target\":\"ollama\"}]}",
		},
		{
			Category:   "plan",
			Thought:    "The user wants logs from the traefik reverse proxy container.",
			Prompt:     CoTPlanPrompt("logs traefik"),
			Completion: "Thought: The user wants logs from the traefik reverse proxy container.\n{\"steps\":[{\"tool\":\"logs\",\"target\":\"traefik\"}]}",
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

// ExtractPlanJSON returns only the JSON object from a completion,
// stripping any leading Thought: sentence or surrounding prose.
func ExtractPlanJSON(completion string) string {
	start := strings.Index(completion, "{")
	end := strings.LastIndex(completion, "}")
	if start < 0 || end <= start {
		return ""
	}
	return completion[start : end+1]
}

// ValidatePlanJSON verifies that an instruction's plan completion conforms to schema.
func ValidatePlanJSON(completion string) error {
	jsonStr := ExtractPlanJSON(completion)
	if jsonStr == "" {
		return fmt.Errorf("missing JSON object")
	}
	var raw struct {
		Steps []struct {
			Tool   string `json:"tool"`
			Target string `json:"target,omitempty"`
		} `json:"steps"`
	}
	decoder := json.NewDecoder(strings.NewReader(jsonStr))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&raw); err != nil {
		return fmt.Errorf("invalid plan JSON: %w", err)
	}
	if len(raw.Steps) == 0 {
		return fmt.Errorf("plan has 0 steps")
	}
	return nil
}
