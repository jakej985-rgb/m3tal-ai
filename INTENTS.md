# Assistant intents and safety classes

> The M3TAL assistant described here lives in the M3TAL Plugins repository
> at `m3tal-plugins/m3tal-ai` (module `m3tal-plugin-ai`) — it moved out of
> this repo. This file is kept as the canonical intent/safety reference
> for that codebase; this repo only provides the optional model it may
> consume (see `docs/ECOSYSTEM_INTEGRATION.md`).

The assistant uses deterministic routing for clear phrases and a small trained
classifier for other requests. The classifier has a confidence threshold; an
unclear request falls back to a clarification prompt.

## Read-only intents

| Intent | Behavior | Safety class |
| --- | --- | --- |
| `docs` | Search the local M3TAL documentation and source knowledge base. | Read-only local files |
| `stats` | Read system statistics from M3TAL Core. | Read-only API request |
| `health` | Read health status from M3TAL Core. | Read-only API request |
| `containers` | List services and containers from M3TAL Core. | Read-only API request |
| `logs` | Read recent logs for a named container from M3TAL Core. | Read-only API request |

## Mutating intents

Every mutating request must identify exactly one target and action. Before the
action, the assistant checks that the target appears in the corresponding live
M3TAL Core list. It then requires the exact, case-sensitive confirmation
`confirm <action> <target-type> <name>`. A missing callback, declined
confirmation, failed allowlist check, or ambiguous request prevents mutation.

| Intent | Target type | Action | Safety class |
| --- | --- | --- | --- |
| `start_stack` | stack | start | Mutating; exact confirmation required |
| `stop_stack` | stack | stop | Mutating; exact confirmation required |
| `restart_stack` | stack | restart | Mutating; exact confirmation required |
| `start_container` | container | start | Mutating; exact confirmation required |
| `stop_container` | container | stop | Mutating; exact confirmation required |
| `restart_container` | container | restart | Mutating; exact confirmation required |
| `start_service` | system service | start | Mutating; exact confirmation required |
| `stop_service` | system service | stop | Mutating; exact confirmation required |
| `restart_service` | system service | restart | Mutating; exact confirmation required |

## Fallback

`unknown` is not an action. It asks the user to clarify and makes no API
request. Documentation freshness flags are informational and do not establish
whether a claim is current or correct.

## Optional local language model

The experimental Go model is loaded only when `--model` or `M3TAL_MODEL_PATH`
is provided. It can draft a documentation response from retrieved excerpts or
propose a short plan for a request containing multiple steps. The plan must use
the fixed tool names handled by the assistant; invalid output runs no tools.
Every mutating plan step is independently checked against Core's live allowlist,
requires the exact confirmation phrase above, and is checked again immediately
before the mutation. Declining a step stops the remaining plan. The prototype
is not a production-quality language model and its generated documentation
drafts should be checked against the listed sources.
