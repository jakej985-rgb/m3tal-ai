# M3TAL AI roadmap

This roadmap builds from the current Go prototype: deterministic intent
routing, local documentation search, read-only M3TAL Core queries, confirmed
Core mutations, and an optional experimental from-scratch language model.

Work is organized by outcomes rather than dates. A phase is ready to advance
when its exit criteria are met and the result is useful to a real M3TAL user.

## Product direction

M3TAL AI should be a dependable local assistant for understanding and operating
the M3TAL ecosystem. It should make the source of an answer visible, keep
operational actions constrained and reviewable, and work well without a hosted
model. The local language model remains an experiment until it demonstrates
measurable value over the existing deterministic and retrieval-based behavior.

## Phase 1 — Make current behavior dependable

**Goal:** Make it easy to install, configure, and diagnose the assistant that
already exists.

- Document supported Go versions, configuration, Core API expectations, and
  common failure modes in one getting-started path.
- Validate startup configuration and give actionable errors for missing or
  unreadable documentation roots, invalid checkpoints, and unreachable Core.
- Define a small set of representative assistant scenarios and record expected
  outcomes, including ambiguous requests and denied mutations.
- Add release packaging and version reporting once the CLI behavior is stable.

**Exit criteria:** A new user can build and run the CLI from the docs, diagnose
configuration problems, and understand what the assistant will and will not
do. Core mutations remain explicitly confirmed and constrained to live targets.

## Phase 2 — Improve documentation answers

**Goal:** Make answers easier to trust and keep current.

- Return concise answers with file and section references for retrieved
  documentation.
- Improve retrieval ranking using headings, exact terms, and ecosystem context;
  keep source code and general README files from crowding out specific guides.
- Make indexing behavior visible, including skipped files, duplicate sources,
  and freshness review flags.
- Build a small evaluation set from real M3TAL questions and expected source
  passages; use it to compare retrieval changes.

**Exit criteria:** Common setup and troubleshooting questions return relevant,
traceable sources, and retrieval changes can be compared against a repeatable
evaluation set.

## Phase 3 — Make Core operations easier to review

**Goal:** Support useful operational workflows while preserving clear control
over every change.

- Improve intent handling for target names, aliases, and follow-up questions;
  ask for clarification when the target or operation is ambiguous.
- Add a read-only preview of the exact target and action before confirmation.
- Provide structured outcomes for successful actions, rejected targets,
  cancellations, and Core errors.
- Add an opt-in local action history that records what was requested and the
  result without storing API tokens or unrelated sensitive data.
- Expand the supported action set only when Core has a well-defined API and the
  action has a clear confirmation policy.

**Exit criteria:** Users can see the target, action, and result for each
operation; ambiguous or unauthorized requests cause no mutation.

## Phase 4 — Establish a stable integration surface

**Goal:** Make M3TAL AI usable from more than a one-shot local CLI.

- Separate assistant capabilities from CLI input/output so they can be reused
  by other M3TAL interfaces.
- Define a versioned request/response contract for integrations, with explicit
  capability and safety boundaries.
- Add an interactive session experience with useful context and clear recovery
  after errors.
- Evaluate a local service mode only if an actual M3TAL client needs it; define
  authentication and network exposure constraints before implementation.

**Exit criteria:** At least one M3TAL client can integrate through a documented
contract without duplicating routing or safety logic.

## Phase 5 — Evaluate language-model options

**Goal:** Use a language model only where it measurably improves the assistant.

- First evaluate the existing retrieval and deterministic baseline on the same
  question set.
- Measure answer relevance, source support, plan validity, latency, memory, and
  operational safety on held-out examples.
- Compare the scratch model with established local inference options before
  investing in a larger training system.
- Keep generated tool plans schema-constrained, allowlisted, and individually
  confirmed; invalid plans must run no actions.
- Publish model checkpoints only with documented provenance and training-data
  rights.

**Decision gate:** Continue scratch-model work only if it improves a defined
metric enough to justify its runtime and maintenance cost. Otherwise, retain it
as an educational experiment or integrate a suitable local inference runtime.

## Ongoing quality bar

- Keep the assistant useful without an optional language model.
- Preserve source attribution for documentation answers.
- Fail closed on uncertain or invalid operational requests.
- Keep credentials out of logs, examples, and generated artifacts.
- Track quality against representative scenarios when behavior changes.

## Not planned yet

- Hosted-model dependence for basic documentation or Core operations.
- Unattended or background mutations.
- Broad shell, filesystem, or arbitrary HTTP tool access.
- Treating freshness flags or generated text as proof that a source is correct.
