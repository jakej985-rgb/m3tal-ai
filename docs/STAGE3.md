# Stage 3 — Chat format, SFT, sampling, eval, DPO

Stage 3 turns the base model into a chat model: a versioned chat template
with assistant-only loss masks, self-owned instruction data, an SFT
pipeline, richer decoding controls, a held-out conversational eval suite,
and optional DPO preference training.

All data is **self-owned** (`license=self-owned` / `source=self-owned`);
nothing was downloaded. CPU remains the default backend; CUDA is optional.

## Chat template (`internal/lm/chat.go`)

Template `m3tal.chatml.v1` serializes conversations as tokenized
segments (each segment encoded independently so mask boundaries are
exact, and the same segment-wise encoding is used for both training and
generation prompts):

```
<|im_start|>system\n{content}<|im_end|>\n
<|im_start|>user\n{content}<|im_end|>\n
<|im_start|>assistant\n{content}<|im_end|>\n
```

- Roles: `system`, `user`, `assistant`, `tool`.
- `Conversation.Validate()` rejects empty conversations/messages, unknown
  roles, and conversations without user or assistant turns.
- `SerializedParts(tokenizer)` returns `(ids, mask)` where `mask[i]` is
  true only for assistant content tokens and the assistant turn's
  `<|im_end|>` — the positions where SFT/DPO loss applies.
- `EncodePrompt(tokenizer)` returns ids through the last non-assistant
  turn plus the `<|im_start|>assistant\n` header.
- `StopMarkers()` returns `["<|im_end|>"]`; generation detects the marker
  on decoded text and truncates it from output.

## Instruction/chat data (`internal/lm/chatdata.go`)

Chat datasets are JSONL — one `ChatExample` (`{"messages":[...],
"category":"..."}`) per line. `LoadChatJSONL` validates every record,
counts malformed lines, and deduplicates by SHA-256 of the canonical
serialization. `SplitChatExamples` performs a deterministic hash split
(no train/val overlap, reproducible). `chatdata validate|split` exposes
this in the CLI; `instructions --chat FILE` converts the built-in
self-owned instruction set to chat JSONL.

Committed data:

- `data/chat-instructions.jsonl` — 67 examples (51 plan, 16 rag),
  self-owned, generated from `DefaultInstructionSet`.
- `data/chat-eval-suite.jsonl` — held-out suite v1, self-owned, disjoint
  domain from the instruction set (general QA, formatting, multi-turn,
  tool-planning probe).
- `data/preferences.jsonl` — 5 self-owned chosen/rejected pairs for DPO.

## SFT (`internal/lm/sft.go`)

`Model.TrainChat` trains on masked chat windows using the shared
`TrainConfig` schedule (warmup + cosine decay + grad clip + seeded
sampling). `PrepareChatExamples` truncates over-long conversations from
the front (most recent turns kept) and drops windows with no assistant
tokens. `SFTMetrics` logs step, loss, LR, tok/s, grad norm, and cumulative
masked-token counts. Optimizer state and step count persist through
`SaveCheckpoint`/`LoadCheckpoint`, so `--max-steps` is an absolute target
and SFT resumes exactly like pretraining. The base checkpoint is never
modified — SFT writes a new checkpoint.

CLI: `train --base-model CKPT --sft train.jsonl --checkpoint out.json
--max-steps N --learning-rate LR ...`

## Decoding (`internal/lm/sample.go`)

`GenerateOptions` now supports:

- `Temperature` — `0` = greedy argmax (fully deterministic); `>0` = softmax.
- `TopK` — keep k highest-probability tokens (0 = disabled).
- `TopP` — nucleus sampling (0 = disabled, else in (0,1]).
- `Stop` — decoded-text stop markers; generation halts early and the
  marker is truncated from output.
- `RepetitionPenalty`, `Seed` (deterministic given fixed settings).
- `GenerateStream(prompt, opts, emit)` — per-token decoded deltas; an
  emit error cancels generation safely (`stream cancelled: ...`) and the
  partial output is returned with the error.

CLI: `generate --checkpoint CKPT --prompt "..." --chat --stream
--temperature 0.8 --top-p 0.9 --top-k 40 --stop "<|im_end|>"`

Context-limit behavior is preserved: KV-cache stepping until the context
window is exhausted, then full-forward fallback with truncation.

## Conversational eval (`internal/lm/chateval.go`)

`chat-eval --checkpoint CKPT --suite suite.jsonl --report out.json
[--baseline prev.json]` runs the held-out suite:

- Entries ending with an assistant turn are **scored**: mean
  next-token loss over assistant (masked) tokens only, plus perplexity.
- Entries ending with a user turn are **probes**: greedy/sampled
  generation recorded verbatim with automated checks (empty output,
  stop-marker leak, max-token exhaustion, stop hit).
- The report records template version, tokenizer fingerprint, checkpoint
  path, decoding settings, and an explicit note that automated metrics
  are heuristics — relevance/coherence/adherence require human review.
- `--baseline` compares assistant loss and fails on regression beyond
  `--max-loss-increase`.

## DPO (`internal/lm/dpo.go`) — optional milestone

Preference pairs are JSONL (`{"id","source","license","prompt":[...],
"chosen","rejected"}`) validated by `LoadPreferenceJSONL`. `TrainDPO`
uses a frozen reference model (a second `LoadCheckpoint` of the same
file — verified never to change) and the standard DPO loss
`-log σ(β·((lp_c−lp_r)_policy − (lp_c−lp_r)_ref))` over assistant-masked
token log-probability sums. Gradient scaling was verified against
central finite differences (`TestDPOLossFiniteDifference`).

CLI: `train --base-model CKPT --dpo pairs.jsonl --checkpoint out.json`.

## Stage 3 run evidence (GTX 960 / CPU, 2.9M-param base)

Base: `/tmp/ckpt-s2-run2.json` (Stage 2 run, step 1800).

- **SFT**: 49 train examples → 49 windows, 3023 masked assistant tokens,
  template `m3tal.chatml.v1`; 60 steps, lr 5e-4→5e-5 cosine, loss
  ~6.3 → ~3.1–5.2 (noisy, small data). Checkpoint `/tmp/ckpt-s3-sft.json`.
- **chat-eval base vs SFT**: held-out assistant loss **7.2144 → 5.1918**
  (Δ −2.02, ppl 1359 → 180). Baseline regression check passes.
- **DPO**: 5 self-owned pairs, 20 steps, DPO loss ~0.66. chat-eval
  assistant loss 5.19 → 5.66 — **regression flagged by the baseline
  check**, as expected: 5 pairs cannot move preference margins without
  degrading likelihood, and DPO optimizes margins not SFT loss. The
  machinery is verified; real preference data is future work.
- Probes still exhaust max tokens (no `<|im_end|>` learned in 60 steps)
  and generations remain story-flavored — honest limitation of a 2.9M
  model + 67 examples. Human review of probe outputs is required for
  quality claims; automated metrics only measure loss/heuristics.
