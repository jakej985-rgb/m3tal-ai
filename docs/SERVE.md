# M3TAL inference server

`m3tal-lm serve` exposes a trained checkpoint over HTTP so M3TAL
components (router, plugins) can route language-model requests without
coupling to model internals. CPU-only deployment is the default; CUDA is
available via the `cuda` build tag (`docs/CUDA.md`).

## Endpoints

| Method | Path           | Purpose                                        |
|--------|----------------|------------------------------------------------|
| GET    | `/healthz`     | Liveness + parameter count                      |
| POST   | `/v1/generate` | Raw prompt completion                           |
| POST   | `/v1/chat`     | Chat completion (chatml.v1); `stream:true` → SSE |

### POST /v1/generate

```json
{"prompt": "Once upon", "max_tokens": 64, "temperature": 0.8,
 "top_k": 0, "top_p": 0.0, "seed": 19, "stop": ["<|im_end|>"]}
```

→ `{"text": "..."}`

`temperature: 0` selects greedy argmax. `top_k`/`top_p` are optional
filters (0 = disabled). `seed` makes sampling deterministic.

### POST /v1/chat

```json
{"messages": [{"role":"user","content":"Hello"}],
 "max_tokens": 64, "temperature": 0.0, "stream": false}
```

→ `{"text": "...", "template": "m3tal.chatml.v1"}`

With `"stream": true` the response is `text/event-stream`: each
`data:` line carries `{"delta": "..."}` and the stream ends with
`data: [DONE]`. Generation always appends the chat stop markers, so
`<|im_end|>` terminates output and is stripped.

### Errors

All errors return `{"error": "..."}` with an appropriate status:
`400` malformed JSON/invalid request, `413` prompt or body over the
safety limits, `500` internal.

## Safety policy (configurable, not in weights)

`server.Policy` bounds every request independently of the checkpoint:

- `MaxBodyBytes` — request body cap (default 1 MiB)
- `MaxPromptTokens` — encoded prompt cap (default 4096)
- `MaxOutputTokens` — per-request generation cap (default 512)
- `RequestTimeout` — per-request deadline (default 2m)

CLI flags: `--max-prompt-tokens`, `--max-output-tokens`,
`--max-body-mb`, `--timeout`. The safety layer lives in
`internal/server` and can be tuned or replaced without touching model
weights.

## Go client

`internal/llmclient` wraps the HTTP API and includes a `Generator`
adapter whose signature matches the `m3tal-ai` assistant `Generator`
interface (`Generate(prompt, maxTokens, temperature, seed)`) — the
assistant lives in `m3tal-plugins/m3tal-ai` — so plugins can route
requests to a local server without this repo depending on it:

```go
c := llmclient.New("http://127.0.0.1:8080")
gen := llmclient.NewGenerator(c)
// In m3tal-plugins/m3tal-ai: assistant.SetLanguageModel(gen)
```

`Client.ChatStream` supports SSE consumption; an emit error cancels the
stream cleanly. Server unavailability surfaces as ordinary Go errors —
callers should treat the model as optional and fall back to deterministic
routing (the plugin already does this when no language model is loaded).

## Run

```sh
go run ./cmd/m3tal-lm serve --checkpoint models/stage0-tiny-stories.json \
  --addr 127.0.0.1:8080
```

Integration tests: `internal/server/server_test.go` (endpoints, SSE,
validation, limits) and `internal/llmclient/client_test.go` (client,
streaming, unavailability).
