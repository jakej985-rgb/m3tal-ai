# CUDA backend

M3TAL-LLM is CPU-first. CUDA support is an optional build-time backend for
NVIDIA GPUs, implemented with CUDA C kernels dispatched from Go through cgo.
Everything works without it: the default build contains no CUDA code and
behaves exactly as before.

## Building with CUDA

```sh
CGO_ENABLED=1 go build -tags cuda ./...
```

Requirements:

- Go with cgo enabled and a C toolchain (`gcc`).
- The CUDA toolkit (`nvcc`) providing `cuda_runtime.h` and `libcudart`.
- An NVIDIA driver new enough for the toolkit you install (`nvidia-smi`).

### Toolkit version vs. GPU architecture

| GPU generation | Compute capability | Minimum toolkit | Maximum toolkit |
|---|---|---|---|
| Maxwell (GTX 9xx, e.g. GTX 960) | 5.0–5.3 | CUDA 8.0 | CUDA 12.x (dropped in CUDA 13) |
| Pascal (GTX 10xx) | 6.0–6.1 | CUDA 8.0 | CUDA 12.x |
| Volta (V100, Titan V) | 7.0–7.2 | CUDA 9.0 | CUDA 12.x |
| Turing (RTX 20xx) and newer | 7.5+ | CUDA 10.0 | latest |

On Ubuntu 24.04 the distribution package works for older cards:

```sh
sudo apt install nvidia-cuda-toolkit   # CUDA 12.x
```

## Selecting a backend

Backend selection lives in `internal/tensor/kernel` and is process-wide:

- Default: `cpu`.
- Environment variable: `M3TAL_LLM_BACKEND=cpu|cuda|auto` (`auto` picks CUDA
  when a device is present, otherwise CPU).
- Programmatic: `kernel.Select("cuda")`, `kernel.Probe()`,
  `kernel.Report()`.

`Select` validates against a runtime probe and returns an actionable error
when the backend was not compiled in or no device is usable — it never
silently produces wrong results. Operations the CUDA backend does not
implement natively fall back to the CPU kernels.

## Runtime detection

`kernel.Probe()` reports every compiled backend with its availability and a
human-readable reason (driver mismatch, no devices, not compiled in, etc.).
Detection uses the CUDA runtime API (`cudaGetDeviceCount`,
`cudaGetDeviceProperties`), so it reflects what the *runtime* can actually
use, not just what `nvidia-smi` shows.

## Testing without a GPU

CI runners have no GPU and no toolkit. All CUDA code sits behind the `cuda`
build tag, so the default CI build (`go build ./...`, `go vet ./...`,
`go test ./...`, staticcheck, govulncheck) is unaffected. GPU parity tests
additionally skip at runtime when `kernel.Probe()` reports CUDA
unavailable, so the same test binary is safe to run anywhere.

To exercise the CUDA path locally:

```sh
CGO_ENABLED=1 go test -tags cuda ./internal/tensor/kernel/ -v
M3TAL_LLM_BACKEND=cuda go run ./cmd/m3tal-lm generate --checkpoint models/m3tal-small.json --prompt "M3TAL is"
```
