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

After installing the toolkit, build the CUDA kernels once per machine
(the generated `kernels_amd64.o` is git-ignored and arch-specific):

```sh
GOFLAGS=-tags=cuda go generate ./internal/gpu
CGO_ENABLED=1 go build -tags cuda ./...
```

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

## Implemented kernels

`internal/gpu/kernels.cu` provides CUDA kernels for the hot paths used by
the transformer and LM inference stacks; Go wrappers live in
`internal/gpu` and dispatch through `internal/tensor/kernel`:

- GEMM (float32 and float64), elementwise add/sub/mul/div, scale.
- Sum/max reductions.
- Embedding gather, RoPE, RMSNorm.
- Fused attention forward (QK^T scores, softmax, PV) used by the CUDA
  inference backend in `internal/lm`.

Operations without a CUDA kernel fall back to the CPU implementation for
that call, so coverage grows incrementally without breaking correctness.

## Numerics and precision

- Kernels accumulate float32 GEMMs in float32 (no float64 promotion), so
  CUDA float32 results differ from the CPU path only by reduction order —
  typically ~1e-6 relative on well-conditioned inputs. Parity tests use
  tolerances of 1e-4 (float32) and 1e-9 (float64).
- Float64 tensors run through dedicated float64 kernels with the same
  tolerance discipline as CPU.
- There is no mixed-precision (fp16/bf16) path today. Activation and
  gradient tensors stay in their declared dtype end-to-end; introducing
  fp16 would require explicit cast points and is a future optimization,
  not silent downcasting.
- Reductions use atomic adds, so GPU sums are not bitwise deterministic
  run-to-run; magnitudes stay well inside the parity tolerances.
- Numerical-stability behavior (RMSNorm epsilon, softmax max-subtraction)
  matches the CPU reference implementations kernel-for-kernel.

## Benchmarks

`internal/tensor/kernel` contains paired CPU/CUDA benchmarks that measure
the full dispatch path (host→device transfer, kernel, device→host), not
just the kernel launch:

```sh
CGO_ENABLED=1 go test -tags cuda -bench 'CUDA|CPU' -benchmem ./internal/tensor/kernel/
```

On a GTX 960 (Maxwell, sm_52) a representative float32 GEMM measures
~2.5 ms/op end-to-end on CUDA vs ~287 ms/op on the scalar CPU fallback.
