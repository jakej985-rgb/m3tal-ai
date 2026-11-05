# Go quality baseline

Baseline run on 2026-11-05 with Go 1.23, against the current working tree.

| Check | Result |
| --- | --- |
| `go build ./...` | Pass |
| `go vet ./...` | Pass |
| `go test ./...` | Pass |
| `staticcheck ./...` | Not run: `staticcheck` is not installed in this environment |
| `govulncheck ./...` | Not run: `govulncheck` is not installed in this environment |

No failures were reported by the checks that ran. The CI workflow installs and
runs Staticcheck and govulncheck on pushes and pull requests.

## CPU Kernels Benchmark Suite (Issue #5)

Benchmark results run on Intel(R) Xeon(R) CPU @ 2.30GHz (x86_64, AVX2+FMA detected):

### Matrix Multiplication (Float32 GEMM)

| Matrix Size | Fallback Throughput | Optimized (Parallel + AVX2) Throughput | Speedup |
| --- | --- | --- | --- |
| 32x32 | 36,172 ns/op (7.2 GB/s) | 11,421 ns/op (22.9 GB/s) | **3.17x** |
| 64x64 | 165,709 ns/op (12.6 GB/s) | 81,048 ns/op (25.8 GB/s) | **2.04x** |
| 128x128 | 705,765 ns/op (23.7 GB/s) | 283,387 ns/op (59.2 GB/s) | **2.49x** |
| 256x256 | 5,011,793 ns/op (26.7 GB/s) | 1,554,603 ns/op (86.3 GB/s) | **3.22x** |
| 512x512 | 45,098,913 ns/op (23.8 GB/s) | 18,765,267 ns/op (57.2 GB/s) | **2.40x** |

### Elementwise Addition (Float32 Add)

| Elements (N) | Fallback Latency | Optimized (AVX2) Latency | Speedup |
| --- | --- | --- | --- |
| 1,024 | 1,076 ns/op (7.6 GB/s) | 150 ns/op (54.6 GB/s) | **7.17x** |
| 65,536 | 52,665 ns/op (9.9 GB/s) | 10,529 ns/op (49.8 GB/s) | **5.00x** |
| 1,048,576 | 897,512 ns/op (9.3 GB/s) | 324,388 ns/op (25.8 GB/s) | **2.77x** |

### Build and Runtime Requirements
- Generic fallback is always compiled and available as a baseline on all platforms and architectures.
- SIMD acceleration (AVX2 + FMA) is feature-detected via CPUID on `amd64` systems.
- NEON acceleration is enabled on `arm64` systems.
- Testing toggle `cpu.SetForceFallback(true)` enables runtime testing and verification against reference fallback implementations.
