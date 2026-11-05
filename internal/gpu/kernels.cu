// CUDA kernels for m3tal-llm tensor primitives.
//
// This file is compiled by nvcc into kernels_amd64.o via:
//
//	go generate ./internal/gpu
//
// and linked into the Go binary through cgo (see gpu_amd64.go). All entry
// points are extern "C" so the Go side can call them directly.
//
// Conventions: all tensors are row-major, float32/float64, contiguous.
// Kernels launch on the default stream; callers synchronize via the
// blocking cudaMemcpy used for device->host downloads.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>

#define GPU_TILE 16
#define GPU_BLOCK 256

// ---------------------------------------------------------------------------
// GEMM: C[M,N] = A[M,K] * B[K,N]
// ---------------------------------------------------------------------------

__global__ void gemm_f32_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ float As[GPU_TILE][GPU_TILE + 1];
    __shared__ float Bs[GPU_TILE][GPU_TILE + 1];

    int row = blockIdx.y * GPU_TILE + threadIdx.y;
    int col = blockIdx.x * GPU_TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += GPU_TILE) {
        int aRow = blockIdx.y * GPU_TILE + threadIdx.y;
        int aCol = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (aRow < M && aCol < K) ? A[(long long)aRow * K + aCol] : 0.0f;

        int bRow = k0 + threadIdx.y;
        int bCol = blockIdx.x * GPU_TILE + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && bCol < N) ? B[(long long)bRow * N + bCol] : 0.0f;

        __syncthreads();

        if (row < M && col < N) {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < GPU_TILE; k++) {
                s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            sum += s;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(long long)row * N + col] = sum;
    }
}

__global__ void gemm_f64_kernel(const double* __restrict__ A,
                                const double* __restrict__ B,
                                double* __restrict__ C,
                                int M, int N, int K) {
    __shared__ double As[GPU_TILE][GPU_TILE + 1];
    __shared__ double Bs[GPU_TILE][GPU_TILE + 1];

    int row = blockIdx.y * GPU_TILE + threadIdx.y;
    int col = blockIdx.x * GPU_TILE + threadIdx.x;
    double sum = 0.0;

    for (int k0 = 0; k0 < K; k0 += GPU_TILE) {
        int aRow = blockIdx.y * GPU_TILE + threadIdx.y;
        int aCol = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (aRow < M && aCol < K) ? A[(long long)aRow * K + aCol] : 0.0;

        int bRow = k0 + threadIdx.y;
        int bCol = blockIdx.x * GPU_TILE + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && bCol < N) ? B[(long long)bRow * N + bCol] : 0.0;

        __syncthreads();

        if (row < M && col < N) {
            double s = 0.0;
            #pragma unroll
            for (int k = 0; k < GPU_TILE; k++) {
                s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            sum += s;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(long long)row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Elementwise binary ops: out[i] = a[i] OP b[i]
// op: 0=add, 1=sub, 2=mul, 3=div
// ---------------------------------------------------------------------------

__device__ __forceinline__ float apply_op_f32(int op, float x, float y) {
    switch (op) {
        case 0: return x + y;
        case 1: return x - y;
        case 2: return x * y;
        default: return x / y;
    }
}

__device__ __forceinline__ double apply_op_f64(int op, double x, double y) {
    switch (op) {
        case 0: return x + y;
        case 1: return x - y;
        case 2: return x * y;
        default: return x / y;
    }
}

__global__ void elemwise_f32_kernel(int op,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out,
                                    long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = apply_op_f32(op, a[i], b[i]);
    }
}

__global__ void elemwise_f64_kernel(int op,
                                    const double* __restrict__ a,
                                    const double* __restrict__ b,
                                    double* __restrict__ out,
                                    long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = apply_op_f64(op, a[i], b[i]);
    }
}

// ---------------------------------------------------------------------------
// Scale: out[i] = src[i] * alpha
// ---------------------------------------------------------------------------

__global__ void scale_f32_kernel(const float* __restrict__ src,
                                 float alpha,
                                 float* __restrict__ out,
                                 long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = src[i] * alpha;
    }
}

__global__ void scale_f64_kernel(const double* __restrict__ src,
                                 double alpha,
                                 double* __restrict__ out,
                                 long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = src[i] * alpha;
    }
}

// ---------------------------------------------------------------------------
// Reductions. Sum uses atomicAdd (float: any arch; double: sm_60+ only, so
// the f64 variants run as a single block to stay correct on sm_52).
// Max encodes floats as order-preserving uint keys for atomicMax.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int f2u_key(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Block-wide reduce helper: returns warp-reduced value via shared memory.
template <typename T, typename Op>
__device__ __forceinline__ T block_reduce(T val, Op op) {
    __shared__ T sh[GPU_BLOCK / 32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        T other = __shfl_down_sync(0xffffffffu, val, off);
        val = op(val, other);
    }
    if (lane == 0) sh[wid] = val;
    __syncthreads();
    int nwarp = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        val = (lane < nwarp) ? sh[lane] : op.identity();
        for (int off = 16; off > 0; off >>= 1) {
            T other = __shfl_down_sync(0xffffffffu, val, off);
            val = op(val, other);
        }
    }
    return val;
}

struct SumOpF32 {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
    __device__ __forceinline__ float identity() const { return 0.0f; }
};
struct MaxOpF32 {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
    __device__ __forceinline__ float identity() const { return -CUDART_INF_F; }
};

__global__ void sum_f32_kernel(const float* __restrict__ src,
                               float* __restrict__ out,
                               long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        acc += src[i];
    }
    SumOpF32 op;
    float total = block_reduce(acc, op);
    if ((threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0) {
        atomicAdd(out, total);
    }
}

__global__ void max_f32_kernel(const float* __restrict__ src,
                               unsigned int* __restrict__ outKey,
                               long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    float acc = -CUDART_INF_F;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        acc = fmaxf(acc, src[i]);
    }
    MaxOpF32 op;
    float m = block_reduce(acc, op);
    if ((threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0) {
        atomicMax(outKey, f2u_key(m));
    }
}

// f64 reductions: single block (no atomicAdd(double) on sm_52).
__global__ void sum_f64_kernel(const double* __restrict__ src,
                               double* __restrict__ out,
                               long long n) {
    long long stride = (long long)blockDim.x;
    double acc = 0.0;
    for (long long i = threadIdx.x; i < n; i += stride) {
        acc += src[i];
    }
    __shared__ double sh[GPU_BLOCK / 32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) sh[wid] = acc;
    __syncthreads();
    int nwarp = blockDim.x >> 5;
    if (wid == 0) {
        acc = (lane < nwarp) ? sh[lane] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) *out = acc;
    }
}

__global__ void max_f64_kernel(const double* __restrict__ src,
                               double* __restrict__ out,
                               long long n) {
    long long stride = (long long)blockDim.x;
    double acc = -CUDART_INF;
    for (long long i = threadIdx.x; i < n; i += stride) {
        acc = fmax(acc, src[i]);
    }
    __shared__ double sh[GPU_BLOCK / 32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        acc = fmax(acc, __shfl_down_sync(0xffffffffu, acc, off));
    }
    if (lane == 0) sh[wid] = acc;
    __syncthreads();
    int nwarp = blockDim.x >> 5;
    if (wid == 0) {
        acc = (lane < nwarp) ? sh[lane] : -CUDART_INF;
        for (int off = 16; off > 0; off >>= 1) {
            acc = fmax(acc, __shfl_down_sync(0xffffffffu, acc, off));
        }
        if (lane == 0) *out = acc;
    }
}

// ---------------------------------------------------------------------------
// extern "C" launchers: compute grid size, launch on stream 0, return the
// immediate cudaError_t so Go can detect launch-configuration errors.
// Device pointers are CUdeviceptr-sized (void* works via casts below).
// ---------------------------------------------------------------------------

static inline dim3 gpu_grid_1d(long long n, int block) {
    long long blocks = (n + block - 1) / block;
    if (blocks > 65535LL * 32) blocks = 65535LL * 32;  // cap; stride loop covers rest
    return dim3((unsigned)blocks);
}

extern "C" int m3tal_launch_gemm_f32(const void* A, const void* B, void* C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    dim3 block(GPU_TILE, GPU_TILE);
    dim3 grid((N + GPU_TILE - 1) / GPU_TILE, (M + GPU_TILE - 1) / GPU_TILE);
    gemm_f32_kernel<<<grid, block>>>(
        (const float*)A, (const float*)B, (float*)C, M, N, K);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_gemm_f64(const void* A, const void* B, void* C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    dim3 block(GPU_TILE, GPU_TILE);
    dim3 grid((N + GPU_TILE - 1) / GPU_TILE, (M + GPU_TILE - 1) / GPU_TILE);
    gemm_f64_kernel<<<grid, block>>>(
        (const double*)A, (const double*)B, (double*)C, M, N, K);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_elemwise_f32(int op, const void* a, const void* b,
                                         void* out, long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    elemwise_f32_kernel<<<grid, GPU_BLOCK>>>(
        op, (const float*)a, (const float*)b, (float*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_elemwise_f64(int op, const void* a, const void* b,
                                         void* out, long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    elemwise_f64_kernel<<<grid, GPU_BLOCK>>>(
        op, (const double*)a, (const double*)b, (double*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_scale_f32(const void* src, float alpha, void* out,
                                      long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    scale_f32_kernel<<<grid, GPU_BLOCK>>>((const float*)src, alpha, (float*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_scale_f64(const void* src, double alpha, void* out,
                                      long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    scale_f64_kernel<<<grid, GPU_BLOCK>>>((const double*)src, alpha, (double*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_sum_f32(const void* src, void* out, long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    sum_f32_kernel<<<grid, GPU_BLOCK>>>((const float*)src, (float*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_max_f32(const void* src, void* outKey, long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    max_f32_kernel<<<grid, GPU_BLOCK>>>((const float*)src, (unsigned int*)outKey, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_sum_f64(const void* src, void* out, long long n) {
    if (n <= 0) return 0;
    sum_f64_kernel<<<1, GPU_BLOCK>>>((const double*)src, (double*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_max_f64(const void* src, void* out, long long n) {
    if (n <= 0) return 0;
    max_f64_kernel<<<1, GPU_BLOCK>>>((const double*)src, (double*)out, n);
    return (int)cudaGetLastError();
}
