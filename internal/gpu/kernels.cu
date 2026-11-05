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
// RMSNorm (forward): out[r,c] = gamma[c] * x[r,c] / sqrt(mean(x[r,:]^2)+eps)
// One thread block per row.
// ---------------------------------------------------------------------------

__global__ void rmsnorm_f32_kernel(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ out,
                                   int rows, int width, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (long long)row * width;
    float* orow = out + (long long)row * width;

    float ss = 0.0f;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float v = xr[d];
        ss += v * v;
    }
    SumOpF32 sop;
    float total = block_reduce(ss, sop);
    __shared__ float rmsS;
    if (threadIdx.x == 0) {
        rmsS = 1.0f / sqrtf(total / (float)width + eps);
    }
    __syncthreads();
    float r = rmsS;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        orow[d] = gamma[d] * xr[d] * r;
    }
}

// ---------------------------------------------------------------------------
// RoPE (forward): rotate interleaved (2d, 2d+1) pairs per (row, head).
// batch variant uses pos = row; single variant uses a fixed position.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void rope_apply(float* q, float* k, int width,
                                           int headDim, int h, int d,
                                           float c, float s) {
    int base = h * headDim + 2 * d;
    float q0 = q[base], q1 = q[base + 1];
    q[base]     = q0 * c - q1 * s;
    q[base + 1] = q0 * s + q1 * c;
    float k0 = k[base], k1 = k[base + 1];
    k[base]     = k0 * c - k1 * s;
    k[base + 1] = k0 * s + k1 * c;
}

__device__ __forceinline__ void rope_pair(float* qrow, float* krow, int width,
                                          int headDim, int heads, int row,
                                          float pos) {
    int half = headDim / 2;
    long long total = (long long)heads * half;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x) {
        int d = (int)(idx % half);
        int t = (int)(idx / half);
        int h = t % heads;
        (void)row;
        float theta = powf(10000.0f, -2.0f * (float)d / (float)headDim);
        float ang = pos * theta;
        rope_apply(qrow + (long long)row * width,
                   krow + (long long)row * width,
                   width, headDim, h, d, cosf(ang), sinf(ang));
    }
}

__global__ void rope_batch_f32_kernel(float* q, float* k, int rows,
                                      int width, int headDim, int heads) {
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        rope_pair(q, k, width, headDim, heads, row, (float)row);
    }
}

__global__ void rope_single_f32_kernel(float* q, float* k, int pos,
                                       int width, int headDim, int heads) {
    rope_pair(q, k, width, headDim, heads, 0, (float)pos);
}

// ---------------------------------------------------------------------------
// SwiGLU: out[i] = silu(gate[i]) * up[i], silu(z) = z * sigmoid(z)
// ---------------------------------------------------------------------------

__global__ void swiglu_f32_kernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ out,
                                  long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// ---------------------------------------------------------------------------
// Bias add: C[m,n] += bias[n]  (C is [M,N] row-major)
// ---------------------------------------------------------------------------

__global__ void bias_add_f32_kernel(float* __restrict__ c,
                                    const float* __restrict__ bias,
                                    int M, int N) {
    long long total = (long long)M * N;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        c[i] += bias[i % N];
    }
}

// ---------------------------------------------------------------------------
// Embedding gather: out[r,:] = emb[ids[r],:]  (ids are int32 on device)
// ---------------------------------------------------------------------------

__global__ void embed_f32_kernel(const float* __restrict__ emb,
                                 const int* __restrict__ ids,
                                 float* __restrict__ out,
                                 int rows, int width) {
    long long total = (long long)rows * width;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        int r = (int)(i / width);
        out[i] = emb[(long long)ids[r] * width + (i % width)];
    }
}

// ---------------------------------------------------------------------------
// Causal multi-head attention (forward), three-kernel formulation matching
// the CPU reference: scores -> softmax -> weighted sum of V.
//   q:   [qRows, width]      (post-RoPE queries)
//   k:   [kvLen, width]      (post-RoPE keys; kvLen >= qRows for batch)
//   v:   [kvLen, width]
//   attn: [heads, qRows, kvLen] scratch, overwritten with probabilities
//   out: [qRows, width]
// ---------------------------------------------------------------------------

__global__ void attn_scores_f32_kernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       float* __restrict__ attn,
                                       int heads, int qRows, int kvLen,
                                       int width, int headDim,
                                       float scale, int causal) {
    int idx = blockIdx.x;  // h * qRows + i
    int h = idx / qRows;
    int i = idx % qRows;
    const float* qrow = q + (long long)i * width + h * headDim;
    float* aout = attn + ((long long)h * qRows + i) * kvLen;

    for (int j = threadIdx.x; j < kvLen; j += blockDim.x) {
        if (causal && j > i) {
            aout[j] = -CUDART_INF_F;
            continue;
        }
        const float* krow = k + (long long)j * width + h * headDim;
        float s = 0.0f;
        for (int d = 0; d < headDim; d++) {
            s += qrow[d] * krow[d];
        }
        aout[j] = s * scale;
    }
}

__global__ void attn_softmax_f32_kernel(float* __restrict__ attn,
                                        int heads, int qRows, int kvLen) {
    int idx = blockIdx.x;
    float* a = attn + (long long)idx * kvLen;

    float mx = -CUDART_INF_F;
    for (int j = threadIdx.x; j < kvLen; j += blockDim.x) {
        mx = fmaxf(mx, a[j]);
    }
    MaxOpF32 mop;
    float m = block_reduce(mx, mop);
    __shared__ float ms;
    __syncthreads();
    if (threadIdx.x == 0) ms = m;
    __syncthreads();
    m = ms;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < kvLen; j += blockDim.x) {
        float e = expf(a[j] - m);
        a[j] = e;
        sum += e;
    }
    SumOpF32 sop;
    float z = block_reduce(sum, sop);
    __shared__ float zs;
    __syncthreads();
    if (threadIdx.x == 0) zs = z;
    __syncthreads();
    z = zs;

    for (int j = threadIdx.x; j < kvLen; j += blockDim.x) {
        a[j] /= z;
    }
}

__global__ void attn_weighted_f32_kernel(const float* __restrict__ attn,
                                         const float* __restrict__ v,
                                         float* __restrict__ out,
                                         int heads, int qRows, int kvLen,
                                         int width, int headDim) {
    int idx = blockIdx.x;  // h * qRows + i
    int h = idx / qRows;
    int i = idx % qRows;
    const float* arow = attn + (long long)idx * kvLen;
    float* orow = out + (long long)i * width + h * headDim;

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float s = 0.0f;
        for (int j = 0; j < kvLen; j++) {
            s += arow[j] * v[(long long)j * width + h * headDim + d];
        }
        orow[d] = s;
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

extern "C" int m3tal_launch_rmsnorm_f32(const void* x, const void* gamma,
                                        void* out, int rows, int width,
                                        float eps) {
    if (rows <= 0 || width <= 0) return 0;
    rmsnorm_f32_kernel<<<rows, GPU_BLOCK>>>(
        (const float*)x, (const float*)gamma, (float*)out, rows, width, eps);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_rope_batch_f32(void* q, void* k, int rows,
                                           int width, int headDim, int heads) {
    if (rows <= 0 || width <= 0) return 0;
    int half = headDim / 2;
    long long total = (long long)rows * heads * half;
    dim3 grid((unsigned)gpu_grid_1d(total, GPU_BLOCK).x, (unsigned)rows);
    rope_batch_f32_kernel<<<grid, GPU_BLOCK>>>(
        (float*)q, (float*)k, rows, width, headDim, heads);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_rope_single_f32(void* q, void* k, int pos,
                                            int width, int headDim, int heads) {
    if (width <= 0) return 0;
    int half = headDim / 2;
    long long total = (long long)heads * half;
    dim3 grid = gpu_grid_1d(total, GPU_BLOCK);
    rope_single_f32_kernel<<<grid, GPU_BLOCK>>>(
        (float*)q, (float*)k, pos, width, headDim, heads);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_swiglu_f32(const void* gate, const void* up,
                                       void* out, long long n) {
    if (n <= 0) return 0;
    dim3 grid = gpu_grid_1d(n, GPU_BLOCK);
    swiglu_f32_kernel<<<grid, GPU_BLOCK>>>(
        (const float*)gate, (const float*)up, (float*)out, n);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_bias_add_f32(void* c, const void* bias,
                                         int M, int N) {
    long long total = (long long)M * N;
    if (total <= 0) return 0;
    dim3 grid = gpu_grid_1d(total, GPU_BLOCK);
    bias_add_f32_kernel<<<grid, GPU_BLOCK>>>((float*)c, (const float*)bias, M, N);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_embed_f32(const void* emb, const void* ids,
                                      void* out, int rows, int width) {
    long long total = (long long)rows * width;
    if (total <= 0) return 0;
    dim3 grid = gpu_grid_1d(total, GPU_BLOCK);
    embed_f32_kernel<<<grid, GPU_BLOCK>>>(
        (const float*)emb, (const int*)ids, (float*)out, rows, width);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_attn_scores_f32(const void* q, const void* k,
                                           void* attn, int heads, int qRows,
                                           int kvLen, int width, int headDim,
                                           float scale, int causal) {
    if (heads <= 0 || qRows <= 0 || kvLen <= 0) return 0;
    attn_scores_f32_kernel<<<heads * qRows, GPU_BLOCK>>>(
        (const float*)q, (const float*)k, (float*)attn,
        heads, qRows, kvLen, width, headDim, scale, causal);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_attn_softmax_f32(void* attn, int heads,
                                             int qRows, int kvLen) {
    if (heads <= 0 || qRows <= 0 || kvLen <= 0) return 0;
    attn_softmax_f32_kernel<<<heads * qRows, GPU_BLOCK>>>(
        (float*)attn, heads, qRows, kvLen);
    return (int)cudaGetLastError();
}

extern "C" int m3tal_launch_attn_weighted_f32(const void* attn,
                                              const void* v, void* out,
                                              int heads, int qRows, int kvLen,
                                              int width, int headDim) {
    if (heads <= 0 || qRows <= 0 || kvLen <= 0) return 0;
    attn_weighted_f32_kernel<<<heads * qRows, GPU_BLOCK>>>(
        (const float*)attn, (const float*)v, (float*)out,
        heads, qRows, kvLen, width, headDim);
    return (int)cudaGetLastError();
}
