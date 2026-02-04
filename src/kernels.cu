#include <vector>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <vector>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

template <typename T>
__global__ void trace_reduce_kernel(const T* input, T* output,
                                    size_t rows, size_t cols, size_t diag_len) {
    __shared__ T shm[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    T local_sum = T(0);

    for (size_t i = idx; i < diag_len; i += blockDim.x * gridDim.x) {
        local_sum += input[i * cols + i];
    }

    shm[tid] = local_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shm[0]);
    }
}


template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    size_t diag_len = std::min(rows, cols);

    T* d_input = nullptr;
    T* d_output = nullptr;

    size_t input_bytes = h_input.size() * sizeof(T);

    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, sizeof(T));

    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(T));

    int threads = 256;
    int blocks = std::min(
        (diag_len + threads - 1) / threads,
        1024UL
    );

    trace_reduce_kernel<<<blocks, threads>>>(
        d_input, d_output, rows, cols, diag_len
    );

    T h_output;
    cudaMemcpy(&h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return h_output;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return (float)x;
}
template <>
__device__ __forceinline__ float to_float<half>(half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}
template <>
__device__ __forceinline__ half from_float<half>(float x) {
    return __float2half(x);
}

template <typename T>
__global__ void flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int B, int Tq, int Tk,
    int QH, int KVH, int D,
    bool causal
) {
    int b  = blockIdx.x;
    int t  = blockIdx.y;
    int qh = blockIdx.z;
    int tid = threadIdx.x;

    int kvh = qh * KVH / QH;
    float scale = rsqrtf((float)D);

    extern __shared__ float smem[];
    float* Qs = smem;
    float* Ks = smem + D;
    float* Os = smem + 2 * D;

    for (int d = tid; d < D; d += blockDim.x) {
        int q_idx = ((b * Tq + t) * QH + qh) * D + d;
        Qs[d] = to_float(Q[q_idx]);
        Os[d] = 0.f;
    }
    __syncthreads();

    float m = -CUDART_INF_F;
    float l = 0.f;

    for (int s = 0; s < Tk; ++s) {
        if (causal && s > t) break;

        for (int d = tid; d < D; d += blockDim.x) {
            int k_idx = ((b * Tk + s) * KVH + kvh) * D + d;
            Ks[d] = to_float(K[k_idx]);
        }
        __syncthreads();

        float dot = 0.f;
        for (int i = 0; i < D; ++i)
            dot += Qs[i] * Ks[i];
        dot *= scale;

        float m_new = fmaxf(m, dot);
        float alpha = __expf(m - m_new);
        float beta  = __expf(dot - m_new);

        for (int d = tid; d < D; d += blockDim.x) {
            int v_idx = ((b * Tk + s) * KVH + kvh) * D + d;
            float v = to_float(V[v_idx]);
            Os[d] = Os[d] * alpha + beta * v;
        }

        l = l * alpha + beta;
        m = m_new;
        __syncthreads();
    }

    // write output
    for (int d = tid; d < D; d += blockDim.x) {
        int o_idx = ((b * Tq + t) * QH + qh) * D + d;
        O[o_idx] = from_float<T>(Os[d] / l);
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    size_t q_bytes = h_q.size() * sizeof(T);
    size_t k_bytes = h_k.size() * sizeof(T);
    size_t v_bytes = h_v.size() * sizeof(T);
    size_t o_bytes = h_o.size() * sizeof(T);

    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_k, k_bytes);
    cudaMalloc(&d_v, v_bytes);
    cudaMalloc(&d_o, o_bytes);

    cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice);

    dim3 grid(batch_size, target_seq_len, query_heads);
    int threads = 256;
    dim3 block(threads);

    size_t smem = 3 * head_dim * sizeof(float);

    flash_attention_kernel<T><<<grid, block, smem>>>(
        d_q, d_k, d_v, d_o,
        batch_size,
        target_seq_len,
        src_seq_len,
        query_heads,
        kv_heads,
        head_dim,
        is_causal
    );

    cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
