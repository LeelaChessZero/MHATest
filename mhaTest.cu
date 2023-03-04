// A test/benchmarking app for multi-head attention as used in lc0
// Input is KQV matrix in NHCW layout - i.e, (batch_size, 64, num_heads, depth)
// Output is the result after the MHA operation - i.e:
// (64 x depth) sized matrices (Q) are multiplied to (depth x 64) matrices (K) to produce (64 x 64) sized matrices
// There are total of batch_size * num_heads such matrices, so the size of intermediate data is (batch_size, num_heads, 64, 64)
// Softmax is applied to this result (over the innermost dimension - of 64 elements)
// (64 x 64) result matrices above are multiplied to (64 x depth) matrices (V) to produce (64 x depth) matrices (same dimension as the inputs)

// Test paramaters
constexpr int batch_size = 128;
constexpr int num_heads = 12;
constexpr int depth = 64;

constexpr int iterations = 10000;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <stdio.h>
#include "utils.h"
#include <vector>
#include <cublas_v2.h>
#include <cassert>

constexpr size_t scratch_size = 512 * 1024 * 1024;      // 512 MB scratch allocations

size_t input_size = batch_size * 64 * num_heads * depth * 3 * sizeof(half);
size_t output_size = batch_size * 64 * num_heads * depth * sizeof(half);
size_t intermediate_size = batch_size * num_heads * 64 * 64 * sizeof(half);


const char* CublasGetErrorString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown cublas error";
}

void CublasError(cublasStatus_t status, const char* file, const int& line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
        exit(0);
    }
}

void CudaError(cudaError_t status, const char* file, const int& line) {
    if (status != cudaSuccess) {
        printf("CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
            file, line);
        exit(0);
    }
}

#define ReportCUBLASErrors(status) CublasError(status, __FILE__, __LINE__)
#define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)


void dumpContentsGPU(void* arr, int size, bool fp16)
{
    size_t bytes = size * (fp16 ? sizeof(half) : sizeof(float));
    void* cpuArr = malloc(bytes);
    cudaMemcpy(cpuArr, arr, bytes, cudaMemcpyDeviceToHost);
    dumpContents(cpuArr, size, fp16);
    free(cpuArr);
}


template <typename DataType>
static void cublasXGemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha, DataType** A, int lda,
    DataType** B, int ldb,
    float beta, DataType** C, int ldc, int batchCount) {
    const bool fp16 = std::is_same<half, DataType>::value;

    //printf("\ndoing non-strided batched GEMM of dimension: %d x %d x %d, batch: %d\n", m, n, k, batchCount);

    if (fp16) {
        half alpha_h = (half)(alpha);
        half beta_h = (half)(beta);
        ReportCUBLASErrors(cublasHgemmBatched(
            handle, transa, transb, m, n, k, (const half*)&alpha_h, (half**)A, lda,
            (half**)B, ldb, (const half*)&beta_h, (half**)C, ldc,
            batchCount));
    }
    else {
        ReportCUBLASErrors(cublasSgemmBatched(
            handle, transa, transb, m, n, k, &alpha, (float**)A, lda,
            (float**)B, ldb, &beta, (float**)C, ldc,
            batchCount));
    }
}

// Helper fuction to do vector loads/stores
template <typename T>
__device__ __forceinline__ void copyAs(void* dst, const void* src) {
    *((T*)(dst)) = *((const T*)(src));
}

// fast reduction for the warp
__device__ __forceinline__ float warpReduce(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        x += __shfl_xor_sync(0xFFFFFFFF, x, mask);

    return x;
}

// fast max reduction for the warp
__device__ __forceinline__ float warpMax(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        x = max(x, __shfl_xor_sync(0xFFFFFFFF, x, mask));

    return x;
}


// softmax along C dimension which is assumed to be 64
// each thread processes two elements. Each warp computes a sum (over 64
// elements)
template <typename T>
__global__ void softmax_opt_64_kernel(T* output, const T* input, const T* input2, int N) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N) return;

    float x[4];
    float ex[2];

    // Load from memory
    const bool fp16 = std::is_same<half, T>::value;
    if (fp16) {
        half inp[2];
        copyAs<int>(&inp[0], &input[index * 2]);
        x[0] = (float)inp[0];
        x[1] = (float)inp[1];
        if (input2 != nullptr) {
            copyAs<int>(&inp[0], &input2[index * 2]);
            x[2] = (float)inp[0];
            x[3] = (float)inp[1];
        }
    }
    else {
        copyAs<uint2>(&x[0], &input[index * 2]);
        if (input2 != nullptr) {
            copyAs<uint2>(&x[2], &input2[index * 2]);
        }
    }

    if (input2 != nullptr) {
        x[0] += x[2];
        x[1] += x[3];
    }
    float threadMax = max(x[0], x[1]);
    float maxval = warpMax(threadMax);
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

    ex[0] = exp(x[0] - maxval);
    ex[1] = exp(x[1] - maxval);

    float threadSum = ex[0] + ex[1];
    float Sum = warpReduce(threadSum);
    Sum = __shfl_sync(0xFFFFFFFF, Sum, 0);

    ex[0] = ex[0] / Sum;
    ex[1] = ex[1] / Sum;

    // Store to memory
    if (fp16) {
        half op[2];
        op[0] = (half)ex[0];
        op[1] = (half)ex[1];
        copyAs<int>(&output[index * 2], &op[0]);
    }
    else {
        copyAs<uint2>(&output[index * 2], &ex[0]);
    }
}


template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2, cudaStream_t stream) {
    assert(C == 64);

    int size = N * 32;              // Total no of threads needed
    const int kBlockSize = 256;
    int blocks = divUp(size, kBlockSize);
    softmax_opt_64_kernel<T> << <blocks, kBlockSize, 0, stream >> > (output, input, input2, size);

    ReportCUDAErrors(cudaGetLastError());
}

void testBaselineMHA(half *output, half *input, half *skip, half *inter)
{
    // Run existing implementation of MHA present in lc0 codebase
    half** scratch_rel_ptrs;

    half* mha_q = input;
    half* mha_k = mha_q + (batch_size * 64 * num_heads * depth);
    half* mha_v = mha_k + (batch_size * 64 * num_heads * depth);
    int d_model = num_heads * depth;

    std::vector<half*> offsets(num_heads * batch_size * 5);
    for (int i = 0; i < num_heads * batch_size; i++) {
        int h = i % num_heads;
        int n = i / num_heads;
        offsets[i] = mha_k + h * depth + 64 * d_model * n;
        offsets[i + num_heads * batch_size] = mha_q + h * depth + 64 * d_model * n;
        offsets[i + 2 * num_heads * batch_size] = inter + i * 64 * 64;
        offsets[i + 3 * num_heads * batch_size] = mha_v + h * depth + 64 * d_model * n;
        offsets[i + 4 * num_heads * batch_size] = output + h * depth + 64 * d_model * n;
    }
    ReportCUDAErrors(cudaMalloc((void**)&scratch_rel_ptrs, num_heads * batch_size * 5 * sizeof(half*)));
    ReportCUDAErrors(cudaMemcpy(scratch_rel_ptrs, offsets.data(), num_heads * batch_size * 5 * sizeof(half*),
        cudaMemcpyHostToDevice));


    // create cublas
    cublasHandle_t cublas;
    ReportCUBLASErrors(cublasCreate_v2(&cublas));

    float factor = 1.0f / sqrt((float)depth);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float time = 0;

    // make sure "inter" tensor stays in cache
    cudaStreamAttrValue stream_attribute = {};
    stream_attribute.accessPolicyWindow.base_ptr = inter;
    stream_attribute.accessPolicyWindow.num_bytes = intermediate_size;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    ReportCUDAErrors(cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

    for (int i = 0; i < iterations * 2; i++)
    {
        if (i == iterations)
            cudaEventRecord(startEvent, NULL);

        cublasXGemmBatched<half>(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
            depth /*K*/,  // A/B, and M/N are swapped for row-major to col-major transform
            factor,       // to handle "/ tf.math.sqrt(dk)"
            scratch_rel_ptrs, 
            d_model /*LDA*/,
            scratch_rel_ptrs + num_heads * batch_size,
            d_model /*LDB*/,
            0.0f,
            scratch_rel_ptrs + num_heads * batch_size * 2,
            64 /*LDC*/,
            batch_size * num_heads);

        // Add smolgen weights to the scaled matmul_qk attention logits before softmax.
        Softmax(num_heads * batch_size * 64, 64, inter, inter, skip, 0);

        cublasXGemmBatched<half>(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N, depth /*M*/, 64 /*N*/, 64 /*K*/, 1.0f,
            scratch_rel_ptrs + num_heads * batch_size * 3,
            d_model /*LDA*/,
            scratch_rel_ptrs + num_heads * batch_size * 2, 
            64 /*LDB*/,
            0.0f,
            scratch_rel_ptrs + num_heads * batch_size * 4,
            d_model /*LDC*/,
            batch_size * num_heads);
    }

    cudaEventRecord(stopEvent, NULL);
    cudaEventSynchronize(stopEvent);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, startEvent, stopEvent);

    printf("\nAverage time taken by Baseline implementation: %g ms\n", msecTotal / iterations);

    ReportCUDAErrors(cudaFree(scratch_rel_ptrs));
}


#include "fused_multi_head_attention/kernel_forward.h"

void testCutlassFusedMHA(half* output, half* input, half* skip, half* inter)
{
    cutlass::half_t* mha_q = (cutlass::half_t*)input;
    cutlass::half_t* mha_k = mha_q + (batch_size * 64 * num_heads * depth);
    cutlass::half_t* mha_v = mha_k + (batch_size * 64 * num_heads * depth);


    constexpr int kQueriesPerBlock = 64;
    constexpr int kKeysPerBlock = 64;
    constexpr bool kSingleValueIteration = true;

    using Attention = AttentionKernel<
        cutlass::half_t,      // scalar_t
        cutlass::arch::Sm80,  // ArchTag
        true,                 // Memory is aligned
        kQueriesPerBlock,
        kKeysPerBlock,
        kSingleValueIteration,
        false,                // Supports dropout
        true                  // Supports bias
    >;

    typename Attention::Params p;
    { // set parameters
        p.query_ptr = mha_q;
        p.key_ptr = mha_k;
        p.value_ptr = mha_v;
        p.logsumexp_ptr = nullptr; // Only needed for bw
        p.output_accum_ptr = nullptr;
        if (Attention::kNeedsOutputAccumulatorBuffer) {
            printf("\nAnkan - check this out, allocating more memory for intermediate tensor?!\n");
            cudaMalloc(&p.output_accum_ptr, (output_size / sizeof(half)) * sizeof(typename Attention::output_accum_t));
        }
        p.output_ptr = (cutlass::half_t*)output;
        p.attn_bias_ptr = (cutlass::half_t*)skip;

        p.scale = 1.0f / sqrt((float)depth);

        p.num_heads = num_heads;
        p.num_batches = batch_size;
        p.head_dim = depth;
        p.head_dim_value = depth;
        p.num_queries = 64;
        p.num_keys = 64;
        if (false) {
            // Ankan - check what does this mean??
            p.custom_mask_type = Attention::CausalFromTopLeft;
        }

        // All tensors are in BMHK shapes
        p.q_strideH = depth;
        p.k_strideH = depth;
        p.v_strideH = depth;
        p.q_strideM = depth * num_heads;
        p.k_strideM = depth * num_heads;
        p.v_strideM = depth * num_heads;
        p.q_strideB = p.q_strideM * 64;
        p.k_strideB = p.k_strideM * 64;
        p.v_strideB = p.v_strideM * 64;
        p.o_strideM = p.head_dim_value * p.num_heads;

        // Ankan - TODO: check layout of the skip connection tensor.
        p.bias_strideH = 64;
        p.bias_strideM = 64 * 64;
        p.bias_strideB = num_heads * p.bias_strideM;
    }

    // launch kernel :)
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    if (!Attention::check_supported(p)) {
        std::cerr << "Kernel does not support these inputs" << std::endl;
        exit(0);
    }

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float time = 0;

    for (int i = 0; i < iterations * 2; i++)
    {
        if (i == iterations)
            cudaEventRecord(startEvent, NULL);

        kernel_fn << <p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes >> > (p);
    }

    cudaEventRecord(stopEvent, NULL);
    cudaEventSynchronize(stopEvent);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, startEvent, stopEvent);
    printf("\nAverage time taken by cutlass implementation: %g ms\n", msecTotal / iterations);


    // Wait for completion
    ReportCUDAErrors(cudaDeviceSynchronize());
}

// stride of inner-most dimensions is always assumed to be 1
// the paramaters specify strides of the "other" dimension.
void matrixMultiplyCpu(float* output, float* A, float* B, int H, int W, int K, int strideA, int strideB, int strideOp, float alpha, bool bTrans)
{
    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            float S = 0;
            for (int k = 0; k < K; k++)
            {
                float a = A[h * strideA + k];
                float b = bTrans ? B[w * strideB + k] : B[k * strideB + w];
                S += a * b;
            }
            output[h * strideOp + w] = S * alpha;
        }
    }
}

void testCPUMHA(half* op, half* input, half* sk)
{
    int tensorElements = (batch_size * 64 * num_heads * depth);
    int interElements = (batch_size * num_heads * 64 * 64);

    half* mha_q = input;
    half* mha_k = mha_q + tensorElements;
    half* mha_v = mha_k + tensorElements;

    // first convert tensors to FP32 for better efficiency on CPU;
    float* q = (float*)malloc(tensorElements * sizeof(float));
    float* k = (float*)malloc(tensorElements * sizeof(float));
    float* v = (float*)malloc(tensorElements * sizeof(float));

    float* skip = (float*)malloc(interElements * sizeof(float));
    float* output = (float*)malloc(tensorElements * sizeof(float));
    float* inter = (float*)malloc(interElements * sizeof(float));

    for (int i = 0; i < tensorElements; i++)
    {
        q[i] = (float)mha_q[i];
        k[i] = (float)mha_k[i];
        v[i] = (float)mha_v[i];
    }

    for (int i = 0; i < interElements; i++)
    {
        skip[i] = (float)sk[i];
    }

    float alpha =  1.0f / sqrt((float)depth);

    // 1. matmul_qk = tf.matmul(q, k, transpose_b=True)
    for (int b = 0; b < batch_size; b++)
    {
        for (int h = 0; h < num_heads; h++)
        {
            // batch_size * num_heads many matrix multiplications
            float* matOut = inter + (b * num_heads + h) * 64 * 64;
            float* matA = q + b * 64 * num_heads * depth + h * depth;
            float* matB = k + b * 64 * num_heads * depth + h * depth;
            matrixMultiplyCpu(matOut, matA, matB, 64, 64, depth, num_heads * depth, num_heads * depth, 64, alpha, true);
        }
    }

    // 2. skip connection addition
    for (int i = 0; i < batch_size * num_heads * 64 * 64; i++)
        inter[i] += skip[i];

    // 3. softmax
    int outer_size = batch_size * num_heads * 64;
    for (int o = 0; o < outer_size; o++)
    {
        int startIndex = o * 64;

        float max = inter[startIndex];
        for (int i = 0; i < 64; i++)
        {
            float x = inter[startIndex + i];
            max = std::max(x, max);
        }

        float sumEx = 0;
        for (int i = 0; i < 64; i++)
        {
            float x = inter[startIndex + i];
            sumEx += exp(x - max);
        }

        // Finally compute and write softmax
        for (int i = 0; i < 64; i++)
        {
            float x = inter[startIndex + i];
            x = exp(x - max) / sumEx;
            inter[startIndex + i] = x;
        }
    }

    // 4. output = tf.matmul(attention_weights, v)
    for (int b = 0; b < batch_size; b++)
    {
        for (int h = 0; h < num_heads; h++)
        {
            // batch_size * num_heads many matrix multiplications
            float* matA = inter + (b * num_heads + h) * 64 * 64;
            float* matB = v + b * 64 * num_heads * depth + h * depth;
            float* matOut = output + b * 64 * num_heads * depth + h * depth;
            matrixMultiplyCpu(matOut, matA, matB, 64, depth, 64, 64, num_heads * depth, num_heads * depth, 1.0f, false);
        }
    }

    // convert output from float to half
    for (int i = 0; i < tensorElements; i++)
    {
        op[i] = (half)output[i];
    }

    free(q);
    free(k);
    free(v);
    free(inter);
    free(skip);
    free(output);
}

void testMHA()
{
    half* skip = 0;         // skip connection to be added before softmax
    half *input = 0;        // kqv matrices
    half *output = 0;
    half* inter = 0;        // product of q*k

    half* cpuInput;
    half* cpuOutput;
    half* refOutput;
    half* cpuSkip;

    // allocate CPU resources
    cpuInput = (half*)malloc(input_size);
    cpuOutput = (half*)malloc(output_size);
    refOutput = (half*)malloc(output_size);
    cpuSkip = (half*)malloc(intermediate_size);

    // fill with random data
    fillRandomArray(cpuInput, input_size / 2, true, 1.0f);
    fillRandomArray(cpuSkip, intermediate_size / 2, true, 0.5f);

    // allocate GPU resources
    ReportCUDAErrors(cudaMalloc((void**)&input, input_size));
    ReportCUDAErrors(cudaMalloc((void**)&output, output_size));
    ReportCUDAErrors(cudaMalloc((void**)&skip, intermediate_size));
    ReportCUDAErrors(cudaMalloc((void**)&inter, intermediate_size));

    // copy inputs to GPU
    ReportCUDAErrors(cudaMemset(output, 0, output_size));
    ReportCUDAErrors(cudaMemcpy(input, cpuInput, input_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(skip, cpuSkip, intermediate_size, cudaMemcpyHostToDevice));

    testBaselineMHA(output, input, skip, inter);
    ReportCUDAErrors(cudaMemcpy(cpuOutput, output, output_size, cudaMemcpyDeviceToHost));
    //dumpContents(cpuOutput, 100, true);

    testCutlassFusedMHA(output, input, skip, inter);
    ReportCUDAErrors(cudaMemcpy(cpuOutput, output, output_size, cudaMemcpyDeviceToHost));
    //dumpContents(cpuOutput, 100, true);

    printf("\nComputing on CPU...\n");
    testCPUMHA(refOutput, cpuInput, cpuSkip);
    //dumpContents(refOutput, 100, true);

    compareResults(refOutput, cpuOutput, output_size / sizeof(half), true);

    ReportCUDAErrors(cudaFree(input));
    ReportCUDAErrors(cudaFree(output));
    ReportCUDAErrors(cudaFree(skip));
    ReportCUDAErrors(cudaFree(inter));
    
    free(refOutput);
    free(cpuInput);
    free(cpuOutput);
    free(cpuSkip);
}


int main()
{
    cudaSetDevice(0);

    testMHA();

    cudaDeviceReset();

    return 0;
}
