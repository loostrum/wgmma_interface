#include <iostream>
#include <functional>
#include <random>

#include <cudawrappers/cu.hpp>
#include <cuda_fp16.h>

#include "wgmma.h"

__global__ void kernel_ref(const half *A, const half *B, float *C, const size_t M, const size_t N, const size_t K, const size_t multiplier) {
    size_t m = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n = threadIdx.y + blockIdx.y * blockDim.y;
    if (m >= M | n >= N) {
        return;
    }

    float sum = 0;
    for (int k=0; k < K; k++) {
      sum += static_cast<float>(A[m * K + k]) * static_cast<float>(B[n * K + k]);
    }
    C[m * N + n] = sum * multiplier;
}

template<size_t M, size_t N, size_t K, size_t REPEAT_COUNT=1, size_t WGMMA_COUNT=1>
__global__ void kernel_wgmma(const half *A, const half *B, float *C) {
    const size_t nthreads = blockDim.x * blockDim.y * blockDim.z;
    const size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int a[4];
    wgmma::load_matrix_a(a, A, K);

    const size_t numB = N * K;
    __shared__ __align__(16) half b[numB];
    for (size_t idx=tid; idx < numB; idx += nthreads) {
        const size_t core_matrix_N = 8;
        const size_t core_matrix_K = 8;
        // no swizzle means core matrices have to have adjacent elements
        // like the ccglib transpose kernel: tiles are contiguous in memory
        // B matrix is N x K, K major i.e. contiguous in K
        size_t n = idx / K;
        size_t k = idx % K;
        // calculate output index. First get index of core matrix
        size_t core_matrix_n = n / core_matrix_N;
        size_t core_matrix_k = k / core_matrix_K;
        size_t core_matrix_index = core_matrix_k * (N / core_matrix_N) + core_matrix_n;  // n-major!
        size_t core_matrix_start = core_matrix_index * core_matrix_N * core_matrix_K; // start position of this core matrix
        size_t core_n = n % core_matrix_N;
        size_t core_k = k % core_matrix_K;
        size_t out_idx = core_matrix_start + core_n * core_matrix_K + core_k;
        b[out_idx] = B[idx];
    }
    __syncthreads();

    float c[N/2] = {0};

    unsigned lds = 2 * N * 8; // 2048
    unsigned sds = 128;
    wgmma::SwizzleMode swizzle = wgmma::SwizzleMode::Interleaved;
    unsigned base_offset = 0;
    unsigned long addr = reinterpret_cast<unsigned long>(&b[0]);

    unsigned long descB = wgmma::make_descriptor(addr, lds, sds, base_offset, swizzle);

    for (size_t repeat=0; repeat < REPEAT_COUNT; repeat++) {
        wgmma::arrive();
        for (size_t counter = 0; counter < WGMMA_COUNT; counter++) {
            wgmma::wgmma_async(a, descB, c);
        }
        wgmma::commit();
        wgmma::wait<0>();
    }

    wgmma::store_matrix(c, C, N);
}


int main() {
    constexpr unsigned M = 64;
    constexpr unsigned N = 128;
    constexpr unsigned K = 16;
    constexpr unsigned REPEAT_COUNT = 256;
    constexpr unsigned WGMMA_COUNT = 16;
    constexpr unsigned ITERATIONS = 16;

    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_BLOCKING_SYNC, device);
    cu::Stream stream;

    auto generator = std::bind(std::uniform_int_distribution<int>(-10, 10),
                               std::default_random_engine());

    size_t bytes_a = sizeof(half) * M * K;
    size_t bytes_b = sizeof(half) * N * K;
    size_t bytes_c = sizeof(float) * M * N;

    half *a, *b; 
    float *c, *c_ref, *c_ref_host;
    cudaMallocHost(&a, bytes_a);
    cudaMallocHost(&b, bytes_b);
    cudaMallocHost(&c, bytes_c);
    cudaMallocHost(&c_ref, bytes_c);
    cudaMallocHost(&c_ref_host, bytes_c);

    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    for (size_t i = 0; i < M * K; i++) {
        a[i] = (half)generator();
    }   

    for (size_t i = 0; i < N * K; i++) {
        b[i] = (half)generator();
    }   

    dim3 threads_ref{32, 32, 1}; 
    dim3 grid_ref{M / threads_ref.x + 1, N / threads_ref.y + 1, 1}; 

    cudaMemcpy(d_a, a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes_b, cudaMemcpyHostToDevice);
    kernel_ref<<<grid_ref, threads_ref>>>(d_a, d_b, d_c, M, N, K, REPEAT_COUNT * WGMMA_COUNT);
    cudaDeviceSynchronize();
    cudaMemcpy(c_ref, d_c, bytes_c, cudaMemcpyDeviceToHost);

    dim3 threads{128, 1, 1}; 
    dim3 grid{1, 1, 1}; 
    cudaMemset(d_c, 0, bytes_c);
    kernel_wgmma<M, N, K, REPEAT_COUNT, WGMMA_COUNT><<<grid, threads>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    int errs = 0;
    for (size_t m=0; m < M; m++) {
        for (size_t n=0; n < N; n++) {
            float diff = c[m * N + n] - c_ref[m * N + n];
            if (diff != 0) errs++;
        }
    }
    std::cout << "Result " << (errs > 0 ? "Not " : "") << "OK" << std::endl;

    // benchmark
    int multiProcessorCount = device.getAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);

    // Kernel dimensions
    int nr_thread_blocks = multiProcessorCount * 512;
    dim3 grid_bench(nr_thread_blocks);
    dim3 threads_bench(128);
    double gops = 1e-9 * 2 * M * N * K * WGMMA_COUNT * REPEAT_COUNT * nr_thread_blocks;
    std::array<double, ITERATIONS> tflops;
    cu::Event start, end;
    for (size_t i=0; i < ITERATIONS; i++) {
        stream.record(start);
        kernel_wgmma<M, N, K, REPEAT_COUNT, WGMMA_COUNT><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
        stream.record(end);
        end.synchronize();
        stream.synchronize();
        float time = end.elapsedTime(start);
        double perf = gops / time; // time in ms converts giga to tera
        tflops[i] = perf;
        std::cout << "TFLOPS: " << perf << std::endl;
    } 
    double tflops_avg = 0;
    double tflops_sq = 0;
    for (auto & item : tflops) {
        tflops_avg += item;
        tflops_sq += item * item;
    }
    tflops_avg /= ITERATIONS;
    tflops_sq /= ITERATIONS;
    // stddev = mean of sq - sq of mean
    double tflops_stddev = tflops_sq - tflops_avg * tflops_avg;
    std::cout << "Average TFLOPS: " << tflops_avg << " +/- " << tflops_stddev << std::endl;

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
}

