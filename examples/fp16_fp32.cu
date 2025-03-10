#include <iostream>
#include <functional>
#include <random>

#include <cudawrappers/cu.hpp>
#include <cuda_fp16.h>

#include "wgmma.h"

__global__ void kernel_ref(const half *A, const half *B, float *C, const size_t M, const size_t N, const size_t K) {
    size_t m = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n = threadIdx.y + blockIdx.y * blockDim.y;
    if (m >= M | n >= N) {
        return;
    }

    float sum = 0;
    for (int k=0; k < K; k++) {
      sum += static_cast<float>(A[m * K + k]) * static_cast<float>(B[n * K + k]);
    }
    C[m * N + n] = sum;
}

template<size_t M_PER_BLOCK, size_t N_PER_BLOCK,
          size_t M_PER_WG, size_t N_PER_WG,
          size_t M_WGMMA, size_t N_WGMMA, size_t K_WGMMA>
__global__ void kernel_wgmma(const half *A, const half *B, float *C) {
    const size_t nthreads = blockDim.x * blockDim.y * blockDim.z;
    const size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    wgmma::fragment<wgmma::matrix_a, M_WGMMA, N_WGMMA, K_WGMMA, half, wgmma::row_major> a;
    __shared__ __align__(16) wgmma::fragment<wgmma::matrix_b, M_WGMMA, N_WGMMA, K_WGMMA, half, wgmma::col_major> b;
    wgmma::fragment<wgmma::accumulator, M_WGMMA, N_WGMMA, K_WGMMA, float> c;
    wgmma::fill_fragment(c, 0);

    wgmma::SwizzleMode swizzle = wgmma::SwizzleMode::Interleaved;

    wgmma::load_matrix(a, A, K_WGMMA);
    wgmma::load_matrix(b, B, K_WGMMA, swizzle, tid, nthreads);
    __syncthreads();
    wgmma::smem_fence();

    unsigned long descB = wgmma::make_descriptor(b, swizzle);
    wgmma::arrive();
    wgmma::mma_async(a, descB, c);
    wgmma::commit();
    wgmma::wait();
    wgmma::store_matrix(c, C, N_WGMMA, wgmma::mem_row_major);
}


int main() {
    //constexpr unsigned M = 256;
    //constexpr unsigned N = 256;
    //constexpr unsigned K = 256;
    constexpr unsigned M = 64;
    constexpr unsigned N = 128;
    constexpr unsigned K = 16;

    constexpr unsigned M_PER_BLOCK = M;
    constexpr unsigned N_PER_BLOCK = N;

    constexpr unsigned M_PER_WG = M;
    constexpr unsigned N_PER_WG = N;

    constexpr unsigned M_WGMMA = 64;
    constexpr unsigned N_WGMMA = 128;
    constexpr unsigned K_WGMMA = 16;

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
    kernel_ref<<<grid_ref, threads_ref>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(c_ref, d_c, bytes_c, cudaMemcpyDeviceToHost);

    dim3 threads{128, N_PER_BLOCK/N_PER_WG, M_PER_BLOCK/M_PER_WG};
    dim3 grid{N/N_PER_BLOCK, M/M_PER_BLOCK, 1};
    cudaMemset(d_c, 0, bytes_c);
    kernel_wgmma<M_PER_BLOCK, N_PER_BLOCK, M_PER_WG, N_PER_WG, M_WGMMA, N_WGMMA, K_WGMMA><<<grid, threads>>>(d_a, d_b, d_c);
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
    double gops = 1e-9 * 2 * M * N * K * nr_thread_blocks;
    cu::Event start, end;
    stream.record(start);
    kernel_wgmma<M_PER_BLOCK, N_PER_BLOCK, M_PER_WG, N_PER_WG, M_WGMMA, N_WGMMA, K_WGMMA><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
    stream.record(end);
    end.synchronize();
    stream.synchronize();
    float time = end.elapsedTime(start);
    double tflops = gops / time; // time in ms converts giga to tera
    std::cout << "TFLOPS: " << tflops << std::endl;

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
}
