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

template<size_t M, size_t N, size_t K,
         size_t M_PER_BLOCK, size_t N_PER_BLOCK,
         size_t M_PER_WG, size_t N_PER_WG,
         size_t M_WGMMA, size_t N_WGMMA, size_t K_WGMMA>
__global__ void kernel_wgmma(const half *A, const half *B, float *C) {
    const size_t blockN = blockIdx.x;
    const size_t blockM = blockIdx.y;
    const size_t wgN = threadIdx.y;
    const size_t wgM = threadIdx.z;

    constexpr size_t M_TILES = M_PER_WG / M_WGMMA;
    constexpr size_t N_TILES = N_PER_WG / N_WGMMA;
    constexpr size_t K_TILES = K / K_WGMMA;

    const size_t nthreads = blockDim.x * blockDim.y * blockDim.z;
    const size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    wgmma::fragment<wgmma::matrix_a, M_WGMMA, N_WGMMA, K_WGMMA, half, wgmma::row_major> a[M_TILES];
    __shared__ __align__(16) wgmma::fragment<wgmma::matrix_b, M_WGMMA, N_WGMMA, K_WGMMA, half, wgmma::col_major> b[N_TILES];
    wgmma::fragment<wgmma::accumulator, M_WGMMA, N_WGMMA, K_WGMMA, float> c[M_TILES][N_TILES];

    for (size_t m = 0; m < M_TILES; m++) {
        for (size_t n = 0; n < N_TILES; n++) {
            wgmma::fill_fragment(c[m][n], 0);
        }
    }

    wgmma::SwizzleMode swizzle = wgmma::SwizzleMode::Interleaved;
    unsigned long descB[N_TILES];
    for (size_t n = 0; n < N_TILES; n++) {
        descB[n] = wgmma::make_descriptor(b[n], swizzle);
    }

    for (size_t k = 0; k < K_TILES; k++) {
        const size_t k_index = k * K_WGMMA;

        for (size_t m = 0; m < M_TILES; m++) {
            const size_t global_m = blockM * M_PER_BLOCK + wgM * M_PER_WG + m * M_WGMMA;
            wgmma::load_matrix(a[m], &A[global_m * K + k_index], K);
        }

        for (size_t n = 0; n < N_TILES; n++) {
            const size_t global_n = blockN * N_PER_BLOCK + wgN * N_PER_WG + n * N_WGMMA;
            wgmma::load_matrix(b[n], &B[global_n * K + k_index], K, swizzle, tid, nthreads);
        }

        __syncthreads();
        wgmma::smem_fence();

        wgmma::arrive();
        for (size_t m = 0; m < M_TILES; m++) {
            for (size_t n = 0; n < N_TILES; n++) {
                wgmma::mma_async(a[m], descB[n], c[m][n]);
            }
        }
        wgmma::commit();
        wgmma::wait();
    }

    for (size_t m = 0; m < M_TILES; m++) {
        for (size_t n = 0; n < N_TILES; n++) {
            const size_t global_m = blockM * M_PER_BLOCK + wgM * M_PER_WG + m * M_WGMMA;
            const size_t global_n = blockN * N_PER_BLOCK + wgN * N_PER_WG + n * N_WGMMA;
            wgmma::store_matrix(c[m][n], &C[global_m * N + global_n], N, wgmma::mem_row_major);
        }
    }
}


int main() {
    constexpr unsigned M = 64;
    constexpr unsigned N = 128;
    constexpr unsigned K = 16;
    //constexpr unsigned M = 64;
    //constexpr unsigned N = 128;
    //constexpr unsigned K = 16;

    constexpr unsigned M_PER_BLOCK = M;
    constexpr unsigned N_PER_BLOCK = N;

    constexpr unsigned M_PER_WG = M;
    constexpr unsigned N_PER_WG = N;

    constexpr unsigned M_WGMMA = 64;
    constexpr unsigned N_WGMMA = 128;
    constexpr unsigned K_WGMMA = 16;

    constexpr unsigned M_TILES = M_PER_WG / M_WGMMA;
    constexpr unsigned N_TILES = N_PER_WG / N_WGMMA;
    constexpr unsigned N_WGMMA_PER_GROUP = M_TILES * N_TILES;
    std::cout << "Number of WGMMA instructions per async group: " << N_WGMMA_PER_GROUP << std::endl;

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
    cudaMemset(d_c, 0, bytes_c);

    dim3 threads{128, N_PER_BLOCK/N_PER_WG, M_PER_BLOCK/M_PER_WG};
    dim3 grid{N/N_PER_BLOCK, M/M_PER_BLOCK, 1};

    double gops = 1e-9 * 2 * M * N * K;
    cu::Event start, end;
    stream.record(start);
    kernel_wgmma<M, N, K, M_PER_BLOCK, N_PER_BLOCK, M_PER_WG, N_PER_WG, M_WGMMA, N_WGMMA, K_WGMMA><<<grid, threads, 0, stream>>>(d_a, d_b, d_c);
    stream.record(end);
    end.synchronize();
    stream.synchronize();
    float time = end.elapsedTime(start);
    double tflops = gops / time; // time in ms converts giga to tera
    std::cout << "TFLOPS: " << tflops << std::endl;
    cudaMemcpy(c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    int errs = 0;
    for (size_t m=0; m < M; m++) {
        for (size_t n=0; n < N; n++) {
            float diff = c[m * N + n] - c_ref[m * N + n];
            if (diff != 0) errs++;
        }
    }
    std::cout << "Result " << (errs > 0 ? "Not " : "") << "OK" << std::endl;

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
}
