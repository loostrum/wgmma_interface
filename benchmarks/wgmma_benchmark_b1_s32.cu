#include <iostream>
#include <functional>
#include <random>
#include <limits>
#include <cmath>

#include <cudawrappers/cu.hpp>

#include "wgmma.hpp"

__global__ void kernel_ref(const int *A, const int *B, int *C, const size_t M, const size_t N, const size_t K, const size_t multiplier) {
    size_t m = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n = threadIdx.y + blockIdx.y * blockDim.y;
    if (m >= M | n >= N) {
        return;
    }
    const size_t packing_factor = 32;
    const size_t K_packed = K / packing_factor;

    int sum = 0;
    for (int k = 0; k < K_packed; k++) {
      sum += __popc(A[m * K_packed + k] & B[n * K_packed + k]);
    }
    C[m * N + n] = sum * multiplier;
}

template<size_t M, size_t N, size_t K, size_t REPEAT_COUNT=1, size_t WGMMA_COUNT=1>
__global__ void kernel_wgmma(const int *A, const int *B, int *C) {
    const size_t nthreads = blockDim.x * blockDim.y * blockDim.z;
    const size_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    wgmma::fragment<wgmma::matrix_a, M, N, K, wgmma::precision::b1, wgmma::row_major> a;
    __shared__ __align__(16) wgmma::fragment<wgmma::matrix_b, M, N, K, wgmma::precision::b1, wgmma::col_major> b;
    wgmma::fragment<wgmma::accumulator, M, N, K, int> c;
    wgmma::fill_fragment(c, 0);

    wgmma::SwizzleMode swizzle = wgmma::SwizzleMode::Interleaved;

    wgmma::load_matrix(a, A, K);
    wgmma::load_matrix(b, B, K, swizzle, tid, nthreads);
    __syncthreads();
    wgmma::smem_fence();

    unsigned long descB = wgmma::make_descriptor(b, swizzle);

    for (size_t repeat=0; repeat < REPEAT_COUNT; repeat++) {
        wgmma::arrive();
        for (size_t counter = 0; counter < WGMMA_COUNT; counter++) {
            wgmma::mma_async(a, descB, c);
        }
        wgmma::commit();
        wgmma::wait();
    }

    wgmma::store_matrix(C, c, N, wgmma::mem_row_major);
}


int main() {
    constexpr unsigned M = 64;
    constexpr unsigned K = 256;
    constexpr unsigned REPEAT_COUNT = 256;
    constexpr unsigned WGMMA_COUNT = 16;
    constexpr unsigned ITERATIONS = 4;

    constexpr std::array<unsigned, 4> N_values{8, 64, 128, 256};
    const unsigned maxN = *std::max_element(N_values.begin(), N_values.end());

    cu::init();
    cu::Device device(0);
    cu::Context context(CU_CTX_BLOCKING_SYNC, device);
    cu::Stream stream;

    auto generator = std::bind(std::uniform_int_distribution<int>(INT_MIN, INT_MAX),
                               std::default_random_engine());

    const size_t packing_factor = 32;
    const size_t K_packed = K / packing_factor;
    const size_t bytes_a = sizeof(int) * M * K_packed;
    const size_t bytes_b = sizeof(int) * maxN * K_packed;
    const size_t bytes_c = sizeof(int) * M * maxN;

    int *a, *b;
    int *c, *c_ref, *c_ref_host;
    cudaMallocHost(&a, bytes_a);
    cudaMallocHost(&b, bytes_b);
    cudaMallocHost(&c, bytes_c);
    cudaMallocHost(&c_ref, bytes_c);
    cudaMallocHost(&c_ref_host, bytes_c);

    int *d_a, *d_b;
    int *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    for (size_t i = 0; i < M * K_packed; i++) {
        a[i] = generator();
    }

    for (size_t i = 0; i < maxN * K_packed; i++) {
        b[i] = generator();
    }

    std::cout << "Performance is average of " << ITERATIONS << " iterations." << std::endl;
    for (const unsigned &N : N_values) {
        std::cout << "MxNxK = " << M << "x" << N << "x" << K << std::endl;
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
        switch(N) {
            case 8:
                kernel_wgmma<M,   8, K, REPEAT_COUNT, WGMMA_COUNT><<<grid, threads>>>(d_a, d_b, d_c);
                break;
            case 64:
                kernel_wgmma<M,  64, K, REPEAT_COUNT, WGMMA_COUNT><<<grid, threads>>>(d_a, d_b, d_c);
                break;
            case 128:
                kernel_wgmma<M, 128, K, REPEAT_COUNT, WGMMA_COUNT><<<grid, threads>>>(d_a, d_b, d_c);
                break;
            case 256:
                kernel_wgmma<M, 256, K, REPEAT_COUNT, WGMMA_COUNT><<<grid, threads>>>(d_a, d_b, d_c);
                break;
        }
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
        dim3 threads_bench(wgmma::WARPGROUP_SIZE);
        double gops = 1e-9 * 2 * M * N * K * WGMMA_COUNT * REPEAT_COUNT * nr_thread_blocks;
        std::array<double, ITERATIONS> tops;
        cu::Event start, end;
        for (size_t i=0; i < ITERATIONS; i++) {
            stream.record(start);
            switch(N) {
                case 8:
                    kernel_wgmma<M,   8, K, REPEAT_COUNT, WGMMA_COUNT><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
                    break;
                case 64:
                    kernel_wgmma<M,  64, K, REPEAT_COUNT, WGMMA_COUNT><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
                    break;
                case 128:
                    kernel_wgmma<M, 128, K, REPEAT_COUNT, WGMMA_COUNT><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
                    break;
                case 256:
                    kernel_wgmma<M, 256, K, REPEAT_COUNT, WGMMA_COUNT><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
                    break;
            }
            stream.record(end);
            end.synchronize();
            stream.synchronize();
            float time = end.elapsedTime(start);
            tops[i] = gops / time; // time in ms converts giga to tera
        }
        double tops_avg = 0;
        double tops_sq = 0;
        for (auto & item : tops) {
            tops_avg += item;
            tops_sq += item * item;
        }
        tops_avg /= ITERATIONS;
        tops_sq /= ITERATIONS;
        // stddev = sqrt(mean of sq - sq of mean)
        double tops_stddev = std::sqrt(tops_sq - tops_avg * tops_avg);
        std::cout << "TOPS: " << tops_avg << " +/- " << tops_stddev << std::endl << std::endl;
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
}
