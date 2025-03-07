#include <cuda_fp16.h>

#include "wgmma_common.h"

namespace wgmma {

  template<> class fragment<wgmma::matrix_a, 64, 128, 16, half, wgmma::row_major> : public Storage<int, 4> {};
  template<> __shared__ __align__(16)  class fragment<wgmma::matrix_b, 64, 128, 16, half, wgmma::col_major> {
    public:
      half x[128*16];
  };

  // 64x128x16
  // A
  template<> inline __device__ void load_matrix(fragment<wgmma::matrix_a, 64, 128, 16, half, wgmma::row_major> &frag, const half *A, const size_t ldm) {
    size_t laneid = threadIdx.x % 128;
    size_t first_row = laneid / 4 + 8 * (laneid / 32);
    size_t first_col = 2 * (laneid % 4);
    size_t idx;

    idx = first_row * ldm + first_col;
    reinterpret_cast<half2 *>(frag.x)[0] = __halves2half2(A[idx], A[idx+1]);
    idx = (first_row + 8) * ldm + first_col;
    reinterpret_cast<half2 *>(frag.x)[1] = __halves2half2(A[idx], A[idx+1]);
    idx = first_row * ldm + (first_col + 8);
    reinterpret_cast<half2 *>(frag.x)[2] = __halves2half2(A[idx], A[idx+1]);
    idx = (first_row + 8) * ldm + (first_col + 8);
    reinterpret_cast<half2 *>(frag.x)[3] = __halves2half2(A[idx], A[idx+1]);
  }

  // B
  template<> inline __device__ void load_matrix(fragment<wgmma::matrix_b, 64, 128, 16, half, wgmma::col_major> &frag, const half *B, const size_t ldm, const size_t tid, const size_t nthreads) {
    for (size_t idx=tid; idx < 128*16; idx += nthreads) {
      const size_t core_matrix_N = 8;
      const size_t core_matrix_K = 8;
      // no swizzle means core matrices have to have adjacent elements
      // like the ccglib transpose kernel: tiles are contiguous in memory
      // B matrix is N x K, K major i.e. contiguous in K
      size_t n = idx / ldm;
      size_t k = idx % ldm;
      // calculate output index. First get index of core matrix
      size_t core_matrix_n = n / core_matrix_N;
      size_t core_matrix_k = k / core_matrix_K;
      size_t core_matrix_index = core_matrix_k * (128 / core_matrix_N) + core_matrix_n;  // n-major!
      size_t core_matrix_start = core_matrix_index * core_matrix_N * core_matrix_K; // start position of this core matrix
      size_t core_n = n % core_matrix_N;
      size_t core_k = k % core_matrix_K;
      size_t out_idx = core_matrix_start + core_n * core_matrix_K + core_k;
      frag.x[out_idx] = B[idx];
    }
  }

  inline __device__ void store_matrix(const float *c, float *C, const size_t N) {
    size_t laneid = threadIdx.x % 128;
    size_t first_row = laneid / 4 + 8 * (laneid / 32);
    size_t first_col = 2 * (laneid % 4);

    for (size_t i = 0; i < N / 2; i++) {
      size_t row = first_row + 8 * ((i % 4) / 2);
      size_t col = first_col + i % 2 + 8 * (i / 4);
      C[row * N + col] = c[i];
    }
  }

  inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 128, 16, half, wgmma::row_major> &a, const unsigned long descB, float *c) {
    constexpr int scaleA = 1;
    constexpr int scaleB = 1;
    constexpr int transB = 0;
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %72, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, %69, %70, %71;\n"
         "}"
         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]), "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]), "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]), "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]), "+f"(c[32]), "+f"(c[33]), "+f"(c[34]), "+f"(c[35]), "+f"(c[36]), "+f"(c[37]), "+f"(c[38]), "+f"(c[39]), "+f"(c[40]), "+f"(c[41]), "+f"(c[42]), "+f"(c[43]), "+f"(c[44]), "+f"(c[45]), "+f"(c[46]), "+f"(c[47]), "+f"(c[48]), "+f"(c[49]), "+f"(c[50]), "+f"(c[51]), "+f"(c[52]), "+f"(c[53]), "+f"(c[54]), "+f"(c[55]), "+f"(c[56]), "+f"(c[57]), "+f"(c[58]), "+f"(c[59]), "+f"(c[60]), "+f"(c[61]), "+f"(c[62]), "+f"(c[63])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleA), "n"(scaleB), "n"(transB), "n"(scaleD));
  }

}  // end namespace wgmma
