#include "wgmma_common.h"

namespace wgmma {

  // 64xNx256
  template<unsigned N> class fragment<wgmma::matrix_a, 64, N, 256, wgmma::precision::b1, wgmma::row_major> : public Storage<int, 4> {};
  template<unsigned N> class fragment<wgmma::matrix_b, 64, N, 256, wgmma::precision::b1, wgmma::col_major> : public Storage<int, N*8 /* N*K/packing_factor */> {};
  template<unsigned N> class fragment<wgmma::accumulator, 64, N, 256, int> : public Storage<int, N/2> {};


  template<unsigned N> inline __device__ void load_matrix(fragment<wgmma::matrix_a, 64, N, 256, wgmma::precision::b1, wgmma::row_major> &frag, const int *A, const size_t ldm) {
    size_t laneid = threadIdx.x % 128;
    size_t first_row = laneid / 4 + 8 * (laneid / 32);
    //size_t first_col = 2 * (laneid % 4);
    size_t first_col = laneid % 4;
    size_t idx;
    const size_t packing_factor = 32;  // 32 values per int
    const size_t ldm_packed = ldm / packing_factor;

    idx = first_row * ldm_packed + first_col;
    frag.x[0] = A[idx];
    idx = (first_row + 8) * ldm_packed + first_col;
    frag.x[1] = A[idx];
    idx = first_row * ldm_packed + (first_col + 4);
    frag.x[2] = A[idx];
    idx = (first_row + 8) * ldm_packed + (first_col + 4);
    frag.x[3] = A[idx];
    //printf("%u %lu %lu\n", threadIdx.x, first_row, first_col*packing_factor);
  }

  template<unsigned N> inline __device__ void load_matrix(fragment<wgmma::matrix_b, 64, N, 256, wgmma::precision::b1, wgmma::col_major> &frag, const int *B, const size_t ldm, const unsigned swizzle_mode, const size_t tid, const size_t nthreads) {
    if (swizzle_mode != wgmma::SwizzleMode::Interleaved) return;
    const size_t packing_factor = 32;
    const size_t ldm_packed = ldm / packing_factor;
    const size_t K_packed = 256 / packing_factor;

    for (size_t idx=tid; idx < N*K_packed; idx += nthreads) {
      const size_t core_matrix_N = 8;
      const size_t core_matrix_K = 128 / packing_factor;
      // no swizzle means core matrices have to have adjacent elements
      // like the ccglib transpose kernel: tiles are contiguous in memory
      // B matrix is N x K, K major i.e. contiguous in K
      size_t n = idx / ldm_packed;
      size_t k = idx % ldm_packed;
      // calculate output index. First get index of core matrix
      size_t core_matrix_n = n / core_matrix_N;
      size_t core_matrix_k = k / core_matrix_K;
      size_t core_matrix_index = core_matrix_k * (N / core_matrix_N) + core_matrix_n;  // n-major!
      size_t core_matrix_start = core_matrix_index * core_matrix_N * core_matrix_K; // start position of this core matrix
      size_t core_n = n % core_matrix_N;
      size_t core_k = k % core_matrix_K;
      size_t out_idx = core_matrix_start + core_n * core_matrix_K + core_k;
      frag.x[out_idx] = B[idx];
    }
  }

  template<unsigned N> inline __device__ void store_matrix(const fragment<wgmma::accumulator, 64, N, 256, int> &c, int *C, const size_t ldm, const unsigned mem_order) {
    size_t laneid = threadIdx.x % wgmma::WARPGROUP_SIZE;
    size_t first_row = laneid / 4 + 8 * (laneid / 32);
    size_t first_col = 2 * (laneid % 4);

    for (size_t i = 0; i < N / 2; i++) {
      size_t row = first_row + 8 * ((i % 4) / 2);
      size_t col = first_col + i % 2 + 8 * (i / 4);
      if (mem_order == wgmma::mem_row_major) {
        C[row * ldm + col] = c.x[i];
      } else if (mem_order == wgmma::mem_col_major) {
        C[col * ldm + row] = c.x[i];
      }
    }
  }

  template<unsigned N>
  inline __device__ unsigned long make_descriptor(const fragment<wgmma::matrix_b, 64, N, 256, wgmma::precision::b1, wgmma::col_major> &frag, const unsigned swizzle_mode) {
    if (swizzle_mode != wgmma::SwizzleMode::Interleaved) return 0;

    const unsigned long addr = reinterpret_cast<unsigned long>(&frag.x[0]);
    const unsigned lds = N * 128 / 8; // size of one row or core matrices in bytes
    const unsigned sds = 128; // core matrix size in bytes
    const unsigned base_offset = 0;


    Descriptor desc;
    desc.bits.start_address = (addr & 0x3FFFF) >> 4;
    desc.bits.leading_dimension_byte_offset = (lds & 0x3FFFF) >> 4;
    desc.bits.stride_dimension_byte_offset = (sds & 0x3FFFF) >> 4;
    desc.bits.matrix_base_offset = base_offset & 0x7;
    desc.bits.swizzle_mode = swizzle_mode & 0x3;

    return desc.descriptor;
  }

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 8, 256, wgmma::precision::b1, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 8, 256, int> &c) {
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %9, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n8k256.s32.b1.b1.and.popc {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p;\n"
         "}"
         : "+r"(c.x[0]), "+r"(c.x[1]), "+r"(c.x[2]), "+r"(c.x[3])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleD));
  }
}  // end namespace wgmma
