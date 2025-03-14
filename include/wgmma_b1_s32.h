#include "wgmma_common.h"

namespace wgmma {

  // 64xNx256
  template<unsigned N> class fragment<wgmma::matrix_a, 64, N, 256, wgmma::precision::b1, wgmma::row_major> : public Storage<int, 4> {};
  template<unsigned N> class fragment<wgmma::matrix_b, 64, N, 256, wgmma::precision::b1, wgmma::col_major> : public Storage<int, N*8 /* N*K/packing_factor */> {};
  template<unsigned N> class fragment<wgmma::accumulator, 64, N, 256, int> : public Storage<int, N/2> {};


  template<unsigned N> inline __device__ void load_matrix(fragment<wgmma::matrix_a, 64, N, 256, wgmma::precision::b1, wgmma::row_major> &frag, const int *A, const size_t ldm) {
    size_t laneid = threadIdx.x % 128;
    size_t first_row = laneid / 4 + 8 * (laneid / 32);
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

  template<unsigned N> inline __device__ void store_matrix(int *C, const fragment<wgmma::accumulator, 64, N, 256, int> &c, const size_t ldm, const unsigned mem_order) {
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

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 128, 256, wgmma::precision::b1, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 128, 256, int> &c) {
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %69, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n128k256.s32.b1.b1.and.popc {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p;\n"
         "}"
         : "+r"(c.x[0]), "+r"(c.x[1]), "+r"(c.x[2]), "+r"(c.x[3]), "+r"(c.x[4]), "+r"(c.x[5]), "+r"(c.x[6]), "+r"(c.x[7]), 
           "+r"(c.x[8]), "+r"(c.x[9]), "+r"(c.x[10]), "+r"(c.x[11]), "+r"(c.x[12]), "+r"(c.x[13]), "+r"(c.x[14]), "+r"(c.x[15]), 
           "+r"(c.x[16]), "+r"(c.x[17]), "+r"(c.x[18]), "+r"(c.x[19]), "+r"(c.x[20]), "+r"(c.x[21]), "+r"(c.x[22]), "+r"(c.x[23]), 
           "+r"(c.x[24]), "+r"(c.x[25]), "+r"(c.x[26]), "+r"(c.x[27]), "+r"(c.x[28]), "+r"(c.x[29]), "+r"(c.x[30]), "+r"(c.x[31]), 
           "+r"(c.x[32]), "+r"(c.x[33]), "+r"(c.x[34]), "+r"(c.x[35]), "+r"(c.x[36]), "+r"(c.x[37]), "+r"(c.x[38]), "+r"(c.x[39]), 
           "+r"(c.x[40]), "+r"(c.x[41]), "+r"(c.x[42]), "+r"(c.x[43]), "+r"(c.x[44]), "+r"(c.x[45]), "+r"(c.x[46]), "+r"(c.x[47]), 
           "+r"(c.x[48]), "+r"(c.x[49]), "+r"(c.x[50]), "+r"(c.x[51]), "+r"(c.x[52]), "+r"(c.x[53]), "+r"(c.x[54]), "+r"(c.x[55]), 
           "+r"(c.x[56]), "+r"(c.x[57]), "+r"(c.x[58]), "+r"(c.x[59]), "+r"(c.x[60]), "+r"(c.x[61]), "+r"(c.x[62]), "+r"(c.x[63])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleD));
  }

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 256, 256, wgmma::precision::b1, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 256, 256, int> &c) {
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %133, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n256k256.s32.b1.b1.and.popc {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p;\n"
         "}"
         : "+r"(c.x[0]), "+r"(c.x[1]), "+r"(c.x[2]), "+r"(c.x[3]), "+r"(c.x[4]), "+r"(c.x[5]), "+r"(c.x[6]), "+r"(c.x[7]), 
           "+r"(c.x[8]), "+r"(c.x[9]), "+r"(c.x[10]), "+r"(c.x[11]), "+r"(c.x[12]), "+r"(c.x[13]), "+r"(c.x[14]), "+r"(c.x[15]), 
           "+r"(c.x[16]), "+r"(c.x[17]), "+r"(c.x[18]), "+r"(c.x[19]), "+r"(c.x[20]), "+r"(c.x[21]), "+r"(c.x[22]), "+r"(c.x[23]), 
           "+r"(c.x[24]), "+r"(c.x[25]), "+r"(c.x[26]), "+r"(c.x[27]), "+r"(c.x[28]), "+r"(c.x[29]), "+r"(c.x[30]), "+r"(c.x[31]), 
           "+r"(c.x[32]), "+r"(c.x[33]), "+r"(c.x[34]), "+r"(c.x[35]), "+r"(c.x[36]), "+r"(c.x[37]), "+r"(c.x[38]), "+r"(c.x[39]), 
           "+r"(c.x[40]), "+r"(c.x[41]), "+r"(c.x[42]), "+r"(c.x[43]), "+r"(c.x[44]), "+r"(c.x[45]), "+r"(c.x[46]), "+r"(c.x[47]), 
           "+r"(c.x[48]), "+r"(c.x[49]), "+r"(c.x[50]), "+r"(c.x[51]), "+r"(c.x[52]), "+r"(c.x[53]), "+r"(c.x[54]), "+r"(c.x[55]), 
           "+r"(c.x[56]), "+r"(c.x[57]), "+r"(c.x[58]), "+r"(c.x[59]), "+r"(c.x[60]), "+r"(c.x[61]), "+r"(c.x[62]), "+r"(c.x[63]), 
           "+r"(c.x[64]), "+r"(c.x[65]), "+r"(c.x[66]), "+r"(c.x[67]), "+r"(c.x[68]), "+r"(c.x[69]), "+r"(c.x[70]), "+r"(c.x[71]), 
           "+r"(c.x[72]), "+r"(c.x[73]), "+r"(c.x[74]), "+r"(c.x[75]), "+r"(c.x[76]), "+r"(c.x[77]), "+r"(c.x[78]), "+r"(c.x[79]), 
           "+r"(c.x[80]), "+r"(c.x[81]), "+r"(c.x[82]), "+r"(c.x[83]), "+r"(c.x[84]), "+r"(c.x[85]), "+r"(c.x[86]), "+r"(c.x[87]), 
           "+r"(c.x[88]), "+r"(c.x[89]), "+r"(c.x[90]), "+r"(c.x[91]), "+r"(c.x[92]), "+r"(c.x[93]), "+r"(c.x[94]), "+r"(c.x[95]), 
           "+r"(c.x[96]), "+r"(c.x[97]), "+r"(c.x[98]), "+r"(c.x[99]), "+r"(c.x[100]), "+r"(c.x[101]), "+r"(c.x[102]), "+r"(c.x[103]), 
           "+r"(c.x[104]), "+r"(c.x[105]), "+r"(c.x[106]), "+r"(c.x[107]), "+r"(c.x[108]), "+r"(c.x[109]), "+r"(c.x[110]), "+r"(c.x[111]), 
           "+r"(c.x[112]), "+r"(c.x[113]), "+r"(c.x[114]), "+r"(c.x[115]), "+r"(c.x[116]), "+r"(c.x[117]), "+r"(c.x[118]), "+r"(c.x[119]), 
           "+r"(c.x[120]), "+r"(c.x[121]), "+r"(c.x[122]), "+r"(c.x[123]), "+r"(c.x[124]), "+r"(c.x[125]), "+r"(c.x[126]), "+r"(c.x[127])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleD));
  }
}  // end namespace wgmma
