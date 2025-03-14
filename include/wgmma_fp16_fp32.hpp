#ifndef WGMMA_FP16_FP32_H_
#define WGMMA_FP16_FP32_H_

#include <cuda_fp16.h>

#include "wgmma_common.hpp"

namespace wgmma {

  // 64xNx16
  template<unsigned N> class fragment<wgmma::matrix_a, 64, N, 16, half, wgmma::row_major> : public Storage<int, 4> {};
  template<unsigned N> class fragment<wgmma::matrix_b, 64, N, 16, half, wgmma::col_major> : public Storage<half, N*16 /* N*K */> {};
  template<unsigned N> class fragment<wgmma::accumulator, 64, N, 16, float> : public Storage<float, N/2> {};


  template<unsigned N> inline __device__ void load_matrix(fragment<wgmma::matrix_a, 64, N, 16, half, wgmma::row_major> &frag, const half *A, const size_t ldm) {
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

  template<unsigned N> inline __device__ void load_matrix(fragment<wgmma::matrix_b, 64, N, 16, half, wgmma::col_major> &frag, const half *B, const size_t ldm, const unsigned swizzle_mode, const size_t tid, const size_t nthreads) {
    if (swizzle_mode != wgmma::SwizzleMode::Interleaved) return;

    for (size_t idx=tid; idx < N*16 /* N*K */; idx += nthreads) {
      const size_t core_matrix_N = 8;
      const size_t core_matrix_K = 8;
      // no swizzle means core matrices have to have adjacent elements
      // like the ccglib transpose kernel: tiles are contiguous in memory
      // B matrix is N x K, K major i.e. contiguous in K
      size_t n = idx / 16;
      size_t k = idx % 16;
      size_t global_idx = n * ldm + k;
      // calculate output index. First get index of core matrix
      // local n,k determined with K = K_WGMMA as opposed to ldm
      size_t core_matrix_n = n / core_matrix_N;
      size_t core_matrix_k = k / core_matrix_K;
      size_t core_matrix_index = core_matrix_k * (N / core_matrix_N) + core_matrix_n;  // n-major!
      size_t core_matrix_start = core_matrix_index * core_matrix_N * core_matrix_K; // start position of this core matrix
      size_t core_n = n % core_matrix_N;
      size_t core_k = k % core_matrix_K;
      size_t out_idx = core_matrix_start + core_n * core_matrix_K + core_k;
      frag.x[out_idx] = B[global_idx];
    }
  }

  template<unsigned N> inline __device__ void store_matrix(float *C, const fragment<wgmma::accumulator, 64, N, 16, float> &c, const size_t ldm, const unsigned mem_order) {
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
  inline __device__ unsigned long make_descriptor(const fragment<wgmma::matrix_b, 64, N, 16, half, wgmma::col_major> &frag, const unsigned swizzle_mode) {
    if (swizzle_mode != wgmma::SwizzleMode::Interleaved) return 0;

    const unsigned long addr = reinterpret_cast<unsigned long>(&frag.x[0]);
    const unsigned lds = sizeof(half) * N * 8; // size of one row or core matrices in bytes
    const unsigned sds = sizeof(half) * 8 * 8; // core matrix size in bytes
    const unsigned base_offset = 0;

    Descriptor desc;
    desc.bits.start_address = (addr & 0x3FFFF) >> 4;
    desc.bits.leading_dimension_byte_offset = (lds & 0x3FFFF) >> 4;
    desc.bits.stride_dimension_byte_offset = (sds & 0x3FFFF) >> 4;
    desc.bits.matrix_base_offset = base_offset & 0x7;
    desc.bits.swizzle_mode = swizzle_mode & 0x3;

    return desc.descriptor;
  }

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 8, 16, half, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 8, 16, float> &c) {
    constexpr int scaleA = 1;
    constexpr int scaleB = 1;
    constexpr int transB = 0;
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %12, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, %9, %10, %11;\n"
         "}"
         : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleA), "n"(scaleB), "n"(transB), "n"(scaleD));
  }

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 64, 16, half, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 64, 16, float> &c) {
    constexpr int scaleA = 1;
    constexpr int scaleB = 1;
    constexpr int transB = 0;
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %40, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, %37, %38, %39;\n"
         "}"
         : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3]), "+f"(c.x[4]), "+f"(c.x[5]), "+f"(c.x[6]), "+f"(c.x[7]),
           "+f"(c.x[8]), "+f"(c.x[9]), "+f"(c.x[10]), "+f"(c.x[11]), "+f"(c.x[12]), "+f"(c.x[13]), "+f"(c.x[14]), "+f"(c.x[15]),
           "+f"(c.x[16]), "+f"(c.x[17]), "+f"(c.x[18]), "+f"(c.x[19]), "+f"(c.x[20]), "+f"(c.x[21]), "+f"(c.x[22]), "+f"(c.x[23]),
           "+f"(c.x[24]), "+f"(c.x[25]), "+f"(c.x[26]), "+f"(c.x[27]), "+f"(c.x[28]), "+f"(c.x[29]), "+f"(c.x[30]), "+f"(c.x[31])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleA), "n"(scaleB), "n"(transB), "n"(scaleD));
  }

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 128, 16, half, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 128, 16, float> &c) {
    constexpr int scaleA = 1;
    constexpr int scaleB = 1;
    constexpr int transB = 0;
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %72, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, %69, %70, %71;\n"
         "}"
         : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3]), "+f"(c.x[4]), "+f"(c.x[5]), "+f"(c.x[6]), "+f"(c.x[7]),
           "+f"(c.x[8]), "+f"(c.x[9]), "+f"(c.x[10]), "+f"(c.x[11]), "+f"(c.x[12]), "+f"(c.x[13]), "+f"(c.x[14]), "+f"(c.x[15]),
           "+f"(c.x[16]), "+f"(c.x[17]), "+f"(c.x[18]), "+f"(c.x[19]), "+f"(c.x[20]), "+f"(c.x[21]), "+f"(c.x[22]), "+f"(c.x[23]),
           "+f"(c.x[24]), "+f"(c.x[25]), "+f"(c.x[26]), "+f"(c.x[27]), "+f"(c.x[28]), "+f"(c.x[29]), "+f"(c.x[30]), "+f"(c.x[31]),
           "+f"(c.x[32]), "+f"(c.x[33]), "+f"(c.x[34]), "+f"(c.x[35]), "+f"(c.x[36]), "+f"(c.x[37]), "+f"(c.x[38]), "+f"(c.x[39]),
           "+f"(c.x[40]), "+f"(c.x[41]), "+f"(c.x[42]), "+f"(c.x[43]), "+f"(c.x[44]), "+f"(c.x[45]), "+f"(c.x[46]), "+f"(c.x[47]),
           "+f"(c.x[48]), "+f"(c.x[49]), "+f"(c.x[50]), "+f"(c.x[51]), "+f"(c.x[52]), "+f"(c.x[53]), "+f"(c.x[54]), "+f"(c.x[55]),
           "+f"(c.x[56]), "+f"(c.x[57]), "+f"(c.x[58]), "+f"(c.x[59]), "+f"(c.x[60]), "+f"(c.x[61]), "+f"(c.x[62]), "+f"(c.x[63])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleA), "n"(scaleB), "n"(transB), "n"(scaleD));
  }

  template<> inline __device__ void mma_async(const fragment<wgmma::matrix_a, 64, 256, 16, half, wgmma::row_major> &a, const unsigned long descB, fragment<wgmma::accumulator, 64, 256, 16, float> &c) {
    constexpr int scaleA = 1;
    constexpr int scaleB = 1;
    constexpr int transB = 0;
    constexpr int scaleD = 1;
     asm("{\n\t"
         ".reg.pred p;\n\t"
         "setp.ne.b32 p, %136, 0;\n\t"
         "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, %133, %134, %135;\n"
         "}"
         : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3]), "+f"(c.x[4]), "+f"(c.x[5]), "+f"(c.x[6]), "+f"(c.x[7]),
           "+f"(c.x[8]), "+f"(c.x[9]), "+f"(c.x[10]), "+f"(c.x[11]), "+f"(c.x[12]), "+f"(c.x[13]), "+f"(c.x[14]), "+f"(c.x[15]),
           "+f"(c.x[16]), "+f"(c.x[17]), "+f"(c.x[18]), "+f"(c.x[19]), "+f"(c.x[20]), "+f"(c.x[21]), "+f"(c.x[22]), "+f"(c.x[23]),
           "+f"(c.x[24]), "+f"(c.x[25]), "+f"(c.x[26]), "+f"(c.x[27]), "+f"(c.x[28]), "+f"(c.x[29]), "+f"(c.x[30]), "+f"(c.x[31]),
           "+f"(c.x[32]), "+f"(c.x[33]), "+f"(c.x[34]), "+f"(c.x[35]), "+f"(c.x[36]), "+f"(c.x[37]), "+f"(c.x[38]), "+f"(c.x[39]),
           "+f"(c.x[40]), "+f"(c.x[41]), "+f"(c.x[42]), "+f"(c.x[43]), "+f"(c.x[44]), "+f"(c.x[45]), "+f"(c.x[46]), "+f"(c.x[47]),
           "+f"(c.x[48]), "+f"(c.x[49]), "+f"(c.x[50]), "+f"(c.x[51]), "+f"(c.x[52]), "+f"(c.x[53]), "+f"(c.x[54]), "+f"(c.x[55]),
           "+f"(c.x[56]), "+f"(c.x[57]), "+f"(c.x[58]), "+f"(c.x[59]), "+f"(c.x[60]), "+f"(c.x[61]), "+f"(c.x[62]), "+f"(c.x[63]),
           "+f"(c.x[64]), "+f"(c.x[65]), "+f"(c.x[66]), "+f"(c.x[67]), "+f"(c.x[68]), "+f"(c.x[69]), "+f"(c.x[70]), "+f"(c.x[71]),
           "+f"(c.x[72]), "+f"(c.x[73]), "+f"(c.x[74]), "+f"(c.x[75]), "+f"(c.x[76]), "+f"(c.x[77]), "+f"(c.x[78]), "+f"(c.x[79]),
           "+f"(c.x[80]), "+f"(c.x[81]), "+f"(c.x[82]), "+f"(c.x[83]), "+f"(c.x[84]), "+f"(c.x[85]), "+f"(c.x[86]), "+f"(c.x[87]),
           "+f"(c.x[88]), "+f"(c.x[89]), "+f"(c.x[90]), "+f"(c.x[91]), "+f"(c.x[92]), "+f"(c.x[93]), "+f"(c.x[94]), "+f"(c.x[95]),
           "+f"(c.x[96]), "+f"(c.x[97]), "+f"(c.x[98]), "+f"(c.x[99]), "+f"(c.x[100]), "+f"(c.x[101]), "+f"(c.x[102]), "+f"(c.x[103]),
           "+f"(c.x[104]), "+f"(c.x[105]), "+f"(c.x[106]), "+f"(c.x[107]), "+f"(c.x[108]), "+f"(c.x[109]), "+f"(c.x[110]), "+f"(c.x[111]),
           "+f"(c.x[112]), "+f"(c.x[113]), "+f"(c.x[114]), "+f"(c.x[115]), "+f"(c.x[116]), "+f"(c.x[117]), "+f"(c.x[118]), "+f"(c.x[119]),
           "+f"(c.x[120]), "+f"(c.x[121]), "+f"(c.x[122]), "+f"(c.x[123]), "+f"(c.x[124]), "+f"(c.x[125]), "+f"(c.x[126]), "+f"(c.x[127])
         : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "l"(descB), "n"(scaleA), "n"(scaleB), "n"(transB), "n"(scaleD));
  }
}  // end namespace wgmma
#endif  // WGMMA_FP16_FP32_H_
