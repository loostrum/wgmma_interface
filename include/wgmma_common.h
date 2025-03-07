#ifndef WGMMA_COMMON_H_
#define WGMMA_COMMON_H_

namespace wgmma {
  static constexpr size_t WARPGROUP_SIZE = 128;

  enum SwizzleMode { Interleaved = 0,
                     B128,
                     B64,
                     B32};

  enum Layout { mem_row_major = 0,
                mem_col_major};

  struct row_major {};
  struct col_major {};
  struct matrix_a {};
  struct matrix_b {};
  struct accumulator {};

  union Descriptor {
    struct {
      unsigned short start_address : 14, : 2;
      unsigned short leading_dimension_byte_offset : 14, : 2;
      unsigned short stride_dimension_byte_offset : 14, : 2;
      unsigned char : 1, matrix_base_offset : 3, : 4;
      unsigned char : 6, swizzle_mode: 2;
    } bits;

    unsigned long descriptor;
  };

  template<typename T, unsigned N>
  struct Storage {
    T x[N];
  };

  template<typename matrixT, unsigned M, unsigned N, unsigned K, typename T, typename LayoutT=void> class fragment;

  template<class Frag, typename T>
  inline __device__ void load_matrix(Frag &frag, const T *A, const size_t ldm);

  template<class Frag, typename T>
  inline __device__ void load_matrix(Frag &frag, const T *A, const size_t ldm, const size_t tid, const size_t nthreads);

  template<class Frag, typename T>
  inline __device__ void store_matrix(const Frag &c, T *C, const size_t ldm, const unsigned mem_order);

  template<class FragA, class FragC>
  inline __device__ void mma_async(const FragA &a, const unsigned long descB, FragC &c);

  inline __host__ __device__ unsigned long make_descriptor(unsigned long start_address, unsigned leading_dimension_offset, unsigned stride_dimension_offset, unsigned matrix_base_offset, SwizzleMode swizzle_mode) {
    Descriptor desc;
    desc.bits.start_address = (start_address & 0x3FFFF) >> 4;
    desc.bits.leading_dimension_byte_offset = (leading_dimension_offset & 0x3FFFF) >> 4;
    desc.bits.stride_dimension_byte_offset = (stride_dimension_offset & 0x3FFFF) >> 4;
    desc.bits.matrix_base_offset = matrix_base_offset & 0x7;
    desc.bits.swizzle_mode = swizzle_mode & 0x3;

    return desc.descriptor;
  }

  inline __device__ void smem_fence() {
    // the async proxy fence is required between writing to shared memory and reading that shared memory in an
    // wgmma.mma_async instruction
    asm("fence.proxy.async;");
  }

  inline __device__ void arrive() {
    // the wgmma fence is required
    // 1. before the first wgmma.mma_async instruction in a warpgroup
    // 2. between a register access and wgmma.mma_async instruction that uses the same registers
    // this applies to the accumulator fragment and A matrix when in registers. Not required when all wgmma.mma_async
    // instructions use the same matrix shape
    asm("wgmma.fence.sync.aligned;");
  }

  inline __device__ void commit() {
    asm("wgmma.commit_group.sync.aligned;");
  }

  template<unsigned N=0>
  inline __device__ void wait() {
    // N is the amount of wgmma instruction that is still allowed to be pending
    asm("wgmma.wait_group.sync.aligned %0;" :: "n"(N));
  }

}  // end namespace wgmma

#endif  // WGMMA_COMMON_H_
