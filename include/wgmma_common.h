#ifndef WGMMA_COMMON_H_
#define WGMMA_COMMON_H_

namespace wgmma {
  static constexpr size_t WARPGROUP_SIZE = 128;

  enum SwizzleMode { Interleaved = 0,
                     B128,
                     B64,
                     B32};

  enum MemOrder { row_major,
                  col_major};

  enum Matrix {matrix_a,
               matrix_b,
               accumulator};

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

  template<unsigned M, unsigned N, unsigned K, typename T, MemOrder mem_order> class fragment;

  inline __host__ __device__ unsigned long make_descriptor(unsigned long start_address, unsigned leading_dimension_offset, unsigned stride_dimension_offset, unsigned matrix_base_offset, SwizzleMode swizzle_mode) {

  Descriptor desc;
  desc.bits.start_address = (start_address & 0x3FFFF) >> 4;
  desc.bits.leading_dimension_byte_offset = (leading_dimension_offset & 0x3FFFF) >> 4;
  desc.bits.stride_dimension_byte_offset = (stride_dimension_offset & 0x3FFFF) >> 4;
  desc.bits.matrix_base_offset = matrix_base_offset & 0x7;
  desc.bits.swizzle_mode = swizzle_mode & 0x3;

  return desc.descriptor;
  }

  inline __device__ void arrive() {
    asm("wgmma.fence.sync.aligned;\n\t"
        "fence.proxy.async;");
  }

  inline __device__ void commit() {
    asm("wgmma.commit_group.sync.aligned;");
  }

  template<unsigned N>
  inline __device__ void wait() {
    static_assert(0 < N < 8, "N must be between 0 and 7");
    asm("wgmma.wait_group.sync.aligned %0;" :: "n"(N));
  }

}  // end namespace wgmma

#endif  // WGMMA_COMMON_H_
