set(WGMMA_BENCHMARKS "wgmma_benchmark_fp16_fp32" "wgmma_benchmark_b1_s32")

foreach(wgmma_example IN LISTS WGMMA_BENCHMARKS)
    add_executable(${wgmma_example} ${wgmma_example}.cu)
    target_link_libraries(${wgmma_example} cudawrappers::cu)
    target_include_directories(${wgmma_example} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)
endforeach()
