set(WGMMA_BENCHMARKS "benchmark_fp16_fp32" "benchmark_b1_s32")

foreach(wgmma_example IN LISTS WGMMA_BENCHMARKS)
    add_executable(wgmma_${wgmma_example} ${wgmma_example}.cu)
    target_link_libraries(wgmma_${wgmma_example} cudawrappers::cu)
    target_include_directories(wgmma_${wgmma_example} PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_compile_options(wgmma_${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)
endforeach()
