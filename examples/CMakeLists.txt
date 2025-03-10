set(WGMMA_EXAMPLES "fp16_fp32")

foreach(wgmma_example IN LISTS WGMMA_EXAMPLES)
    add_executable(wgmma_${wgmma_example} ${wgmma_example}.cu)
    target_link_libraries(wgmma_${wgmma_example} cudawrappers::cu)
    target_include_directories(wgmma_${wgmma_example} PUBLIC ${CMAKE_SOURCE_DIR}/include)

    if(WGMMA_KEEP_INTERMEDIATES)
        target_compile_options(wgmma_${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
    endif()
    target_compile_options(wgmma_${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)

endforeach()
