set(WGMMA_EXAMPLES "wgmma_fp16_fp32")

foreach(wgmma_example IN LISTS WGMMA_EXAMPLES)
    add_executable(${wgmma_example} ${wgmma_example}.cu)
    target_link_libraries(${wgmma_example} cudawrappers::cu)
    target_include_directories(${wgmma_example} PRIVATE ${CMAKE_SOURCE_DIR}/include)

    if(WGMMA_KEEP_INTERMEDIATES)
        target_compile_options(${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
    endif()
    # to avoid context deprecation warning in cudawrappers
    target_compile_options(${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)

endforeach()
