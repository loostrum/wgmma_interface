cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(
  wgmma_examples
  DESCRIPTION "WGMMA examples"
  VERSION 0.1
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 90a)

option(WGMMA_KEEP_INTERMEDIATES "Keep intermediate build files, such as PTX" OFF)

find_package(CUDAToolkit REQUIRED)

# fetch cudawrappers
set(CUDAWRAPPERS_BACKEND, "CUDA")
include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG "main")
FetchContent_MakeAvailable(cudawrappers)

set(WGMMA_EXAMPLES "fp16_fp32")

foreach(wgmma_example IN LISTS WGMMA_EXAMPLES)
    add_executable(wgmma_${wgmma_example} examples/${wgmma_example}.cu)
    target_link_libraries(wgmma_${wgmma_example} cudawrappers::cu)
    target_include_directories(wgmma_${wgmma_example} PUBLIC ${CMAKE_SOURCE_DIR}/include)

    if(WGMMA_KEEP_INTERMEDIATES)
        target_compile_options(wgmma_${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
    endif()
    target_compile_options(wgmma_${wgmma_example} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)

endforeach()
