# wgmma interface

With the Hopper generation, NVIDIA introduced a new interface to the GPU tensor cores, called wgmma. This interface is accessible through inline PTX, or through libraries such as CUTLASS. Unlike the older wmma interface, there is no C++ interface to wgmma. This repository provides a C++ interface to a limited subset of wgmma instructions, as well as to the related load/store operations.

## Build
Build this project with CMake:
```
cmake -S . -B build
make -C build
```

After building the project, examples can be found in `build/examples`. These showcase the use of WGMMA on a multiplication of two large matrices.  
The `build/benchmark` folder contains examples that run many WGMMA instructions but minimal I/O. These are effectively WGMMA performance benchmarks.

## Supported matrix types and shapes
| precision (in) | precision (out) | M | N | K |
| -------------  | --------------  | - | - | - |
| fp16 | fp32    | 64 | 8, 64, 128, 256 | 16  |
| b1   | s32     | 64 | 8, 64, 128, 256 | 256 |
