# wgmma examples

With the Hopper generation, NVIDIA introduced a new interface to the GPU tensor cores, called wgmma. This interface is accessible through inline PTX, or through libraries such as CUTLASS. Unlike the older wmma interface, there is no C++ interface to wgmma. This repository provides a C++ interface to a limited subset of wgmma instructions, as well as to the related load/store operations.
