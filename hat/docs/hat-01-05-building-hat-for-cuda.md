# Building HAT

----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Building HAT](hat-01-03-building-hat.md)
        * [Enabling the CUDA Backend](hat-01-05-building-hat-for-cuda.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

----

This guide provides step-by-step instructions to configure Babylon/HAT for CUDA-based GPU acceleration.

### Prerequisites

To run Babylon/HAT with the CUDA backend, ensure the following components are properly installed:

1. **The NVIDIA GPU Driver:** Download and install the latest appropriate driver for your operating system from the [NVIDIA Drivers page](https://www.nvidia.com/en-us/drivers/)
2. **The CUDA Toolkit (SDK):** Download the matching CUDA Toolkit from the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads).

The CUDA Toolkit version must be compatible with your installed NVIDIA driver.
For example, as of June 2025, the stable NVIDIA driver for Linux is `570.169`, which supports CUDA Toolkit version `12.8`.


Always verify compatibility before installation to prevent runtime errors:
- You can find previous CUDA Toolkit versions on the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- Review supported CUDA versions and PTX ISA implementations in the [NVIDIA Parallel Thread Execution documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes).


### Troubleshooting: Unsupported CUDA/PTX Versions

If Babylon/HAT runs with an incompatible CUDA/PTX version, you may encounter an error similar to:

```bash
cuModuleLoadDataEx CUDA error = 222 CUDA_ERROR_UNSUPPORTED_PTX_VERSION
      /<path/to/babylon>/hat/backends/ffi/cuda/src/main/native/cpp/cuda_backend.cpp line 220
```

If this is your case, you can:
  - **update your GPU driver**.
  - **or downgrade your CUDA version.**

### Building HAT for the CUDA backend

If the NVIDIA driver and the CUDA Toolkit SDK are installed, the HAT build process will automatically compile
all sources to dispatch with the CUDA backend.

```bash
java @hat/bld
```

### Run HAT with the CUDA Backend

You can enable the CUDA backend by using the `ffi-cuda` option from HAT.
For example, to run the Matrix Multiplication example:

```bash
java @hat/run ffi-cuda matmul 1D
```
