<!--
Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.

This code is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License version 2 only, as
published by the Free Software Foundation.  Oracle designates this
particular file as subject to the "Classpath" exception as provided
by Oracle in the LICENSE file that accompanied this code.

This code is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
version 2 for more details (a copy is included in the LICENSE file that
accompanied this code).

You should have received a copy of the GNU General Public License version
2 along with this work; if not, write to the Free Software Foundation,
Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.

Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
or visit www.oracle.com if you need additional information or have any
questions.
-->

# Building HAT (NVidia Notes)
[Back to Index ../](../index.md)

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
mvn clean package
```

### Run HAT examples with the CUDA Backend
You can enable the CUDA backend by using the `ffi-cuda` option from HAT.
For example, to run the Matrix Multiplication example:

```bash
java @.ffi-cuda-example matmul.Main --kernel=1D
```

### CUDA source compiler selection

The CUDA backend compiles generated CUDA source with `nvcc` by default, which
preserves the original backend behavior. To use NVRTC instead, set
`HAT_CUDA_COMPILER=nvrtc` when launching the Java process:

```bash
HAT_CUDA_COMPILER=nvrtc java @.ffi-cuda-example matmul.Main --kernel=1D
```

The supported values are:

- `HAT_CUDA_COMPILER=nvcc`: compile generated CUDA source with the `nvcc`
  executable.
- `HAT_CUDA_COMPILER=nvrtc`: compile generated CUDA source in-process with
  NVRTC.

Any other `HAT_CUDA_COMPILER` value is treated as a configuration error.

When the NVRTC path is selected, `libnvrtc.so` is loaded lazily at runtime.
The backend first tries the CUDA Toolkit library directory found at build time
and then lets the platform loader search its default paths. If needed, set
`HAT_CUDA_NVRTC_LIBRARY` to the path or loader-visible name of the NVRTC shared
library. When this variable is set, that value is passed directly to `dlopen()`
and loading fails if it is invalid:

```bash
HAT_CUDA_COMPILER=nvrtc HAT_CUDA_NVRTC_LIBRARY=/path/to/libnvrtc.so \
    java @.ffi-cuda-example matmul.Main --kernel=1D
```
