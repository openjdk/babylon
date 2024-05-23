/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

#include "cuda_shared.h"

class ContextDevice {
public:
    CUdevice device;
    CUcontext context;

    ContextDevice() {
        CUresult err = cuInit(0);
        int deviceCount = 0;
        if (err == CUDA_SUCCESS) {
            checkCudaErrors(cuDeviceGetCount(&deviceCount));
        }
        checkCudaErrors(cuDeviceGet(&device, 0));
        checkCudaErrors(cuCtxCreate(&context, 0, device));
        std::cout << "created context" << std::endl;
    }

    ~ContextDevice() {
        std::cout << "freeing context" << std::endl;
        checkCudaErrors(cuCtxDestroy(context));
    }
};

int main(int argc, char **argv) {
    std::cout << "CUDA info" << std::endl;
    ContextDevice contextDevice;
    char name[100];
    cuDeviceGetName(name, 100, contextDevice.device);
    std::cout << "> Using device 0: " << name << std::endl;

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, contextDevice.device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, contextDevice.device));
    std::cout << "> GPU Device has major=" << major << " minor=" << minor << " compute capability" << std::endl;

    int warpSize;
    checkCudaErrors(cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, contextDevice.device));
    std::cout << "> GPU Device has warpSize " << warpSize << std::endl;

    int threadsPerBlock;
    checkCudaErrors(cuDeviceGetAttribute(&threadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, contextDevice.device));
    std::cout << "> GPU Device has threadsPerBlock " << threadsPerBlock << std::endl;

    int cores;
    checkCudaErrors(cuDeviceGetAttribute(&cores, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, contextDevice.device));
    std::cout << "> GPU Cores " << cores << std::endl;

    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, contextDevice.device));
    std::cout << "  Total amount of global memory:   " << (unsigned long long) totalGlobalMem << std::endl;
    std::cout << "  64-bit Memory Address:           " <<
              ((totalGlobalMem > (unsigned long long) 4 * 1024 * 1024 * 1024L) ? "YES" : "NO") << std::endl;

}

