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

#include <sys/wait.h>
#include <chrono>
#include "cuda_backend.h"


PtxSource::PtxSource()
    : Text(0L) {
}

PtxSource::PtxSource(size_t len)
    : Text(len) {
}

PtxSource::PtxSource(char *text)
    : Text(text, false) {
}

PtxSource::PtxSource(size_t len, char *text)
    : Text(len, text, true) {
}
PtxSource::PtxSource(size_t len, char *text, bool isCopy)
    : Text(len, text, isCopy) {
}

CudaSource::CudaSource(size_t len)
    : Text(len) {
}

CudaSource::CudaSource(char *text)
    : Text(text, false) {
}

CudaSource::CudaSource(size_t len, char *text, bool isCopy)
    : Text(len, text, isCopy) {
}

CudaSource::CudaSource()
    : Text(0) {
}

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string tmpFileName(uint64_t time, const std::string &suffix) {
    std::stringstream timestamp;
    timestamp << "./tmp" << time << suffix;
    return timestamp.str();
}



CudaBackend::CudaBackend(int configBits)
    : Backend(new Config(configBits), new CudaQueue(this)), initStatus(cuInit(0)), device(), context() {
    int deviceCount = 0;

    if (initStatus == CUDA_SUCCESS) {
        WHERE{
            .f = __FILE__, .l = __LINE__,
            .e = cuDeviceGetCount(&deviceCount),
            .t = "cuDeviceGetCount"
        }.report();
        std::cout << "CudaBackend device count = " << deviceCount << std::endl;
        WHERE{
            .f = __FILE__, .l = __LINE__,
            .e = cuDeviceGet(&device, 0),
            .t = "cuDeviceGet"
        }.report();
        WHERE{
            .f = __FILE__, .l = __LINE__,
            .e = cuCtxCreate(&context, 0, device),
            .t = "cuCtxCreate"
        }.report();
        std::cout << "CudaBackend context created ok (id=" << context << ")" << std::endl;
        dynamic_cast<CudaQueue *>(queue)->init();
    } else {
        WHERE{
            .f = __FILE__, .l = __LINE__,
            .e = initStatus,
            "cuInit() failed we seem to have the runtime library but no device"
        }.report();
    }
}


CudaBackend::~CudaBackend() {
    std::cout << "freeing context" << std::endl;
    WHERE{
        .f = __FILE__, .l = __LINE__,
        .e = cuCtxDestroy(context),
        .t = "cuCtxDestroy"
    }.report();
}

void CudaBackend::info() {
    char name[100];
    cuDeviceGetName(name, sizeof(name), device);
    std::cout << "> Using device 0: " << name << std::endl;

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    std::cout << "> GPU Device has major=" << major << " minor=" << minor << " compute capability" << std::endl;

    int warpSize;
    cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    std::cout << "> GPU Device has warpSize " << warpSize << std::endl;

    int threadsPerBlock;
    cuDeviceGetAttribute(&threadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    std::cout << "> GPU Device has threadsPerBlock " << threadsPerBlock << std::endl;

    int cores;
    cuDeviceGetAttribute(&cores, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    std::cout << "> GPU Cores " << cores << std::endl;

    size_t totalGlobalMem;
    cuDeviceTotalMem(&totalGlobalMem, device);
    std::cout << "  Total amount of global memory:   " << (unsigned long long) totalGlobalMem << std::endl;
    std::cout << "  64-bit Memory Address:           " <<
            ((totalGlobalMem > static_cast<unsigned long long>(4) * 1024 * 1024 * 1024L) ? "YES" : "NO") << std::endl;
}




PtxSource *CudaBackend::nvcc(const CudaSource *cudaSource) {
  //std::cout << "inside nvcc" << std::endl;
    const uint64_t time = timeSinceEpochMillisec();
    const std::string ptxPath = tmpFileName(time, ".ptx");
    const std::string cudaPath = tmpFileName(time, ".cu");
    int pid;
    cudaSource->write(cudaPath);
    if ((pid = fork()) == 0) { //child
        const auto path = "/usr/local/cuda/bin/nvcc";
        const char *argv[]{  "/usr/local/cuda/bin/nvcc", "-ptx", "-Wno-deprecated-gpu-targets", cudaPath.c_str(), "-o", ptxPath.c_str(), nullptr};
        const int stat = execvp(path, (char *const *) argv);
        std::cerr << " nvcc stat = " << stat << " errno=" << errno << " '" << std::strerror(errno) << "'" << std::endl;
        std::exit(errno);
    } else if (pid < 0) {// fork failed.
        std::cerr << "fork of nvcc failed" << std::endl;
        std::exit(1);
    } else { //parent
        int status;
        pid_t result = wait(&status);
        auto *ptx = new PtxSource();
        ptx->read(ptxPath);
        return ptx;
    }
}

CudaBackend::CudaModule *CudaBackend::compile(const CudaSource &cudaSource) {
    return compile(&cudaSource);
}

CudaBackend::CudaModule *CudaBackend::compile(const CudaSource *cudaSource) {
    const PtxSource *ptxSource = nvcc(cudaSource);
    return compile(ptxSource);
}

CudaBackend::CudaModule *CudaBackend::compile(const PtxSource &ptxSource) {
    return compile(&ptxSource);
}

CudaBackend::CudaModule *CudaBackend::compile(const  PtxSource *ptx) {

    CUmodule module;
     // std::cout << "inside compile" << std::endl;
    // std::cout << "cuda " << cudaSource->text << std::endl;
    if (ptx->text != nullptr) {
       // std::cout << "ptx " << ptx->text << std::endl;
        const Log *infLog = new Log(8192);
        const Log *errLog = new Log(8192);
        constexpr unsigned int optc = 5;
        const auto jitOptions = new CUjit_option[optc];
        auto jitOptVals = new void *[optc];


        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        jitOptVals[0] = reinterpret_cast<void *>(infLog->len);
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        jitOptVals[1] = infLog->text;
        jitOptions[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        jitOptVals[2] = reinterpret_cast<void *>(errLog->len);
        jitOptions[3] = CU_JIT_ERROR_LOG_BUFFER;
        jitOptVals[3] = errLog->text;
        jitOptions[4] = CU_JIT_GENERATE_LINE_INFO;
        jitOptVals[4] = reinterpret_cast<void *>(1);

        WHERE{
            .f = __FILE__, .l = __LINE__,
            .e = cuModuleLoadDataEx(&module, ptx->text, optc, jitOptions, (void **) jitOptVals),
            .t = "cuModuleLoadDataEx"
        }.report();
        if (*infLog->text!='\0'){
           std::cout << "> PTX JIT inflog:" << std::endl << infLog->text << std::endl;
        }
        if (*errLog->text!='\0'){
           std::cout << "> PTX JIT errlog:" << std::endl << errLog->text << std::endl;
        }
        return new CudaModule(this, ptx->text, infLog->text, true, module);

        //delete ptx;
    } else {
        std::cout << "no ptx content!" << std::endl;
        exit(1);
    }
}

//Entry point from HAT.  We use the config PTX bit to determine which Source type

Backend::CompilationUnit *CudaBackend::compile(const int len, char *source) {
    if (config->traceCalls) {
        std::cout << "inside compileProgram" << std::endl;
    }

    if (config->ptx){
        if (config->trace) {
            std::cout << "compiling from provided  ptx " << std::endl;
        }
        PtxSource ptxSource(len, source, false);
        return compile(ptxSource);
    }else{
        if (config->trace) {
            std::cout << "compiling from provided  cuda " << std::endl;
        }
        CudaSource cudaSource(len , source, false);
        return compile(cudaSource);
    }
}

/*

    if (config->ptx) {

    } else {
        if (config->trace) {
            std::cout << "compiling from cuda c99 " << std::endl;
        }
        if (config->showCode) {
            std::cout << "cuda " << source << std::endl;
        }
        auto* cuda = new CudaSource(len, source, false);
        ptx = nvcc(cuda);
    }
    if (config->showCode) {
        std::cout << "ptx " << ptx->text << std::endl;
    }
    CUmodule module;


    if (ptx->text != nullptr) {
        constexpr unsigned int jitNumOptions = 2;
        const auto jitOptions = new CUjit_option[jitNumOptions];
        const auto jitOptVals = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        constexpr int jitLogBufferSize = 8192;
        jitOptVals[0] = reinterpret_cast<void *>(jitLogBufferSize);

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        auto jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;
        cuCtxSetCurrent(context);

        WHERE{
            .f = __FILE__, .l = __LINE__,
            .e = cuModuleLoadDataEx(&module, ptx->text, jitNumOptions, jitOptions, jitOptVals),
            .t = "cuModuleLoadDataEx"
        }.report();
        if (jitLogBuffer != nullptr && *jitLogBuffer!='\0'){
             std::cout << "PTX log:" << jitLogBuffer << std::endl;
        }
        return new CudaModule(this, ptx->text, jitLogBuffer, true, module);
    } else {
        std::cout << "no ptx content!" << std::endl;
        exit(1);
    }
} */

extern "C" long getBackend(int mode) {
    long backendHandle = reinterpret_cast<long>(new CudaBackend(mode));
    //  std::cout << "getBackend() -> backendHandle=" << std::hex << backendHandle << std::dec << std::endl;
    return backendHandle;
}

void clCallback(void *) {
    std::cerr << "start of compute" << std::endl;
}


void CudaBackend::computeEnd() {
    queue->computeEnd();
}

void CudaBackend::computeStart() {
    queue->computeStart();
}

bool CudaBackend::getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) {
    if (config->traceCalls) {
        std::cout << "getBufferFromDeviceIfDirty(" << std::hex << reinterpret_cast<long>(memorySegment) << "," <<
                std::dec << memorySegmentLength << "){" << std::endl;
    }
    if (config->minimizeCopies) {
        const BufferState *bufferState = BufferState::of(memorySegment, memorySegmentLength);
        if (bufferState->state == BufferState::DEVICE_OWNED) {
            queue->copyFromDevice(static_cast<Backend::Buffer *>(bufferState->vendorPtr));
            if (config->traceEnqueues | config->traceCopies) {
                std::cout << "copying buffer from device (from java access) " << std::endl;
            }
            queue->wait();
            queue->release();
        } else {
            std::cout << "HOW DID WE GET HERE 1 attempting  to get buffer but buffer is not device dirty" << std::endl;
            std::exit(1);
        }
    } else {
        std::cerr <<
                "HOW DID WE GET HERE ? java side should avoid calling getBufferFromDeviceIfDirty as we are not minimising buffers!"
                << std::endl;
        std::exit(1);
    }
    if (config->traceCalls) {
        std::cout << "}getBufferFromDeviceIfDirty()" << std::endl;
    }
    return true;
}

CudaBackend *CudaBackend::of(const long backendHandle) {
    return reinterpret_cast<CudaBackend *>(backendHandle);
}

CudaBackend *CudaBackend::of(Backend *backend) {
    return dynamic_cast<CudaBackend *>(backend);
}

CudaBackend::CudaBuffer *CudaBackend::getOrCreateBuffer(BufferState *bufferState) {
    CudaBuffer *cudaBuffer = nullptr;
    if (bufferState->vendorPtr == nullptr || bufferState->state == BufferState::NEW_STATE) {
        cudaBuffer = new CudaBuffer(this, bufferState);
        if (config->trace) {
            std::cout << "We allocated arg buffer " << std::endl;
        }
    } else {
        if (config->trace) {
            std::cout << "Were reusing  buffer  buffer " << std::endl;
        }
        cudaBuffer = static_cast<CudaBuffer *>(bufferState->vendorPtr);
    }
    return cudaBuffer;
}
