/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
#include <unistd.h>
#include <chrono>
#include "cuda_backend.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr bool DEFAULT_USE_NVRTC = false;

using nvrtcProgram = void *;
using nvrtcResult = int;
constexpr nvrtcResult NVRTC_SUCCESS = 0;

template <typename T>
T loadNvrtcSymbol(void *handle, const char *name) {
    dlerror();
    void *symbol = dlsym(handle, name);
    if (const char *error = dlerror()) {
        std::cerr << "Failed to load NVRTC symbol " << name << ": " << error << std::endl;
        std::exit(1);
    }
    return reinterpret_cast<T>(symbol);
}

void *loadNvrtcLibrary() {
    std::vector<std::string> candidates;
    if (const char *envLibrary = std::getenv("HAT_CUDA_NVRTC_LIBRARY")) {
        dlerror();
        if (void *handle = dlopen(envLibrary, RTLD_NOW | RTLD_LOCAL)) {
            return handle;
        }
        std::cerr << "Failed to load NVRTC from HAT_CUDA_NVRTC_LIBRARY='"
                  << envLibrary << "'";
        if (const char *error = dlerror()) {
            std::cerr << ": " << error;
        }
        std::cerr << std::endl;
        std::exit(1);
    }
    // First try the CUDA toolkit library directory found by CMake.
    // Then let dlopen search the system loader path.
#ifdef HAT_CUDA_LIBRARY_DIR
    candidates.emplace_back(std::string(HAT_CUDA_LIBRARY_DIR) +
                            "/libnvrtc.so");
#endif
    candidates.emplace_back("libnvrtc.so");

    std::vector<std::string> errors;
    for (const std::string &candidate : candidates) {
        if (void *handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL)) {
            return handle;
        }
        if (const char *error = dlerror()) {
            errors.emplace_back(candidate + ": " + error);
        }
    }

    std::cerr << "Failed to load NVRTC. Set HAT_CUDA_NVRTC_LIBRARY to "
              << "the NVRTC shared library path or name, "
              << "or use HAT_CUDA_COMPILER=nvcc." << std::endl;
    for (const std::string &error : errors) {
        std::cerr << "  " << error << std::endl;
    }
    std::exit(1);
}

std::vector<std::string> splitDelimited(const char *value,
                                        const char delimiter) {
    std::vector<std::string> result;
    if (value == nullptr || *value == '\0') {
        return result;
    }

    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

// Minimal NVRTC ABI surface used through dlsym. This avoids a build-time
// dependency on nvrtc.h and a link-time dependency on libnvrtc.so.
struct NvrtcApi {
    using CreateProgram = nvrtcResult (*)(nvrtcProgram *,
                                          const char *,
                                          const char *,
                                          int,
                                          const char * const *,
                                          const char * const *);
    using CompileProgram = nvrtcResult (*)(nvrtcProgram, int, const char * const *);
    using DestroyProgram = nvrtcResult (*)(nvrtcProgram *);
    using GetErrorString = const char *(*)(nvrtcResult);
    using GetPTX = nvrtcResult (*)(nvrtcProgram, char *);
    using GetPTXSize = nvrtcResult (*)(nvrtcProgram, size_t *);
    using GetProgramLog = nvrtcResult (*)(nvrtcProgram, char *);
    using GetProgramLogSize = nvrtcResult (*)(nvrtcProgram, size_t *);

    void *handle;
    CreateProgram createProgram;
    CompileProgram compileProgram;
    DestroyProgram destroyProgram;
    GetErrorString getErrorString;
    GetPTX getPTX;
    GetPTXSize getPTXSize;
    GetProgramLog getProgramLog;
    GetProgramLogSize getProgramLogSize;

    explicit NvrtcApi(void *handle)
        : handle(handle),
          createProgram(loadNvrtcSymbol<CreateProgram>(handle, "nvrtcCreateProgram")),
          compileProgram(loadNvrtcSymbol<CompileProgram>(handle, "nvrtcCompileProgram")),
          destroyProgram(loadNvrtcSymbol<DestroyProgram>(handle, "nvrtcDestroyProgram")),
          getErrorString(loadNvrtcSymbol<GetErrorString>(handle, "nvrtcGetErrorString")),
          getPTX(loadNvrtcSymbol<GetPTX>(handle, "nvrtcGetPTX")),
          getPTXSize(loadNvrtcSymbol<GetPTXSize>(handle, "nvrtcGetPTXSize")),
          getProgramLog(loadNvrtcSymbol<GetProgramLog>(handle, "nvrtcGetProgramLog")),
          getProgramLogSize(loadNvrtcSymbol<GetProgramLogSize>(handle, "nvrtcGetProgramLogSize")) {
    }
};

// Load and cache NVRTC API entry points on first use
NvrtcApi &nvrtcApi() {
    static NvrtcApi api(loadNvrtcLibrary());
    return api;
}

void nvrtcCheck(const NvrtcApi &api, const nvrtcResult result,
                const char *functionName) {
    if (result != NVRTC_SUCCESS) {
        std::cerr << functionName << " NVRTC error = " << result << " "
                  << api.getErrorString(result) << std::endl;
        std::exit(1);
    }
}

std::string getNvrtcLog(const NvrtcApi &api, nvrtcProgram program) {
    size_t logSize = 0;
    nvrtcCheck(api,
               api.getProgramLogSize(program, &logSize),
               "nvrtcGetProgramLogSize");
    std::string log;
    if (logSize > 1) {
        log.resize(logSize, '\0');
        nvrtcCheck(api,
                   api.getProgramLog(program, log.data()),
                   "nvrtcGetProgramLog");
    }
    return log;
}
}

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

CudaSource::CudaSource(size_t len, char *text, bool isCopy, bool lineinfo)
    : Text(len, text, isCopy) {
    _lineInfo = lineinfo;
}

CudaSource::CudaSource()
    : Text(0) {
}

bool CudaSource::lineInfo() const {
    return _lineInfo;
}

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string tmpFileName(uint64_t time, const std::string directoryName, const std::string &suffix) {
    std::stringstream timestamp;
    timestamp << directoryName << "/tmp_" << time << suffix;
    return timestamp.str();
}

CudaBackend::CudaBackend(int configBits)
    : Backend(new Config(configBits), new CudaQueue(this)), initStatus(cuInit(0)), device(), context() {
    int deviceCount = 0;

    if (initStatus == CUDA_SUCCESS) {
        CUDA_CHECK(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
        if (config->info) {
            std::cout << "CudaBackend device count = " << deviceCount << std::endl;
        }
        CUDA_CHECK(cuDeviceGet(&device, 0), "cuDeviceGet");
        #if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
            CUctxCreateParams ctxCreateParams = {};
            CUDA_CHECK(cuCtxCreate_v4(&context, &ctxCreateParams, 0, device), "cuCtxCreate");
        #else
            // Invoke previous implementation with 3 parameters
            CUDA_CHECK(cuCtxCreate(&context, 0, device), "cuCtxCreate");
        #endif
        if (config->info) {
            std::cout << "CudaBackend context created ok (id=" << context << ")" << std::endl;
        }
        dynamic_cast<CudaQueue *>(queue)->init();
    } else {
        CUDA_CHECK(initStatus, "cuInit() failed we seem to have the runtime library but no device");
    }
}

CudaBackend::~CudaBackend() {
    std::cout << "freeing context" << std::endl;
    CUDA_CHECK(cuCtxDestroy(context), "cuCtxDestroy");
}

void CudaBackend::shortDeviceInfo() {
    char name[100];
    CUDA_CHECK(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    std::cout << "[INFO] Using NVIDIA GPU: " << name << std::endl;
}

void CudaBackend::showDeviceInfo() {
    char name[100];
    CUDA_CHECK(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");

    std::cout << "> Using device 0: " << name << std::endl;

    // get compute capabilities and the device name
    int major = 0, minor = 0;
    CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute");
    CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute");
    std::cout << "> GPU Device has major=" << major << " minor=" << minor << " compute capability" << std::endl;

    int warpSize;
    CUDA_CHECK(cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device), "cuDeviceGetAttribute");
    std::cout << "> GPU Device has warpSize " << warpSize << std::endl;

    int threadsPerBlock;
    CUDA_CHECK(cuDeviceGetAttribute(&threadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device), "cuDeviceGetAttribute");
    std::cout << "> GPU Device has threadsPerBlock " << threadsPerBlock << std::endl;

    int cores;
    CUDA_CHECK(cuDeviceGetAttribute(&cores, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "cuDeviceGetAttribute");
    std::cout << "> GPU Cores " << cores << std::endl;

    size_t totalGlobalMem;
    CUDA_CHECK(cuDeviceTotalMem(&totalGlobalMem, device), "cuDeviceTotalMem");
    std::cout << "  Total amount of global memory:   " << (unsigned long long) totalGlobalMem << std::endl;
    std::cout << "  64-bit Memory Address:           " <<
            ((totalGlobalMem > static_cast<unsigned long long>(4) * 1024 * 1024 * 1024L) ? "YES" : "NO") << std::endl;
}

bool CudaBackend::useNvrtcCompiler() const {
    const char *compiler = std::getenv("HAT_CUDA_COMPILER");
    if (compiler != nullptr) {
        if (std::strcmp(compiler, "nvcc") == 0) {
            return false;
        }
        if (std::strcmp(compiler, "nvrtc") == 0) {
            return true;
        }
        std::cerr << "Unknown HAT_CUDA_COMPILER='" << compiler
                  << "', expected 'nvrtc' or 'nvcc'." << std::endl;
        std::exit(1);
    }

    return DEFAULT_USE_NVRTC;
}

PtxSource *CudaBackend::nvcc(const CudaSource *cudaSource) {

    // create var/cuda directory
    std::string localDirectory = "./var/cuda";
    std::filesystem::create_directories(localDirectory);
    // create temp file for cuda generarated code
    const uint64_t time = timeSinceEpochMillisec();
    const std::string ptxPath = tmpFileName(time, localDirectory, ".ptx");
    const std::string cudaPath = tmpFileName(time, localDirectory, ".cu");

    // compile the generated code
    int pid;
    cudaSource->write(cudaPath);
    if ((pid = fork()) == 0) { //child
        const auto path = "nvcc";
        std::vector<std::string> command;
        command.push_back(path);
        command.push_back("-ptx");
        command.push_back("-Wno-deprecated-gpu-targets");
        command.push_back(cudaPath);
        if (cudaSource->lineInfo()) {
            command.push_back("-lineinfo");
        }
        command.push_back("-o");
        command.push_back(ptxPath);

        // conver to char*[]
        const char* args[command.size() + 1];
        for (int i = 0; i < command.size(); i++) {
            args[i] = command[i].c_str();
        }
        args[command.size()] = nullptr;
        const int stat = execvp(path, (char *const *) args);
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

PtxSource *CudaBackend::nvrtc(const CudaSource *cudaSource) {
    NvrtcApi &api = nvrtcApi();
    std::string source(cudaSource->text, cudaSource->len);

    // Keep generated CUDA/PTX artifacts under the same directory as the nvcc path.
    std::string localDirectory = "./var/cuda";
    std::filesystem::create_directories(localDirectory);
    const uint64_t time = timeSinceEpochMillisec();
    const std::string ptxPath = tmpFileName(time, localDirectory, ".ptx");
    const std::string cudaPath = tmpFileName(time, localDirectory, ".cu");
    cudaSource->write(cudaPath);

    nvrtcProgram program;
    nvrtcCheck(api,
               api.createProgram(&program,
                                 source.c_str(),
                                 cudaPath.c_str(),
                                 0,
                                 nullptr,
                                 nullptr),
               "nvrtcCreateProgram");

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cuDeviceGetAttribute(&major,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                    device),
               "cuDeviceGetAttribute");
    CUDA_CHECK(cuDeviceGetAttribute(&minor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                    device),
               "cuDeviceGetAttribute");

    std::vector<std::string> options;
    options.emplace_back("--std=c++17");
    options.emplace_back("--gpu-architecture=compute_" +
                         std::to_string(major) +
                         std::to_string(minor));
#ifdef HAT_CUDA_INCLUDE_DIRS
    // CMake serializes the CUDA include list with '|' so it can be passed as a
    // single compile definition.
    for (const std::string &includeDir :
            splitDelimited(HAT_CUDA_INCLUDE_DIRS, '|')) {
        options.emplace_back("-I" + includeDir);
    }
#endif
    if (cudaSource->lineInfo()) {
        options.emplace_back("--generate-line-info");
    }

    std::vector<const char *> optionPtrs;
    optionPtrs.reserve(options.size());
    for (const std::string &option : options) {
        optionPtrs.push_back(option.c_str());
    }

    nvrtcResult compileResult =
            api.compileProgram(program,
                               static_cast<int>(optionPtrs.size()),
                               optionPtrs.data());
    if (compileResult != NVRTC_SUCCESS) {
        std::cerr << "NVRTC compilation failed: "
                  << api.getErrorString(compileResult)
                  << ". CUDA source saved to " << cudaPath << std::endl;
        std::string log = getNvrtcLog(api, program);
        if (!log.empty()) {
            std::cerr << "> NVRTC log:" << std::endl << log << std::endl;
        }
        nvrtcCheck(api, api.destroyProgram(&program), "nvrtcDestroyProgram");
        std::exit(1);
    } else if (config->info || config->trace) {
        std::string log = getNvrtcLog(api, program);
        if (!log.empty()) {
            std::cout << "> NVRTC log:" << std::endl << log << std::endl;
        }
    }

    size_t ptxSize = 0;
    nvrtcCheck(api, api.getPTXSize(program, &ptxSize), "nvrtcGetPTXSize");
    auto *ptx = new PtxSource(ptxSize);
    nvrtcCheck(api, api.getPTX(program, ptx->text), "nvrtcGetPTX");
    if (ptxSize == 0 || ptx->text[ptxSize - 1] != '\0') {
        std::cerr << "NVRTC returned invalid PTX buffer" << std::endl;
        nvrtcCheck(api, api.destroyProgram(&program), "nvrtcDestroyProgram");
        std::exit(1);
    }

    // nvrtcGetPTXSize includes the trailing NUL. Keep it in memory for the
    // driver API, but omit it from the debug artifact to match nvcc output.
    PtxSource ptxFile(ptxSize > 0 ? ptxSize - 1 : 0, ptx->text, false);
    ptxFile.write(ptxPath);

    nvrtcCheck(api, api.destroyProgram(&program), "nvrtcDestroyProgram");
    return ptx;
}

CudaBackend::CudaModule *CudaBackend::compile(const CudaSource &cudaSource) {
    return compile(&cudaSource);
}

CudaBackend::CudaModule *CudaBackend::compile(const CudaSource *cudaSource) {
    const bool useNvrtc = useNvrtcCompiler();
    if (config->info) {
        std::cout << "[INFO] CUDA source compiler: "
                  << (useNvrtc ? "NVRTC" : "NVCC") << std::endl;
    }
    const PtxSource *ptxSource = useNvrtc ? nvrtc(cudaSource) : nvcc(cudaSource);
    return compile(ptxSource);
}

CudaBackend::CudaModule *CudaBackend::compile(const PtxSource &ptxSource) {
    return compile(&ptxSource);
}

CudaBackend::CudaModule *CudaBackend::compile(const  PtxSource *ptx) {
    CUmodule module;
    if (ptx->text != nullptr) {
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

        CUDA_CHECK(cuCtxSetCurrent(context), "cuCtxSetCurrent");
        CUDA_CHECK(cuModuleLoadDataEx(&module, ptx->text, optc, jitOptions, (void **) jitOptVals), "cuModuleLoadDataEx");

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
        CudaSource cudaSource(len , source, false, config->profile);
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
        bufferState->state = BufferState::NEW_STATE;
    } else {
        if (config->trace) {
            std::cout << "Were reusing  buffer  buffer " << std::endl;
        }
        cudaBuffer = static_cast<CudaBuffer *>(bufferState->vendorPtr);
    }
    return cudaBuffer;
}
