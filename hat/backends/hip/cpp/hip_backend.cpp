#include <sys/wait.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include "hip_backend.h"

#define CHECK_RET_CODE(call, ret_code)                                                             \
  {                                                                                                \
    if ((call) != ret_code) {                                                                      \
      std::cout << "Failed in call: " << #call << std::endl;                                       \
      std::abort();                                                                                \
    }                                                                                              \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)
#define HIPRTC_CHECK(call) CHECK_RET_CODE(call, HIPRTC_SUCCESS)

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

HIPBackend::HIPProgram::HIPKernel::HIPBuffer::HIPBuffer(Backend::Program::Kernel *kernel, Arg_s *arg)
        : Buffer(kernel, arg), devicePtr() {
    /*
     *   (void *) arg->value.buffer.memorySegment,
     *   (size_t) arg->value.buffer.sizeInBytes);
     */
#ifdef VERBOSE
    std::cout << "hipMalloc()" << std::endl;
#endif
    HIP_CHECK(hipMalloc(&devicePtr, (size_t) arg->value.buffer.sizeInBytes));
#ifdef VERBOSE
    std::cout << "devptr " << std::hex<<  (long)devicePtr <<std::dec <<std::endl;
#endif
    arg->value.buffer.vendorPtr = static_cast<void *>(this);
}

HIPBackend::HIPProgram::HIPKernel::HIPBuffer::~HIPBuffer() {

#ifdef VERBOSE
    std::cout << "hipFree()"
              << "devptr " << std::hex<<  (long)devicePtr <<std::dec
              << std::endl;
#endif
    HIP_CHECK(hipFree(devicePtr));
    arg->value.buffer.vendorPtr = nullptr;
}

void HIPBackend::HIPProgram::HIPKernel::HIPBuffer::copyToDevice() {
    auto hipKernel = dynamic_cast<HIPKernel*>(kernel);
#ifdef VERBOSE
    std::cout << "copyToDevice() 0x"   << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
              << "devptr " << std::hex<<  (long)devicePtr <<std::dec
              << std::endl;
#endif
    char *ptr = (char*)arg->value.buffer.memorySegment;

    unsigned long ifacefacade1 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-16);
    unsigned long ifacefacade2 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-8);

    if (ifacefacade1 != 0x1face00000facadeL && ifacefacade1 != ifacefacade2) {
        std::cerr<<"End of buf marker before HtoD"<< std::hex << ifacefacade1 << ifacefacade2<< " buffer corrupt !" <<std::endl
                <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

    HIP_CHECK(hipMemcpyHtoDAsync(devicePtr, arg->value.buffer.memorySegment, arg->value.buffer.sizeInBytes, hipKernel->hipStream));
}

void HIPBackend::HIPProgram::HIPKernel::HIPBuffer::copyFromDevice() {
    auto hipKernel = dynamic_cast<HIPKernel*>(kernel);
#ifdef VERBOSE
    std::cout << "copyFromDevice() 0x" << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
              << "devptr " << std::hex<<  (long)devicePtr <<std::dec
              << std::endl;
#endif
    char *ptr = (char*)arg->value.buffer.memorySegment;

    unsigned long ifacefacade1 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-16);
    unsigned long ifacefacade2 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-8);

    if (ifacefacade1 != 0x1face00000facadeL || ifacefacade1 != ifacefacade2) {
        std::cerr<<"end of buf marker before  DtoH"<< std::hex << ifacefacade1 << ifacefacade2<< std::dec<< " buffer corrupt !"<<std::endl
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    HIP_CHECK(hipMemcpyDtoHAsync(arg->value.buffer.memorySegment, devicePtr, arg->value.buffer.sizeInBytes, hipKernel->hipStream));

    ifacefacade1 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-16);
    ifacefacade2 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-8);

    if (ifacefacade1 != 0x1face00000facadeL || ifacefacade1 != ifacefacade2) {
        std::cerr<<"end of buf marker after  DtoH"<< std::hex << ifacefacade1 << ifacefacade2<< std::dec<< " buffer corrupt !"<<std::endl
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

HIPBackend::HIPProgram::HIPKernel::HIPKernel(Backend::Program *program, char * name, hipFunction_t kernel)
        : Backend::Program::Kernel(program, name), kernel(kernel),hipStream() {
}

HIPBackend::HIPProgram::HIPKernel::~HIPKernel() = default;

long HIPBackend::HIPProgram::HIPKernel::ndrange(void *argArray) {
#ifdef VERBOSE
    std::cout << "ndrange(" << range << ") " << name << std::endl;
#endif

    hipStreamCreate(&hipStream);
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    void *argslist[argSled.argc()];
    NDRange *ndrange = nullptr;
#ifdef VERBOSE
    std::cerr << "there are " << argSled.argc() << "args " << std::endl;
#endif
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                if (arg->idx == 0){
                    ndrange = static_cast<NDRange *>(arg->value.buffer.memorySegment);
                }
                auto hipBuffer = new HIPBuffer(this, arg);
                hipBuffer->copyToDevice();
                argslist[arg->idx] = static_cast<void *>(&hipBuffer->devicePtr);
                break;
            }
            case 'I':
            case 'F':
            case 'J':
            case 'D':
            case 'C':
            case 'S': {
                argslist[arg->idx] = static_cast<void *>(&arg->value);
                break;
            }
            default: {
                std::cerr << " unhandled variant " << (char) arg->variant << std::endl;
                break;
            }
        }
    }

    int range = ndrange->maxX;
    int rangediv1024 = range / 1024;
    int rangemod1024 = range % 1024;
    if (rangemod1024 > 0) {
        rangediv1024++;
    }

#ifdef VERBOSE
    std::cout << "Running the kernel..." << std::endl;
    std::cout << "   Requested range   = " << range << std::endl;
    std::cout << "   Range mod 1024    = " << rangemod1024 << std::endl;
    std::cout << "   Actual range 1024 = " << (rangediv1024 * 1024) << std::endl;
#endif

    HIP_CHECK(hipModuleLaunchKernel(kernel, rangediv1024, 1, 1, 1024, 1, 1, 0, hipStream, argslist, 0));

#ifdef VERBOSE
    std::cout << "Kernel complete..."<<hipGetErrorString(t)<<std::endl;
#endif

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            static_cast<HIPBuffer *>(arg->value.buffer.vendorPtr)->copyFromDevice();

        }
    }

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            delete static_cast<HIPBuffer *>(arg->value.buffer.vendorPtr);
            arg->value.buffer.vendorPtr = nullptr;
        }
    }
    HIP_CHECK(hipStreamSynchronize(hipStream));
    HIP_CHECK(hipStreamDestroy(hipStream));

    return (long) 0;
}


HIPBackend::HIPProgram::HIPProgram(Backend *backend, BuildInfo *buildInfo, hipModule_t module)
        : Backend::Program(backend, buildInfo), module(module) {
}

HIPBackend::HIPProgram::~HIPProgram() = default;

long HIPBackend::HIPProgram::getKernel(int nameLen, char *name) {

    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, name));
    long kernelHandle =  reinterpret_cast<long>(new HIPKernel(this, name, kernel));

    return kernelHandle;
}

bool HIPBackend::HIPProgram::programOK() {
    return true;
}

HIPBackend::HIPBackend(HIPBackend::HIPConfig *hipConfig, int
configSchemaLen, char *configSchema)
        : Backend((Backend::Config*) hipConfig, configSchemaLen, configSchema), device(),context()  {
#ifdef VERBOSE
    std::cout << "HIPBackend constructor " << ((hipConfig == nullptr) ? "hipConfig== null" : "got hipConfig")
              << std::endl;
#endif
    int deviceCount = 0;
    hipError_t err = hipInit(0);
    if (err == HIP_SUCCESS) {
        hipGetDeviceCount(&deviceCount);
        std::cout << "HIPBackend device count" << std::endl;
        hipDeviceGet(&device, 0);
        std::cout << "HIPBackend device ok" << std::endl;
        hipCtxCreate(&context, 0, device);
        std::cout << "HIPBackend context created ok" << std::endl;
    } else {
        std::cout << "HIPBackend failed, we seem to have the runtime library but no device, no context, nada "
                  << std::endl;
        exit(1);
    }
}

HIPBackend::HIPBackend() : HIPBackend(nullptr, 0, nullptr) {

}

HIPBackend::~HIPBackend() {
#ifdef VERBOSE
    std::cout << "freeing context" << std::endl;
#endif
    auto status = hipCtxDestroy(context);
    if (HIP_SUCCESS != status) {
        std::cerr << "hipCtxDestroy(() HIP error = " << status
                  <<" " << hipGetErrorString(static_cast<hipError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

int HIPBackend::getMaxComputeUnits() {
    std::cout << "getMaxComputeUnits()" << std::endl;
    int value = 1;
    return value;
}

void HIPBackend::info() {
    char name[100];
    hipDeviceGetName(name, sizeof(name), device);
    std::cout << "> Using device 0: " << name << std::endl;

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device);
    hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device);
    std::cout << "> HIP Device has major=" << major << " minor=" << minor << " compute capability" << std::endl;

    int warpSize;
    hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, device);
    std::cout << "> HIP Device has wave front size " << warpSize << std::endl;

    int threadsPerBlock;
    hipDeviceGetAttribute(&threadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, device);
    std::cout << "> HIP Device has threadsPerBlock " << threadsPerBlock << std::endl;

    int cores;
    hipDeviceGetAttribute(&cores, hipDeviceAttributeMultiprocessorCount, device);
    std::cout << "> HIP Cores " << cores << std::endl;

    size_t totalGlobalMem;
    hipDeviceTotalMem(&totalGlobalMem, device);
    std::cout << "  Total amount of global memory:   " << (unsigned long long) totalGlobalMem << std::endl;
    std::cout << "  64-bit Memory Address:           " <<
              ((totalGlobalMem > (unsigned long long) 4 * 1024 * 1024 * 1024L) ? "YES" : "NO") << std::endl;

}

long HIPBackend::compileProgram(int len, char *source) {

#ifdef VERBOSE
    std::cout << "inside compileProgram" << std::endl;
    std::cout << "hip " << source << std::endl;
#endif
    hiprtcProgram prog;
    auto status = hiprtcCreateProgram(&prog,
                    source,
                    "hip_kernel.hip",
                    0,
                    nullptr,
                    nullptr);
    if (status != HIPRTC_SUCCESS){
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);

        std::cerr << "hiprtcCreateProgram(() HIP error = " << std::endl;
        if (logSize) {
            std::string log(logSize, '\0');
            hiprtcGetProgramLog(prog, &log[0]);
            std::cerr <<" " << log
                      <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        }
        exit(-1);
    }

    status = hiprtcCompileProgram(prog, 0, nullptr);
    if (status != HIPRTC_SUCCESS){
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);

        std::cerr << "hiprtcCompileProgram(() HIP error = " << std::endl;
        if (logSize) {
            std::string log(logSize, '\0');
            hiprtcGetProgramLog(prog, &log[0]);
            std::cerr <<" " << log
                      <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        }
        exit(-1);
    }

    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);
#ifdef VERBOSE
    std::cerr << "HIP compiled code size " << codeSize << std::endl;
#endif

    std::vector<char> kernel_binary(codeSize);
    hiprtcGetCode(prog, kernel_binary.data());

    hipModule_t module;
    hipModuleLoadData(&module, kernel_binary.data());
    hiprtcDestroyProgram(&prog);

    return reinterpret_cast<long>(new HIPProgram(this, nullptr, module));
}

long getBackend(void *config, int configSchemaLen, char *configSchema) {
    long backendHandle = reinterpret_cast<long>(
            new HIPBackend(static_cast<HIPBackend::HIPConfig *>(config), configSchemaLen,
                            configSchema));
#ifdef VERBOSE
    std::cout << "getBackend() -> backendHandle=" << std::hex << backendHandle << std::dec << std::endl;
#endif
    return backendHandle;
}



