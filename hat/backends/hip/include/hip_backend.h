#pragma once
#define HIP_TYPES
#ifdef __APPLE__

#define LongUnsignedNewline "%llu\n"
#define Size_tNewline "%lu\n"
#define LongHexNewline "(0x%llx)\n"
#define alignedMalloc(size, alignment) memalign(alignment, size)
#define SNPRINTF snprintf
#else

#include <malloc.h>

#define LongHexNewline "(0x%lx)\n"
#define LongUnsignedNewline "%lu\n"
#define Size_tNewline "%lu\n"
#if defined (_WIN32)
#include "windows.h"
#define alignedMalloc(size, alignment) _aligned_malloc(size, alignment)
#define SNPRINTF _snprintf
#else
#define alignedMalloc(size, alignment) memalign(alignment, size)
#define SNPRINTF  snprintf
#endif
#endif

#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_TYPES

#include "shared.h"

#include <fstream>

#include<vector>

class HIPBackend : public Backend {
public:
    class HIPConfig : public Backend::Config {
    public:
        boolean gpu;
    };

    class HIPProgram : public Backend::Program {
        class HIPKernel : public Backend::Program::Kernel {
            class HIPBuffer : public Backend::Program::Kernel::Buffer {
            public:
                hipDeviceptr_t devicePtr;

                HIPBuffer(Backend::Program::Kernel *kernel, Arg_s *arg);

                void copyToDevice();

                void copyFromDevice();

                virtual ~HIPBuffer();
            };

        private:
            hipFunction_t kernel;
            hipStream_t hipStream;
        public:
            HIPKernel(Backend::Program *program, char* name, hipFunction_t kernel);

            ~HIPKernel() override;

            long ndrange( void *argArray);
        };

    private:
        hipModule_t module;

    public:
        HIPProgram(Backend *backend, BuildInfo *buildInfo, hipModule_t module);

        ~HIPProgram();

        long getKernel(int nameLen, char *name);

        bool programOK();
    };

private:
    hipDevice_t device;
    hipCtx_t context;
public:

    HIPBackend(HIPConfig *config, int configSchemaLen, char *configSchema);

    HIPBackend();

    ~HIPBackend();

    int getMaxComputeUnits();

    void info();

    long compileProgram(int len, char *source);

};

