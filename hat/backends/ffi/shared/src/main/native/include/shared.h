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

#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <stack>

#include "strutil.h"
#include "config.h"

#ifdef __APPLE__
#define SNPRINTF snprintf
#else
#include <malloc.h>
#if defined (_WIN32)
#include "windows.h"
#define SNPRINTF _snprintf
#else
#define SNPRINTF  snprintf
#endif
#endif

typedef char s8_t;
typedef char byte;
typedef char boolean;
typedef char z1_t;
typedef unsigned char u8_t;
typedef short s16_t;
typedef unsigned short u16_t;
typedef unsigned int u32_t;
typedef int s32_t;
typedef float f32_t;
typedef double f64_t;
typedef long s64_t;
typedef unsigned long u64_t;

extern void hexdump(void *ptr, int buflen);

class Text {
public:
    size_t len;
    char *text;
    bool isCopy;

    Text(size_t len, char *text, bool isCopy);

    Text(char *text, bool isCopy);

    explicit Text(size_t len);

    void write(const std::string &filename) const;

    void read(const std::string &filename);

    virtual ~Text();
};

class Log : public Text {
public:
    explicit Log(size_t len);

    explicit Log(char *text);

    ~Log() override = default;
};


#define UNKNOWN_BYTE 0
#define RO_BYTE (1<<1)
#define WO_BYTE (1<<2)
#define RW_BYTE (RO_BYTE|WO_BYTE)

struct Buffer_s {
    void *memorySegment; // Address of a Buffer/MemorySegment
    long sizeInBytes; // The size of the memory segment in bytes
    u8_t access; // see hat/buffer/ArgArray.java  UNKNOWN_BYTE=0, RO_BYTE =1<<1,WO_BYTE =1<<2,RW_BYTE =RO_BYTE|WO_BYTE;
};

union Value_u {
    boolean z1; // 'Z'
    u8_t s8; // 'B'
    u16_t u16; // 'C'
    s16_t s16; // 'S'
    u16_t x16; // 'C' or 'S"
    s32_t s32; // 'I'
    s32_t x32; // 'I' or 'F'
    f32_t f32; // 'F'
    f64_t f64; // 'D'
    s64_t s64; // 'J'
    s64_t x64; // 'D' or 'J'
    Buffer_s buffer; // '&'
};

struct KernelArg {
    u32_t idx; // 0..argc
    u8_t variant; // which variant 'I','Z','S','J','F', '&' implies Buffer/MemorySegment
    u8_t pad8[8];
    Value_u value;
    u8_t pad6[6];

    size_t size() const {
        size_t sz;
        switch (variant) {
            case 'I':
            case 'F':
                sz = sizeof(u32_t);
                break;
            case 'S':
            case 'C':
                sz = sizeof(u16_t);
                break;
            case 'D':
            case 'J':
                return sizeof(u64_t);
                break;
            case 'B':
                return sizeof(u8_t);
                break;
            default:
                std::cerr << "Bad variant " << variant << "arg::size" << std::endl;
                exit(1);
        }

        return sz;
    }
};

struct BufferState {
    static constexpr long MAGIC = 0x4a71facebffab175;
    static constexpr int NO_STATE = 0;
    static constexpr int NEW_STATE = 1;
    static constexpr int HOST_OWNED = 2;
    static constexpr int DEVICE_OWNED = 3;
    static constexpr int DEVICE_VALID_HOST_HAS_COPY = 4;
    const static char *stateNames[]; // See below for out of line definition

    long magic1;
    void *ptr;
    long length;
    int bits;
    int state;
    void *vendorPtr;
    long magic2;

    bool ok() const {
        return ((magic1 == MAGIC) && (magic2 == MAGIC));
    }

    void setState(int newState) {
        state = newState;
    }

    int getState() const {
        return state;
    }

    void dump(const char *msg) const {
        if (ok()) {
            printf("{%s,ptr:%016lx,length: %016lx,  state:%08x, vendorPtr:%016lx}\n", msg, (long) ptr, length, state,
                   (long) vendorPtr);
        } else {
            printf("%s bad magic \n", msg);
            printf("(magic1:%016lx,", magic1);
            printf("{%s, ptr:%016lx, length: %016lx,  state:%08x, vendorPtr:%016lx}", msg, (long) ptr, length, state,
                   (long) vendorPtr);
            printf("magic2:%016lx)\n", magic2);
        }
    }

    static BufferState *of(void *ptr, size_t sizeInBytes) {
        return reinterpret_cast<BufferState *>(static_cast<char *>(ptr) + sizeInBytes - sizeof(BufferState));
    }

    static BufferState *of(const KernelArg *arg) {
        // access?
        BufferState *bufferState = BufferState::of(
            arg->value.buffer.memorySegment,
            arg->value.buffer.sizeInBytes
        );


        //Sanity check the buffers
        // These sanity check finds errors passing memory segments which are not Buffers

        if (bufferState->ptr != arg->value.buffer.memorySegment) {
            std::cerr << "bufferState->ptr !=  arg->value.buffer.memorySegment" << std::endl;
            std::exit(1);
        }

        if ((bufferState->vendorPtr == nullptr) && (bufferState->state != BufferState::NEW_STATE)) {
            std::cerr << "Warning:  Unexpected initial state for buffer "
                    //<<" of kernel '"<<(dynamic_cast<Backend::CompilationUnit::Kernel*>(this))->name<<"'"
                    << " state=" << bufferState->state << " '"
                    << BufferState::stateNames[bufferState->state] << "'"
                    << " vendorPtr" << bufferState->vendorPtr << std::endl;
        }
        // End of sanity checks
        return bufferState;
    }
};

#ifdef shared_cpp
const char *BufferState::stateNames[] = {
    "NO_STATE",
    "NEW_STATE",
    "HOST_OWNED",
    "DEVICE_OWNED",
    "DEVICE_VALID_HOST_HAS_COPY"
};
#endif

struct ArgArray_s {
    u32_t argc;
    u8_t pad12[12];
    KernelArg argv[0/*argc*/];
};

class ArgSled {
private:
    ArgArray_s *argArray;

public:
    int argc() const {
        return argArray->argc;
    }

    KernelArg *arg(int n) const {
        KernelArg *a = (argArray->argv + n);
        return a;
    }

    void hexdumpArg(int n) const {
        hexdump(arg(n), sizeof(KernelArg));
    }

    void dumpArg(int n) const {
        KernelArg *a = arg(n);
        int idx = (int) a->idx;
        std::cout << "arg[" << idx << "]";
        char variant = (char) a->variant;
        switch (variant) {
            case 'F':
                std::cout << " f32 " << a->value.f32 << std::endl;
                break;
            case 'I':
                std::cout << " s32 " << a->value.s32 << std::endl;
                break;
            case 'D':
                std::cout << " f64 " << a->value.f64 << std::endl;
                break;
            case 'J':
                std::cout << " s64 " << a->value.s64 << std::endl;
                break;
            case 'C':
                std::cout << " u16 " << a->value.u16 << std::endl;
                break;
            case 'S':
                std::cout << " s16 " << a->value.s32 << std::endl;
                break;
            case 'Z':
                std::cout << " z1 " << a->value.z1 << std::endl;
                break;
            case '&':
                std::cout << " buffer {"
                        << " void *address = 0x" << std::hex << (long) a->value.buffer.memorySegment << std::dec
                        << ", long bytesSize= 0x" << std::hex << (long) a->value.buffer.sizeInBytes << std::dec
                        << ", char access= 0x" << std::hex << (unsigned char) a->value.buffer.access << std::dec
                        << "}" << std::endl;
                break;
            default:
                std::cout << (char) variant << std::endl;
                break;
        }
    }

    void *afterArgsPtrPtr() const {
        KernelArg *a = arg(argc());
        return (void *) a;
    }

    int *schemaLenPtr() const {
        int *schemaLenP = (int *) ((char *) afterArgsPtrPtr() /*+ sizeof(void *) */);
        return schemaLenP;
    }

    int schemaLen() const {
        return *schemaLenPtr();
    }

    char *schema() const {
        int *schemaLenP = ((int *) ((char *) afterArgsPtrPtr() /*+ sizeof(void *)*/) + 1);
        return (char *) schemaLenP;
    }

    explicit ArgSled(ArgArray_s *argArray)
        : argArray(argArray) {
    }
};


class Timer {
    struct timeval startTV, endTV;

public:
    unsigned long elapsed_us{};

    Timer(): startTV(), endTV() {
    }

    void start() {
        gettimeofday(&startTV, nullptr);
    }

    unsigned long end() {
        gettimeofday(&endTV, nullptr);
        elapsed_us = (endTV.tv_sec - startTV.tv_sec) * 1000000; // sec to us
        elapsed_us += (endTV.tv_usec - startTV.tv_usec);
        return elapsed_us;
    }
};


//extern void hexdump(void *ptr, int buflen);

class Sled {
public:
    static void show(std::ostream &out, void *argArray);
};

class KernelContext {
public:
    int x;
    int maxX;
    int y;
    int maxY;
    int z;
    int maxZ;
    int dimensions;

    // global index
    int gix;
    int giy;
    int giz;

    // global sizes
    int gsx;
    int gsy;
    int gsz;

    // local index
    int lix;
    int liy;
    int liz;

    // local size
    int lsx;
    int lsy;
    int lsz;

    // Group index
    int bix;
    int biy;
    int biz;
};

class Backend {
public:
    class Config final : public BasicConfig {
    public:
        explicit Config(int mode);

        ~Config() override;
    };

    class Buffer {
    public:
        Backend *backend;
        BufferState *bufferState;

        Buffer(Backend *backend, BufferState *bufferState)
            : backend(backend), bufferState(bufferState) {
        }

        virtual ~Buffer() = default;
    };

    class CompilationUnit {
    public:
        class Kernel {
        public:
            char *name;

            CompilationUnit *compilationUnit;

            virtual bool setArg(KernelArg *arg, Buffer *openCLBuffer) = 0;

            virtual bool setArg(KernelArg *arg) = 0;

            virtual long ndrange(void *argArray) final;

            Kernel(CompilationUnit *compilationUnit, char *name)
                : name(strutil::clone(name)), compilationUnit(compilationUnit) {
            }

            virtual ~Kernel() {
                delete[] name;
            }
        };

    public:
        Backend *backend;
        char *src;
        char *log;
        bool ok;

        virtual Kernel *getKernel(int nameLen, char *name) = 0;

        virtual bool compilationUnitOK() final {
            return ok;
        }

        CompilationUnit(Backend *backend, char *src, char *log, bool ok)
            : backend(backend), src(src), log(log), ok(ok) {
        }

        virtual ~CompilationUnit() {
            delete[] src;
            delete[] log;
        };
    };

    class Queue {
    public:
        Backend *backend;

        explicit Queue(Backend *backend);

        virtual void wait() = 0;

        virtual void release() = 0;

        virtual void computeStart() = 0;

        virtual void computeEnd() = 0;

        virtual void copyToDevice(Buffer *buffer) =0;

        virtual void copyFromDevice(Buffer *buffer) =0;

        virtual void dispatch(KernelContext *kernelContext, CompilationUnit::Kernel *kernel) = 0;

        virtual ~Queue();
    };

    class ProfilableQueue : public Queue {
    public:
        static constexpr int START_BIT_IDX = 20;
        static constexpr int CopyToDeviceBits = 1 << START_BIT_IDX;
        static constexpr int CopyFromDeviceBits = 1 << 21;
        static constexpr int NDRangeBits = 1 << 22;
        static constexpr int StartComputeBits = 1 << 23;
        static constexpr int EndComputeBits = 1 << 24;
        static constexpr int EnterKernelDispatchBits = 1 << 25;
        static constexpr int LeaveKernelDispatchBits = 1 << 26;
        static constexpr int HasConstCharPtrArgBits = 1 << 27;
        static constexpr int hasIntArgBits = 1 << 28;
        static constexpr int END_BIT_IDX = 27;

        size_t eventMax;
        size_t eventc;
        int *eventInfoBits;
        const char **eventInfoConstCharPtrArgs;

        virtual void showEvents(int width) = 0;

        virtual void inc(int bits) = 0;

        virtual void inc(int bits, const char *arg) = 0;

        virtual void marker(int bits) = 0;

        virtual void marker(int bits, const char *arg) = 0;


        virtual void markAsStartComputeAndInc() = 0;

        virtual void markAsEndComputeAndInc() = 0;

        virtual void markAsEnterKernelDispatchAndInc() = 0;

        virtual void markAsLeaveKernelDispatchAndInc() = 0;

        ProfilableQueue(Backend *backend, int eventMax)
            : Queue(backend),
              eventMax(eventMax),
              eventInfoBits(new int[eventMax]),
              eventInfoConstCharPtrArgs(new const char *[eventMax]),
              eventc(0) {
        }

        ~ProfilableQueue() override {
            delete[]eventInfoBits;
            delete[]eventInfoConstCharPtrArgs;
        }
    };

    Config *config;
    Queue *queue;

    Backend(Config *config, Queue *queue)
        : config(config), queue(queue) {
    }

    virtual Buffer *getOrCreateBuffer(BufferState *bufferState) = 0;

    virtual void info() = 0;

    virtual void computeStart() = 0;

    virtual void computeEnd() = 0;

    virtual CompilationUnit *compile(int len, char *source) = 0;

    virtual bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) = 0;

    virtual ~Backend() = default;
};



template<typename T>
T *bufferOf(const char *name) {
    size_t lenIncludingBufferState = sizeof(T);
    size_t lenExcludingBufferState = lenIncludingBufferState - sizeof(BufferState);
    T *buffer = reinterpret_cast<T *>(new unsigned char[lenIncludingBufferState]);
    auto *bufferState = reinterpret_cast<BufferState *>(reinterpret_cast<char *>(buffer) + lenExcludingBufferState);
    bufferState->magic1 = bufferState->magic2 = BufferState::MAGIC;
    bufferState->ptr = buffer;
    bufferState->length = sizeof(T) - sizeof(BufferState);
    bufferState->state = BufferState::NEW_STATE;
    bufferState->vendorPtr = nullptr;
    bufferState->dump(name);
    return buffer;
}
