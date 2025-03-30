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


 #define UNKNOWN_BYTE 0
 #define RO_BYTE (1<<1)
 #define WO_BYTE (1<<2)
 #define RW_BYTE (RO_BYTE|WO_BYTE)

 struct Buffer_s {
    void *memorySegment;   // Address of a Buffer/MemorySegment
    long sizeInBytes;     // The size of the memory segment in bytes
    u8_t access;          // see hat/buffer/ArgArray.java  UNKNOWN_BYTE=0, RO_BYTE =1<<1,WO_BYTE =1<<2,RW_BYTE =RO_BYTE|WO_BYTE;
} ;

 union Value_u {
    boolean z1;  // 'Z'
    u8_t s8;  // 'B'
    u16_t u16;  // 'C'
    s16_t s16;  // 'S'
    u16_t x16;  // 'C' or 'S"
    s32_t s32;  // 'I'
    s32_t x32;  // 'I' or 'F'
    f32_t f32; // 'F'
    f64_t f64; // 'D'
    s64_t s64; // 'J'
    s64_t x64; // 'D' or 'J'
    Buffer_s buffer; // '&'
} ;

 struct Arg_s {
    u32_t idx;          // 0..argc
    u8_t variant;      // which variant 'I','Z','S','J','F', '&' implies Buffer/MemorySegment
    u8_t pad8[8];
    Value_u value;
    u8_t pad6[6];
    size_t size(){
       size_t sz;
       switch(variant){
          case 'I': case'F':sz= sizeof(u32_t);break;
          case 'S': case 'C':sz= sizeof(u16_t);break;
          case 'D':case 'J':return sizeof(u64_t);break;
          case 'B':return sizeof (u8_t);break;
        default:
           std::cerr <<"Bad variant " <<variant << "arg::size" << std::endl;
           exit(1);

      }

      return sz;
      }
};

 struct BufferState_s{
   static const long  MAGIC =0x4a71facebffab175;
   static const int NO_STATE = 0;
   static const int NEW_STATE = 1;
   static const int HOST_OWNED = 2;
   static const int DEVICE_OWNED = 3;
   static const int DEVICE_VALID_HOST_HAS_COPY = 4;
   const static  char *stateNames[]; // See below for out of line definition

   long magic1;
   void *ptr;
   long length;
   int bits;
   int state;
   void *vendorPtr;
   long magic2;
   bool ok(){
      return ((magic1 == MAGIC) && (magic2 == MAGIC));
   }
   void setState(int newState) {
      state = newState;
   }
   int getState() {
      return state;
   }

   void dump(const char *msg){
     if (ok()){
        printf("{%s,ptr:%016lx,length: %016lx,  state:%08x, vendorPtr:%016lx}\n", msg, (long)ptr, length,  state, (long)vendorPtr);
     }else{
        printf("%s bad magic \n", msg);
        printf("(magic1:%016lx,", magic1);
        printf("{%s, ptr:%016lx, length: %016lx,  state:%08x, vendorPtr:%016lx}", msg, (long)ptr, length,  state, (long)vendorPtr);
        printf("magic2:%016lx)\n", magic2);
     }
   }
   static BufferState_s* of(void *ptr, size_t sizeInBytes){
      return (BufferState_s*) (((char*)ptr)+sizeInBytes-sizeof(BufferState_s));
   }

     static BufferState_s* of(Arg_s *arg){ // access?
        return BufferState_s::of(
           arg->value.buffer.memorySegment,
           arg->value.buffer.sizeInBytes
           );
      }
};
#ifdef shared_cpp
const  char *BufferState_s::stateNames[] = {
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
    Arg_s argv[0/*argc*/];
};

class ArgSled {
private:
    ArgArray_s *argArray;
public:
    int argc() {
        return argArray->argc;
    }

    Arg_s *arg(int n) {
        Arg_s *a = (argArray->argv + n);
        return a;
    }

    void hexdumpArg(int n) {
        hexdump(arg(n), sizeof(Arg_s));
    }

    void dumpArg(int n) {
        Arg_s *a = arg(n);
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

    void *afterArgsPtrPtr() {
        Arg_s *a = arg(argc());
        return (void *) a;
   }

    int *schemaLenPtr() {
        int *schemaLenP = (int *) ((char *) afterArgsPtrPtr() /*+ sizeof(void *) */);
        return schemaLenP;
    }

    int schemaLen() {
        return *schemaLenPtr();
    }

    char *schema() {
        int *schemaLenP = ((int *) ((char *) afterArgsPtrPtr() /*+ sizeof(void *)*/) + 1);
        return (char *) schemaLenP;
    }

    ArgSled(ArgArray_s *argArray)
            : argArray(argArray) {}
};


class Timer {
    struct timeval startTV, endTV;
public:
    unsigned long elapsed_us;

    void start() {
        gettimeofday(&startTV, NULL);
    }

    unsigned long end() {
        gettimeofday(&endTV, NULL);
        elapsed_us = (endTV.tv_sec - startTV.tv_sec) * 1000000;      // sec to us
        elapsed_us += (endTV.tv_usec - startTV.tv_usec);
        return elapsed_us;
    }
};




extern void hexdump(void *ptr, int buflen);

class Sled {
public:
    static void show(std::ostream &out, void *argArray);
};


class NDRange {
public:
    int x;
    int maxX;
};

class Backend {
public:
    class Config{
    public:
        // These must sync with hat/backend/ffi/Mode.java
        // Bits 0-3 select platform id 0..5
        // Bits 4-7 select device id 0..15
        const static  int START_BIT_IDX = 16;
        const static  int MINIMIZE_COPIES_BIT =1<<START_BIT_IDX;
        const static  int TRACE_BIT =1<<17;
        const static  int PROFILE_BIT =1<<18;
        const static  int SHOW_CODE_BIT = 1 << 19;
        const static  int SHOW_KERNEL_MODEL_BIT = 1 << 20;
        const static  int SHOW_COMPUTE_MODEL_BIT = 1 <<21;
        const static  int INFO_BIT = 1<<22;
        const static  int TRACE_COPIES_BIT = 1 <<23;
        const static  int TRACE_SKIPPED_COPIES_BIT = 1 <<24;
        const static  int TRACE_ENQUEUES_BIT = 1 <<25;
        const static  int TRACE_CALLS_BIT = 1 <<26;
        const static  int SHOW_WHY_BIT = 1 <<27;
        const static  int SHOW_STATE_BIT = 1 <<28;
        const static  int END_BIT_IDX = 29;

        const static  char *bitNames[]; // See below for out of line definition
        int configBits;
        bool minimizeCopies;
        bool alwaysCopy;
        bool trace;
        bool profile;
        bool showCode;
        bool info;
        bool traceCopies;
        bool traceSkippedCopies;
        bool traceEnqueues;
        bool traceCalls;
        bool showWhy;
        bool showState;
        int platform; //0..15
        int device; //0..15
        Config(int mode);
        virtual ~Config();
    };
    class Queue {
    public:
        const static  int START_BIT_IDX =20;
        static const int CopyToDeviceBits= 1<<START_BIT_IDX;
        static const int CopyFromDeviceBits= 1<<21;
        static const int NDRangeBits =1<<22;
        static const int StartComputeBits= 1<<23;
        static const int EndComputeBits= 1<<24;
        static const int EnterKernelDispatchBits= 1<<25;
        static const int LeaveKernelDispatchBits= 1<<26;
        static const int HasConstCharPtrArgBits = 1<<27;
        static const int hasIntArgBits = 1<<28;
        const static  int END_BIT_IDX = 27;
        Backend *backend;
        size_t eventMax;
        size_t eventc;

        int *eventInfoBits;
        const char **eventInfoConstCharPtrArgs;
        Queue(Backend *openclBackend);
        virtual void showEvents(int width)=0;
        virtual void wait()=0;
        virtual void release()=0;
        virtual void computeStart()=0;
        virtual void computeEnd()=0;
        virtual void inc(int bits)=0;
        virtual void inc(int bits, const char *arg)=0;
        virtual  void inc(int bits, int arg)=0;
        virtual void marker(int bits)=0;
        virtual void marker(int bits, const char *arg)=0;
        virtual void marker(int bits, int arg)=0;
        virtual void markAsCopyToDeviceAndInc(int argn)=0;
        virtual  void markAsCopyFromDeviceAndInc(int argn)=0;
        virtual void markAsNDRangeAndInc()=0;
        virtual void markAsStartComputeAndInc()=0;
        virtual  void markAsEndComputeAndInc()=0;
        virtual  void markAsEnterKernelDispatchAndInc()=0;
        virtual void markAsLeaveKernelDispatchAndInc()=0;
         virtual ~Queue();
    };
    class Buffer {
    public:
        Backend *backend;
        Arg_s *arg;
        BufferState_s *bufferState;

        virtual void copyToDevice() = 0;

        virtual void copyFromDevice() = 0;

        Buffer(Backend *backend, Arg_s *arg, BufferState_s *bufferState)
                : backend(backend), arg(arg), bufferState(bufferState) {
        }

        virtual ~Buffer() {}
    };
    class CompilationUnit {
    public:
        class Kernel {
        public:
            char *name;// strduped!

            CompilationUnit *compilationUnit;

            virtual long ndrange(void *argArray) = 0;

            Kernel(CompilationUnit *compilationUnit, char *name)
                    : compilationUnit(compilationUnit), name(strutil::clone(name)) {
            }

            virtual ~Kernel() {
                if (name) {
                    delete[] name;
                }
            }
        };
  public:
        Backend *backend;
        char *src;
        char *log;
        bool ok;
        virtual long getKernel(int nameLen, char *name) = 0;

        bool compilationUnitOK(){
           return ok;
        }

        CompilationUnit(Backend *backend, char *src, char *log, bool ok)
                : backend(backend), src(src),log(log),ok(ok) {
        }

        virtual ~CompilationUnit() {
           if (src != nullptr) {
              delete[] src;
           }
           if (log != nullptr) {
              delete[] log;
           }
        };
    };
    int mode;

    Backend(int mode): mode(mode){}

    virtual void info() = 0;

    virtual void computeStart() = 0;

    virtual void computeEnd() = 0;

    virtual long compile(int len, char *source) = 0;

    virtual bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength)=0;

    virtual ~Backend() {};
};

#ifdef shared_cpp
const  char *Backend::Config::bitNames[] = {
              "MINIMIZE_COPIES",
              "TRACE",
              "PROFILE",
              "SHOW_CODE",
              "SHOW_KERNEL_MODEL",
              "SHOW_COMPUTE_MODEL",
              "INFO",
              "TRACE_COPIES",
              "TRACE_SKIPPED_COPIES",
              "TRACE_ENQUEUES",
              "TRACE_CALLS"
              "SHOW_WHY_BIT",
              "USE_STATE_BIT",
              "SHOW_STATE_BIT"
        };
#endif
