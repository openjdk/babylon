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
 // hat iface buffer bits
 // hat iface bffa   bits
 // 4a7 1face bffa   b175

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
   static const int   NONE = 0;
   static const int   BIT_HOST_NEW =1<<0;
   static const int   BIT_DEVICE_NEW =1<<1;
   static const int   BIT_HOST_DIRTY =1<<2;
   static const int   BIT_DEVICE_DIRTY =1<<3;
   static const int   BIT_HOST_CHECKED =1<<4;


   long magic1;
   int bits;
   int unused;
   void *vendorPtr;
   long magic2;
   bool ok(){
      return ((magic1 == MAGIC) && (magic2 == MAGIC));
   }

   void assignBits(int bitBits) {
      bits=bitBits;
   }
   void setBits(int bitBits) {
      bits|=bitBits;
   }
   void  xorBits(int bitsToReset) {
      // say bits = 0b0111 (7) and bitz = 0b0100 (4)
      int xored = bits^bitsToReset;  // xored = 0b0011 (3)
      bits =  xored;
   }
    void  resetBits(int bitsToReset) {
         // say bits = 0b0111 (7) and bitz = 0b0100 (4)
         bits = bits&~bitsToReset;  // xored = 0b0011 (3)
         //bits =  xored;
      }
   int getBits() {
      return bits;
   }
   bool areBitsSet(int bitBits) {
      return (bits&bitBits)==bitBits;
   }
   void setHostDirty(){
      setBits(BIT_HOST_DIRTY);
   }
   bool isHostDirty(){
      return  areBitsSet(BIT_HOST_DIRTY);
   }
   void clearHostDirty(){
      resetBits(BIT_HOST_DIRTY);
   }
   void clearHostChecked(){
      resetBits(BIT_HOST_CHECKED);
   }
   void clear(){
       bits=0;
   }
   bool isHostNew(){
      return  areBitsSet(BIT_HOST_NEW);
   }
   void clearHostNew(){
      resetBits(BIT_HOST_NEW);
   }
   bool isHostNewOrDirty() {
      return areBitsSet(BIT_HOST_NEW|BIT_HOST_DIRTY);
   }

   void setDeviceDirty(){
      setBits(BIT_DEVICE_DIRTY);
   }

   bool isDeviceDirty(){
      return areBitsSet(BIT_DEVICE_DIRTY);
   }
   void clearDeviceDirty(){
      resetBits(BIT_DEVICE_DIRTY);
   }


   void dump(const char *msg){
     if (ok()){
        printf("{%s, bits:%08x, unused:%08x, vendorPtr:%016lx}\n", msg, bits, unused, (long)vendorPtr);
     }else{
        printf("%s bad magic \n", msg);
        printf("(magic1:%016lx,", magic1);
        printf("{%s, bits:%08x, unused:%08x, vendorPtr:%016lx}", msg, bits, unused, (long)vendorPtr);
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


class BuildInfo {
public:
    char *src;
    char *log;
    bool ok;

    BuildInfo(char *src, char *log, bool ok)
            : src(src), log(log), ok(ok) {
    }

    ~BuildInfo() {
        if (src) {
            delete[] src;
        }
        if (log) {
            delete[] log;
        }
    }

};


//extern "C" void dumpArgArray(void *ptr);


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

    class Program {
    public:
        class Kernel {
        public:
            class Buffer {
            public:
                Kernel *kernel;
                Arg_s *arg;

                virtual void copyToDevice() = 0;

                virtual void copyFromDevice() = 0;

                Buffer(Kernel *kernel, Arg_s *arg)
                        : kernel(kernel), arg(arg) {
                }

                virtual ~Buffer() {}
            };

            char *name;// strduped!

            Program *program;

            virtual long ndrange(void *argArray) = 0;
            static char *copy(char *name){
                size_t len =::strlen(name);
                char *buf = new char[len+1];
                memcpy(buf, name, len);
                buf[len]='\0';
                return buf;
            }
            Kernel(Program *program, char *name)
                    : program(program), name(copy(name)) {
            }

            virtual ~Kernel() {
                if (name) {
                    delete[] name;
                }
            }
        };

    public:
        Backend *backend;
        BuildInfo *buildInfo;

        virtual long getKernel(int nameLen, char *name) = 0;

        virtual bool programOK() = 0;

        Program(Backend *backend, BuildInfo *buildInfo)
                : backend(backend), buildInfo(buildInfo) {
        }

        virtual ~Program() {
            if (buildInfo != nullptr) {
                delete buildInfo;
            }
        };

    };
    int mode;

    Backend(int mode)
            : mode(mode){}

    virtual void info() = 0;

     virtual void computeStart() = 0;
      virtual void computeEnd() = 0;

    virtual int getMaxComputeUnits() = 0;

    virtual long compileProgram(int len, char *source) = 0;

    virtual bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength)=0;

    virtual ~Backend() {};
};

extern "C" void info(long backendHandle);
extern "C" int getMaxComputeUnits(long backendHandle);
extern "C" long compileProgram(long backendHandle, int len, char *source);
extern "C" long getKernel(long programHandle, int len, char *name);
extern "C" void releaseBackend(long backendHandle);
extern "C" void releaseProgram(long programHandle);
extern "C" bool programOK(long programHandle);
extern "C" void releaseKernel(long kernelHandle);
extern "C" long ndrange(long kernelHandle, void *argArray);
extern "C" void computeStart(long backendHandle);
extern "C" void computeEnd(long backendHandle);
extern "C" bool getBufferFromDeviceIfDirty(long backendHandle, long memorySegmentHandle, long memorySegmentLength);

