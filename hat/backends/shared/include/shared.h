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

typedef struct Buffer_s {
    void *memorySegment;   // Address of a Buffer/MemorySegment
    long sizeInBytes;     // The size of the memory segment in bytes
    void *vendorPtr;       // The vendor side can reference vendor into
    u8_t access;          // 0=??/1=RO/2=WO/3=RW if this is a buffer
    u8_t state;           // 0=UNKNOWN/1=GPUDIRTY/2=JAVADIRTY
} Buffer_t;

typedef union value_u {
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
    Buffer_t buffer; // '&'
} Value_t;

typedef struct Arg_s {
    u32_t idx;          // 0..argc
    u8_t variant;      // which variant 'I','Z','S','J','F', '&' implies Buffer/MemorySegment
    u8_t pad8[8];
    Value_t value;
    u8_t pad6[6];
} Arg_t;

typedef struct ArgArray_s {
    u32_t argc;
    u8_t pad12[12];
    Arg_t argv[0/*argc*/];
    // void * vendorPtr;
    // int schemaLen
    // char schema[schemaLen]
} ArgArray_t;

class ArgSled {
private:
    ArgArray_t *argArray;
public:
    int argc() {
        return argArray->argc;
    }

    Arg_t *arg(int n) {
        Arg_t *a = (argArray->argv + n);
        return a;
    }

    void hexdumpArg(int n) {
        hexdump(arg(n), sizeof(Arg_t));
    }

    void dumpArg(int n) {
        Arg_t *a = arg(n);
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
                          << "}" << std::endl;
                break;
            default:
                std::cout << (char) variant << std::endl;
                break;
        }
    }

    void *vendorPtrPtr() {
        Arg_t *a = arg(argc());
        return (void *) a;
    }

    int *schemaLenPtr() {
        int *schemaLenP = (int *) ((char *) vendorPtrPtr() + sizeof(void *));
        return schemaLenP;
    }

    int schemaLen() {
        return *schemaLenPtr();
    }

    char *schema() {
        int *schemaLenP = ((int *) ((char *) vendorPtrPtr() + sizeof(void *)) + 1);
        return (char *) schemaLenP;
    }

    ArgSled(ArgArray_t *argArray)
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

/*

struct State {
   enum StateType {
      NONE, STRUCT_OR_UNION, SEQUENCE,MEMBER
   };
   StateType stateType;
   char *start;
   void *dataStart;
   int count;
   int of;
   int sz = 0;
   char name[128];

   State(StateType stateType, char *start, void *dataStart, int count, int sz)
      : stateType(stateType), start(start), dataStart(dataStart), count(0), of(count), sz(sz) {
         name[0] = '\0';
      }

   bool isMember() {
      return stateType == MEMBER;
   }

   bool isI08() {
      return stateType == MEMBER && *start == 'b' && sz == 8;
   }

   bool isI16() {
      return stateType == MEMBER && *start == 'i' && sz == 16;
   }

   bool isI32() {
      return stateType == MEMBER && *start == 'i' && sz == 32;
   }

   bool isI64() {
      return stateType == MEMBER && *start == 'i' && sz == 64;
   }
   bool isS08() {
      return stateType == MEMBER && *start == 's' && sz == 8;
   }

   bool isS16() {
      return stateType == MEMBER && *start == 's' && sz == 16;
   }

   bool isS32() {
      return stateType == MEMBER && *start == 's' && sz == 32;
   }

   bool isS64() {
      return stateType == MEMBER && *start == 's' && sz == 64;
   }

   bool isF32() {
      return stateType == MEMBER && *start == 'f' && sz == 32;
   }

   bool isF64() {
      return stateType == MEMBER && *start == 'f' && sz == 64;
   }

   bool isSequence() {
      return stateType == SEQUENCE;
   }

   bool isEndOfSequence(){
      return isSequence() && ((count+1) ==of);
   }
   bool isMidSequence(){
      return isSequence() && ((count+1)<of);
   }

   bool isStructOrUnion() {
      return stateType == STRUCT_OR_UNION;
   }

   void value(std::ostream &s, void *data){
      s << std::setw(8) <<std::left<< typeName() << std::setw(30) << std::left<< name << " ";
      if (isF32()) {
         float *fp = (float *) data;
         s <<std::setw(10) << std::right<< *fp << std::endl;
      } else if (isF64()) {
         double *dp = (double *) data;
         s <<std::setw(10) << std::right << *dp << std::endl;
      } else if (isI08()) {
         char *cp = (char *) data;
         s <<std::setw(10) << std::right<< ((int) *cp) << std::endl;
      } else if (isI16()) {
         short *sp = (short *) data;
         s <<std::setw(10) << std::right<< *sp << std::endl;
      } else if (isI32()) {
         int *ip = (int *) data;
         s <<std::setw(10) << std::right<< *ip << std::endl;;
      } else if (isI64()) {
         long *lp = (long *) data;
         s <<std::setw(10) << std::right<< *lp << std::endl;
      } else if (isS08()) {
         char *cp = (char *) data;
         s <<std::setw(10) << std::right<< ((int) *cp) << std::endl;
      } else if (isS16()) {
         short *sp = (short *) data;
         s <<std::setw(10) << std::right<< *sp << std::endl;
      } else if (isS32()) {
         int *ip = (int *) data;
         s <<std::setw(10) << std::right<< *ip << std::endl;;
      } else if (isS64()) {
         long *lp = (long *) data;
         s <<std::setw(10) << std::right<< *lp << std::endl;
      } else {
         s << "??     ";
      }
   }

   const char* state() {
      switch (stateType) {
         case NONE:return "NONE";
         case SEQUENCE:return "SEQUENCE";
         case MEMBER:return "MEMBER";
         case STRUCT_OR_UNION:return "STRUCT_OR_UNION";
      }
      return "?";
   }
   const char* typeName() {
      if (isF32()) {
         return "f32_t";
      } else if (isF64()) {
         return "f64_t";
      } else if (isI08()) {
         return "i08_t";
      } else if (isI16()) {
         return "i16_t";
      } else if (isI32()) {
         return "i32_t";;
      } else if (isI64()) {
         return "i64_t";
      } else if (isS08()) {
         return "s08_t";
      } else if (isS16()) {
         return "s16_t";
      } else if (isS32()) {
         return "s32_t";;
      } else if (isS64()) {
         return "s64_t";
      } else {
         return "??";
      }
   }
   static State *sequence(char *start, void *dataStart,  int count) {
      return new State(SEQUENCE, start, dataStart, count, 0);
   }

   static State *structOrUnion(char *start, void *dataStart) {
      return new State(STRUCT_OR_UNION, start,dataStart, 0, 0);
   }

   static State *member(char *start,void *dataStart, int sz) {
      return new State(MEMBER, start, dataStart,0, sz);
   }
};*/
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


extern "C" void dumpArgArray(void *ptr);


extern void hexdump(void *ptr, int buflen);

class Schema {
public:
    static std::map<int, std::string> stateNameMap;

    static int replaceStateBit(int state, int remove, int set);
    static int newState(int state, int to);
    static std::ostream &stateType(std::ostream &out, int state);
    static std::ostream &stateDescribe(std::ostream &out, int state);
    static char *strduprange(char *start, char *end);

    static std::ostream &indent(std::ostream &out, int depth);

    static std::ostream &dump(std::ostream &out, char *start, char *end);

    static std::ostream &dump(std::ostream &out, char *label, char *start, char *end);

    static void dumpSled(std::ostream &out, void *argArray);

    static char *dumpSchema(std::ostream &out, int, int depth, char *ptr, void *data);

    static char *dumpSchema(std::ostream &out, char *ptr, void *data);

    static char *dumpSchema(std::ostream &out, char *ptr);
};

class NDRange{
public:
    int x;
    int maxX;
};

class Backend {
public:
    class Config {
    public:
    };

    class Program {
    public:
        class Kernel {
        public:
            class Buffer {
            public:
                Kernel *kernel;
                Arg_t *arg;

                virtual void copyToDevice() = 0;

                virtual void copyFromDevice() = 0;

                Buffer(Kernel *kernel, Arg_t *arg) : kernel(kernel), arg(arg) {
                }

                virtual ~Buffer() {}
            };

            char *name;// strduped!

            Program *program;

            virtual long ndrange( void *argArray) = 0;

            Kernel(Program *program, char * name)
                    : program(program), name(strdup(name)) {
            }

            virtual ~Kernel(){
                if (name){
                    free(name);
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

    Config *config;
    int configSchemaLen;
    char *configSchema;

    Backend(Config *config, int configSchemaLen, char *configSchema)
            : config(config), configSchemaLen(configSchemaLen), configSchema(configSchema) {}

    virtual ~Backend() {};

    virtual void info() = 0;

    virtual int getMaxComputeUnits() = 0;

    virtual long compileProgram(int len, char *source) = 0;


};

extern "C" long getBackend(void *config, int configSchemaLen, char *configSchema);
extern "C" void info(long backendHandle);
extern "C" int getMaxComputeUnits(long backendHandle);
extern "C" long compileProgram(long backendHandle, int len, char *source);
extern "C" long getKernel(long programHandle, int len, char *name);
extern "C" void releaseBackend(long backendHandle);
extern "C" void releaseProgram(long programHandle);
extern "C" bool programOK(long programHandle);
extern "C" void releaseKernel(long kernelHandle);
extern "C" long ndrange(long kernelHandle,  void *argArray);

