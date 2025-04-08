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
#include "shared.h"

class SpirvBackend : public Backend {
public:
    class SpirvQueue: public Backend::Queue{
    public:
        void wait()override{};
        void release()override{};
        void computeStart()override{};
        void computeEnd()override{};
    //   bool copyToDevice(Buffer *buffer, int accessBits) override {return false;};
     //   bool copyFromDevice(Buffer *buffer, int accessBits) override {return false;};
        explicit SpirvQueue(Backend *backend):Queue(backend){}
        ~SpirvQueue() override =default;
    };
    class SpirvProgram : public Backend::CompilationUnit {
        class SpirvKernel : public Backend::CompilationUnit::Kernel {
        public:
            SpirvKernel(Backend::CompilationUnit *compilationUnit, char *name)
                    : Backend::CompilationUnit::Kernel(compilationUnit, name) {
            }

            ~SpirvKernel() {
            }
            bool setArg(Arg_s *arg, Buffer *buffer) override{
                return false ;
            }
            bool setArg(Arg_s *arg) override{
                return false ;
            }
            long ndrange(void *argArray) override {
                std::cout << "spirv ndrange() " << std::endl;
                return 0;
            }
        };

    public:
        SpirvProgram(Backend *backend, char *src, char *log, bool ok)
                : Backend::CompilationUnit(backend,src,log,ok) {
        }

        ~SpirvProgram() {
        }

        Kernel *getKernel(int nameLen, char *name) {
            return new SpirvKernel(this, name);
        }

        bool compilationUnitOK() {
            return true;
        }
    };

public:
    SpirvBackend(int mode): Backend(new Config(mode), new SpirvQueue(this)) {
    }

    ~SpirvBackend() {
    }

    Buffer * getOrCreateBuffer(BufferState_s *bufferState) override{
        Buffer *buffer = nullptr;

      /* if (bufferState->vendorPtr == 0L || bufferState->state == BufferState_s::NEW_STATE){
            openclBuffer = new OpenCLBuffer(this,  bufferState);
            if (openclConfig.trace){
                std::cout << "We allocated arg buffer "<<std::endl;
            }
        }else{
            if (openclConfig.trace){
                std::cout << "Were reusing  buffer  buffer "<<std::endl;
            }
            openclBuffer=  static_cast<OpenCLBuffer*>(bufferState->vendorPtr);
        }*/
        return buffer;
    }
    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) override {
        std::cout << "attempting  to get buffer from SpirvBackend "<<std::endl;
        return false;
    }
    void info() override{
        std::cout << "spirv info()" << std::endl;
    }
     void computeStart() override{
       std::cout << "spirv compute start()" << std::endl;
     }
        void computeEnd() override {
          std::cout << "spirv compute start()" << std::endl;
        }

    CompilationUnit* compile(int len, char *source) override{
        std::cout << "spirv compile()" << std::endl;
        size_t srcLen = ::strlen(source);
        char *src = new char[srcLen + 1];
        ::strncpy(src, source, srcLen);
        src[srcLen] = '\0';
        std::cout << "native compiling " << src << std::endl;

        SpirvProgram *spirvProgram = new SpirvProgram(this,  src, nullptr, false);
        return dynamic_cast<CompilationUnit*>(spirvProgram);
    }
};

long getBackend(int mode) {
    return reinterpret_cast<long>(new SpirvBackend(mode));
}
