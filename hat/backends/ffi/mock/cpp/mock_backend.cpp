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

class MockBackend : public Backend {
public:
    class MockProgram : public Backend::CompilationUnit {
        class MockKernel : public Backend::CompilationUnit::Kernel {
        public:
            MockKernel(Backend::CompilationUnit *compilationUnit, char *name)
                    : Backend::CompilationUnit::Kernel(compilationUnit, name) {
            }
            ~MockKernel() {}
            long ndrange(void *argArray) {
                std::cout << "mock ndrange() " << std::endl;
                return 0;
            }
        };

    public:
        MockProgram(Backend *backend, char *src, char *log, bool ok )
                : Backend::CompilationUnit(backend, src,log, ok) {
        }

        ~MockProgram() {
        }

        Kernel* getKernel(int nameLen, char *name) {
            return new MockKernel(this, name);
        }

        bool compilationUnitOK() {
            return true;
        }
    };

public:

    MockBackend(int configBits): Backend(configBits) {
    }

    ~MockBackend() {
    }

    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) override {
        std::cout << "attempting  to get buffer from Mockbackend "<<std::endl;
        return false;
    }


    void info() override {
        std::cout << "mock info()" << std::endl;
    }

    void computeStart() override{
        std::cout << "mock compute start()" << std::endl;
    }

    void computeEnd() override{
        std::cout << "mock compute start()" << std::endl;
    }

    CompilationUnit *compile(int len, char *source) override{
        std::cout << "mock compileProgram()" << std::endl;
        size_t srcLen = ::strlen(source);
        char *src = new char[srcLen + 1];
        ::strncpy(src, source, srcLen);
        src[srcLen] = '\0';
        std::cout << "native compiling " << src << std::endl;
        MockProgram *mockProgram = new MockProgram(this,src, nullptr, false);
        return dynamic_cast<CompilationUnit*>(mockProgram);
    }
};

long getBackend(int configBits) {
    return reinterpret_cast<long>(new MockBackend(configBits));
}
