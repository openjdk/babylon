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
#define shared_cpp
#include "shared.h"

#define INFO 0


void hexdump(void *ptr, int buflen) {
    unsigned char *buf = (unsigned char *) ptr;
    int i, j;
    for (i = 0; i < buflen; i += 16) {
        printf("%06x: ", i);
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%02x ", buf[i + j]);
            else
                printf("   ");
        printf(" ");
        for (j = 0; j < 16; j++)
            if (i + j < buflen)
                printf("%c", isprint(buf[i + j]) ? buf[i + j] : '.');
        printf("\n");
    }
}

void Sled::show(std::ostream &out, void *argArray) {
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                out << "Buf: of " << arg->value.buffer.sizeInBytes << " bytes " << std::endl;
                break;
            }
            case 'B': {
                out << "S8:" << arg->value.s8 << std::endl;
                break;
            }
            case 'Z': {
                out << "Z:" << arg->value.z1 << std::endl;
                break;
            }
            case 'C': {
                out << "U16:" << arg->value.u16 << std::endl;
                break;
            }
            case 'S': {
                out << "S16:" << arg->value.s16 << std::endl;
                break;
            }
            case 'I': {
                out << "S32:" << arg->value.s32 << std::endl;
                break;
            }
            case 'F': {
                out << "F32:" << arg->value.f32 << std::endl;
                break;
            }
            case 'J': {
                out << "S64:" << arg->value.s64 << std::endl;
                break;
            }
            case 'D': {
                out << "F64:" << arg->value.f64 << std::endl;
                break;
            }
            default: {
                std::cerr << "unexpected variant (shared.cpp) '" << (char) arg->variant << "'" << std::endl;
                exit(1);
            }
        }
    }
    out << "schema len = " << argSled.schemaLen() << std::endl;

    out << "schema = " << argSled.schema() << std::endl;
}



extern "C" void info(long backendHandle) {
    if (INFO){
       std::cout << "trampolining through backendHandle to backend.info()" << std::endl;
    }
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    backend->info();
}
extern "C" void computeStart(long backendHandle) {
    if (INFO){
       std::cout << "trampolining through backendHandle to backend.computeStart()" << std::endl;
    }
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    backend->computeStart();
}
extern "C" void computeEnd(long backendHandle) {
    if (INFO){
       std::cout << "trampolining through backendHandle to backend.computeEnd()" << std::endl;
    }
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    backend->computeEnd();
}
extern "C" void releaseBackend(long backendHandle) {
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    delete backend;
}
extern "C" long compile(long backendHandle, int len, char *source) {
    if (INFO){
       std::cout << "trampolining through backendHandle to backend.compile() "
           <<std::hex<<backendHandle<< std::dec <<std::endl;
    }
    auto *backend = reinterpret_cast<Backend*>(backendHandle);
    auto compilationUnitHandle = backend->compile(len, source);
    if (INFO){
       std::cout << "compilationUnitHandle = "<<std::hex<<compilationUnitHandle<< std::dec <<std::endl;
    }
    return compilationUnitHandle;
}
extern "C" long getKernel(long compilationUnitHandle, int nameLen, char *name) {
    if (INFO){
        std::cout << "trampolining through programHandle to compilationUnit.getKernel()"
            <<std::hex<<compilationUnitHandle<< std::dec <<std::endl;
    }
    auto compilationUnit = reinterpret_cast<Backend::CompilationUnit *>(compilationUnitHandle);
    return compilationUnit->getKernel(nameLen, name);
}

extern "C" long ndrange(long kernelHandle, void *argArray) {
    if (INFO){
       std::cout << "trampolining through kernelHandle to kernel.ndrange(...) " << std::endl;
    }
    auto kernel = reinterpret_cast<Backend::CompilationUnit::Kernel *>(kernelHandle);
    kernel->ndrange( argArray);
    return (long) 0;
}
extern "C" void releaseKernel(long kernelHandle) {
    if (INFO){
       std::cout << "trampolining through to releaseKernel " << std::endl;
    }
    auto kernel = reinterpret_cast<Backend::CompilationUnit::Kernel *>(kernelHandle);
    delete kernel;
}

extern "C" void releaseCompilationUnit(long compilationUnitHandle) {
    if (INFO){
       std::cout << "trampolining through to releaseCompilationUnit " << std::endl;
    }
    auto compilationUnit = reinterpret_cast<Backend::CompilationUnit *>(compilationUnitHandle);
    delete compilationUnit;
}
extern "C" bool compilationUnitOK(long compilationUnitHandle) {
    if (INFO){
       std::cout << "trampolining through to compilationUnitHandleOK " << std::endl;
    }
    auto compilationUnit = reinterpret_cast<Backend::CompilationUnit *>(compilationUnitHandle);
    return compilationUnit->compilationUnitOK();
}

extern "C" bool getBufferFromDeviceIfDirty(long backendHandle, long memorySegmentHandle, long memorySegmentLength) {
    if (INFO){
       std::cout << "trampolining through to getBuffer " << std::endl;
    }
    auto backend = reinterpret_cast<Backend *>(backendHandle);
    auto memorySegment = reinterpret_cast<void *>(memorySegmentHandle);
    return backend->getBufferFromDeviceIfDirty(memorySegment, memorySegmentLength);
}



Backend::Config::Config(int configBits):
        configBits(configBits),
        minimizeCopies((configBits&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
        alwaysCopy(!minimizeCopies),
        trace((configBits&TRACE_BIT)==TRACE_BIT),
        traceCopies((configBits&TRACE_COPIES_BIT)==TRACE_COPIES_BIT),
        traceEnqueues((configBits&TRACE_ENQUEUES_BIT)==TRACE_ENQUEUES_BIT),
        traceCalls((configBits&TRACE_CALLS_BIT)==TRACE_CALLS_BIT),
        traceSkippedCopies((configBits&TRACE_SKIPPED_COPIES_BIT)==TRACE_SKIPPED_COPIES_BIT),
        info((configBits&INFO_BIT)==INFO_BIT),
        showCode((configBits&SHOW_CODE_BIT)==SHOW_CODE_BIT),
        profile((configBits&PROFILE_BIT)==PROFILE_BIT),
        showWhy((configBits&SHOW_WHY_BIT)==SHOW_WHY_BIT),
        showState((configBits&SHOW_STATE_BIT)==SHOW_STATE_BIT),

        platform((configBits&0xf)),
        device((configBits&0xf0)>>4){
    if (info){
        std::cout << "native showCode " << showCode <<std::endl;
        std::cout << "native info " << info<<std::endl;
        std::cout << "native minimizeCopies " << minimizeCopies<<std::endl;
        std::cout << "native alwaysCopy " << alwaysCopy<<std::endl;
        std::cout << "native trace " << trace<<std::endl;
        std::cout << "native traceSkippedCopies " << traceSkippedCopies<<std::endl;
        std::cout << "native traceCalls " << traceCalls<<std::endl;
        std::cout << "native traceCopies " << traceCopies<<std::endl;
        std::cout << "native traceEnqueues " << traceEnqueues<<std::endl;
        std::cout << "native profile " << profile<<std::endl;
        std::cout << "native showWhy " << showWhy<<std::endl;
        std::cout << "native showState " << showState<<std::endl;
        std::cout << "native platform " << platform<<std::endl;
        std::cout << "native device " << device<<std::endl;
    }
}
Backend::Config::~Config(){
}

Backend::Queue::Queue(Backend *backend)
        :backend(backend),
         eventMax(10000),
         eventInfoBits(new int[eventMax]),
         eventInfoConstCharPtrArgs(new const char *[eventMax]),
         eventc(0){
}
Backend::Queue::~Queue() {
    delete[]eventInfoBits;
    delete[]eventInfoConstCharPtrArgs;
}
