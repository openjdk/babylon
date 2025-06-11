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

#include <sys/wait.h>
#include <chrono>
#include "cuda_backend.h"


CudaBackend::CudaModule::CudaModule(Backend *backend, char *src, char  *log,  bool ok, CUmodule module)
        : Backend::CompilationUnit(backend, src, log, ok), cudaSource(src), ptxSource(),log(log), module(module) {
}

CudaBackend::CudaModule::~CudaModule() = default;
CudaBackend::CudaModule * CudaBackend::CudaModule::of(long moduleHandle){
    return reinterpret_cast<CudaBackend::CudaModule *>(moduleHandle);
}
Backend::CompilationUnit::Kernel * CudaBackend::CudaModule::getKernel(int len, char *name) {
    CudaKernel* cudaKernel= getCudaKernel(len, name);
    return dynamic_cast<Backend::CompilationUnit::Kernel *>(cudaKernel);

}
CudaBackend::CudaModule::CudaKernel *CudaBackend::CudaModule::getCudaKernel(char *name) {
    return getCudaKernel(std::strlen(name), name);
}
CudaBackend::CudaModule::CudaKernel *CudaBackend::CudaModule::getCudaKernel(int nameLen, char *name) {
    CUfunction function;
    WHERE{.f=__FILE__, .l=__LINE__,
          .e=cuModuleGetFunction(&function, module, name),
          .t="cuModuleGetFunction"
    }.report();
    return new CudaKernel(this,name, function);

}

bool CudaBackend::CudaModule::programOK() {
    return true;
}
