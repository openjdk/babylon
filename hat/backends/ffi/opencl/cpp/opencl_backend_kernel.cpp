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
#include "opencl_backend.h"


/*
  OpenCLKernel
  */

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLKernel(Backend::CompilationUnit *compilationUnit, char* name, cl_kernel kernel)
    : Backend::CompilationUnit::Kernel(compilationUnit, name), kernel(kernel){
}

OpenCLBackend::OpenCLProgram::OpenCLKernel::~OpenCLKernel() {
    clReleaseKernel(kernel);
}


/*
void dispatchKernel(Kernel kernel, KernelContext kc, Arg ... args) {
    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (alwaysCopyBuffers || (((arg.flags &JavaDirty)==JavaDirty) && kernel.readsFrom(arg))) {
         enqueueCopyToDevice(arg);
      }
    }
    enqueueKernel(kernel);
    waitForKernel();

    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (alwaysCopyBuffers){
         enqueueCopyFromDevice(arg);
         arg.flags = 0;
      }else{
          if (kernel.writesTo(arg)) {
             arg.flags = DeviceDirty;
          }else{
             arg.flags = 0;
          }
      }
    }

}
*/

bool OpenCLBackend::OpenCLProgram::OpenCLKernel::setArg(KernelArg *arg, Buffer *buffer){
    auto * openCLBuffer = dynamic_cast<OpenCLBuffer *>(buffer);
    cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(cl_mem), &openCLBuffer->clMem);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        return false;
    }
    return true;
}
bool OpenCLBackend::OpenCLProgram::OpenCLKernel::setArg(KernelArg *arg) {
    cl_int status = clSetKernelArg(kernel, arg->idx, arg->size(), (void *) &arg->value);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        return false;
    }
    return true;
}