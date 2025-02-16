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
package hat.backend.ffi;


import hat.Accelerator;
import hat.ComputeContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.Schema;

public class OpenCLBackend extends C99FFIBackend implements BufferTracker {

    public OpenCLBackend() {
        super("opencl_backend");
        Mode mode = Mode.valueOf(System.getProperty("Mode", Mode.GPU_TRACE.toString()));
        getBackend(mode.value,0, 0 );
        info();
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        //System.out.println("OpenCL backend received computeContext");
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        //System.out.println("OpenCL backend dispatching kernel " + kernelCallGraph.entrypoint.method);
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createCode(kernelCallGraph, new OpenCLHatKernelBuilder(), args);
            System.out.println(code);
            long programHandle = compileProgram(code);
            if (programOK(programHandle)) {
                long kernelHandle = getKernel(programHandle, kernelCallGraph.entrypoint.method.getName());
                return new CompiledKernel(this, kernelCallGraph, code, kernelHandle, args);
            } else {
                throw new IllegalStateException("opencl failed to compile ");
            }
        });
        compiledKernel.dispatch(ndRange,args);

    }

    @Override
    public void preMutate(Buffer b) {
        throw new UnsupportedOperationException("OpenCL Backend does not support pre-mutate");
    }

    @Override
    public void postMutate(Buffer b) {
        throw new UnsupportedOperationException("OpenCL Backend does not support post-mutate");
    }

    @Override
    public void preAccess(Buffer b) {
        throw new UnsupportedOperationException("OpenCL Backend does not support pre-access");
    }

    @Override
    public void postAccess(Buffer b) {
        throw new UnsupportedOperationException("OpenCL Backend does not support post-access");
    }

    @Override
    public void preEscape(Buffer b) {
        throw new UnsupportedOperationException("OpenCL Backend does not support pre-escape");
    }

    @Override
    public void postEscape(Buffer b) {
        throw new UnsupportedOperationException("OpenCL Backend does not support post-escape");
    }
}
