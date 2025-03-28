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


import hat.ComputeContext;
import hat.NDRange;
import hat.callgraph.KernelCallGraph;

import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class CudaBackend extends C99FFIBackend {

    final FFILib.LongIntMethodPtr getBackend_MPtr;
    public long getBackend(int mode) {
            backendBridge.handle = getBackend_MPtr.invoke(mode);
            return backendBridge.handle;
    }
    public CudaBackend() {
        super("cuda_backend");
        getBackend_MPtr  =  ffiLib.longIntFunc("getBackend");
        getBackend(0);
        backendBridge.info();
    }

    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        //System.out.println("Cuda backend received computeContext");
        injectBufferTracking(computeContext.computeCallGraph.entrypoint, true,true);

    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        // System.out.println("Cuda backend dispatching kernel " + kernelCallGraph.entrypoint.method);
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createCode(kernelCallGraph, new CudaHatKernelBuilder(), args, true);
            var compilationUnit = backendBridge.compile(code);
            if (compilationUnit.ok()) {
                var kernel = compilationUnit.getKernel(kernelCallGraph.entrypoint.method.getName());
                return new CompiledKernel(this, kernelCallGraph,  kernel, args);
            } else {
                throw new IllegalStateException("cuda failed to compile ");
            }
        });
        compiledKernel.dispatch(ndRange,args);
    }
}
