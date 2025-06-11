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
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.BufferState;

public class OpenCLBackend extends C99FFIBackend {

   // final Config config;


    public OpenCLBackend(String configSpec) {
        this(Config.of(configSpec));
    }

    public OpenCLBackend(Config config) {
        super("opencl_backend", config);

    }


    public OpenCLBackend() {
        this(Config.of());
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        // System.out.println("OpenCL backend received computeContext minimizing = "+ config.isMINIMIZE_COPIES());
        injectBufferTracking(computeContext.computeCallGraph.entrypoint, config.isSHOW_COMPUTE_MODEL(), config.isMINIMIZE_COPIES());
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        //System.out.println("OpenCL backend dispatching kernel " + kernelCallGraph.entrypoint.method);
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createCode(kernelCallGraph, new OpenCLHatKernelBuilder(), args, config.isSHOW_KERNEL_MODEL());
            if (config.isSHOW_CODE()) {
                System.out.println(code);
            }
            var compilationUnit = backendBridge.compile(code);
            if (compilationUnit.ok()) {
                var kernel = compilationUnit.getKernel( kernelCallGraph.entrypoint.method.getName());
                return new CompiledKernel(this, kernelCallGraph, kernel, args);
            } else {
                throw new IllegalStateException("opencl failed to compile ");
            }
        });
        compiledKernel.dispatch(ndRange, args);

    }

}
