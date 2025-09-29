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

public class OpenCLBackend extends C99FFIBackend {

    public OpenCLBackend(String configSpec) {
        this(FFIConfig.of(configSpec));
    }

    public OpenCLBackend() {
        this(FFIConfig.of());
    }

    public OpenCLBackend(FFIConfig config) {
        super("opencl_backend", config);
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createC99(kernelCallGraph,  args);
            if (FFIConfig.SHOW_CODE.isSet(config.bits())) {
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

    String createC99(KernelCallGraph kernelCallGraph,  Object[] args){
        return createCode(kernelCallGraph, new OpenCLHATKernelBuilder(), args);
    }

}
