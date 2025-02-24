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
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class MockBackend extends FFIBackend {
    final MethodHandle getBackend_MH;
    public long getBackend(int mode) {
        try {
            backendHandle = (long) getBackend_MH.invoke(mode);
        } catch (Throwable throwable) {
            throw new IllegalStateException(throwable);
        }
        return backendHandle;
    }


    public MockBackend() {
        super("mock_backend");
        getBackend_MH  =  nativeLibrary.longFunc("getMockBackend",JAVA_INT);
        getBackend(0);
    }

    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        System.out.println("Mock backend recieved closed closure");
        System.out.println("Mock backend will mutate  " + computeContext.computeCallGraph.entrypoint + computeContext.computeCallGraph.entrypoint.method);
        injectBufferTracking(computeContext.computeCallGraph.entrypoint, true);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        System.out.println("Mock dispatch kernel");
        // Here we receive a callgraph from the kernel entrypoint
        // The first time we see this we need to convert the kernel entrypoint
        // and rechable methods to a form that our mock backend can execute.
        kernelCallGraph.kernelReachableResolvedStream().forEach(kr -> {

        });
    }
}
