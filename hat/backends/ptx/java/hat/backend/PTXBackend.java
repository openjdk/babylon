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
package hat.backend;


import hat.ComputeContext;
import hat.NDRange;
import hat.callgraph.KernelCallGraph;

public class PTXBackend extends NativeBackend {
    public PTXBackend() {
        super("ptx_backend");
        getBackend(null);
    }

    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        System.out.println("PTX backend recieved closed closure");
        System.out.println("PTX backend will mutate  " + computeContext.computeCallGraph.entrypoint + computeContext.computeCallGraph.entrypoint.method);
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        System.out.println("PTX dispatch kernel");
        // Here we recieve a callgraph from the kernel entrypoint
        // The first time we see this we need to convert the kernel entrypoint
        // and rechable methods to PTX.

        // sort the dag by rank means that we get the methods called by the entrypoint in dependency order
        // of course there may not be any of these
        kernelCallGraph.kernelReachableResolvedStream()
                .sorted((lhs, rhs) -> rhs.rank - lhs.rank)
                .forEach(kernelReachableResolvedMethod ->
                        System.out.println(" call to -> "+kernelReachableResolvedMethod.method.getName())
                );

        System.out.println("Entrypoint ->"+kernelCallGraph.entrypoint.method.getName());
        System.out.println(kernelCallGraph.entrypoint.funcOpWrapper().toText());
        System.out.println("Add your code to "+PTXBackend.class.getName()+".dispatchKernel() to actually run! :)");
        System.exit(1);
    }
}
