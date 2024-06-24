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
import hat.optools.*;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.util.Optional;

public class PTXBackend extends NativeBackend {
    int major;
    int minor;
    String target;
    int addressSize;

    public PTXBackend() {
        super("ptx_backend");
        major = 7;
        minor = 5;
        target = "sm_52";
        addressSize = 64;
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
        String code = createCode(kernelCallGraph, new PTXCodeBuilder(), args);
        System.out.println("\nCod Builder Output: \n\n" + code);
        System.out.println("Add your code to "+PTXBackend.class.getName()+".dispatchKernel() to actually run! :)");
        System.exit(1);
    }

    public String createCode(KernelCallGraph kernelCallGraph, PTXCodeBuilder builder, Object[] args) {

        String out, body;

        Optional<CoreOp.FuncOp> o = Optional.ofNullable(kernelCallGraph.entrypoint.funcOpWrapper().op());
        FuncOpWrapper f = new FuncOpWrapper(o.orElseThrow());
        FuncOpWrapper lowered = f.lower();
        FuncOpWrapper ssa = lowered.ssa();
        System.out.println(ssa.toText());


        System.out.println();
        MethodHandles.Lookup l = MethodHandles.lookup();
        System.out.println(l);
        for (Block.Parameter p : ssa.op().parameters()) {
//            System.out.print(p.type() + " :: ");
            try {
                if ((p.type() instanceof JavaType jt) && (jt.resolve(l) instanceof Class<?> c)) {
                    for (Method m : c.getMethods()) {
                        System.out.print(m.getName() + ", ");
                    }
                    System.out.println();
//                    System.out.println(jt.toNominalDescriptor().);
//                    if (c.isInterface()) System.out.println(memberLayouts());
                }
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }


        //building header
        builder.ptxHeader(major, minor, target, addressSize);

        //building fn info (name, params)
        builder.functionHeader(lowered.functionName());

        // printing out params
        builder.parameters(lowered.paramTable().list());

        //building body of fn
        builder.functionPrologue();

        out = builder.getTextAndReset();
        ssa.firstBody().blocks().forEach(block -> builder.blockBody(block, block.ops().stream().map(OpWrapper::wrap)));

        builder.functionEpilogue();
        body = builder.getTextAndReset();

        builder.ptxRegisterDecl();
        out += builder.getText() + body;

        return out;
    }
}
