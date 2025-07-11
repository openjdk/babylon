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

package hat.backend.jextracted;

import hat.NDRange;
import hat.codebuilders.C99HATKernelBuilder;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;
import hat.buffer.KernelContext;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.Schema;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public abstract class C99JExtractedBackend extends JExtractedBackend {
    public C99JExtractedBackend(String libName) {
        super(libName);
    }

    public static class CompiledKernel {
        public final C99JExtractedBackend c99NativeBackend;
        public final KernelCallGraph kernelCallGraph;
        public final String text;
        public final long kernelHandle;
        public final ArgArray argArray;
        public final KernelContext kernelContext;

        public CompiledKernel(C99JExtractedBackend c99NativeBackend, KernelCallGraph kernelCallGraph, String text, long kernelHandle, Object[] ndRangeAndArgs) {
            this.c99NativeBackend = c99NativeBackend;
            this.kernelCallGraph = kernelCallGraph;
            this.text = text;
            this.kernelHandle = kernelHandle;
            this.kernelContext = KernelContext.create(kernelCallGraph.computeContext.accelerator, 0, 0);
            ndRangeAndArgs[0] = this.kernelContext;
            this.argArray = ArgArray.create(kernelCallGraph.computeContext.accelerator, kernelCallGraph,  ndRangeAndArgs);
        }

        public void dispatch(NDRange ndRange, Object[] args) {
            kernelContext.maxX(ndRange.kid.maxX);
            args[0] = this.kernelContext;
            ArgArray.update(argArray,kernelCallGraph,  args);
         //   c99NativeBackend.ndRange(kernelHandle, this.argArray);
        }
    }

    public Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();

    public <T extends C99HATKernelBuilder<T>> String createCode(KernelCallGraph kernelCallGraph, T builder, Object[] args) {
        builder.defines().pragmas().types();
        Set<Schema.IfaceType> already = new LinkedHashSet<>();
        Arrays.stream(args)
                .filter(arg -> arg instanceof Buffer)
                .map(arg -> (Buffer) arg)
                .forEach(ifaceBuffer -> {
                    BoundSchema<?> boundSchema = Buffer.getBoundSchema(ifaceBuffer);
                    boundSchema.schema().rootIfaceType.visitTypes(0, t -> {
                        if (!already.contains(t)) {
                            builder.typedef(boundSchema, t);
                            already.add(t);
                        }
                    });
                });

        // Sorting by rank ensures we don't need forward declarations
        kernelCallGraph.kernelReachableResolvedStream().sorted((lhs, rhs) -> rhs.rank - lhs.rank)
                .forEach(kernelReachableResolvedMethod -> builder.nl().kernelMethod(kernelReachableResolvedMethod).nl());

        builder.nl().kernelEntrypoint(kernelCallGraph.entrypoint, args).nl();

        System.out.println("Original");
        System.out.println(kernelCallGraph.entrypoint.funcOpWrapper().op().toText());
        System.out.println("Lowered");
        System.out.println(kernelCallGraph.entrypoint.funcOpWrapper().lower().op().toText());

        return builder.toString();
    }
}
