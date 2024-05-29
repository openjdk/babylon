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

import hat.NDRange;
import hat.backend.c99codebuilders.C99HatKernelBuilder;
import hat.backend.c99codebuilders.Typedef;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public abstract class C99NativeBackend extends NativeBackend {
    public C99NativeBackend(String libName) {
        super(libName);
    }

    static class CompiledKernel {
        public final C99NativeBackend c99NativeBackend;
        public final KernelCallGraph kernelCallGraph;
        public final String text;
        public final long kernelHandle;
        public final ArgArray argArray;

        CompiledKernel(C99NativeBackend c99NativeBackend, KernelCallGraph kernelCallGraph, String text, long kernelHandle, Object[] ndRangeAndArgs) {
            this.c99NativeBackend = c99NativeBackend;
            this.kernelCallGraph = kernelCallGraph;
            this.text = text;
            this.kernelHandle = kernelHandle;

            Object[] args = new Object[ndRangeAndArgs.length - 1];
            System.arraycopy(ndRangeAndArgs, 1, args, 0, ndRangeAndArgs.length - 1);
            this.argArray = ArgArray.create(kernelCallGraph.computeContext.accelerator, args);
        }

        public void dispatch(Object[] ndRangeAndArgs) {
            // Strip arg0 NDRange.
            NDRange ndRange = (NDRange) ndRangeAndArgs[0];

            Object[] args = new Object[ndRangeAndArgs.length - 1];
            System.arraycopy(ndRangeAndArgs, 1, args, 0, ndRangeAndArgs.length - 1);
            ArgArray.update(argArray, args);
            //    System.out.println(this.argArray.dump());
            // c99NativeBackend.dumpArgArray(argArray);
            //System.out.println("requesting dispatch range "+ndRange.kid.maxX);
            c99NativeBackend.ndRange(kernelHandle, ndRange.kid.maxX, this.argArray);

        }
    }

    Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();

    public <T extends C99HatKernelBuilder<T>> String createCode(KernelCallGraph kernelCallGraph, T builder, Object[] args) {
        builder.defines().pragmas().types();
        Map<String, Typedef> scope = new LinkedHashMap<>();
        Arrays.stream(args)
                .filter(arg -> arg instanceof Buffer)
                .map(arg -> (Buffer) arg)
                .forEach(ifaceBuffer -> builder.typedef(scope, ifaceBuffer));

        // The sort below ensures we don't need forward declarations
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
