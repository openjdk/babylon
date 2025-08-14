/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.LocalMesh1D;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

public class LocalIds {

    // NOTE: Do not forget to add accessor for each parameter.
    @CodeReflection
    private static void assignLocalIds(@RO KernelContext context, @RW S32Array array) {
        int gix = context.gix;
        int lix = context.lix;
        int bsx = context.bsx;
        array.array(gix, bsx);
    }

    @CodeReflection
    private static void mySimpleCompute(@RO ComputeContext cc, @RW S32Array array) {
        // 2 groups of 16 threads each
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(32), new LocalMesh1D(4));
        cc.dispatchKernel(computeRange, kc -> assignLocalIds(kc, array));
    }

    public static void main(String[] args) {
        System.out.println("Experiment: local IDs and local groups");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 32;
        S32Array array = S32Array.create(accelerator, size);

        // Set value to 0
        for (int i = 0; i < size; i++) {
            array.array(i, 0);
        }

        accelerator.compute( cc -> LocalIds.mySimpleCompute(cc, array));

        System.out.println("Finished");
        System.out.println("Result: ");
        for (int i = 0; i < array.length(); i++) {
            System.out.println(array.array(i));
        }
    }

}
