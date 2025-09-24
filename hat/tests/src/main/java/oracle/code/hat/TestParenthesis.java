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
package oracle.code.hat;

import hat.Accelerator;
import hat.ComputeContext;
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface.RO;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.invoke.MethodHandles;

import static hat.ifacemapper.MappableIface.*;

public class TestParenthesis {

    @CodeReflection
    public static void compute(@RO KernelContext context, @RW S32Array data) {
        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX / (TN * TF);
        data.array(context.x, c);
    }

    @CodeReflection
    public static void compute2(@RO KernelContext context, @RW S32Array data) {
        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX / ((TN * TF) / (TN * TN));
        data.array(context.x, c);
    }

    @CodeReflection
    public static void compute3(@RO KernelContext context, @RW S32Array data) {
        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX * (TF + 2) / ((TN * TF) / (TN * TN));
        data.array(context.x, c);
    }

    @CodeReflection
    public static void compute(@RO ComputeContext cc, @RW S32Array data) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(data.length()));
        cc.dispatchKernel(computeRange,kc -> compute(kc, data));
    }

    @CodeReflection
    public static void compute2(@RO ComputeContext cc, @RW S32Array data) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(data.length()));
        cc.dispatchKernel(computeRange,kc -> compute2(kc, data));
    }

    @CodeReflection
    public static void compute3(@RO ComputeContext cc, @RW S32Array data) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(data.length()));
        cc.dispatchKernel(computeRange,kc -> compute3(kc, data));
    }

    @HatTest
    public static void testParenthesis01() {
        final int size = 1;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var data = S32Array.create(accelerator, size);

        // Initialize array
        data.fill(_ -> 0);

        accelerator.compute(cc -> TestParenthesis.compute(cc, data));

        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX / (TN * TF);
        HatAsserts.assertEquals(c, data.array(0));
    }

    @HatTest
    public static void testParenthesis02() {
        final int size = 1;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var data = S32Array.create(accelerator, size);

        // Initialize array
        data.fill(_ -> 0);

        accelerator.compute(cc -> TestParenthesis.compute2(cc, data));

        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX / ((TN * TF) / (TN * TN));
        HatAsserts.assertEquals(c, data.array(0));
    }

    @HatTest
    public static void testParenthesis03() {
        final int size = 1;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var data = S32Array.create(accelerator, size);

        // Initialize array
        data.fill(_ -> 0);

        accelerator.compute(cc -> TestParenthesis.compute3(cc, data));

        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX * (TF + 2) / ((TN * TF) / (TN * TN));
        HatAsserts.assertEquals(c, data.array(0));
    }

}
