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
package hat.test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.NDRange;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import optkl.ifacemapper.MappableIface.RO;
import jdk.incubator.code.Reflect;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;

import java.lang.invoke.MethodHandles;

import static optkl.ifacemapper.MappableIface.RW;

public class TestParenthesis {

    @Reflect
    public static void compute(@RO KernelContext context, @RW S32Array data) {
        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX / (TN * TF);
        data.array(context.gix, c);
    }

    @Reflect
    public static void compute2(@RO KernelContext context, @RW S32Array data) {
        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX / ((TN * TF) / (TN * TN));
        data.array(context.gix, c);
    }

    @Reflect
    public static void compute3(@RO KernelContext context, @RW S32Array data) {
        final int TN = 2;
        final int TF = 128;
        final int MAX = 1024;
        int c = MAX * (TF + 2) / ((TN * TF) / (TN * TN));
        data.array(context.gix, c);
    }

    @Reflect
    public static void compute(@RO ComputeContext cc, @RW S32Array data) {
        cc.dispatchKernel(NDRange.of1D(data.length()),kc -> compute(kc, data));
    }

    @Reflect
    public static void compute2(@RO ComputeContext cc, @RW S32Array data) {
        cc.dispatchKernel(NDRange.of1D(data.length()),kc -> compute2(kc, data));
    }

    @Reflect
    public static void compute3(@RO ComputeContext cc, @RW S32Array data) {
        cc.dispatchKernel(NDRange.of1D(data.length()),kc -> compute3(kc, data));
    }

    @HatTest
    @Reflect
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
        HATAsserts.assertEquals(c, data.array(0));
    }

    @HatTest
    @Reflect
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
        HATAsserts.assertEquals(c, data.array(0));
    }

    @HatTest
    @Reflect
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
        HATAsserts.assertEquals(c, data.array(0));
    }

}
