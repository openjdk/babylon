/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
import hat.HATMath;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.Accelerator.Compute;

public class TestConstantFloat {

    @Reflect
    public static void mathConstantFloat(F32Array input, F32Array output) {
        final int idx = KernelContext.GIX();
        final float maxValue = HATMath.maxf(input.array(idx), 1e-5f);
        output.array(idx, maxValue);
    }

    @Reflect
    public static void mathConstantFloat(ComputeContext context, F32Array input, F32Array output) {
        context.dispatchKernel(NDRange.of1D(input.length()), () -> mathConstantFloat(input, output));
    }

    @HatTest
    public static void test01() {
        final var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        F32Array input = F32Array.create(accelerator, size);
        F32Array output = F32Array.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            input.array(i, r.nextFloat());
        }

        accelerator.compute((@Reflect Compute)
                cc -> mathConstantFloat(cc, input, output));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals(HATMath.maxf(input.array(i), 1e-5f), output.array(i), 0.001f);
        }
    }

}
