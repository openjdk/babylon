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
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;

/**
 * How to run? This test is meant to be launched with minimized copies ON:
 *
 * <code>HAT=MC java @.ffi-cuda-test hat.test.TestLookups</code>
 */
public class TestLookups {

    private static class Foo {

        @Reflect
        static void myKernel(F32Array array) {
            final int gidx = KernelContext.GIX();
            array.array(gidx, -100);
        }

        @Reflect
        static void myCompute(ComputeContext cc, F32Array array) {
            cc.dispatchKernel(NDRange.of1D(array.length()), () -> myKernel(array));
            cc.copyOutOnExit(array);
        }
    }

    @HatTest
    public void test() {
        final int size = 8192;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var array = F32Array.create(accelerator, size);

        // Invoke a private method from another class
        // This used to break if the MC option was enabled.
        accelerator.compute((@Reflect Compute) cc -> Foo.myCompute(cc, array));

        for (int i = 0; i < array.length(); i++) {
            HATAsserts.assertEquals(-100, array.array(i), 0.0f);
        }
    }

}
