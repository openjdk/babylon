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
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.invoke.MethodHandles;

public class TestArrays {

    @CodeReflection
    public static int squareit(int v) {
        return  v * v;

    }

    @CodeReflection
    public static void squareKernel(@MappableIface.RO KernelContext kc, @MappableIface.RW S32Array s32Array) {
        if (kc.x < kc.gsx){
            int value = s32Array.array(kc.x);       // arr[cc.x]
            s32Array.array(kc.x, squareit(value));  // arr[cc.x]=value*value
        }
    }

    @CodeReflection
    public static void square(@MappableIface.RO ComputeContext cc, @MappableIface.RW S32Array s32Array) {
        cc.dispatchKernel(s32Array.length(),
                kc -> squareKernel(kc, s32Array)
        );
    }

    @HatTest
    public static void testHelloHat() {

        final int size = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var array = S32Array.create(accelerator, size);

        // Initialize array
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        // Blocking call
        accelerator.compute(cc -> TestArrays.square(cc, array));

        S32Array test = S32Array.create(accelerator, size);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, squareit(i));
        }

        for (int i = 0; i < test.length(); i++) {
            HatAsserts.assertEquals(test.array(i), array.array(i));
        }
    }

    @HatTest
    public static void testVectorAddition() {

        final int size = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var array = S32Array.create(accelerator, size);

        // Initialize array
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        // Blocking call
        accelerator.compute(cc -> TestArrays.square(cc, array));

        S32Array test = S32Array.create(accelerator, size);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, squareit(i));
        }

        for (int i = 0; i < test.length(); i++) {
            HatAsserts.assertEquals(test.array(i), array.array(i));
        }
    }

}
