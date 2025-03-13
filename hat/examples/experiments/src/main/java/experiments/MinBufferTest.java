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
package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.ffi.OpenCLBackend;
import hat.buffer.S32Array;
import static hat.ifacemapper.MappableIface.*;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

import static hat.backend.ffi.OpenCLConfig.*;

public class MinBufferTest {


    public static class Compute {
        @CodeReflection
        public static void inc(@RO KernelContext kc, @RW S32Array s32Array, int len) {
            if (kc.x < kc.maxX) {
                s32Array.array(kc.x, s32Array.array(kc.x) + 1);
            }
        }

        @CodeReflection
        public static void add(ComputeContext cc, @RW S32Array s32Array, int len, int n) {
            for (int i = 0; i < n; i++) {
                cc.dispatchKernel(len, kc -> inc(kc, s32Array, len));
                System.out.println(i);//s32Array.array(0));
            }
        }
    }

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
                new OpenCLBackend(of(
                      //  TRACE(),
                        TRACE_COPIES(),
                        GPU(),
                        MINIMIZE_COPIES()
                ))

        );
        int len = 10000000;
        int valueToAdd = 10;
        S32Array s32Array = S32Array.create(accelerator, len,i->i);
        accelerator.compute(
                cc -> Compute.add(cc, s32Array, len, valueToAdd)
        );
        // Quite an expensive way of adding 20 to each array alement
        for (int i = 0; i < 20; i++) {
            System.out.println(i + "=" + s32Array.array(i));
        }
    }

}
