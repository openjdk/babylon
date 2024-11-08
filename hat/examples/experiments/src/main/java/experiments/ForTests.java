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
import hat.buffer.F32Array;

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.CodeReflection;

public class ForTests {

    public static class Compute {

        @CodeReflection
        static void breakAndContinue(KernelContext kc, F32Array a) {
            long i = kc.x;
            long size = kc.maxX;
            outer:
            for (long j = 0; j < size; j++) {
                float sum = 0f;
                for (long k = 0; k < size; k++) {
                    if (k == 6) {
                        sum += 3;
                        break outer;
                    } else if (k == 4) {
                        sum += 2;
                        continue outer;
                    } else if (k == 0) {
                        sum += 0;
                    } else {
                        sum += 4;
                    }
                    sum++;
                }
            }
        }

        @CodeReflection
        static void counted(KernelContext kc, F32Array a) {
            for (int j = 0; j < a.length(); j = j + 1) {
                float sum = j;
            }
        }

        @CodeReflection
        static void tuple(KernelContext kc, F32Array a) {
            for (int j = 1, i = 2, k = 3; j < a.length(); k += 1, i += 2, j += 3) {
                float sum = k + i + j;
            }
        }

        @CodeReflection
        static void compute(ComputeContext computeContext, F32Array a) {
            computeContext.dispatchKernel(a.length(), (kc) -> counted(kc, a));
            computeContext.dispatchKernel(a.length(), (kc) -> tuple(kc, a));
            computeContext.dispatchKernel(a.length(), (kc) -> breakAndContinue(kc, a));
        }

    }

    public static void main(String[] args) {

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
                //  Backend.JAVA_MULTITHREADED
                (backend) -> backend.getClass().getSimpleName().startsWith("OpenCL")
        );
        var a = F32Array.create(accelerator,100);
        accelerator.compute(
                cc -> Compute.compute(cc, a)
        );

    }

}
