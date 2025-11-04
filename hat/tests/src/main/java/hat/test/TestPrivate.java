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
import hat.Global1D;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.buffer.F32Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;

import java.lang.invoke.MethodHandles;

import static hat.ifacemapper.MappableIface.RW;

public class TestPrivate {

    private interface PrivateArray extends Buffer {
        void array(long index, float value);
        float array(long index);

        Schema<PrivateArray> schema = Schema.of(PrivateArray.class,
                myPrivateArray -> myPrivateArray
                        .array("array", 1));

        static PrivateArray create(Accelerator accelerator) {
            return schema.allocate(accelerator, 1);
        }

        static PrivateArray createPrivate() {
            return create(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    @CodeReflection
    private static void compute(@RO KernelContext kernelContext, @RW F32Array data) {
        PrivateArray privateArray = PrivateArray.createPrivate();
        int lix = kernelContext.lix;
        int blockId = kernelContext.bix;
        int blockSize = kernelContext.lsx;
        privateArray.array(0, lix);
        data.array(lix + (long) blockId * blockSize, privateArray.array(0));
    }

    @CodeReflection
    private static void myCompute(@RO ComputeContext computeContext, @RW F32Array data) {
        NDRange ndRange = new NDRange(new Global1D(32));
        computeContext.dispatchKernel(ndRange,
                kernelContext -> compute(kernelContext, data)
        );
    }

    @HatTest
    public void testPrivate() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        F32Array data = F32Array.create(accelerator, 32);
        accelerator.compute(computeContext -> {
            TestPrivate.myCompute(computeContext, data);
        });

        // Check result
        boolean isCorrect = true;
        int jIndex = 0;
        for (int i = 0; i < data.length(); i++) {
            if (data.array(i) != jIndex++) {
                IO.println(data.array(i) + " != " + jIndex);
                isCorrect = false;
                //break;
            }
        }
        HatAsserts.assertTrue(isCorrect);
    }

}
