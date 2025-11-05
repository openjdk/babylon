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
import hat.buffer.Buffer;
import hat.buffer.F32Array;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;

import java.lang.invoke.MethodHandles;

public class TestLocal {

    private interface MySharedArray extends Buffer {
        void array(long index, float value);
        float array(long index);

        Schema<MySharedArray> schema = Schema.of(MySharedArray.class,
                myPrivateArray -> myPrivateArray
                        .array("array", 16));

        static MySharedArray create(Accelerator accelerator) {
            return schema.allocate(accelerator, 1);
        }

        static MySharedArray createLocal() {
            return create(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }


    @CodeReflection
    private static void compute(@MappableIface.RO KernelContext kernelContext, @MappableIface.RW F32Array data) {
        MySharedArray mySharedArray = MySharedArray.createLocal();
        int lix = kernelContext.lix;
        int blockId = kernelContext.bix;
        int blockSize = kernelContext.lsx;
        mySharedArray.array(lix, lix);
        kernelContext.barrier();
        data.array(lix + (long) blockId * blockSize, mySharedArray.array(lix));
    }

    @CodeReflection
    private static void myCompute(@MappableIface.RO ComputeContext computeContext, @MappableIface.RW F32Array data) {
        NDRange ndRange = NDRange.of(new NDRange.Global1D(32), new NDRange.Local1D(16));
        computeContext.dispatchKernel(ndRange,
                kernelContext -> compute(kernelContext, data)
        );
    }

    @HatTest
    public void testLocal() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        F32Array data = F32Array.create(accelerator, 32);
        accelerator.compute(computeContext -> {
            TestLocal.myCompute(computeContext, data);
        });

        // Check result
        boolean isCorrect = true;
        int jIndex = 0;
        for (int i = 0; i < data.length(); i++) {
            if (data.array(i) != jIndex) {
                isCorrect = false;
                break;
            }
            jIndex++;
            if (jIndex == 16) {
                jIndex = 0;
            }
        }
        HatAsserts.assertTrue(isCorrect);
    }

}
