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
import hat.buffer.Buffer;
import hat.buffer.F32Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

/**
 * Example of how to declare and use a custom data type in a method kernel on the GPU.
 * This is just a proof of concept.
 */
public class LocalArray {

    private interface MySharedArray extends Buffer {
        int length();
        void array(long index, float value);
        float array(long index);

        Schema<MySharedArray> schema = Schema.of(MySharedArray.class,
                myPrivateArray -> myPrivateArray
                        .arrayLen("length")
                        .pad(12)
                        .array("array"));

        static MySharedArray create(Accelerator accelerator, int length) {
            return schema.allocate(accelerator, length);
        }

        static <T extends Buffer> MySharedArray createLocal(Class<T> klass, int length) {
            return (MySharedArray) Buffer.createLocal(klass, length);
        }
    }

    @CodeReflection
    private static void compute(@RO KernelContext kernelContext, @RW F32Array data) {
        MySharedArray mySharedArray = MySharedArray.createLocal(MySharedArray.class, 18);
        mySharedArray.array(0, kernelContext.lix);
        data.array(kernelContext.gix, mySharedArray.array(0));
    }

    @CodeReflection
    private static void myCompute(@RO ComputeContext computeContext, @RW F32Array data) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(32), new LocalMesh1D(16));
        computeContext.dispatchKernel(computeRange,
                kernelContext -> compute(kernelContext, data)
        );
    }

    static void main(String[] args) {
        System.out.println("Testing Shared Data Structures Mapping");
        System.out.println("Schema description");
        MySharedArray.schema.toText(System.out::print);
        System.out.println(" ==================");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        F32Array data = F32Array.create(accelerator, 32);
        accelerator.compute(computeContext -> {
            LocalArray.myCompute(computeContext, data);
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
        if (isCorrect) {
            System.out.println("Correct result");
        } else {
            System.out.println("Wrong result");
        }
    }

}