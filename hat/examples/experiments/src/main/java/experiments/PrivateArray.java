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
import hat.Space;
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
public class PrivateArray {

    /**
     * Bound iFaceMapping
     */
    private interface MyPrivateArray extends Buffer {

        void array(long index, float value);
        float array(long index);

        Schema<MyPrivateArray> schema = Schema.of(MyPrivateArray.class,
                myPrivateArray -> myPrivateArray
                        .array("array", 256));

        static MyPrivateArray create(Accelerator accelerator) {
            return schema.allocate(accelerator, 1);
        }

        static <T extends MyPrivateArray> T create(Space space) {
            return Buffer.create(space);
        }

        static String schemaDescription() {
            StringBuilder sb = new StringBuilder();
            schema.toText(sb::append);
            return sb.toString();
        }
    }

    @CodeReflection
    private static void compute(@RO KernelContext kernelContext, @RW F32Array data) {

        // Initial Approach: this approach requires instance objects at compilation time, which is not possible
        // MyPrivateArray myPrivateArray = kernelContext.allocateSchema(Space.PRIVATE, MyPrivateArray.schema, 1);

        // We can use this factory to create a new private local array
        //MyPrivateArray myPrivateArray = Buffer.createPrivate(MyPrivateArray.class, 18);

        // Alternatively, the user can create a new static method with the same signature
        MyPrivateArray myPrivateArray = MyPrivateArray.create(Space.PRIVATE);

        myPrivateArray.array(0, kernelContext.gix);
        data.array(kernelContext.gix, myPrivateArray.array(0));
    }

    @CodeReflection
    private static void myCompute(@RO ComputeContext computeContext, @RW F32Array data) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(32));
        computeContext.dispatchKernel(computeRange,
                kernelContext -> compute(kernelContext, data)
        );
    }

    static void main(String[] args) {
        System.out.println("Testing Private DT Mapping");

        System.out.println("Schema description");
        MyPrivateArray.schema.toText(System.out::print);

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        F32Array data = F32Array.create(accelerator, 32);
        accelerator.compute(computeContext -> {
            PrivateArray.myCompute(computeContext, data);
        });

        // Check result
        boolean isCorrect = true;
        for (int i = 0; i < data.length(); i++) {
            if (data.array(i) != i) {
                isCorrect = false;
                break;
            }
        }
        if (isCorrect) {
            System.out.println("Correct result");
        } else {
            System.out.println("Wrong result");
        }
    }

}