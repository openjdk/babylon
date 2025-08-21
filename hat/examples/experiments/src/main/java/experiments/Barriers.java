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
import hat.buffer.S32Array;
import hat.buffer.S32LocalArray;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;
import java.util.stream.IntStream;

/**
 * How to test?
 * <code>
 *     HAT=SHOW_CODE java -cp job.jar hat.java exp ffi-opencl Barriers
 *     HAT=SHOW_CODE java -cp job.jar hat.java exp ffi-cuda Barriers
 * </code>
 */
public class Barriers {

    private static boolean PRINT_RESULTS = true;

    /**
     * Example of a simple reduction using accelerator's global memory. This is inefficient, but it shows
     * the constructs needed to support this case, such as accessing to local ids, sizes and blocks.
     *
     * @param context
     * @param input
     * @param partialSums
     */
    @CodeReflection
    private static void reduce(@RO KernelContext context, @RW S32Array input, @RW S32Array partialSums) {
        int localId = context.lix;
        int localSize = context.lsx;
        int blockId = context.bix;
        int baseIndex = localSize * blockId + localId;

        for (int offset = localSize / 2; offset > 0; offset /= 2) {
            if (localId < offset) {
                int val = input.array(baseIndex);
                val += input.array((baseIndex + offset));
                input.array(baseIndex, val);
            }
            context.barrier();
        }
        if (localId == 0) {
            // copy from shared memory to global memory
            partialSums.array(blockId,  input.array(baseIndex));
        }
    }

    /**
     * Example of a simple reduction using accelerator's global memory. This is inefficient, but it shows
     * the constructs needed to support this case, such as accessing to local ids, sizes and blocks.
     *
     * @param context
     * @param input
     * @param partialSums
     */
    @CodeReflection
    private static void reduceLocal(@RO KernelContext context, @RW S32Array input, @RW S32Array partialSums) {
        int localId = context.lix;
        int localSize = context.lsx;
        int blockId = context.bix;

        // Prototype: allocate in shared memory an array of 16 ints
        //int[] sharedArray = context.createLocalIntArray(16);
        S32Array sharedArray = context.createLocalIntArray(16);
        // another approach is to return a new type S32LocalArray
        //S32LocalArray sharedArray = context.createLocalArray(INT, 16);

        // Copy from global to shared memory
        //sharedArray[localId] = input.array(baseIndex);
        sharedArray.array(localId, input.array(context.gix));

        for (int offset = localSize / 2; offset > 0; offset /= 2) {
            context.barrier();
            if (localId < offset) {
                sharedArray.array(localId,  sharedArray.array(localId) +  sharedArray.array(localId + offset));
            }
        }
        if (localId == 0) {
            // copy from shared memory to global memory
            partialSums.array(blockId,  sharedArray.array(0));
        }
    }

    private static final int BLOCK_SIZE = 16;

    @CodeReflection
    private static void mySimpleCompute(@RO ComputeContext cc,  @RW S32Array input, @RW S32Array partialSums) {
        // 2 groups of 16 threads each
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(32), new LocalMesh1D(16));
        cc.dispatchKernel(computeRange, kc -> reduceLocal(kc, input, partialSums));
    }

    public static void main(String[] args) {
        System.out.println("Experiment: local IDs and local groups");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 32;
        S32Array input = S32Array.create(accelerator, size);
        S32Array inputOrig = S32Array.create(accelerator, size);
        S32Array partialSums = S32Array.create(accelerator, 2);

        for (int i = 0; i < size; i++) {
            input.array(i, i);
            inputOrig.array(i, i);
        }
        partialSums.fill(_ -> 0);

        // Compute on the accelerator
        accelerator.compute( cc -> Barriers.mySimpleCompute(cc, input, partialSums));

        int[] results = new int[2]; // 2 groups
        int sum = 0;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += inputOrig.array(i);
        }
        results[0] = sum;
        sum = 0;
        for (int i = BLOCK_SIZE; i < inputOrig.length(); i++) {
            sum += inputOrig.array(i);
        }
        results[1] = sum;

        if (PRINT_RESULTS) {
            System.out.println("Result Locals: ");
            System.out.println("SEQ: " + results[0] + " vs " + partialSums.array(0));
            System.out.println("SEQ: " + results[1] + " vs " + partialSums.array(1));
        }
    }

}
