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
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.WO;

import java.lang.invoke.MethodHandles;

import static hat.NDRange.Global1D;
import static hat.NDRange.Local1D;
import static hat.NDRange.NDRange1D;

public class TestGrids {

    @Reflect
    private static void compute(@RO KernelContext kernelContext, @WO F32Array output) {
        int idx = kernelContext.gix;
        int bsx = kernelContext.bsx;
        // Write the number of blocks
        output.array(idx, bsx);
    }

    @Reflect
    private static void myCompute(@RO ComputeContext computeContext, @WO F32Array output, int numThreads, int localBlockSize) {
        var ndRange = NDRange1D.of(Global1D.of(numThreads), Local1D.of(localBlockSize));
        computeContext.dispatchKernel(ndRange, kernelContext -> compute(kernelContext, output));
    }

    @HatTest
    public void testgrid_01() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int numThreads = 128;
        final int localBlockSize = 16;

        F32Array data = F32Array.create(accelerator, numThreads);

        accelerator.compute((@Reflect Compute) computeContext -> {
            TestGrids.myCompute(computeContext, data, numThreads, localBlockSize);
        });

        float expectedValue = (float) numThreads / localBlockSize;
        for (int i = 0; i < data.length(); i++) {
            HATAsserts.assertEquals(expectedValue, data.array(i), 0.001f);
        }
    }
}
