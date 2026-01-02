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
import hat.KernelContext;
import hat.NDRange;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.Schema;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;

public class TestWriteOnly {

    public interface CellGrid extends Buffer {
        int width();
        int height();
        byte array(long idx);
        void array(long idx, byte b);

        Schema<CellGrid> schema = Schema.of(CellGrid.class, lifeData -> lifeData
                .arrayLen("width", "height").stride(2).array("array")
        );

        static CellGrid create(Accelerator accelerator, int width, int height) {
            return schema.allocate(accelerator, width, height);
        }

        default byte[][] arrayView() {
            return null;
        }
    }

    @Reflect
    public static void life(@RO KernelContext kc, @RO CellGrid cellGrid, @MappableIface.WO CellGrid cellGridRes) {
    }

    @Reflect
    static public void compute(final @RO ComputeContext cc, @RO CellGrid grid, @MappableIface.WO CellGrid gridRes) {
        int range = grid.width() * grid.height();
        cc.dispatchKernel(NDRange.of1D(range), kc -> life(kc, grid, gridRes));
    }

    @HatTest
    @Reflect
    public static void testWriteOnly() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());

        int size = 1028;
        CellGrid cellGridRes = CellGrid.create(accelerator, size, size);
        CellGrid cellGrid = CellGrid.create(accelerator, size, size);

        accelerator.compute( cc -> compute(cc, cellGrid, cellGridRes));

        // System.out.println("cellGrid width, height are " + cellGrid.width() + ", " + cellGrid.height());
        // System.out.println("cellGridRes width, height are " + cellGridRes.width() + ", " + cellGridRes.height());

        HATAsserts.assertEquals(cellGridRes.width(), size, 0.01f);
        HATAsserts.assertEquals(cellGridRes.height(), size, 0.01f);
    }

}
