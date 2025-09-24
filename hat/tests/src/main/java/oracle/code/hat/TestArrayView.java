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
import hat.buffer.Buffer;
import hat.buffer.S32Array;
import hat.buffer.S32Array2D;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public class TestArrayView {

    @CodeReflection
    public static void squareKernel(@RO  KernelContext kc, @RW S32Array s32Array) {
        if (kc.x<kc.maxX){
            int[] arr = s32Array.arrayView();
            arr[kc.x] *= arr[kc.x];
        }
    }

    @CodeReflection
    public static void square(@RO ComputeContext cc, @RW S32Array s32Array) {
        cc.dispatchKernel(s32Array.length(),
                kc -> squareKernel(kc, s32Array)
        );
    }

    @HatTest
    public static void testSquare() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var arr = S32Array.create(accelerator, 32);
        for (int i = 0; i < arr.length(); i++) {
            arr.array(i, i);
        }
        accelerator.compute(
                cc -> square(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.length(); i++) {
            HatAsserts.assertEquals(arr.array(i), i * i);
        }
    }

    @CodeReflection
    public static void square2DKernel(@RO  KernelContext kc, @RW S32Array2D s32Array2D) {
        if (kc.x<kc.maxX){
            int[][] arr = s32Array2D.arrayView();
            arr[kc.x][kc.y] *= arr[kc.x][kc.y];
        }
    }

    @CodeReflection
    public static void square2D(@RO ComputeContext cc, @RW S32Array2D s32Array2D) {
        cc.dispatchKernel(s32Array2D.width() * s32Array2D.height(),
                kc -> square2DKernel(kc, s32Array2D)
        );
    }

    @HatTest
    public static void testSquare2D() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var arr = S32Array2D.create(accelerator, 5, 5);
        for (int i = 0; i < arr.height(); i++) {
            for (int j = 0; j < arr.width(); j++) {
                arr.set(i, j, i * 5 + j);
            }
        }
        accelerator.compute(
                cc -> square2D(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.height(); i++) {
            for (int j = 0; j < arr.width(); j++) {
                HatAsserts.assertEquals(arr.get(i, j), (i * 5 + j) * (i * 5 + j));
            }
        }
    }


    public final static byte ALIVE = (byte) 0xff;
    public final static byte DEAD = 0x00;

    public interface CellGrid extends Buffer {
        /*
         * struct CellGrid{
         *     int width;
         *     int height;
         *     byte[width*height*2] cellArray;
         *  }
         */
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

        ValueLayout valueLayout = JAVA_BYTE;

        default byte[] arrayView() {
            int size = this.width() * this.height();
            byte[] arr = new byte[size];
            for (int i = 0; i < size; i++) {
                arr[i] = this.array(i);
            }
            return arr;
        }
    }

    public interface Control extends Buffer {
        /*
         * struct Control{
         *     int from;
         *     int to;
         *  }
         */
        int from();

        void from(int from);

        int to();

        void to(int to);

        Schema<Control> schema = Schema.of(
                Control.class, control ->
                        control.fields("from", "to"));//, "generation", "requiredFrameRate", "maxGenerations"));

        static Control create(Accelerator accelerator, CellGrid cellGrid) {
            var instance = schema.allocate(accelerator);
            instance.from(cellGrid.width() * cellGrid.height());
            instance.to(0);
            return instance;
        }
    }

    public static class Compute {
        @CodeReflection
        public static void lifePerIdx(int idx, @RO Control control, @RW CellGrid cellGrid) {
            int w = cellGrid.width();
            int h = cellGrid.height();
            int from = control.from();
            int to = control.to();
            int x = idx % w;
            int y = idx / w;

            // byte[] bytes = cellGrid.arrayView();
            // byte cell = bytes[idx + from];
            // byte[] lookup = new byte[]{};
            // if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
            //     int lookupIdx =
            //             (bytes[(y - 1) * w + x - 1 + from]&1 <<0)
            //                     |(bytes[(y + 0) * w + x - 1 + from]&1 <<1)
            //                     |(bytes[(y + 1) * w + x - 1 + from]&1 <<2)
            //                     |(bytes[(y - 1) * w + x + 0 + from]&1 <<3)
            //                     |(bytes[(y - 0) * w + x + 0 + from]&1 <<4) // current cell added
            //                     |(bytes[(y + 1) * w + x + 0 + from]&1 <<5)
            //                     |(bytes[(y + 0) * w + x + 1 + from]&1 <<6)
            //                     |(bytes[(y - 1) * w + x + 1 + from]&1 <<7)
            //                     |(bytes[(y + 1) * w + x + 1 + from]&1 <<8) ;
            //     // conditional removed!
            //     bytes[idx + to] = lookup[lookupIdx];
            // }

            byte[] bytes = cellGrid.arrayView();
            byte cell = bytes[idx];
            if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                int count =
                        (bytes[(y - 1) * w + (x - 1)] & 1)
                                + (bytes[(y + 0) * w + (x - 1)] & 1)
                                + (bytes[(y + 1) * w + (x - 1)] & 1)
                                + (bytes[(y - 1) * w + (x + 0)] & 1)
                                + (bytes[(y + 1) * w + (x + 0)] & 1)
                                + (bytes[(y - 1) * w + (x + 1)] & 1)
                                + (bytes[(y + 0) * w + (x + 1)] & 1)
                                + (bytes[(y + 1) * w + (x + 1)] & 1);
                cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
            }
            bytes[idx] = cell;
        }

        @CodeReflection
        public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
            if (kc.x < kc.maxX) {
                Compute.lifePerIdx(kc.x, control, cellGrid);
            }
        }

        @CodeReflection
        static public void compute(final @RO ComputeContext cc, @RO Control ctrl, @RW CellGrid grid) {
            int range = grid.width() * grid.height();
            cc.dispatchKernel(range, kc -> Compute.life(kc, ctrl, grid));
        }
    }

    @HatTest
    public static void testLife() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());//,new OpenCLBackend("INFO,MINIMIZE_COPIES,SHOW_COMPUTE_MODEL"));

        // We oversize the grid by adding 1 to n,e,w and s
        CellGrid cellGrid = CellGrid.create(accelerator,
                17,
                17);

        byte[][] actualGrid = new byte[][]{
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
        };

        // By shifting all cells +1,+1 so we only need to scan 1..width-1, 1..height-1
        // we don't worry about possibly finding cells in 0,n width,n or n,0 height,n
        for (int i = 0; i < cellGrid.height(); i++) {
            for (int j = 0; j < cellGrid.width(); j++) {
                cellGrid.array(((long) i * cellGrid.width()) + j, actualGrid[i][j]);
            }
        }

        Control control = Control.create(accelerator, cellGrid);

        accelerator.compute(cc -> Compute.compute(cc, control, cellGrid));

        byte[][] resultGrid = new byte[][]{
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD},
                {DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD},
                {DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
        };

        for (int i = 0; i < cellGrid.height(); i++) {
            for (int j = 0; j < cellGrid.width(); j++) {
                HatAsserts.assertEquals(cellGrid.array(((long) i * cellGrid.width()) + j), resultGrid[i][j]);
            }
        }
    }
}
