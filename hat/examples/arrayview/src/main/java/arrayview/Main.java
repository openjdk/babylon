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
package arrayview;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.Buffer;
import hat.ifacemapper.BufferState;
import hat.ifacemapper.Schema;
import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;
import jdk.incubator.code.CodeReflection;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.stream.IntStream;

import static hat.backend.Backend.FIRST;
import static hat.ifacemapper.MappableIface.RO;
import static hat.ifacemapper.MappableIface.RW;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public class Main {

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
        long headerOffset = JAVA_INT.byteOffset() * 2;

        default void copySliceTo(byte[] bytes, int to) {
            long offset = headerOffset + to * valueLayout.byteOffset();
            MemorySegment.copy(Buffer.getMemorySegment(this), valueLayout, offset, bytes, 0, width() * height());

        }

        default int wxh() {
            return width() * height();
        }

        default byte[] arrayView() {
            byte[] arr = new byte[]{};
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
        public static final String codeHeader = """
                #define ALIVE -1
                #define DEAD 0
                 typedef struct control_s{
                     int from;
                     int to;
                     long generation;
                 }control_t;

                 typedef struct cellGrid_s{
                     int width;
                     int height;
                     signed char cellArray[0];
                 }cellGrid_t;

                """;

        final static String codeVal = """
                 inline int val(__global cellGrid_t *CLWrapCellGrid, int from, int w, int x, int y) {
                     return CLWrapCellGrid->cellArray[((y * w) + x + from)] & 1;
                 }
                """;


        @CodeReflection
        public static int val(@RO CellGrid grid, int from, int w, int x, int y) {
            return grid.array(((long) y * w) + x + from) & 1;
        }

        final static String codeLifePerIdx = """
                 __kernel void life( __global  cellGrid_t *CLWrapCellGrid ,__global control_t *CLWrapControl ){
                    int kcx = get_global_id(0);
                    int w = CLWrapCellGrid->width;
                    int h = CLWrapCellGrid->height;
                    int from = CLWrapControl->from;
                    int to = CLWrapControl->to;
                    int x = kcx % w;
                    int y = kcx / w;
                    signed char cell = CLWrapCellGrid->cellArray[kcx + from];
                    if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                        int count =
                                val(CLWrapCellGrid, from, w, x - 1, y - 1)
                                        + val(CLWrapCellGrid, from, w, x - 1, y + 0)
                                        + val(CLWrapCellGrid, from, w, x - 1, y + 1)
                                        + val(CLWrapCellGrid, from, w, x + 0, y - 1)
                                        + val(CLWrapCellGrid, from, w, x + 0, y + 1)
                                        + val(CLWrapCellGrid, from, w, x + 1, y + 0)
                                        + val(CLWrapCellGrid, from, w, x + 1, y - 1)
                                        + val(CLWrapCellGrid, from, w, x + 1, y + 1);
                        cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
                    }
                    CLWrapCellGrid->cellArray[kcx + to]=  cell;
                }
                """;

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
            byte cell = bytes[idx + from];
            if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                int count =
                        (bytes[(y - 1) * w + (x - 1) + from] & 1)
                                + (bytes[(y + 0) * w + (x - 1) + from] & 1)
                                + (bytes[(y + 1) * w + (x - 1) + from] & 1)
                                + (bytes[(y - 1) * w + (x + 0) + from] & 1)
                                + (bytes[(y + 1) * w + (x + 0) + from] & 1)
                                + (bytes[(y - 1) * w + (x + 1) + from] & 1)
                                + (bytes[(y + 0) * w + (x + 1) + from] & 1)
                                + (bytes[(y + 1) * w + (x + 1) + from] & 1);
                cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
            }
            bytes[idx + to] = cell;
        }


        @CodeReflection
        public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
            if (kc.x < kc.maxX) {
                Compute.lifePerIdx(kc.x, control, cellGrid);
            }
        }




        @CodeReflection
        static public void compute(final @RO ComputeContext cc,
                                   Viewer viewer, @RO Control ctrl, @RW CellGrid grid) {
            viewer.state.timeOfLastChange = System.currentTimeMillis();
            int range = grid.width() * grid.height();
            while (viewer.stillRunning()) {
                cc.dispatchKernel(range, kc -> Compute.life(kc, ctrl, grid));

                int to = ctrl.from(); ctrl.from(ctrl.to()); ctrl.to(to);

                long now = System.currentTimeMillis();
                if (viewer.isReadyForUpdate(now)){
                    viewer.update(now,grid,to);
                }
            }
        }
    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());//,new OpenCLBackend("INFO,MINIMIZE_COPIES,SHOW_COMPUTE_MODEL"));

        Arena arena = Arena.global();
        PatternData patternData = RleParser.readPatternData(
                Main.class.getClassLoader().getResourceAsStream("orig.rle")
        );

        // We oversize the grid by adding 1 to n,e,w and s
        CellGrid cellGrid = CellGrid.create(accelerator,
                patternData.getMetaData().getWidth() + 2,
                patternData.getMetaData().getHeight() + 2);

        // By shifting all cells +1,+1 so we only need to scan 1..width-1, 1..height-1
        // we don't worry about possibly finding cells in 0,n width,n or n,0 height,n
        patternData.getLiveCells().getCoordinates().stream().forEach(c ->
                cellGrid.array((1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), ALIVE)
        );

        Control control = Control.create(accelerator, cellGrid);

        Viewer.State state = new Viewer.State();
        Viewer viewer = new Viewer("Life", cellGrid, state);

        var tempFrom = control.from();
        control.from(control.to());
        control.to(tempFrom);

        viewer.mainPanel.repaint();
        viewer.waitForStart();
         accelerator.compute(cc -> Compute.compute(cc, viewer, control, cellGrid));

    }
}
