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
package life;

import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;
import wrap.Scalar;
import wrap.Sequence;
import wrap.clwrap.CLPlatform;
import wrap.clwrap.ComputeContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static wrap.LayoutBuilder.structOf;

public class Main {
    final static int ZeroBase = 0;

    public static class CellGrid {
        /*
         * struct CellGrid{
         *     int width;
         *     int height;
         *     byte[width*height*2] cellArray;
         *  }
         */
        final MemoryLayout layout;
        final MemorySegment segment;
        final Scalar width;
        final Scalar height;
        final Sequence cellArray;


        final private int w;
        final private int h;
        final private int wxh;

        CellGrid(Arena arena, int w, int h) {
            this.w = w;
            this.h = h;
            this.wxh = w * h;
            this.layout = structOf("cellGrid", $ -> $
                    .i32("width")
                    .i32("height")
                    .i8Seq("cellArray", (long) wxh * 2)
            );
            this.segment = arena.allocate(layout);
            this.width = Scalar.of(segment, layout, "width", this.w);
            this.height = Scalar.of(segment, layout, "height", this.h);
            this.cellArray = Sequence.of(segment, layout, "cellArray");
        }

        int width() {
            return w;//width.i32();
        }

        int height() {
            return h;//height.i32();
        }

        byte cell(int idx) {
            return cellArray.i8(idx);
        }

        void cell(int idx, byte v) {
            cellArray.set(idx, v);//
        }

        CellGrid copySliceTo(byte[] bytes, int to) {
            MemorySegment.copy(segment, JAVA_BYTE,
                    JAVA_INT.byteSize() + JAVA_INT.byteSize() + to * JAVA_BYTE.byteSize(),
                    bytes, 0, wxh);
            return this;
        }

        public int wxh() {
            return wxh;
        }
    }

    public static class Control {
        final MemorySegment segment;
        final MemoryLayout layout;
        final Scalar from;
        final Scalar to;
        final Scalar generation;


        Control(Arena arena, CellGrid cellGrid) {
            this.layout = structOf("control", $ -> $
                    .i32("from")
                    .i32("to")
                    .i64("generation")
            );
            this.segment = arena.allocate(this.layout);
            this.from = Scalar.of(this.segment, this.layout, "from", cellGrid.width() * cellGrid.height());
            this.to = Scalar.of(this.segment, this.layout, "to", 0);
            this.generation = Scalar.of(this.segment, this.layout, "generation", 0);

        }

        int from() {
            return this.from.i32();
        }

        int to() {
            return this.to.i32();
        }

        void generation(long generation) {
            this.generation.set(generation);
        }

        void swap() {
            int from = from();
            int to = to();
            this.to.set(from);
            this.from.set(to);
        }

    }

    public final static byte ALIVE = (byte) 0xff;
    public final static byte DEAD = 0x00;

    public static int val(CellGrid grid, int from, int w, int x, int y) {
        return grid.cell((y * w) + x + from) & 1;
    }

    public static void life(int kcx, Control control, CellGrid cellGrid) {

        int w = cellGrid.width();
        int h = cellGrid.height();
        int from = control.from();
        int to = control.to();
        int x = kcx % w;
        int y = kcx / w;
        byte cell = cellGrid.cell(kcx + from);
        if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
            int count =
                    val(cellGrid, from, w, x - 1, y - 1)
                            + val(cellGrid, from, w, x - 1, y + 0)
                            + val(cellGrid, from, w, x - 1, y + 1)
                            + val(cellGrid, from, w, x + 0, y - 1)
                            + val(cellGrid, from, w, x + 0, y + 1)
                            + val(cellGrid, from, w, x + 1, y + 0)
                            + val(cellGrid, from, w, x + 1, y - 1)
                            + val(cellGrid, from, w, x + 1, y + 1);
            cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
        }
        cellGrid.cell(kcx + to, cell);
        //  }
    }


    public static void main(String[] args) {
        Arena arena = Arena.global();
        PatternData patternData = RleParser.readPatternData(
                Main.class.getClassLoader().getResourceAsStream("orig.rle")
        );
        // We oversize the grid by adding 1 to n,e,w and s
        CellGrid cellGrid = new CellGrid(
                Arena.global(),
                patternData.getMetaData().getWidth() + 2,
                patternData.getMetaData().getHeight() + 2
        );

        // By shifting all cells +1,+1 so we only need to scan 1..width-1, 1..height-1
        // we don't worry about possibly finding cells in 0,n width,n or n,0 height,n
        patternData.getLiveCells().getCoordinates().stream().forEach(c -> {
                    cellGrid.cell((1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), ALIVE);
                    //  cellGrid.cell(cellGrid.wxh + (1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), ALIVE);
                }
        );

        Control control = new Control(arena, cellGrid);
        Viewer viewer = new Viewer("Life", cellGrid);

        ComputeContext computeContext = new ComputeContext(arena, 20);

        List<CLPlatform> platforms = CLPlatform.platforms(arena);
        System.out.println("platforms " + platforms.size());
        CLPlatform platform = platforms.get(0);
        platform.devices.forEach(device -> {
            System.out.println("   Compute Units     " + device.computeUnits());
            System.out.println("   Device Name       " + device.deviceName());
            System.out.println("   Device Vendor       " + device.deviceVendor());
            System.out.println("   Built In Kernels  " + device.builtInKernels());
        });
        CLPlatform.CLDevice device = platform.devices.get(0);
        System.out.println("   Compute Units     " + device.computeUnits());
        System.out.println("   Device Name       " + device.deviceName());
        System.out.println("   Device Vendor       " + device.deviceVendor());

        System.out.println("   Built In Kernels  " + device.builtInKernels());
        CLPlatform.CLDevice.CLContext context = device.createContext();

        var code = """
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
                
                 inline int val(__global cellGrid_t *cellGrid, int from, int w, int x, int y) {
                     return cellGrid->cellArray[((y * w) + x + from)] & 1;
                 }
                 __kernel void life( __global  cellGrid_t *cellGrid ,__global control_t *control ){
                      int kcx = get_global_id(0);
                      int w = cellGrid->width;
                      int h = cellGrid->height;
                      int from = control->from;
                      int to = control->to;
                      int x = kcx % w;
                      int y = kcx / w;
                      signed char cell = cellGrid->cellArray[kcx + from];
                      if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                          int count =
                                 val(cellGrid, from, w, x - 1, y - 1)
                                 + val(cellGrid, from, w, x - 1, y + 0)
                                 + val(cellGrid, from, w, x - 1, y + 1)
                                 + val(cellGrid, from, w, x + 0, y - 1)
                                 + val(cellGrid, from, w, x + 0, y + 1)
                                 + val(cellGrid, from, w, x + 1, y + 0)
                                 + val(cellGrid, from, w, x + 1, y - 1)
                                 + val(cellGrid, from, w, x + 1, y + 1);
                          cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
                      }
                      cellGrid->cellArray[kcx + to]=  cell;
                   }
                """;
        var program = context.buildProgram(code);
        CLPlatform.CLDevice.CLContext.CLProgram.CLKernel kernel = program.getKernel("life");
        ComputeContext.MemorySegmentState cellGridState = computeContext.register(cellGrid.segment);
        ComputeContext.MemorySegmentState controlState = computeContext.register(control.segment);


        cellGrid.copySliceTo(viewer.mainPanel.rasterData, control.to());
        control.swap();
        viewer.mainPanel.repaint();
        viewer.waitForStart();

        long start = System.currentTimeMillis();
        long generationCounter = 0;

        long requiredFrameRate = 10;
        long generations = 1000000;
        long generationsSinceLastChange = 0;
        long framesSinceLastChange = 0;

        long msPerFrame = 1000 / requiredFrameRate;
        long lastFrame = start;
        controlState.copyToDevice = true;
        controlState.copyFromDevice = true;
        cellGridState.copyToDevice = true;
        viewer.mainPanel.state = Viewer.MainPanel.State.Done;
        while (generationCounter < generations) {
            boolean alwaysCopy = viewer.controls.alwaysCopy();
            long now = System.currentTimeMillis();
            boolean displayThisGeneration =
                    viewer.mainPanel.state.equals(Viewer.MainPanel.State.Done)
                            && (now - lastFrame >= msPerFrame);

            if (viewer.controls.useGPU()) {
                cellGridState.copyToDevice = alwaysCopy || generationCounter == 0; // only first
                cellGridState.copyFromDevice = alwaysCopy || displayThisGeneration;
                kernel.run(computeContext, cellGrid.wxh, cellGridState, controlState);
            } else {
                IntStream.range(0, cellGrid.wxh()).parallel().forEach(kcx ->
                        life(kcx, control, cellGrid)
                );
            }
            control.generation(generationCounter);
            control.swap();
            ++generationCounter;
            ++generationsSinceLastChange;
            if (displayThisGeneration) {
                if (viewer.controls.updated) {
                    // When the user changes something we have to update FPS
                    generationsSinceLastChange = 0;
                    framesSinceLastChange = 0;
                    viewer.controls.updated = false;
                }
                viewer.controls.updateGenerationCounter(generationsSinceLastChange, framesSinceLastChange, msPerFrame);
                cellGrid.copySliceTo(viewer.mainPanel.rasterData, control.from());
                viewer.mainPanel.state = Viewer.MainPanel.State.Scheduled;
                viewer.mainPanel.repaint();
                lastFrame = now;
                framesSinceLastChange++;
            }
        }
    }
}
