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

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import static hat.ifacemapper.MappableIface.*;

import hat.buffer.Buffer;
import hat.ifacemapper.Schema;
import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;
import jdk.incubator.code.CodeReflection;
import wrap.Scalar;
import wrap.Sequence;
import wrap.clwrap.CLPlatform;
import wrap.clwrap.CLWrapComputeContext;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static wrap.LayoutBuilder.structOf;

public class Main {
    final static int ZeroBase = 0;

    public final static byte ALIVE = (byte) 0xff;
    public final static byte DEAD = 0x00;

    public interface CellGrid extends Buffer {
        int width();

        int height();

        byte cell(long idx);

        void cell(long idx, byte b);

        Schema<CellGrid> schema = Schema.of(CellGrid.class, lifeData -> lifeData
                .arrayLen("width", "height").stride(2).array("cell")
        );

        static CellGrid create(Accelerator accelerator, int width, int height) {
            return schema.allocate(accelerator, width, height);
        }

        ValueLayout valueLayout = JAVA_BYTE;
        long headerOffset = JAVA_INT.byteOffset() * 2;

        default CellGrid copySliceTo(byte[] bytes, int to) {
            long offset = headerOffset + to * valueLayout.byteOffset();
            MemorySegment.copy(Buffer.getMemorySegment(this), valueLayout, offset, bytes, 0, width() * height());
            return this;
        }
    }

    public interface Control extends Buffer {
        int from();

        void from(int from);

        int to();

        void to(int to);

        Schema<Control> schema = Schema.of(Control.class, lifeSupport -> lifeSupport.fields("from", "to"));

        static Control create(Accelerator accelerator, CellGrid CLWrapCellGrid) {
            var instance = schema.allocate(accelerator);
            instance.to(CLWrapCellGrid.width() * CLWrapCellGrid.height());
            instance.from(0);
            return instance;
        }
    }

    public static class Compute {
        @CodeReflection
        public static int val(@RO CellGrid grid, int from, int w, int x, int y) {
            return grid.cell( ((long) y * w)  + x +from)&1;
        }

        @CodeReflection
        public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
            if (kc.x < kc.maxX) {
                int w = cellGrid.width();
                int h = cellGrid.height();
                int from = control.from();
                int to = control.to();
                int x = kc.x % w;
                int y = kc.x / w;
                byte cell = cellGrid.cell(kc.x + from);
                if (x>0 && x<(w-1) && y>0 && y<(h-1)) { // passports please
                    int count =
                            val(cellGrid,from,w,x-1,y-1)
                                    +val(cellGrid,from,w,x-1,y+0)
                                    +val(cellGrid,from,w,x-1,y+1)
                                    +val(cellGrid,from,w,x+0,y-1)
                                    +val(cellGrid,from,w,x+0,y+1)
                                    +val(cellGrid,from,w,x+1,y+0)
                                    +val(cellGrid,from,w,x+1,y-1)
                                    +val(cellGrid,from,w,x+1,y+1);
                    cell =  ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
                }
                cellGrid.cell(kc.x + to, cell);
            }
        }


        @CodeReflection
        static public void compute(final ComputeContext cc, Viewer viewer, Control ctrl, CellGrid grid) {
            //  while (viewer.isVisible()) {
            cc.dispatchKernel(
                    grid.width() * grid.height(),
                    kc -> Compute.life(kc, ctrl, grid)
            );
            int to = ctrl.from(); ctrl.from(ctrl.to()); ctrl.to(to); //swap from/to


              //  if (start==0L) {
                 //   start = System.currentTimeMillis();
              //  }else {
               //     this.controls.generation.setText(String.format("%8d", ++generationCounter));
                 //   this.controls.generationsPerSecond.setText(
                   //         String.format("%5.2f", (generationCounter * 1000f) / (System.currentTimeMillis() - start))
                   // );
                    viewer.mainPanel.repaint();
              //  }

           // if (viewer.isReadyForUpdate()) {
           //     viewer.update(grid, to);
          //  }
            //   }
        }
    }
    public static class CLWrapCellGrid {
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

        CLWrapCellGrid(Arena arena, int w, int h) {
            this.w = w;
            this.h = h;
            this.wxh = w * h;
            this.layout = structOf("CLWrapCellGrid", $ -> $
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

        CLWrapCellGrid copySliceTo(byte[] bytes, int to) {
            MemorySegment.copy(segment, JAVA_BYTE,
                    JAVA_INT.byteSize() + JAVA_INT.byteSize() + to * JAVA_BYTE.byteSize(),
                    bytes, 0, wxh);
            return this;
        }

        public int wxh() {
            return wxh;
        }
    }

    public static class CLWrapControl {
        final MemorySegment segment;
        final MemoryLayout layout;
        final Scalar from;
        final Scalar to;
        final Scalar generation;


        CLWrapControl(Arena arena, CLWrapCellGrid CLWrapCellGrid) {
            this.layout = structOf("CLWrapControl", $ -> $
                    .i32("from")
                    .i32("to")
                    .i64("generation")
            );
            this.segment = arena.allocate(this.layout);
            this.from = Scalar.of(this.segment, this.layout, "from", CLWrapCellGrid.width() * CLWrapCellGrid.height());
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

    public static int val(CLWrapCellGrid grid, int from, int w, int x, int y) {
        return grid.cell((y * w) + x + from) & 1;
    }

    public static void life(int kcx, CLWrapControl CLWrapControl, CLWrapCellGrid CLWrapCellGrid) {

        int w = CLWrapCellGrid.width();
        int h = CLWrapCellGrid.height();
        int from = CLWrapControl.from();
        int to = CLWrapControl.to();
        int x = kcx % w;
        int y = kcx / w;
        byte cell = CLWrapCellGrid.cell(kcx + from);
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
        CLWrapCellGrid.cell(kcx + to, cell);
        //  }
    }


    public static void main(String[] args) {
        Arena arena = Arena.global();
        PatternData patternData = RleParser.readPatternData(
                Main.class.getClassLoader().getResourceAsStream("orig.rle")
        );
        // We oversize the grid by adding 1 to n,e,w and s
        CLWrapCellGrid CLWrapCellGrid = new CLWrapCellGrid(
                Arena.global(),
                patternData.getMetaData().getWidth() + 2,
                patternData.getMetaData().getHeight() + 2
        );

        // By shifting all cells +1,+1 so we only need to scan 1..width-1, 1..height-1
        // we don't worry about possibly finding cells in 0,n width,n or n,0 height,n
        patternData.getLiveCells().getCoordinates().stream().forEach(c -> {
                    CLWrapCellGrid.cell((1 + c.getX()) + (1 + c.getY()) * CLWrapCellGrid.width(), ALIVE);
                    //  CLWrapCellGrid.cell(CLWrapCellGrid.wxh + (1 + c.getX()) + (1 + c.getY()) * CLWrapCellGrid.width(), ALIVE);
                }
        );

        CLWrapControl CLWrapControl = new CLWrapControl(arena, CLWrapCellGrid);
        Viewer viewer = new Viewer("Life", CLWrapCellGrid);

        CLWrapComputeContext CLWrapComputeContext = new CLWrapComputeContext(arena, 20);

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
                
                 inline int val(__global cellGrid_t *CLWrapCellGrid, int from, int w, int x, int y) {
                     return CLWrapCellGrid->cellArray[((y * w) + x + from)] & 1;
                 }
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
        var program = context.buildProgram(code);
        CLPlatform.CLDevice.CLContext.CLProgram.CLKernel kernel = program.getKernel("life");
        CLWrapComputeContext.MemorySegmentState cellGridState = CLWrapComputeContext.register(CLWrapCellGrid.segment);
        CLWrapComputeContext.MemorySegmentState controlState = CLWrapComputeContext.register(CLWrapControl.segment);


        CLWrapCellGrid.copySliceTo(viewer.mainPanel.rasterData, CLWrapControl.to());
        CLWrapControl.swap();
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
                kernel.run(CLWrapComputeContext, CLWrapCellGrid.wxh, cellGridState, controlState);
            } else {
                IntStream.range(0, CLWrapCellGrid.wxh()).parallel().forEach(kcx ->
                        life(kcx, CLWrapControl, CLWrapCellGrid)
                );
            }
            CLWrapControl.generation(generationCounter);
            CLWrapControl.swap();
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
                CLWrapCellGrid.copySliceTo(viewer.mainPanel.rasterData, CLWrapControl.from());
                viewer.mainPanel.state = Viewer.MainPanel.State.Scheduled;
                viewer.mainPanel.repaint();
                lastFrame = now;
                framesSinceLastChange++;
            }
        }
    }
}
