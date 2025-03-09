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
import hat.backend.ffi.OpenCLBackend;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;
import hat.ifacemapper.SegmentMapper;
import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;
import jdk.incubator.code.CodeReflection;
import wrap.clwrap.CLPlatform;
import wrap.clwrap.CLWrapComputeContext;

import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Polygon;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.stream.IntStream;

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

        default void copySliceTo(byte[] bytes, int to) {
            long offset = headerOffset + to * valueLayout.byteOffset();
            MemorySegment.copy(Buffer.getMemorySegment(this), valueLayout, offset, bytes, 0, width() * height());

        }

        default int wxh() {
            return width() * height();
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
            return grid.cell(((long) y * w) + x + from) & 1;
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
            byte cell = cellGrid.cell(idx + from);
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
            cellGrid.cell(idx + to, cell);
        }


        @CodeReflection
        public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
            if (kc.x < kc.maxX) {
                Compute.lifePerIdx(kc.x, control, cellGrid);
            }
        }


        @CodeReflection
        static public void compute(final @RO ComputeContext cc,
                                   Viewer viewer, @RO Control ctrl, @RW CellGrid cellGrid) {
            viewer.state.framesSinceLastChange = 0;
            while (viewer.state.generation < viewer.state.maxGenerations) {
                cc.dispatchKernel(cellGrid.width() * cellGrid.height(), kc -> Compute.life(kc, ctrl, cellGrid));
                int to = ctrl.from();
                ctrl.from(ctrl.to());
                ctrl.to(to);
                long now = System.currentTimeMillis();
                if (viewer.needsUpdating(now)) {
                    viewer.state.lastFrame = now;
                    viewer.controls.updateGenerationCounter();
                    cellGrid.copySliceTo(viewer.mainPanel.rasterData, ctrl.from());
                    viewer.state.redrawState = Viewer.State.RedrawState.RepaintRequested;
                    viewer.mainPanel.repaint();
                }
                viewer.state.framesSinceLastChange++;
                viewer.state.generation++;
            }
        }
    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), new OpenCLBackend("GPU,MINIMIZE_COPIES"));

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
                cellGrid.cell((1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), ALIVE)
        );

        Control control = Control.create(accelerator, cellGrid);
        CLWrapComputeContext clWrapComputeContext = new CLWrapComputeContext(arena, 20);
        List<CLPlatform> platforms = CLPlatform.platforms(arena);
        CLPlatform platform = platforms.get(0);
        CLPlatform.CLDevice device = platform.devices.get(0);
        CLPlatform.CLDevice.CLContext context = device.createContext();

        var program = context.buildProgram(Compute.codeHeader + Compute.codeVal + Compute.codeLifePerIdx);
        CLPlatform.CLDevice.CLContext.CLProgram.CLKernel kernel = program.getKernel("life");
        boolean useHat = false;
        Viewer viewer = new Viewer("Life", cellGrid, useHat);
        var tempFrom = control.from();
        control.from(control.to());
        control.to(tempFrom);
        viewer.mainPanel.repaint();
        viewer.waitForStart();
        if (useHat) {
            accelerator.compute(cc -> Compute.compute(cc, viewer, control, cellGrid));
        } else {

            while (viewer.state.generation < viewer.state.maxGenerations) {
                boolean alwaysCopy = !viewer.state.minimizingCopies;
                long now = System.currentTimeMillis();
                boolean displayThisGeneration = viewer.needsUpdating(now);

                if (viewer.state.usingGPU) {
                    SegmentMapper.BufferState bufferState = SegmentMapper.BufferState.of(cellGrid);
                    bufferState.setHostDirty(alwaysCopy || (viewer.state.generation == 0)); // only first
                    bufferState.setDeviceDirty(alwaysCopy || displayThisGeneration);
                    kernel.run(clWrapComputeContext, cellGrid.wxh(), cellGrid, control);
                } else {
                    IntStream.range(0, cellGrid.wxh()).parallel().forEach(kcx ->
                            Compute.lifePerIdx(kcx, control, cellGrid)
                    );
                }
                tempFrom = control.from();
                control.from(control.to());
                control.to(tempFrom);
                viewer.state.generation++;
                ++viewer.state.generationsSinceLastChange;
                if (displayThisGeneration) {
                    if (viewer.state.updated) {
                        // When the user changes something we have to update FPS
                        viewer.state.generationsSinceLastChange = 0;
                        viewer.state.framesSinceLastChange = 0;
                        viewer.state.updated = false;
                    }
                    viewer.controls.updateGenerationCounter();
                    cellGrid.copySliceTo(viewer.mainPanel.rasterData, control.from());
                    viewer.state.redrawState = Viewer.State.RedrawState.RepaintRequested;
                    viewer.mainPanel.repaint();
                    viewer.state.lastFrame = now;
                    viewer.state.framesSinceLastChange++;
                }
            }
        }
    }
}
