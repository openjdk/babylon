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
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;
import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public class Life {

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

        static Control create(Accelerator accelerator, CellGrid cellGrid) {
            var instance = schema.allocate(accelerator);
            instance.to(cellGrid.width() * cellGrid.height());
            instance.from(0);
            return instance;
        }
    }


    public final static byte ALIVE = (byte) 0xff;
    public final static byte DEAD = 0x00;

    public static class Compute {
        @CodeReflection
        public static int val(CellGrid grid, int from, int w, int x, int y) {
            return grid.cell( ((long) y * w)  + x +from)&1;
        }

        @CodeReflection
        public static void life(KernelContext kc, Control control, CellGrid cellGrid) {
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
        static public void compute(final ComputeContext computeContext, Viewer viewer, Control control, CellGrid cellGrid) {
            long start = System.currentTimeMillis();
            int generation = 0;
            while (true) {
                computeContext.dispatchKernel(
                        cellGrid.width() * cellGrid.height(),
                        kc -> Compute.life(kc, control, cellGrid)
                );
                int to = control.from();control.from(control.to());control.to(to); //swap from/to
                viewer.setGeneration(generation++, System.currentTimeMillis() - start);
             //   if (generation % 50 == 0) {
                    viewer.update(cellGrid, to);
              //  }
            }
        }
    }


    public static void main(String[] args) {
        boolean headless = Boolean.getBoolean("headless") || (args.length > 0 && args[0].equals("--headless"));

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), /*Backend.JAVA_MULTITHREADED);//*/Backend.FIRST);

        PatternData patternData = RleParser.readPatternData(
                Life.class.getClassLoader().getResourceAsStream("orig.rle")
        );
        CellGrid cellGrid = CellGrid.create(accelerator,
                  patternData.getMetaData().getWidth() + 2,
                patternData.getMetaData().getHeight() + 2

        );
        patternData.getLiveCells().getCoordinates().stream().forEach(c ->
                cellGrid.cell((1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), ALIVE)
        );

        Control control = Control.create(accelerator, cellGrid);
        final Viewer viewer = new Viewer("Life", control, cellGrid);
        viewer.update(cellGrid, 0);
        viewer.waitForStart();
        accelerator.compute(cc -> Compute.compute(cc, viewer, control, cellGrid));

    }
}
