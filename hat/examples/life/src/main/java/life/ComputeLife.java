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

import hat.ComputeContext;
import hat.NDRange;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import java.util.concurrent.locks.LockSupport;

import static hat.KernelContext.GIX;
import static hat.KernelContext.GSX;

public class ComputeLife {
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
             int val(__global cellGrid_t *CLWrapCellGrid, int from, int w, int x, int y) {
                 return CLWrapCellGrid->cellArray[((y * w) + x + from)] & 1;
             }
            """;


    @Reflect
    public static int val(CellGrid grid, int from, int w, int x, int y) {
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

    @Reflect
    public static void lifePerIdx(int idx, Control control, CellGrid cellGrid) {
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
            cell = ((count == 3) || ((count == 2) && (cell == CellGrid.ALIVE))) ? CellGrid.ALIVE : CellGrid.DEAD;// B3/S23.
        }
        cellGrid.cell(idx + to, cell);
    }

    @Reflect
    public static void life(Control control, CellGrid cellGrid) {
        if (GIX() < GSX()) {
            ComputeLife.lifePerIdx(GIX(), control, cellGrid);
        }
    }

    // Keep the host slice swap in each compute entrypoint: buffer tracking is
    // currently injected there, not into nested compute methods.
    @Reflect
    public static void step(final @MappableIface.RO ComputeContext cc, @MappableIface.RW Control ctrl, @MappableIface.RO CellGrid grid) {
        cc.dispatchKernel(NDRange.of1D(grid.width() * grid.height()), () -> ComputeLife.life(ctrl, grid));
        int from = ctrl.from();
        ctrl.from(ctrl.to());
        ctrl.to(from);
    }

    @Reflect
    public static void steps(final @MappableIface.RO ComputeContext cc, @MappableIface.RW Control ctrl, @MappableIface.RO CellGrid grid,
                             int generations) {
        for (int generation = 0; generation < generations; generation++) {
            cc.dispatchKernel(NDRange.of1D(grid.width() * grid.height()),
                    () -> ComputeLife.life(ctrl, grid));
            int from = ctrl.from();
            ctrl.from(ctrl.to());
            ctrl.to(from);
        }
    }

    @Reflect
    public static void compute(final @MappableIface.RO ComputeContext cc,
                               Viewer viewer, @MappableIface.RW Control ctrl, @MappableIface.RO CellGrid grid, int frameDelayMillis) {
        viewer.state.timeOfLastChange = System.currentTimeMillis();
        while (viewer.stillRunning()) {
            cc.dispatchKernel(NDRange.of1D(grid.width() * grid.height()), () -> ComputeLife.life(ctrl, grid));
            int from = ctrl.from();
            ctrl.from(ctrl.to());
            ctrl.to(from);
            // Note we can't call step above, because HAT does not check calls from compute to see if thay mutate buffers.
            if (frameDelayMillis > 0) {
                LockSupport.parkNanos(frameDelayMillis * 1_000_000L);
            }

            long now = System.currentTimeMillis();
            if (viewer.isReadyForUpdate(now)) {
                viewer.update(now, grid, ctrl.from());
            }
        }
    }
}
