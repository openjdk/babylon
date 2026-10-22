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
import hat.Accelerator.Compute;
import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;

public class Main {
    static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());


        PatternData patternData = RleParser.readPatternData(
                Main.class.getClassLoader().getResourceAsStream("orig.rle")
        );

        // We oversize the grid by adding 1 to n,e,w and s.
        CellGrid cellGrid = CellGrid.create(accelerator,
                patternData.getMetaData().getWidth() + 2,
                patternData.getMetaData().getHeight() + 2);

        // Shift cells by one so the outer border can stay empty.
        patternData.getLiveCells().getCoordinates().forEach(c ->
                cellGrid.cell((1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), CellGrid.ALIVE)
        );

        Control control = Control.create(accelerator, cellGrid);
        Viewer.State state = new Viewer.State();
        Viewer viewer = new Viewer("Life", cellGrid, state);
        var tempFrom = control.from();
        control.from(control.to());
        control.to(tempFrom);
        viewer.mainPanel.repaint();
        if (!Arrays.asList(args).contains("--auto-start")) {
            viewer.waitForStart();
        }
        int frameDelayMillis=0;
        accelerator.compute((@Reflect Compute)
                cc -> ComputeLife.compute(cc, viewer, control, cellGrid, frameDelayMillis));

    }
}
