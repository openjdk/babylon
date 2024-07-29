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
/*
 * Based on code from HealingBrush renderscript example
 *
 * https://github.com/yongjhih/HealingBrush/tree/master
 *
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package heal;

import hat.Accelerator;

import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.NoninvertibleTransformException;

public class Viewer extends Display {
    volatile Selection selection = null;
    volatile  Point bestMatchOffset = null;
    public final Accelerator accelerator;
    public Viewer(ImageData imageData, Accelerator accelerator) {
        super(imageData);
        this.accelerator = accelerator;
        addMouseListener(new MouseAdapter() {
           @Override
            public void mouseReleased(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    bestMatchOffset = SearchCompute.getOffsetOfBestMatch(accelerator, imageData, selection.close());
                    HealCompute.heal(accelerator,imageData, selection, bestMatchOffset);
                    Timer t = new Timer(1000, new ActionListener() {
                        @Override
                        public void actionPerformed(ActionEvent e) {
                            selection = null;
                            bestMatchOffset = null;
                            repaint();
                        }
                    });
                    t.setRepeats(false);
                    t.start();
                    repaint();
                }
            }

            @Override
            public void mousePressed(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    try {
                        var ptDst =  transform.inverseTransform(e.getPoint(), null);
                        selection = new Selection(ptDst);
                    } catch (NoninvertibleTransformException e1) {
                        e1.printStackTrace();
                    }
                }
            }

        });
        addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    try {
                        var ptDst = transform.inverseTransform(e.getPoint(), null);
                        selection.add(ptDst);
                        repaint();
                    } catch (NoninvertibleTransformException e1) {
                        // TODO Auto-generated catch block
                        e1.printStackTrace();
                    }
                }
            }
        });
    }

    protected void paintInScale(Graphics2D g) {
        if (selection != null) {
            Polygon selectionPolygon = new Polygon();
            Polygon solutionPolygon = new Polygon();
            selection.pointList.forEach(point -> {
                selectionPolygon.addPoint(point.x, point.y);
                if (bestMatchOffset != null){
                    solutionPolygon.addPoint(point.x+bestMatchOffset.x, point.y+bestMatchOffset.y);
                }
            });
            g.setColor(Color.RED);
            g.drawPolygon(selectionPolygon);
            if (bestMatchOffset!=null){
                g.setColor(Color.BLUE);
                g.drawPolygon(solutionPolygon);
            }
        }
    }
}
