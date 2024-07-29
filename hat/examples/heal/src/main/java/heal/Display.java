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


import hat.buffer.S32Array2D;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;

public abstract class Display extends JPanel {
    protected BufferedImage image;
    protected  int[] rasterData;
    protected S32Array2D s32Array2D;
    protected AffineTransform transform = new AffineTransform();
    protected float zoom = .95f; // set the zoom factor 1.0 = fit to screen

    protected float xOffset = 0; // 0 is centered -1 is to the left;
    protected float yOffset = 0; // 0 is centered -1 is to the top;

    Point mousePressedPosition;
    Point2D imageRelativeMouseDownPosition = new Point2D.Float();
    Point2D imageRelativeMovePosition = new Point2D.Float();

    public Display(BufferedImage image,S32Array2D s32Array2D) {
        this.image = image;
        this.rasterData = ((DataBufferInt) (image.getRaster().getDataBuffer())).getData();
        this.s32Array2D = s32Array2D;
        s32Array2D.copyFrom(rasterData);
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    mousePressedPosition = e.getPoint();
                    try {
                        imageRelativeMouseDownPosition= transform.inverseTransform(e.getPoint(), null);
                    } catch (NoninvertibleTransformException e1) {
                        e1.printStackTrace();
                    }
                }
            }
        });
        addMouseWheelListener(new MouseWheelListener() {
            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                zoom = zoom * (1 + e.getWheelRotation() / 10f);
                repaint();
            }

        });
        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    Point rightButonPoint = e.getPoint();
                    Dimension deltaFromInitialMousePress = new Dimension(rightButonPoint.x - mousePressedPosition.x, rightButonPoint.y - mousePressedPosition.y);
                    try {
                        imageRelativeMovePosition = transform.inverseTransform(e.getPoint(), null);
                        Dimension displaySize = getSize();
                        Dimension imageSize = new Dimension( s32Array2D.width(),s32Array2D.height());
                        float scale = zoom *
                                Math.min(displaySize.width / (float) imageSize.width,
                                        displaySize.height / (float) imageSize.height);
                        xOffset =  2 * (deltaFromInitialMousePress.width / (displaySize.width - scale * imageSize.width));
                        yOffset =  2 * (deltaFromInitialMousePress.height / (displaySize.height - scale * imageSize.height));
                        xOffset = Math.max(Math.min(xOffset, 1), -1);
                        yOffset = Math.max(Math.min(yOffset, 1), -1);
                        repaint();
                    } catch (NoninvertibleTransformException e1) {
                        e1.printStackTrace();
                    }
                }
            }
        });
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setBackground(Color.BLACK);
        g2d.fillRect(0, 0, getWidth(), getHeight());
        if (s32Array2D != null) {
            Dimension displaySize = getSize();
            Dimension imageSize = new Dimension( s32Array2D.width(),s32Array2D.height());
            AffineTransform safeTransform = g2d.getTransform();
            transform.setToIdentity();
            double scale = zoom *
                    Math.min(displaySize.width / (double) imageSize.width,
                            displaySize.height / (double) imageSize.height);
            transform.translate((1 + xOffset) * (displaySize.width - imageSize.width * scale) / 2,
                    (1 + yOffset) * (displaySize.height - imageSize.height * scale) / 2);
            transform.scale(scale, scale);
            g2d.transform(transform);
            s32Array2D.copyTo(rasterData);
            g.drawImage(image, 0, 0, imageSize.width, imageSize.height, null);
            paintInScale(g2d);
            g2d.setTransform(safeTransform);
        }
    }

    abstract protected void paintInScale(Graphics2D g) ;

}
