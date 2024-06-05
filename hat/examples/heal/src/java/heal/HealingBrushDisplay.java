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

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class HealingBrushDisplay extends Display {
    boolean orig = false;
    Path selectionPath = null;
    Path matchPath = null;
    BufferedImage img;

    public HealingBrushDisplay() {
        addMouseListener(new MouseAdapter() {
            Point2D ptDst = new Point2D.Double();

            @Override
            public void mouseReleased(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    Selection selection = new Selection(new ImageData(img), selectionPath.close());
                    Point p = HealingBrush.getBestMatch(selection);
                    matchPath = new Path();
                    HealingBrush.heal(selection, p.x, p.y);
                    repaint();
                    Timer t = new Timer(1000, new ActionListener() {
                        @Override
                        public void actionPerformed(ActionEvent e) {
                            selectionPath = null;
                            matchPath = null;
                            repaint();
                        }
                    });
                    t.setRepeats(false);
                    t.start();
                }
            }

            @Override
            public void mousePressed(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    try {
                        orig = true;
                        repaint();
                        transform.inverseTransform(e.getPoint(), ptDst);
                        selectionPath = new Path((int) ptDst.getX(), (int) ptDst.getY());
                    } catch (NoninvertibleTransformException e1) {
                        // TODO Auto-generated catch block
                        e1.printStackTrace();
                    }
                }
            }

        });
        addMouseMotionListener(new MouseMotionAdapter() {
            Point2D ptDst = new Point2D.Double();

            public void mouseDragged(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    try {
                        transform.inverseTransform(e.getPoint(), ptDst);
                        selectionPath.extendTo((int) ptDst.getX(), (int) ptDst.getY());
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
        if (selectionPath != null) {
            g.setColor(Color.RED);
            g.drawPolygon(selectionPath.getPolygon());
            if (matchPath != null) {
                g.setColor(Color.BLUE);
                for (int i = 0; i < matchPath.length(); i++) {
                    XYList.XY p = (XYList.XY) matchPath.xy(i);
                    g.drawRect(p.x(), p.y(), 4, 4);
                }
            }
        }
    }

    public void setImage(BufferedImage img) {
        orig = true;
        super.setImage(img);
        this.img = img;
        orig = false;
        repaint();
    }

    public static void main(String[] args) throws IOException {
        JFrame f = new JFrame("Main");
        BufferedImage originalImage = ImageIO.read(HealingBrushDisplay.class.getResourceAsStream("/images/bolton.png"));
        BufferedImage image=null;
        if (originalImage.getType() == BufferedImage.TYPE_INT_RGB){
            image = originalImage;
        }else {
            // there must be a better way!
            image = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_RGB);
            image.getGraphics().drawImage(originalImage, 0, 0, null);
        }
        f.setBounds(new Rectangle(image.getWidth(), image.getHeight()));
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        HealingBrushDisplay healingBrushDisplay = new HealingBrushDisplay();
        f.setContentPane(healingBrushDisplay);
        f.validate();
        f.setVisible(true);
        healingBrushDisplay.setImage(image);
    }
}
