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
import java.awt.*;
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

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

public class HealingBrushDisplay extends Display {
    boolean orig = false;
    Path selectionPatch = null;
    Path matchPath = null;
    BufferedImage img;

    public HealingBrushDisplay() {
        addMouseListener(new MouseAdapter() {
            Point2D ptDst = new Point2D.Double();
            @Override
            public void mouseReleased(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    Selection selection = new Selection(new ImageData(img), selectionPatch.close());
                    Point p = HealingBrush.getBestMatch(selection);
                    matchPath = new Path(selectionPatch,  p.x,  p.y);
                    HealingBrush.heal(selection, p.x, p.y);
                    repaint();
                    Timer t = new Timer(1000, new ActionListener() {
                        @Override
                        public void actionPerformed(ActionEvent e) {
                            selectionPatch = null;
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
                        selectionPatch = new Path((int) ptDst.getX(), (int) ptDst.getY());
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
                        selectionPatch.extendTo((int) ptDst.getX(), (int) ptDst.getY());
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
        if (selectionPatch != null) {
            g.setColor(Color.RED);
            g.drawPolygon(selectionPatch.getPolygon());
            if (matchPath != null) {
                g.setColor(Color.BLUE);
                for (int i=0; i<matchPath.length();i++){
                XYList.XY p = (XYList.XY)matchPath.xy(i);
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

    private static BufferedImage getdata(InputStream is) throws IOException {
        BufferedImage img = ImageIO.read(is);
        BufferedImage ret = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        ret.getGraphics().drawImage(img, 0, 0, null);
        return ret;
    }


    public static void main(String[] args) throws IOException {
        JFrame f = new JFrame("Main");
        BufferedImage image = getdata(HealingBrushDisplay.class.getResourceAsStream("/images/bolton.png"));
        f.setBounds(new Rectangle(image.getWidth(),image.getHeight()));
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        HealingBrushDisplay p = new HealingBrushDisplay();
        f.setContentPane(p);
        f.validate();
        f.setVisible(true);
        p.setImage(image);
    }
}
