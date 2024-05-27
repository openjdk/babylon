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


import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;

public class Display extends JPanel implements MouseListener, MouseMotionListener, MouseWheelListener {
    protected BufferedImage image;
    protected AffineTransform transform = new AffineTransform();
    protected float zoom = .95f; // set the zoom factor 1.0 = fit to screen
    protected float xOffset = 0; // 0 is centered -1 is to the left;
    protected float yOffset = 0; // 0 is centered -1 is to the top;


    int mDownX;
    int mDownY;
    float mDownXOffset;
    float mDownYOffset;
    Point2D mImageDown = new Point2D.Float();
    Point2D mImageMove = new Point2D.Float();

    @Override
    public void mouseReleased(MouseEvent e) {
    }

    @Override
    public void mouseExited(MouseEvent e) {
    }

    @Override
    public void mouseEntered(MouseEvent e) {
    }

    @Override
    public void mouseClicked(MouseEvent e) {
    }

    @Override
    public void mouseMoved(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        if (SwingUtilities.isRightMouseButton(e)) {
            mDownX = e.getX();
            mDownY = e.getY();
            mDownXOffset = xOffset;
            mDownYOffset = yOffset;
            try {
                transform.inverseTransform(e.getPoint(), mImageDown);
            } catch (NoninvertibleTransformException e1) {
                e1.printStackTrace();
            }
        }
    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        zoom = zoom * (1 + e.getWheelRotation() / 10f);
        repaint();
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        if (SwingUtilities.isRightMouseButton(e)) {
            int dx = e.getX() - mDownX;
            int dy = e.getY() - mDownY;
            try {
                transform.inverseTransform(e.getPoint(), mImageMove);
                int sw = getWidth();
                int sh = getHeight();
                int iw = image.getWidth();
                int ih = image.getHeight();
                float scale = zoom * Math.min(sw / (float) iw, sh / (float) ih);
                xOffset = mDownXOffset + 2 * (dx / (sw - scale * iw));
                yOffset = mDownYOffset + 2 * (dy / (sh - scale * ih));

                xOffset = Math.max(Math.min(xOffset, 1), -1);
                yOffset = Math.max(Math.min(yOffset, 1), -1);
                repaint();
            } catch (NoninvertibleTransformException e1) {
                e1.printStackTrace();
            }
        }
    }


    public Display() {
        addMouseListener(this);
        addMouseWheelListener(this);
        addMouseMotionListener(this);
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setBackground(Color.BLACK);
        g2d.fillRect(0, 0, getWidth(), getHeight());
        if (image != null) {
            paintImage(g2d);
        }
    }

    public void paintImage(Graphics2D g) {
        int sw = getWidth();
        int sh = getHeight();
        int iw = image.getWidth();
        int ih = image.getHeight();
        AffineTransform tx = g.getTransform();
        transform.setToIdentity();
        double scale = zoom * Math.min(sw / (double) iw, sh / (double) ih);
        transform.translate((1 + xOffset) * (sw - iw * scale) / 2,
                (1 + yOffset) * (sh - ih * scale) / 2);
        transform.scale(scale, scale);
        g.transform(transform);
        g.drawImage(image, 0, 0, iw, ih, null);
        paintInScale(g);
        g.setTransform(tx);
    }

    protected void paintInScale(Graphics2D g) {

    }

    public void setImage(BufferedImage img) {
        this.image = img;
        repaint();
    }

}
