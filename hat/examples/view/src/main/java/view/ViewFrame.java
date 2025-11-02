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

package view;

import javax.swing.JComponent;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static view.F32.dotprod;
import static view.F32.normal;
import static view.F32.sub;

public abstract class ViewFrame extends JFrame {
    final Renderer renderer;
    private volatile Point point = null;
    private final Object doorBell;
    final JComponent viewer;
    final long startMillis;
    long frames;
    void ringDoorBell(Point point) {
        this.point = point;
        synchronized (doorBell) {
            doorBell.notify();
        }
    }
    ViewFrame(String name, Renderer renderer, Runnable sceneBuilder) {
        super(name);
        startMillis = System.currentTimeMillis();
        this.renderer = renderer;
        this.doorBell = new Object();

        this.viewer = new JComponent() {
            @Override
            public void paintComponent(Graphics g) {
                renderer.view().paint((Graphics2D) g);
            }
        };
        viewer.setPreferredSize(new Dimension(renderer.view().image.getWidth(), renderer.view().image.getHeight()));
        viewer.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                ringDoorBell(e.getPoint());

            }
        });
        getContentPane().add(viewer);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent _windowEvent) {
                System.exit(0);
            }
        });

        sceneBuilder.run();
          }


    static final float thetaDelta = 0.0002f;

    abstract void  update();

}
