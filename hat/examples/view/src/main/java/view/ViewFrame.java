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

import view.f32.F32Mat4;
import view.f32.F32Mat4x4;
import view.f32.tri;
import view.f32.vec3;
import view.f32.F32Triangle2D;
import view.f32.F32Vec2;

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

public class ViewFrame extends JFrame {
    private final Rasterizer rasterizer;
    private volatile Point point = null;
    private final Object doorBell;
    private final JComponent viewer;
    final long startMillis;
    long frames;
    final vec3 cameraVec3;
    final vec3 lookDirVec3;
    final F32Mat4x4 projF32Mat4x4;
    final vec3 centerVec3;
    final vec3 moveAwayVec3;

    ModelHighWaterMark mark;

    private ViewFrame(String name, Rasterizer rasterizer, Runnable sceneBuilder) {
        super(name);
        startMillis = System.currentTimeMillis();
        this.rasterizer = rasterizer;
        this.doorBell = new Object();

        this.viewer = new JComponent() {
            @Override
            public void paintComponent(Graphics g) {
                rasterizer.view.paint((Graphics2D) g);
            }
        };
        viewer.setPreferredSize(new Dimension(rasterizer.view.image.getWidth(), rasterizer.view.image.getHeight()));
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

        cameraVec3 = vec3.of(0f, 0f, .0f);
        lookDirVec3 = vec3.of(0f, 0f, 0f);
        var projF32Mat4x4_1 = F32Mat4.Projection.of(rasterizer.view.image, 0.1f, 1000f, 60f);
        var projF32Mat4x4_2 = F32Mat4x4.mul(projF32Mat4x4_1,F32Mat4.Scale.of(rasterizer.view.image.getHeight() / 4f));
        projF32Mat4x4 = F32Mat4x4.mul(projF32Mat4x4_2, F32Mat4.Transformation.of(rasterizer.view.image.getHeight() / 2f));
        centerVec3 = vec3.of(rasterizer.view.image.getWidth() / 2f,  rasterizer.view.image.getHeight() / 2f, 0);
        moveAwayVec3 = vec3.of(0f, 0f, 30f);
        mark = new ModelHighWaterMark(); // mark all buffers.  transforms create new points so this allows us to garbage colect
    }

    public static ViewFrame of(String name, Rasterizer rasterizer, Runnable sceneBuilder){
        return new ViewFrame(name,rasterizer,sceneBuilder);
    }

    Point waitForPoint(long timeout) {
        while (point == null) {
            synchronized (doorBell) {
                try {
                    if (timeout > 0) {
                        doorBell.wait(timeout);
                    }
                    update();
                } catch (final InterruptedException ie) {
                    ie.getStackTrace();
                }
            }
        }
        Point returnPoint = point;
        point = null;
        return returnPoint;
    }

    void ringDoorBell(Point point) {
        this.point = point;
        synchronized (doorBell) {
            doorBell.notify();
        }
    }

    void update() {
        final long elapsedMillis = System.currentTimeMillis() - startMillis;
        float theta = elapsedMillis * Physics.thetaDelta;

        if ((frames++ % 50) == 0) {
            System.out.println("Frames " + frames + " Theta = " + theta + " FPS = " + ((frames * 1000) / elapsedMillis) + " Vertices " + rasterizer.vec2EntriesCount);
        }

        mark.resetAll();

        F32Mat4x4 xyzRot4x4 = new F32Mat4.Rotation(theta * 2, theta / 2, theta);

        ModelHighWaterMark resetMark = new ModelHighWaterMark();

        List<ZPos> zpos = new ArrayList<>();
        // Loop through the triangles
        boolean showHidden = rasterizer.displayMode == Rasterizer.DisplayMode.WIRE_SHOW_HIDDEN;

        for (tri t : tri.all()) {
            // here we rotate and then move into the Z plane.
            t = t.mul(xyzRot4x4).add(moveAwayVec3);
            float howVisible = 1f;
            boolean isVisible = showHidden;

            if (!showHidden) {
                // here we determine whether the camera can see the plane that the translated triangle is on.
                // so we need the normal to the triangle in the coordinate system

                // Now we work out where the camera is relative to a line projected from the plane to the camera
                // if camera is at 0,0,0 clearly this is a no-op

                // We need a point on the triangle it looks like assume we can use any, I choose the center of the triangle
                // intuition suggests the one with the minimal Z is best no?

                // We subtract the camera from our point on the triangle so we can compare

                vec3 cameraDeltaVec3 = t.center().sub(cameraVec3); // clearly our default camera is 0,0,0

                //  howVisible = cameraDeltaVec3.mul( t.normalSumOfSquares()).sumOf();
                howVisible = cameraDeltaVec3.dotProd(t.normal());
                // howVisible is a 'scalar'
                // it's magnitude indicating how much it is 'facing away from' the camera.
                // it's sign indicates if the camera can indeed see the location.
                isVisible = howVisible < 0.0;
            }

            if (isVisible) {
                // Projected triangle is still in unit 1 space!!
                // now project the 3d triangle to 2d plane.
                // Scale up to quarter screen height then add half height of screen

                t = t.mul(projF32Mat4x4);//  projection matrix also scales to screen and translate half a screen

                zpos.add(new ZPos(t, howVisible));
            }

            resetMark.reset3D(); // do not move this up.
        }


        Collections.sort(zpos);

        for (ZPos z : zpos) {
            z.create();
        }

        rasterizer.triangle2DEntries = F32Triangle2D.entries;
        rasterizer.triangle2DEntriesCount = F32Triangle2D.count;
        rasterizer.vec2Entries = F32Vec2.entries;
        rasterizer.vec2EntriesCount = F32Vec2.count;
        rasterizer.colors = F32Triangle2D.colors;
        rasterizer.execute(rasterizer.range);
        rasterizer.view.update();
        viewer.repaint();
    }
}
