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

import view.f32.F32;
import view.f32.F32x4x4;
import view.f32.F32x3Triangle;
import view.f32.F32x3;
import view.f32.ModelHighWaterMark;
import view.f32.ZPos;
import view.f32.pool.F32PoolBased;
import view.f32.pool.Pool;

import javax.swing.JComponent;
import javax.swing.JFrame;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ViewFrame extends JFrame {

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

    static final float thetaDelta = 0.0002f;

    F32x3 cameraVec3;
    F32x4x4 projF32Mat4x4;
    F32x3 moveAwayVec3;

    ModelHighWaterMark mark;
    public F32 f32;



    private ViewFrame(F32 f32, String name, Renderer renderer, Runnable sceneBuilder) {
        super(name);
        this.f32 = f32;
        startMillis = System.currentTimeMillis();
        this.renderer = renderer;
        this.doorBell = new Object();

        this.viewer = new JComponent() {
            @Override
            public void paintComponent(Graphics g) {
                renderer.paint((Graphics2D) g);
            }
        };
        this.viewer.setPreferredSize(new Dimension(renderer.width(), renderer.height()));
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
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        sceneBuilder.run();
        float farZ = 1000f;
        float nearZ = 0.1f;
        float fieldOfViewDegrees = 60f;
        float originZ = 0f;
        float originY = 0f;
        float originX = 0f;
        float moveAwayZ = 30f;
        float halfHeight = renderer.height() / 2f;
        float quarterHeight = renderer.height() / 4f;

        cameraVec3 = f32.f32x3(originX, originY, originZ);
        var projF32Mat4x4_1 = f32.projection(renderer.width(), renderer.height(), nearZ, farZ, fieldOfViewDegrees);
        var projF32Mat4x4_2 = f32.mul(projF32Mat4x4_1, f32.scale(quarterHeight));
        projF32Mat4x4 = f32.mul(projF32Mat4x4_2, f32.transformation(halfHeight));
        moveAwayVec3 = f32.f32x3(originX, originY, moveAwayZ);
        mark = ModelHighWaterMark.of((F32PoolBased) f32);// mark all buffers.  transforms create new points so this allows us to garbage colect
    }

    public static ViewFrame of(F32 f32, String name, Renderer renderer, Runnable sceneBuilder) {
        return new ViewFrame(f32, name, renderer, sceneBuilder);
    }


    void update() {
        final long elapsedMillis = System.currentTimeMillis() - startMillis;
        float theta = elapsedMillis * thetaDelta;
        if ((frames++ % 50) == 0) {
            System.out.println("Frames " + frames + " Theta = " + theta + " FPS = " + ((frames * 1000) / elapsedMillis));
        }


       // boolean showHidden = ;
        mark.resetAll();
        var xyzRot4x4 = f32.rot(theta * 2, theta / 2, theta);
        ModelHighWaterMark resetMark = ModelHighWaterMark.of((F32PoolBased) f32);
        List<ZPos> zpos = new ArrayList<>();
        // Loop through the triangles

        for (int i = 0; i < ((Pool<?,?>) f32.f32x3TriangleFactory()).count(); i++) {
            var f32x3Triangle = (F32x3Triangle) ((Pool<?,?>) f32.f32x3TriangleFactory()).entry(i);

            // here we rotate and then move into the Z plane.
            f32x3Triangle = f32.add(f32.mul(f32x3Triangle, xyzRot4x4), moveAwayVec3);
            float howVisible = 1f;
            boolean isVisible = renderer.displayMode().equals(Renderer.DisplayMode.WIRE_SHOW_HIDDEN);
            if (!isVisible) {
                // here we determine whether the camera can see the plane that the translated triangle is on.
                // so we need the normal to the triangle in the coordinate system

                // Now we work out where the camera is relative to a line projected from the plane to the camera
                // if camera is at 0,0,0 clearly this is a no-op

                // We need a point on the triangle it looks like assume we can use any, I choose the center of the triangle
                // intuition suggests the one with the minimal Z is best no?

                // We subtract the camera from our point on the triangle so we can compare
                F32x3 cameraDeltaVec3 = f32.sub(f32.centre(f32x3Triangle), cameraVec3);// clearly our default camera is 0,0,0


                //  howVisible = cameraDeltaVec3.mul( t.normalSumOfSquares()).sumOf();

                howVisible = f32.dotProd(cameraDeltaVec3, f32.normal(f32x3Triangle));
                // howVisible is a 'scalar'
                // it's magnitude indicating how much it is 'facing away from' the camera.
                // it's sign indicates if the camera can indeed see the location.
                isVisible = howVisible < 0.0;
            }

            if (isVisible) {
                // Projected triangle is still in unit 1 space!!
                // now project the 3d triangle to 2d plane.
                // Scale up to quarter screen height then add half height of screen

                f32x3Triangle = f32.mul(f32x3Triangle, projF32Mat4x4);//  projection matrix also scales to screen and translate half a screen

                zpos.add(new ZPos(f32,f32x3Triangle, howVisible));
            }
            resetMark.reset3D(); // do not move this up.
        }

        Collections.sort(zpos);
        for (ZPos z : zpos) {
            z.create();
        }
        renderer.render();
        viewer.repaint();
    }
}
