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

import view.f32.F32Matrix4x4;
import view.f32.F32Triangle3D;
import view.f32.F32Vec3;
import view.f32.ModelHighWaterMark;
import view.f32.ZPos;

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

public class ViewFrame extends JFrame {
    final boolean old;
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

    F32Vec3.F32Vec3Impl cameraVec3Old;
   // F32Vec3.vec3 lookDirVec3Old;
    F32Matrix4x4 projF32Mat4x4Old;
   // F32Vec3.vec3 centerVec3Old;
    F32Vec3.F32Vec3Impl moveAwayVec3Old;

    ModelHighWaterMark markOld;

     F32.Vec3 cameraVec3New;
    // F32.Vec3 lookDirVec3New;
     F32.Mat4x4 projF32Mat4x4New;
    // F32.Vec3 centerVec3New;
     F32.Vec3 moveAwayVec3New;

    F32.ModelHighWaterMark markNew;

    private ViewFrame(String name, boolean old, Renderer renderer, Runnable sceneBuilder) {
        super(name);
        this.old = old;
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
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent _windowEvent) {
                System.exit(0);
            }
        });
        sceneBuilder.run();
        float farZ = 1000f;
        float nearZ = 0.1f;
        float fieldOfViewDegrees = 60f;
        float originZ = 0f;
        float originY = 0f;
        float originX = 0f;
        float moveAwayZ=30f;
       // float halfWidth = renderer.width()/2f;
        float halfHeight = renderer.height()/2f;
        float quarterHeight = renderer.height()/4f;//??
        if (old) {
            cameraVec3Old = F32Vec3.F32Vec3Impl.of(originX, originY, originZ);
           // lookDirVec3Old = F32Vec3.vec3.of(originX, originY, originZ);
            var projF32Mat4x4_1 = F32Matrix4x4.projection(renderer.width(),renderer.height(), nearZ,farZ, fieldOfViewDegrees);
            var projF32Mat4x4_2 = F32Matrix4x4.mulMat4(projF32Mat4x4_1, F32Matrix4x4.scale(quarterHeight));
            projF32Mat4x4Old = F32Matrix4x4.mulMat4(projF32Mat4x4_2, F32Matrix4x4.transformation(halfHeight));
            //   centerVec3Old = F32Vec3.vec3.of(halfWidth, halfHeight, originZ);
            moveAwayVec3Old = F32Vec3.F32Vec3Impl.of(originX, originY, moveAwayZ);
            markOld = new ModelHighWaterMark();// mark all buffers.  transforms create new points so this allows us to garbage colect
        }else{
            cameraVec3New = F32.Vec3.of(originX, originY, originZ);
           // lookDirVec3New = F32.Vec3.of(originX, originY, originZ);
            var projF32Mat4x4_1 = F32.Mat4x4.Projection.of(renderer.width(),renderer.height(), nearZ, farZ, fieldOfViewDegrees);
            var projF32Mat4x4_2 = F32.Mat4x4.mul(projF32Mat4x4_1, F32.Mat4x4.Scale.of(quarterHeight));
            projF32Mat4x4New = F32.Mat4x4.mul(projF32Mat4x4_2, F32.Mat4x4.Transformation.of(halfHeight));
          //  centerVec3New = F32.Vec3.of(halfWidth, halfHeight, originZ);
            moveAwayVec3New = F32.Vec3.of(originX, originY, moveAwayZ);
            markNew = new F32.ModelHighWaterMark(); // mark all buffers.  transforms create new points so this allows us to garbage colect
        }
    }

    public static ViewFrame of(String name, boolean old, Renderer renderer, Runnable sceneBuilder) {
        return new ViewFrame(name, old, renderer, sceneBuilder);
    }


    void update() {
        final long elapsedMillis = System.currentTimeMillis() - startMillis;
        float theta = elapsedMillis * thetaDelta;
        if ((frames++ % 50) == 0) {
            System.out.println("Frames " + frames + " Theta = " + theta + " FPS = " + ((frames * 1000) / elapsedMillis));
        }

        // Loop through the triangles
        boolean showHidden = renderer.displayMode() == Renderer.DisplayMode.WIRE_SHOW_HIDDEN;

        if (old) {
            markOld.resetAll();
            var xyzRot4x4 =new F32Matrix4x4.Rotation(theta * 2, theta / 2, theta);
            ModelHighWaterMark resetMark = new ModelHighWaterMark();
            List<ZPos> zpos = new ArrayList<>();
            for (F32Triangle3D.F32Triangle3DImpl t : F32Triangle3D.F32Triangle3DImpl.all()) {
                if (t.rgb()==0){
                    throw new RuntimeException("ti.rgb == 0");
                }
                // here we rotate and then move into the Z plane.
                t = t.mul(xyzRot4x4).add(moveAwayVec3Old);
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

                    F32Vec3.F32Vec3Impl cameraDeltaVec3 = t.center().sub(cameraVec3Old); // clearly our default camera is 0,0,0

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

                    t = t.mul(projF32Mat4x4Old);//  projection matrix also scales to screen and translate half a screen

                    zpos.add(new ZPos(t, howVisible));
                }
                resetMark.reset3D(); // do not move this up.
            }
            Collections.sort(zpos);
            for (ZPos z : zpos) {
                z.create();
            }
        }else{
            List<F32.ZPos> zpos = new ArrayList<>();
            var xyzRot4x4 = F32.Mat4x4.Rotation.of(theta * 2, theta / 2, theta);
            int end = F32.TriangleVec3.arr.size();
            for (int i = 0; i < end; i++) {
                var t = F32.TriangleVec3.arr.get(i);
                // here we rotate and then move into the Z plane.
                t = F32.mul(t, xyzRot4x4);
                t = F32.add(t, moveAwayVec3New);
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

                    F32.Vec3 cameraDeltaVec3 = sub(F32.center(t), cameraVec3New); // clearly our default camera is 0,0,0

                    //  howVisible = cameraDeltaVec3.mul( t.normalSumOfSquares()).sumOf();
                    howVisible = dotprod(cameraDeltaVec3, normal(t));
                    // howVisible is a 'scalar'
                    // it's magnitude indicating how much it is 'facing away from' the camera.
                    // it's sign indicates if the camera can indeed see the location.
                    isVisible = howVisible < 0.0;
                }

                if (isVisible) {
                    // Projected triangle is still in unit 1 space!!
                    // now project the 3d triangle to 2d plane.
                    // Scale up to quarter screen height then add half height of screen
                    t = F32.mul(t, projF32Mat4x4New);//  projection matrix also scales to screen and translate half a screen
                    zpos.add(new F32.ZPos(t, howVisible));
                }
                // resetMark.reset3D(); // do not move this up.
            }


            Collections.sort(zpos);
            List<F32.TriangleVec2> ztri = new ArrayList<>();
            for (F32.ZPos z : zpos) {
                ztri.add(z.create());
            }

        }

        renderer.render(old);

        viewer.repaint();
    }
}
