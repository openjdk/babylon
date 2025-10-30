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
import view.f32.Pool;
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

public class ViewFrameOld extends ViewFrame {

    final F32Vec3.vec3 cameraVec3;
    final F32Vec3.vec3 lookDirVec3;
    final F32Matrix4x4.Projection projF32Mat4x4;
    final F32Vec3.vec3 centerVec3;
    final F32Vec3.vec3 moveAwayVec3;

    ModelHighWaterMark mark;

    private ViewFrameOld(String name, Renderer renderer, Runnable sceneBuilder) {
        super(name,renderer,sceneBuilder);


        cameraVec3 = F32Vec3.vec3.of(0f, 0f, .0f);
        lookDirVec3 = F32Vec3.vec3.of(0f, 0f, 0f);
        F32Matrix4x4.Projection projF32Mat4x4_1 = F32Matrix4x4.Projection.of(renderer.view().image, 0.1f, 1000f, 60f);
        Pool.Idx projF32Mat4x4_2 = F32Matrix4x4.mulMat4(projF32Mat4x4_1.id(), F32Matrix4x4.Scale.of(renderer.view().image.getHeight() / 4f).id());
        projF32Mat4x4 = F32Matrix4x4.Projection.of(F32Matrix4x4.mulMat4(projF32Mat4x4_2, F32Matrix4x4.Transformation.of(renderer.view().image.getHeight() / 2f).id()));
        centerVec3 = F32Vec3.vec3.of(renderer.view().image.getWidth() / 2f,  renderer.view().image.getHeight() / 2f, 0);
        moveAwayVec3 = F32Vec3.vec3.of(0f, 0f, 30f);
        mark = new ModelHighWaterMark(); // mark all buffers.  transforms create new points so this allows us to garbage colect
    }

    public static ViewFrameOld of(String name, Renderer renderer, Runnable sceneBuilder){
        return new ViewFrameOld(name, renderer,sceneBuilder);
    }
@Override
    void update() {
        final long elapsedMillis = System.currentTimeMillis() - startMillis;
        float theta = elapsedMillis * thetaDelta;

        if ((frames++ % 50) == 0) {
            System.out.println("Frames " + frames + " Theta = " + theta + " FPS = " + ((frames * 1000) / elapsedMillis) + " Vertices " + F32Vec2.pool.count);
        }

        mark.resetAll();

        var xyzRot4x4 = new F32Matrix4x4.Rotation(theta * 2, theta / 2, theta);

        ModelHighWaterMark resetMark = new ModelHighWaterMark();

        List<ZPos> zpos = new ArrayList<>();
        // Loop through the triangles
        boolean showHidden = renderer.displayMode() == Renderer.DisplayMode.WIRE_SHOW_HIDDEN;

        for (F32Triangle3D.tri t : F32Triangle3D.tri.all()) {
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

                F32Vec3.vec3 cameraDeltaVec3 = t.center().sub(cameraVec3); // clearly our default camera is 0,0,0

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

        renderer.render();

        viewer.repaint();
    }
}
