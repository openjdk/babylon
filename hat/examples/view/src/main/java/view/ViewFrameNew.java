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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static view.F32.dotprod;
import static view.F32.normal;
import static view.F32.sub;

public class ViewFrameNew extends ViewFrame {
    final F32.Vec3 cameraVec3;
    final F32.Vec3 lookDirVec3;
    final F32.Mat4x4 projF32Mat4x4;
    final F32.Vec3 centerVec3;
    final F32.Vec3 moveAwayVec3;

    F32.ModelHighWaterMark mark;

    private ViewFrameNew(String name, Renderer renderer, Runnable sceneBuilder) {
        super(name, renderer, sceneBuilder);
        cameraVec3 = F32.Vec3.of(0f, 0f, .0f);
        lookDirVec3 = F32.Vec3.of(0f, 0f, 0f);
        var projF32Mat4x4_1 = F32.Mat4x4.Projection.of(renderer.image(), 0.1f, 1000f, 60f);
        var projF32Mat4x4_2 = F32.Mat4x4.mul(projF32Mat4x4_1, F32.Mat4x4.Scale.of(renderer.height() / 4f));
        projF32Mat4x4 = F32.Mat4x4.mul(projF32Mat4x4_2, F32.Mat4x4.Transformation.of(renderer.height() / 2f));
        centerVec3 = F32.Vec3.of(renderer.width() / 2f, renderer.height() / 2f, 0);
        moveAwayVec3 = F32.Vec3.of(0f, 0f, 30f);
        mark = new F32.ModelHighWaterMark(); // mark all buffers.  transforms create new points so this allows us to garbage colect
    }

    public static ViewFrameNew of(String name, Renderer renderer, Runnable sceneBuilder) {
        return new ViewFrameNew(name, renderer, sceneBuilder);
    }

    @Override
    void update() {
        final long elapsedMillis = System.currentTimeMillis() - startMillis;
        float theta = elapsedMillis * thetaDelta;

        if ((frames++ % 50) == 0) {
            System.out.println("Frames " + frames + " Theta = " + theta + " FPS = " + ((frames * 1000) / elapsedMillis));
        }

        //    mark.resetAll();

        var xyzRot4x4 = F32.Mat4x4.Rotation.of(theta * 2, theta / 2, theta);

        //  F32.ModelHighWaterMark resetMark = new F32.ModelHighWaterMark();

        List<F32.ZPos> zpos = new ArrayList<>();
        // Loop through the triangles
        boolean showHidden = renderer.displayMode() == Renderer.DisplayMode.WIRE_SHOW_HIDDEN;

        int end = F32.TriangleVec3.arr.size();
        for (int i = 0; i < end; i++) {
            var t = F32.TriangleVec3.arr.get(i);
            // here we rotate and then move into the Z plane.
            t = F32.mul(t, xyzRot4x4);
            t = F32.add(t, moveAwayVec3);
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

                F32.Vec3 cameraDeltaVec3 = sub(F32.center(t), cameraVec3); // clearly our default camera is 0,0,0

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

                t = F32.mul(t, projF32Mat4x4);//  projection matrix also scales to screen and translate half a screen

                zpos.add(new F32.ZPos(t, howVisible));
            }

            // resetMark.reset3D(); // do not move this up.
        }


        Collections.sort(zpos);
        List<F32.TriangleVec2> ztri = new ArrayList<>();
        for (F32.ZPos z : zpos) {
            ztri.add(z.create());
        }

        renderer.render(false);

        viewer.repaint();
    }
}
