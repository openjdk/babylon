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
package nbody;

import wrap.Wrap;
import wrap.glwrap.GLTexture;
import wrap.glwrap.GLWindow;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.stream.IntStream;

import static opengl.opengl_h.GLUT_DEPTH;
import static opengl.opengl_h.GLUT_DOUBLE;
import static opengl.opengl_h.GLUT_RGB;
import static opengl.opengl_h.GL_COLOR_BUFFER_BIT;
import static opengl.opengl_h.GL_DEPTH_BUFFER_BIT;
import static opengl.opengl_h.GL_MODELVIEW;
import static opengl.opengl_h.GL_TEXTURE_2D;
import static opengl.opengl_h.glBindTexture;
import static opengl.opengl_h.glClear;
import static opengl.opengl_h.glClearColor;
import static opengl.opengl_h.glColor3f;
import static opengl.opengl_h.glDisable;
import static opengl.opengl_h.glEnable;
import static opengl.opengl_h.glMatrixMode;
import static opengl.opengl_h.glRasterPos2f;
import static opengl.opengl_h.glScalef;
import static opengl.opengl_h.glTexCoord2f;
import static opengl.opengl_h.glVertex3f;
import static opengl.opengl_h.glutBitmapCharacter;
import static opengl.opengl_h.glutBitmapTimesRoman24$segment;
import static opengl.opengl_h.glutSwapBuffers;


public class NBodyGLWindow extends GLWindow {
    protected final float delT = .1f;

    protected final float espSqr = 0.1f;

    protected final float mass = .5f;

    final GLTexture particle;
    protected final Wrap.Float4Arr xyzPosFloatArr;
    protected final Wrap.Float4Arr xyzVelFloatArr;

    protected int bodyCount;
    protected int frameCount = 0;
    final long startTime = System.currentTimeMillis();

    protected final Mode mode;

    public NBodyGLWindow(Arena arena, int width, int height, GLTexture particle, int bodyCount, Mode mode) {
        super(arena, width, height, "nbody", GLUT_DOUBLE() | GLUT_RGB() | GLUT_DEPTH(), particle);
        this.particle = particle;
        this.bodyCount = bodyCount;
        this.xyzPosFloatArr = Wrap.Float4Arr.of(arena, bodyCount);
        this.xyzVelFloatArr = Wrap.Float4Arr.of(arena, bodyCount);

        this.mode = mode;
        final float maxDist = 80f;

        System.out.println(bodyCount + " particles");

        for (int body = 0; body < bodyCount; body++) {
            final float theta = (float) (Math.random() * Math.PI * 2);
            final float phi = (float) (Math.random() * Math.PI * 2);
            final float radius = (float) (Math.random() * maxDist);

            var radial = Wrap.Float4Arr.float4.of(
                    (float) (radius * Math.cos(theta) * Math.sin(phi)),
                    (float) (radius * Math.sin(theta) * Math.sin(phi)),
                    (float) (radius * Math.cos(phi)),
                    0f);

            xyzPosFloatArr.set(body, radial);
        }
    }


    float rot = 0f;

    public static void run(int body, int size, Wrap.Float4Arr xyzPos, Wrap.Float4Arr xyzVel, float mass, float delT, float espSqr) {
        float accx = 0.f;
        float accy = 0.f;
        float accz = 0.f;

        final float myPosx = xyzPos.getx(body);
        final float myPosy = xyzPos.gety(body);
        final float myPosz = xyzPos.getz(body);
        final float myVelx = xyzVel.getx(body);
        final float myVely = xyzVel.gety(body);
        final float myVelz = xyzVel.getz(body);

        for (int i = 0; i < size; i++) {
            final float dx = xyzPos.get(i).x() - myPosx;
            final float dy = xyzPos.get(i).y() - myPosy;
            final float dz = xyzPos.get(i).z() - myPosz;
            final float invDist = 1 / (float) Math.sqrt((dx * dx) + (dy * dy) + (dz * dz) + espSqr);
            final float s = mass * invDist * invDist * invDist;
            accx = accx + (s * dx);
            accy = accy + (s * dy);
            accz = accz + (s * dz);
        }
        accx = accx * delT;
        accy = accy * delT;
        accz = accz * delT;

        xyzPos.setx(body, myPosx + (myVelx + accx * .5f) * delT);
        xyzPos.sety(body, myPosy + (myVely + accy * .5f) * delT);
        xyzPos.setz(body, myPosz + (myVelz + accz * .5f) * delT);

        xyzVel.setx(body, myVelx + accx);
        xyzVel.sety(body, myVely + accy);
        xyzVel.setz(body, myVelz + accz);
    }

    public static void runf4(int body, int size, Wrap.Float4Arr xyzPos, Wrap.Float4Arr xyzVel, float mass, float delT, float espSqr) {
        var accf4 = Wrap.Float4Arr.float4.zero;
        var myPosf4 = xyzPos.get(body);
        var myVelf4 = xyzVel.get(body);
        for (int i = 0; i < size; i++) {
            var delta = xyzPos.get(i).sub(myPosf4); // xyz[i]-myPos
            var dSqrd = delta.mul(delta);           // delta^2
            var invDist = 1f / (float) Math.sqrt(dSqrd.x() + dSqrd.y() + dSqrd.z() + espSqr);
            accf4 = accf4.add(delta.mul(mass * invDist * invDist * invDist)); // accf4 += delta*(invDist^3*mass)
        }
        accf4 = accf4.mul(delT);
        xyzPos.set(body, myPosf4.add(myVelf4.mul(delT)).add(accf4.mul(.5f * delT)));
        xyzVel.set(body, myVelf4.add(accf4));
    }

    protected void moveBodies() {
        switch (mode) {
            case JavaMT4 -> IntStream.range(0, bodyCount).parallel().forEach(
                    i -> runf4(i, bodyCount, xyzPosFloatArr, xyzVelFloatArr, mass, delT, espSqr)
            );
            case JavaMT -> IntStream.range(0, bodyCount).parallel().forEach(
                    i -> run(i, bodyCount, xyzPosFloatArr, xyzVelFloatArr, mass, delT, espSqr)
            );
            case JavaSeq4 -> IntStream.range(0, bodyCount).forEach(
                    i -> runf4(i, bodyCount, xyzPosFloatArr, xyzVelFloatArr, mass, delT, espSqr)
            );
            case JavaSeq -> IntStream.range(0, bodyCount).forEach(
                    i -> run(i, bodyCount, xyzPosFloatArr, xyzVelFloatArr, mass, delT, espSqr)
            );
            default -> throw new RuntimeException("Should never get here");
        }
    }

    static final float WEST = 0;
    static final float EAST = 1;
    static final float NORTH = 0;
    static final float SOUTH = 1;
    static float dx = -.5f;
    static float dy = -.5f;
    static float dz = -.5f;




    @Override
    public void display() {
        moveBodies();
        glClearColor(0f, 0f, 0f, 0f);
        glClear(GL_COLOR_BUFFER_BIT() | GL_DEPTH_BUFFER_BIT());
        glEnable(GL_TEXTURE_2D()); // Annoyingly important,
        glBindTexture(GL_TEXTURE_2D(), textureBuf.get(particle.idx));

        glPushMatrix1(() -> {
            glScalef(.01f, .01f, .01f);
            glColor3f(1f, 1f, 1f);
            glQuads(() -> {

                for (int bodyIdx = 0; bodyIdx < bodyCount; bodyIdx++) {
                    var bodyf4 = xyzPosFloatArr.get(bodyIdx);

                            /*
                             * Textures are mapped to a quad by defining the vertices in
                             * the order SW,NW,NE,SE
                             &
                             *   2--->3
                             *   ^    |
                             *   |    v
                             *   1    4
                             *
                             * Here we are describing the 'texture plane' for the body.
                             * Ideally we need to rotate this to point to the camera (see billboarding)
                             */

                    glTexCoord2f(WEST, SOUTH);
                    glVertex3f(bodyf4.x() + WEST + dx, bodyf4.y() + SOUTH + dy, bodyf4.z() + dz);
                    glTexCoord2f(WEST, NORTH);
                    glVertex3f(bodyf4.x() + WEST + dx, bodyf4.y() + NORTH + dy, bodyf4.z() + dz);
                    glTexCoord2f(EAST, NORTH);
                    glVertex3f(bodyf4.x() + EAST + dx, bodyf4.y() + NORTH + dy, bodyf4.z() + dz);
                    glTexCoord2f(EAST, SOUTH);
                    glVertex3f(bodyf4.x() + EAST + dx, bodyf4.y() + SOUTH + dy, bodyf4.z() + dz);
                }
            });
        });

        glDisable(GL_TEXTURE_2D()); // Annoyingly important .. took two days to work that out
        //glUseProgram(0);
        glMatrixMode(GL_MODELVIEW());
        glPushMatrix1(() -> {
            glColor3f(0.0f, 1.0f, 0.0f);
            var font = glutBitmapTimesRoman24$segment();
            long elapsed = System.currentTimeMillis() - startTime;
            float secs = elapsed / 1000f;
            var FPS = "Mode: "+mode.toString()+" Bodies "+bodyCount+" FPS: "+((frameCount / secs));
          // System.out.print(" gw "+glutGet(GLUT_SCREEN_WIDTH())+" gh "+glutGet(GLUT_SCREEN_HEIGHT()));
           // System.out.print(" a "+aspect+",s "+size);
            // System.out.println(" w "+width+" h"+height);

            glRasterPos2f(-.8f, .7f);
            for (int c : FPS.getBytes()) {
                glutBitmapCharacter(font, c);
            }
        });
        glutSwapBuffers();
        frameCount++;

    }

    @Override
    public void onIdle() {
        // rot += 1f;
        super.onIdle();
    }



    public enum Mode  {
        OpenCL, Cuda, OpenCL4, Cuda4, JavaSeq, JavaMT, JavaSeq4, JavaMT4;

        public static Mode of(String s) {
            return switch (s) {
                case "OpenCL" -> Mode.OpenCL;
                case "Cuda" -> Mode.Cuda;
                case "JavaSeq" -> Mode.JavaSeq;
                case "JavaMT" -> Mode.JavaMT;
                case "JavaSeq4" -> Mode.JavaSeq4;
                case "JavaMT4" -> Mode.JavaMT4;
                case "OpenCL4" -> Mode.OpenCL4;
                case "Cuda4" -> Mode.Cuda4;
                default -> throw new IllegalStateException("No mode " + s);
            };
        }
    }
    public static void main(String[] args) throws IOException {
        int particleCount = args.length > 2 ? Integer.parseInt(args[2]) : 32768/2/2;
        Mode mode = Mode.of(args.length>3?args[3]: Mode.JavaMT.toString());
        System.out.println("mode" + mode);
        try (var arena = mode.equals(Mode.JavaMT)||mode.equals(Mode.JavaMT4) ? Arena.ofShared() : Arena.ofConfined()) {
            var particleTexture = new GLTexture(arena, NBodyGLWindow.class.getResourceAsStream("/particle.png"));
            new NBodyGLWindow(arena, 1000, 1000, particleTexture, particleCount, mode).bindEvents().mainLoop();
        }
    }
}

