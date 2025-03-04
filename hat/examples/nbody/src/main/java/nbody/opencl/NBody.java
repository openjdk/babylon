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
package nbody.opencl;


import hat.Accelerator;
import hat.ComputeContext;
import hat.backend.ffi.OpenCLBackend;
import hat.buffer.Buffer;
import hat.KernelContext;
import hat.ifacemapper.MappableIface;
import static hat.ifacemapper.MappableIface.*;
import static opengl.opengl_h.glMatrixMode;
import static opengl.opengl_h.glRasterPos2f;
import static opengl.opengl_h.glScalef;
import static opengl.opengl_h.glTexCoord2f;
import static opengl.opengl_h.glVertex3f;
import static opengl.opengl_h.glutBitmapCharacter;
import static opengl.opengl_h.glutBitmapTimesRoman24$segment;
import static opengl.opengl_h.glutSwapBuffers;
import static opengl.opengl_h_1.glBindTexture;
import static opengl.opengl_h_1.glClear;
import static opengl.opengl_h_1.glClearColor;
import static opengl.opengl_h_1.glColor3f;
import static opengl.opengl_h_1.glDisable;
import static opengl.opengl_h_1.glEnable;
import static opengl.opengl_h_2.GL_COLOR_BUFFER_BIT;
import static opengl.opengl_h_2.GL_DEPTH_BUFFER_BIT;
import static opengl.opengl_h_2.GL_MODELVIEW;
import static opengl.opengl_h_2.GL_TEXTURE_2D;

import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import nbody.NBodyGLWindow;
import wrap.clwrap.CLPlatform;
import wrap.clwrap.CLWrapComputeContext;
import wrap.glwrap.GLTexture;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;

public class NBody {
    public static class CLNBodyGLWindow extends NBodyGLWindow {
        public interface Universe extends Buffer {
            int length();

            interface Body extends Struct {
                float x();

                float y();

                float z();

                float vx();

                float vy();

                float vz();

                void x(float x);

                void y(float y);

                void z(float z);

                void vx(float vx);

                void vy(float vy);

                void vz(float vz);
            }

            Body body(long idx);
            /*
            typedef Body_s{
                float x;
                float y;
                float vx;
                float vy;
            } Body_t;

            typedef Universe_s{
               int length;
               Body body[1];
            }Universe_t;

             */
            Schema<Universe> schema = Schema.of(Universe.class, resultTable -> resultTable

                    .arrayLen("length").array("body", array -> array
                            .fields("x", "y", "z", "vx", "vy", "vz")
                    )
            );

            static Universe create(Accelerator accelerator, int length) {
                return schema.allocate(accelerator, length);
            }

        }


        @CodeReflection
        static public void nbodyKernel(@RO KernelContext kc, @RW Universe universe, float mass, float delT, float espSqr) {
            float accx = 0.0f;
            float accy = 0.0f;
            float accz = 0.0f;
            Universe.Body me = universe.body(kc.x);

            for (int i = 0; i < kc.maxX; i++) {
                Universe.Body body = universe.body(i);
                float dx = body.x() - me.x();
                float dy = body.y() - me.y();
                float dz = body.z() - me.z();
                float invDist = (float) (1.0f / Math.sqrt(((dx * dx) + (dy * dy) + (dz * dz) + espSqr)));
                float s = mass * invDist * invDist * invDist;
                accx = accx + (s * dx);
                accy = accy + (s * dy);
                accz = accz + (s * dz);
            }
            accx = accx * delT;
            accy = accy * delT;
            accz = accz * delT;
            me.x(me.x() + (me.vx() * delT + accx * .5f * delT));
            me.y(me.y() + (me.vy() * delT + accy * .5f * delT));
            me.z(me.z() + (me.vz() * delT + accz * .5f * delT));
            me.vx(me.vx() + accx);
            me.vy(me.vy() + accy);
            me.vz(me.vz() + accz);
        }

        @CodeReflection
        public static void nbodyCompute(@RO ComputeContext cc, @RW Universe universe, float mass, float delT, float espSqr) {
            float cmass = mass;
            float cdelT = delT;
            float cespSqr= espSqr;

            cc.dispatchKernel(universe.length(), kc -> nbodyKernel(kc, universe, cmass, cdelT, cespSqr));
        }


        final CLPlatform.CLDevice.CLContext.CLProgram.CLKernel kernel;
        final CLWrapComputeContext clWrapComputeContext;
        final CLWrapComputeContext.MemorySegmentState vel;
        final CLWrapComputeContext.MemorySegmentState pos;
final Accelerator accelerator;
final Universe universe;
        public CLNBodyGLWindow( Arena arena, int width, int height, GLTexture particle, int bodyCount, Mode mode) {
            super( arena, width, height, particle, bodyCount, mode);
            final float maxDist = 80f;
            if (mode.equals(Mode.HAT)){
                kernel = null;
                clWrapComputeContext = null;
                vel=null;
                pos=null;
                accelerator = new Accelerator(MethodHandles.lookup(),
                        new OpenCLBackend(OpenCLBackend.Mode.of("GPU"))
                );
                universe = Universe.create(accelerator, bodyCount);
                for (int body = 0; body < bodyCount; body++) {
                    Universe.Body b = universe.body(body);
                    final float theta = (float) (Math.random() * Math.PI * 2);
                    final float phi = (float) (Math.random() * Math.PI * 2);
                    final float radius = (float) (Math.random() * maxDist);

                    // get random 3D coordinates in sphere
                    b.x((float) (radius * Math.cos(theta) * Math.sin(phi)));
                    b.y((float) (radius * Math.sin(theta) * Math.sin(phi)));
                    b.z((float) (radius * Math.cos(phi)));
                }


            }else {
                this.universe = null;
                this.accelerator=null;
                this.clWrapComputeContext = new CLWrapComputeContext(arena, 20);
                this.vel = clWrapComputeContext.register(xyzVelFloatArr.ptr());
                this.pos = clWrapComputeContext.register(xyzPosFloatArr.ptr());

                var platforms = CLPlatform.platforms(arena);
                System.out.println("platforms " + platforms.size());
                var platform = platforms.get(0);
                platform.devices.forEach(device -> {
                    System.out.println("   Compute Units     " + device.computeUnits());
                    System.out.println("   Device Name       " + device.deviceName());
                    System.out.println("   Device Vendor       " + device.deviceVendor());
                    System.out.println("   Built In Kernels  " + device.builtInKernels());
                });
                var device = platform.devices.get(0);
                System.out.println("   Compute Units     " + device.computeUnits());
                System.out.println("   Device Name       " + device.deviceName());
                System.out.println("   Device Vendor       " + device.deviceVendor());

                System.out.println("   Built In Kernels  " + device.builtInKernels());
                var context = device.createContext();
                String code = switch (mode) {
                    case Mode.OpenCL -> """
                            __kernel void nbody( __global float *xyzPos ,__global float* xyzVel, float mass, float delT, float espSqr ){
                                int body = get_global_id(0);
                                int STRIDE=4;
                                int Xidx=0;
                                int Yidx=1;
                                int Zidx=2;
                                int bodyStride = body*STRIDE;
                                int bodyStrideX = bodyStride+Xidx;
                                int bodyStrideY = bodyStride+Yidx;
                                int bodyStrideZ = bodyStride+Zidx;

                                float accx = 0.0;
                                float accy = 0.0;
                                float accz = 0.0;
                                float myPosx = xyzPos[bodyStrideX];
                                float myPosy = xyzPos[bodyStrideY];
                                float myPosz = xyzPos[bodyStrideZ];
                                for (int i = 0; i < get_global_size(0); i++) {
                                    int iStride = i*STRIDE;
                                    float dx = xyzPos[iStride+Xidx] - myPosx;
                                    float dy = xyzPos[iStride+Yidx] - myPosy;
                                    float dz = xyzPos[iStride+Zidx] - myPosz;
                                    float invDist =  (float) 1.0/sqrt((float)((dx * dx) + (dy * dy) + (dz * dz) + espSqr));
                                    float s = mass * invDist * invDist * invDist;
                                    accx = accx + (s * dx);
                                    accy = accy + (s * dy);
                                    accz = accz + (s * dz);
                                }
                                accx = accx * delT;
                                accy = accy * delT;
                                accz = accz * delT;
                                xyzPos[bodyStrideX] = myPosx + (xyzVel[bodyStrideX] * delT) + (accx * 0.5 * delT);
                                xyzPos[bodyStrideY] = myPosy + (xyzVel[bodyStrideY] * delT) + (accy * 0.5 * delT);
                                xyzPos[bodyStrideZ] = myPosz + (xyzVel[bodyStrideZ] * delT) + (accz * 0.5 * delT);

                                xyzVel[bodyStrideX] = xyzVel[bodyStrideX] + accx;
                                xyzVel[bodyStrideY] = xyzVel[bodyStrideY] + accy;
                                xyzVel[bodyStrideZ] = xyzVel[bodyStrideZ] + accz;

                            }
                            """;
                   /* case Mode.OpenCL4 -> """
                            __kernel void nbody( __global float4 *xyzPos ,__global float4* xyzVel, float mass, float delT, float espSqr ){
                                float4 acc = (0.0,0.0,0.0,0.0);
                                float4 myPos = xyzPos[get_global_id(0)];
                                float4 myVel = xyzVel[get_global_id(0)];
                                for (int i = 0; i < get_global_size(0); i++) {
                                       float4 delta =  xyzPos[i] - myPos;
                                       float invDist =  (float) 1.0/sqrt((float)((delta.x * delta.x) + (delta.y * delta.y) + (delta.z * delta.z) + espSqr));
                                       float s = mass * invDist * invDist * invDist;
                                       acc= acc + (s * delta);
                                }
                                acc = acc*delT;
                                myPos = myPos + (myVel * delT) + (acc * delT)/2;
                                myVel = myVel + acc;
                                xyzPos[get_global_id(0)] = myPos;
                                xyzVel[get_global_id(0)] = myVel;

                            }
                            """;*/
                    case Mode.OpenCL4 -> """
                            __kernel void nbody( __global float4 *xyzPos ,__global float4* xyzVel, float mass, float delT, float espSqr ){
                                float4 acc = (0.0,0.0,0.0,0.0);
                                float4 myPos = xyzPos[get_global_id(0)];
                                float4 myVel = xyzVel[get_global_id(0)];
                                for (int i = 0; i < get_global_size(0); i++) {
                                       float4 delta =  xyzPos[i] - myPos;
                                       float invDist =  (float) 1.0/sqrt((float)((delta.x * delta.x) + (delta.y * delta.y) + (delta.z * delta.z) + espSqr));
                                       float s = mass * invDist * invDist * invDist;
                                       acc= acc + (s * delta);
                                }
                                acc = acc*delT;
                                myPos = myPos + (myVel * delT) + (acc * delT)/2;
                                myVel = myVel + acc;
                                xyzPos[get_global_id(0)] = myPos;
                                xyzVel[get_global_id(0)] = myVel;

                            }
                            """;
                    default -> throw new IllegalStateException();
                };
                var program = context.buildProgram(code);
                kernel = program.getKernel("nbody");
            }
        }
        @Override
        public void display() {
            if (mode.equals(Mode.HAT)) {
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
                            var bodyf4 = universe.body(bodyIdx);//xyzPosFloatArr.get(bodyIdx);

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
            }else{
                super.display();
            }
        }


        @Override
        protected void moveBodies() {
            if (mode.equals(Mode.HAT)) {
                float cmass = mass;
                float cdelT = delT;
                float cespSqr = espSqr;
                Universe cuniverse = universe;
                accelerator.compute(cc -> nbodyCompute(cc, cuniverse, cmass, cdelT, cespSqr));
            }else if (mode.equals(Mode.OpenCL4) || mode.equals(Mode.OpenCL)) {
                if (frameCount == 0) {
                    vel.copyToDevice = true;
                    pos.copyToDevice = true;
                } else {
                    vel.copyToDevice = false;
                    pos.copyToDevice = false;
                }
                vel.copyFromDevice = false;
                pos.copyFromDevice = true;

                kernel.run(clWrapComputeContext, bodyCount, pos, vel, mass, delT, espSqr);
            } else {
                super.moveBodies();
            }
        }
    }

    public static void main(String[] args) throws IOException {
        int particleCount = args.length > 2 ? Integer.parseInt(args[2]) : 32768;
        NBodyGLWindow.Mode mode = NBodyGLWindow.Mode.of(args.length > 3 ? args[3] : NBodyGLWindow.Mode.OpenCL4.toString());
        System.out.println("mode" + mode);
        try (var arena = mode.equals(NBodyGLWindow.Mode.JavaMT4) || mode.equals(NBodyGLWindow.Mode.JavaMT) ? Arena.ofShared() : Arena.ofConfined()) {
            var particleTexture = new GLTexture(arena, NBody.class.getResourceAsStream("/particle.png"));
            new CLNBodyGLWindow( arena, 1000, 1000, particleTexture, particleCount, mode).bindEvents().mainLoop();
        }
    }
}

