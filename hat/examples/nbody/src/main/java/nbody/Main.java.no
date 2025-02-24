package nbody;
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


import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.Buffer;
import static hat.ifacemapper.MappableIface.*;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static opengl.opengl_h.GL_COLOR_BUFFER_BIT;
import static opengl.opengl_h.GL_DEPTH_BUFFER_BIT;
import static opengl.opengl_h.GL_QUADS;
import static opengl.opengl_h.GL_TEXTURE_2D;
import static opengl.opengl_h.glBegin;
import static opengl.opengl_h.glBindTexture;
import static opengl.opengl_h.glClear;
import static opengl.opengl_h.glColor3f;
import static opengl.opengl_h.glEnd;
import static opengl.opengl_h.glLoadIdentity;
import static opengl.opengl_h.glPopMatrix;
import static opengl.opengl_h.glPushMatrix;
import static opengl.opengl_h.glRotatef;
import static opengl.opengl_h.glScalef;
import static opengl.opengl_h.glTexCoord2f;
import static opengl.opengl_h.glVertex3f;
import static opengl.opengl_h.glutSwapBuffers;
import static opengl.opengl_h_3.C_FLOAT;


public class Main {
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

        Schema<Universe> schema = Schema.of(Universe.class, resultTable -> resultTable

                .arrayLen("length").array("body", array -> array
                        .fields("x", "y", "z", "vx", "vy", "vz")
                )
        );

        static Universe create(Accelerator accelerator, int length) {
            return schema.allocate(accelerator, length);
        }

    }

    public static class NBody extends GLWrap.GLWindow {

        protected final static float delT = .1f;

        protected final static float espSqr = 0.1f;

        protected final static float mass = .5f;

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


        private static int STRIDE = 4;
        private static int Xidx = 0;
        private static int Yidx = 1;
        private static int Zidx = 2;

        final float[] xyzPos;
        final float[] xyzVel;

        final GLWrap.GLTexture particle;
        final MemorySegment xyzPosSeg;
        final MemorySegment xyzVelSeg;
        final Universe universe;
        final Accelerator accelerator;
        final CLWrap.Platform.Device.Context.Program.Kernel kernel;

        int count;
        int frames = 0;
        long startTime = 0l;

        public enum Mode {
            HAT(),
            OpenCL("""
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
                    """),
            OpenCL4("""
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
                    """),
            JavaSeq(false),
            JavaMT(true);
            final public boolean hat;
            final public String code;
            final public boolean isOpenCL;
            final public boolean isJava;
            final public boolean isMultiThreaded;

            Mode() {
                this.hat = true;
                this.code = null;
                this.isOpenCL = false;
                this.isJava = false;
                this.isMultiThreaded = false;
            }

            Mode(String code) {
                this.hat = true;
                this.code = code;
                this.isOpenCL = true;
                this.isJava = false;
                this.isMultiThreaded = false;
            }

            Mode(boolean isMultiThreaded) {
                this.hat = true;
                this.code = null;
                this.isOpenCL = false;
                this.isJava = true;
                this.isMultiThreaded = isMultiThreaded;
            }

            public static Mode of(String name, Mode defaultMode) {
                return switch (name) {
                    case "HAT" -> NBody.Mode.HAT;
                    case "OpenCL" -> NBody.Mode.OpenCL;
                    case "JavaSeq" -> NBody.Mode.JavaSeq;
                    case "JavaMT" -> NBody.Mode.JavaMT;
                    case "OpenCL4" -> NBody.Mode.OpenCL4;
                    default -> defaultMode;
                };
            }
        }

        final Mode mode;

        public NBody(Arena arena, int width, int height, GLWrap.GLTexture particle, int count, Mode mode) {
            super(arena, width, height, "nbody", particle);
            this.particle = particle;
            this.count = count;
            this.xyzPos = new float[count * STRIDE];
            this.xyzVel = new float[count * STRIDE];
            this.mode = mode;
            final float maxDist = 80f;

            System.out.println(count + " particles");

            switch (mode) {
                case OpenCL, OpenCL4, JavaMT, JavaSeq -> {
                    for (int body = 0; body < count; body++) {
                        final float theta = (float) (Math.random() * Math.PI * 2);
                        final float phi = (float) (Math.random() * Math.PI * 2);
                        final float radius = (float) (Math.random() * maxDist);

                        // get random 3D coordinates in sphere
                        xyzPos[(body * STRIDE) + Xidx] = (float) (radius * Math.cos(theta) * Math.sin(phi));
                        xyzPos[(body * STRIDE) + Yidx] = (float) (radius * Math.sin(theta) * Math.sin(phi));
                        xyzPos[(body * STRIDE) + Zidx] = (float) (radius * Math.cos(phi));
                    }
                }
                default -> {
                }

            }
            switch (mode){
                case OpenCL,OpenCL4->{
                    xyzPosSeg = arena.allocateFrom(JAVA_FLOAT, xyzPos);
                    xyzVelSeg = arena.allocateFrom(JAVA_FLOAT, xyzVel);
                    CLWrap openCL = new CLWrap(arena);

                    CLWrap.Platform.Device[] selectedDevice = new CLWrap.Platform.Device[1];
                    openCL.platforms.forEach(platform -> {
                        System.out.println("Platform Name " + platform.platformName());
                        platform.devices.forEach(device -> {
                            System.out.println("   Compute Units     " + device.computeUnits());
                            System.out.println("   Device Name       " + device.deviceName());
                            System.out.println("   Built In Kernels  " + device.builtInKernels());
                            selectedDevice[0] = device;
                        });
                    });
                    var context = selectedDevice[0].createContext();
                    var program = context.buildProgram(mode.code);
                    kernel = program.getKernel("nbody");
                    accelerator = null;
                    universe = null;
                }
                case JavaMT,JavaSeq->{
                    kernel = null;
                    xyzPosSeg = null;
                    xyzVelSeg = null;
                    accelerator = null;
                    universe = null;
                }
                case HAT->{
                    kernel = null;
                    xyzPosSeg = null;
                    xyzVelSeg = null;
                    accelerator = new Accelerator(MethodHandles.lookup(),
                            Backend.FIRST
                    );
                    universe = Universe.create(accelerator, count);
                    for (int body = 0; body < count; body++) {
                        Universe.Body b = universe.body(body);
                        final float theta = (float) (Math.random() * Math.PI * 2);
                        final float phi = (float) (Math.random() * Math.PI * 2);
                        final float radius = (float) (Math.random() * maxDist);

                        // get random 3D coordinates in sphere
                        b.x((float) (radius * Math.cos(theta) * Math.sin(phi)));
                        b.y((float) (radius * Math.sin(theta) * Math.sin(phi)));
                        b.z((float) (radius * Math.cos(phi)));
                    }
                }
                default -> {
                    kernel = null;
                    xyzPosSeg = null;
                    xyzVelSeg = null;
                    accelerator = null;
                    universe = null;
                }
            }
        }


        float rot = 0f;

        public static void run(int body, int size, float[] xyzPos, float[] xyzVel, float mass, float delT, float espSqr) {
            float accx = 0.f;
            float accy = 0.f;
            float accz = 0.f;
            int bodyStride = body * STRIDE;
            int bodyStrideX = bodyStride + Xidx;
            int bodyStrideY = bodyStride + Yidx;
            int bodyStrideZ = bodyStride + Zidx;

            final float myPosx = xyzPos[bodyStrideX];
            final float myPosy = xyzPos[bodyStrideY];
            final float myPosz = xyzPos[bodyStrideZ];

            for (int i = 0; i < size; i++) {
                int iStride = i * STRIDE;
                int iStrideX = iStride + Xidx;
                int iStrideY = iStride + Yidx;
                int iStrideZ = iStride + Zidx;
                final float dx = xyzPos[iStrideX] - myPosx;
                final float dy = xyzPos[iStrideY] - myPosy;
                final float dz = xyzPos[iStrideZ] - myPosz;
                final float invDist = 1 / (float) Math.sqrt((dx * dx) + (dy * dy) + (dz * dz) + espSqr);
                final float s = mass * invDist * invDist * invDist;
                accx = accx + (s * dx);
                accy = accy + (s * dy);
                accz = accz + (s * dz);
            }
            accx = accx * delT;
            accy = accy * delT;
            accz = accz * delT;
            xyzPos[bodyStrideX] = myPosx + (xyzVel[bodyStrideX] * delT) + (accx * .5f * delT);
            xyzPos[bodyStrideY] = myPosy + (xyzVel[bodyStrideY] * delT) + (accy * .5f * delT);
            xyzPos[bodyStrideZ] = myPosz + (xyzVel[bodyStrideZ] * delT) + (accz * .5f * delT);

            xyzVel[bodyStrideX] = xyzVel[bodyStrideX] + accx;
            xyzVel[bodyStrideY] = xyzVel[bodyStrideY] + accy;
            xyzVel[bodyStrideZ] = xyzVel[bodyStrideZ] + accz;
        }

        void display() {
            if (startTime == 0) {
                startTime = System.currentTimeMillis();
            }
            glClear(GL_COLOR_BUFFER_BIT() | GL_DEPTH_BUFFER_BIT());
            glPushMatrix();
            glLoadIdentity();
            glRotatef(-rot / 2f, 0f, 0f, 1f);
            //glRotatef(rot, 0f, 1f, 0f);
            //   glTranslatef(0f, 0f, trans);
            glScalef(.01f, .01f, .01f);
            glColor3f(1f, 1f, 1f);

            switch (mode){
                case JavaMT,JavaSeq ->{
                    if (mode.isMultiThreaded) {
                        IntStream.range(0, count).parallel().forEach(
                                i -> run(i, count, xyzPos, xyzVel, mass, delT, espSqr)
                        );
                    } else {
                        IntStream.range(0, count).forEach(
                                i -> run(i, count, xyzPos, xyzVel, mass, delT, espSqr)
                        );
                    }
                }
                case OpenCL,OpenCL4->{
                    kernel.run(count, xyzPosSeg, xyzVelSeg, mass, delT, espSqr);
                }
                case HAT->{
                    float cmass = mass;
                    float cdelT = delT;
                    float cespSqr = espSqr;
                    Universe cuniverse = universe;
                    accelerator.compute(cc -> nbodyCompute(cc, cuniverse, cmass, cdelT, cespSqr));
                }
            }

            glBegin(GL_QUADS());
            {
                glBindTexture(GL_TEXTURE_2D(), textureBuf.get(JAVA_INT, particle.idx * JAVA_INT.byteSize()));
                float dx = -.5f;
                float dy = -.5f;
                float dz = -.5f;

                for (int i = 0; i < count; i++) {
                    float x=0,y=0,z=0;
                    switch (mode){
                        case OpenCL4 ,OpenCL -> {
                            x = xyzPosSeg.get(C_FLOAT, (i * STRIDE * C_FLOAT.byteSize()) + (Xidx * C_FLOAT.byteSize()));
                            y = xyzPosSeg.get(C_FLOAT, (i * STRIDE * C_FLOAT.byteSize()) + (Yidx * C_FLOAT.byteSize()));
                            z = xyzPosSeg.get(C_FLOAT, (i * STRIDE * C_FLOAT.byteSize()) + (Zidx * C_FLOAT.byteSize()));
                        }
                        case JavaMT, JavaSeq -> {
                            x = xyzPos[(i * STRIDE) + Xidx];
                            y = xyzPos[(i * STRIDE) + Yidx];
                            z = xyzPos[(i * STRIDE) + Zidx];
                        }
                        case HAT ->{
                            Universe.Body body = universe.body(i);
                            x=body.x();
                            y=body.y();
                            z=body.z();
                        }
                    }
                    final int LEFT = 0;
                    final int RIGHT = 1;
                    final int TOP = 0;
                    final int BOTTOM = 1;
                    glTexCoord2f(LEFT, BOTTOM);
                    glVertex3f(x + dx + LEFT, y + dy + BOTTOM, z + dz);
                    glTexCoord2f(LEFT, TOP);
                    glVertex3f(x + dx + LEFT, y + dy + TOP, z + dz);
                    glTexCoord2f(RIGHT, TOP);
                    glVertex3f(x + dx + RIGHT, y + dy + TOP, z + dz);
                    glTexCoord2f(RIGHT, BOTTOM);
                    glVertex3f(x + dx + RIGHT, y + dy + BOTTOM, z + dz);
                }
            }
            glEnd();
            glColor3f(0.8f, 0.1f, 0.1f);
            glPopMatrix();
            glutSwapBuffers();
            frames++;
            long elapsed = System.currentTimeMillis() - startTime;
            if (elapsed > 200 || (frames % 100) == 0) {
                float secs = elapsed / 1000f;
              //  System.out.println((frames / secs) + "fps");
            }
        }

        void onIdle() {
            rot += 1f;
            super.onIdle();
        }
    }

    public void main(String[] args) {
        int particleCount =  32768;
        NBody.Mode mode = NBody.Mode.HAT;//NBody.Mode.OpenCL4;//NBody.Mode.of("HAT", NBody.Mode.OpenCL);
        System.out.println("mode" + mode);
        try (var arena = Arena.ofConfined()) {
            var particleTexture = new GLWrap.GLTexture(arena, NBody.class.getResourceAsStream("/particle.png"));
            new NBody(arena, 1000, 1000, particleTexture, particleCount, mode).mainLoop();
        }
    }
}

