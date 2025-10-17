/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.ifacemapper.MappableIface.RO;
import static hat.ifacemapper.MappableIface.RW;
import static hat.test.TestNbody.Universe.*;

public class TestNbody {

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

    @CodeReflection
    static public void nbodyKernel(@RO KernelContext kc, @RW Universe universe, float mass, float delT, float espSqr) {
        float accx = 0.0f;
        float accy = 0.0f;
        float accz = 0.0f;
        Body body = universe.body(kc.gix);

        for (int i = 0; i < universe.length(); i++) {
            Body otherBody = universe.body(i);
            float dx = otherBody.x() - body.x();
            float dy = otherBody.y() - body.y();
            float dz = otherBody.z() - body.z();
            float invDist = (float) (1.0f / Math.sqrt(((dx * dx) + (dy * dy) + (dz * dz) + espSqr)));
            float s = mass * invDist * invDist * invDist;
            accx = accx + (s * dx);
            accy = accy + (s * dy);
            accz = accz + (s * dz);
        }
        accx = accx * delT;
        accy = accy * delT;
        accz = accz * delT;
        body.x(body.x() + (body.vx() * delT) + accx * .5f * delT);
        body.y(body.y() + (body.vy() * delT) + accy * .5f * delT);
        body.z(body.z() + (body.vz() * delT) + accz * .5f * delT);
        body.vx(body.vx() + accx);
        body.vy(body.vy() + accy);
        body.vz(body.vz() + accz);
    }

    @CodeReflection
    public static void nbodyCompute(@RO ComputeContext cc, @RW Universe universe, final float mass, final float delT, final float espSqr) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(universe.length()));
        cc.dispatchKernel(computeRange, kernelContext -> nbodyKernel(kernelContext, universe, mass, delT, espSqr));
    }

    public static void computeSequential(Universe universe, float mass, float delT, float espSqr) {
        float accx = 0.0f;
        float accy = 0.0f;
        float accz = 0.0f;
        for (int j = 0; j < universe.length(); j++) {
            Body body = universe.body(j);

            for (int i = 0; i < universe.length(); i++) {
                Body otherBody = universe.body(i);
                float dx = otherBody.x() - body.x();
                float dy = otherBody.y() - body.y();
                float dz = otherBody.z() - body.z();
                float invDist = (float) (1.0f / Math.sqrt(((dx * dx) + (dy * dy) + (dz * dz) + espSqr)));
                float s = mass * invDist * invDist * invDist;
                accx = accx + (s * dx);
                accy = accy + (s * dy);
                accz = accz + (s * dz);
            }
            accx = accx * delT;
            accy = accy * delT;
            accz = accz * delT;
            body.x(body.x() + (body.vx() * delT) + accx * .5f * delT);
            body.y(body.y() + (body.vy() * delT) + accy * .5f * delT);
            body.z(body.z() + (body.vz() * delT) + accz * .5f * delT);
            body.vx(body.vx() + accx);
            body.vy(body.vy() + accy);
            body.vz(body.vz() + accz);
        }
    }

    @HatTest
    public void testNbody() {
        final int NUM_BODIES = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        Universe universe = create(accelerator, NUM_BODIES);
        Universe universeSeq = create(accelerator, NUM_BODIES);
        final float delT = .1f;
        final float espSqr = 0.1f;
        final float mass = .5f;

        Random random = new Random(71);
        for (int bodyIdx = 0; bodyIdx < NUM_BODIES; bodyIdx++) {
            Body b = universe.body(bodyIdx);

            final float theta = (float) (Math.random() * Math.PI * 2);
            final float phi = (float) (Math.random() * Math.PI * 2);
            final float radius = (float) (Math.random() * 100.f);

            // get random 3D coordinates in sphere
            b.x((float) (radius * Math.cos(theta) * Math.sin(phi)));
            b.y((float) (radius * Math.sin(theta) * Math.sin(phi)));
            b.z((float) (radius * Math.cos(phi)));
            b.vx(random.nextFloat(1));
            b.vy(random.nextFloat(1));
            b.vz(random.nextFloat(1));

            // Copy random values into the other body to check results
            Body seqBody = universeSeq.body(bodyIdx);
            seqBody.x(b.x());
            seqBody.y(b.y());
            seqBody.z(b.z());

            seqBody.vx(b.vx());
            seqBody.vy(b.vy());
            seqBody.vz(b.vz());

        }

        accelerator.compute(computeContext -> {
            TestNbody.nbodyCompute(computeContext, universe, mass, delT, espSqr);
        });

        computeSequential(universeSeq, espSqr, mass, espSqr);

        // Check results
        float delta = 0.1f;
        for (int i = 0; i < NUM_BODIES; i++) {
            Body hatBody = universe.body(i);
            Body seqBody = universeSeq.body(i);
            IO.println(i);
            HatAsserts.assertEquals(seqBody.x(), hatBody.x(), delta);
            HatAsserts.assertEquals(seqBody.y(), hatBody.y(), delta);
            HatAsserts.assertEquals(seqBody.z(), hatBody.z(), delta);
            HatAsserts.assertEquals(seqBody.vx(), hatBody.vx(), delta);
            HatAsserts.assertEquals(seqBody.vy(), hatBody.vy(), delta);
            HatAsserts.assertEquals(seqBody.vz(), hatBody.vz(), delta);
        }
    }
}
