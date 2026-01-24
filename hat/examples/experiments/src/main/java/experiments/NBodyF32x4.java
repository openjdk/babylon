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
package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;

import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;
import java.util.Random;

import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.RW;

public class NBodyF32x4 {
    public interface Universe extends Buffer {
        long length();

        interface Body extends Struct {
            float x();

            float y();

            float z();

            float w();

            float vx();

            float vy();

            float vz();

            float vw();

            void x(float x);

            void y(float y);

            void z(float z);

            void w(float z);

            void vx(float vx);

            void vy(float vy);

            void vz(float vz);

            void vw(float vw);
        }

        Body body(long idx);

        Schema<Universe> schema = Schema.of(Universe.class, resultTable -> resultTable
                .arrayLen("length")
                    .pad(8)
                    .array("body", array -> array
                        .fields("x", "y", "z", "w", "vx", "vy", "vz", "vw")
                )
        );

        static Universe create(Accelerator accelerator, int length) {
            return BoundSchema.of(accelerator, schema, length).allocate();
        }
    }

    @Reflect
    static public void nbodyKernel(@RO KernelContext kc, @RW Universe universe, float mass, float delT, float espSqr) {
        float accx = 0.0f;
        float accy = 0.0f;
        float accz = 0.0f;
        Universe.Body body = universe.body(kc.gix);

        for (int i = 0; i < universe.length(); i++) {
            Universe.Body otherBody = universe.body(i);
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

    @Reflect
    public static void nbodyCompute(@RO ComputeContext cc, @RW Universe universe, final float mass, final float delT, final float espSqr) {
        var ndrange = NDRange.of1D((int)universe.length());
        cc.dispatchKernel(ndrange, kernelContext -> nbodyKernel(kernelContext, universe, mass, delT, espSqr));
    }

    public static void computeSequential(Universe universe, float mass, float delT, float espSqr) {
        var ndrange = NDRange.of1D((int)universe.length());
        KernelContext kernelContext = new KernelContext(ndrange);
        for (kernelContext.gix = 0; kernelContext.gix < kernelContext.gsx; kernelContext.gix++) {
           nbodyKernel(kernelContext,universe,mass,delT,espSqr);
        }
    }

    @Reflect
    public static void main(String[] args) {
        final int NUM_BODIES = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        Universe universe = Universe.create(accelerator, NUM_BODIES);

        final float delT = .1f;
        final float espSqr = 0.1f;
        final float mass = .5f;

        Random random = new Random(71);
        for (int bodyIdx = 0; bodyIdx < NUM_BODIES; bodyIdx++) {
            Universe.Body b = universe.body(bodyIdx);

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
        }
        Universe universeSeq = Universe.create(accelerator, NUM_BODIES);
        MemorySegment from = MappableIface.getMemorySegment(universe);
        MemorySegment toSeq = MappableIface.getMemorySegment(universeSeq);
        toSeq.copyFrom(from);

        accelerator.compute(computeContext -> nbodyCompute(computeContext, universe, mass, delT, espSqr));

        computeSequential(universeSeq, espSqr, mass, espSqr);

        System.out.println("Delta = "+averageDisplacementError(universe,universeSeq));
    }



        /**
         * Compares two sets of positions and returns the average Euclidean error.
         * @return The Average Displacement Error (ADE)
         */
        public static double averageDisplacementError(Universe lhs, Universe rhs) {
            double totalError = 0;
            for (int i = 0; i < lhs.length(); i++) {
                var rightBody = lhs.body(i);
                var leftBody = rhs.body(i);
                double dx = rightBody.x() - leftBody.x();
                double dy = rightBody.y() - leftBody.y();
                double dz = rightBody.z() - leftBody.z();
                totalError += Math.sqrt(dx * dx + dy * dy + dz * dz);
            }
            return totalError / lhs.length();
        }

}
