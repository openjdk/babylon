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
package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.OpsAndTypes;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.BackendAdaptor;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.Schema;
import hat.buffer.BufferAllocator;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.CodeReflection;

public class PointyHatArray {
    public interface PointArray extends Buffer {
        interface Point extends Struct {

            int x();

            void x(int x);

            int y();

            void y(int y);

            GroupLayout LAYOUT = MemoryLayout.structLayout(

                    ValueLayout.JAVA_INT.withName("x"),
                    ValueLayout.JAVA_INT.withName("y")
            );
        }

        int length();

        void length(int length);

        Point point(long idx);

        GroupLayout LAYOUT = MemoryLayout.structLayout(
                ValueLayout.JAVA_INT.withName("length"),
                MemoryLayout.sequenceLayout(100, Point.LAYOUT.withName(Point.class.getSimpleName())).withName("point")
        ).withName(PointArray.class.getSimpleName());


        Schema<PointArray> schema = Schema.of(PointArray.class, (pointArray)-> pointArray
                .arrayLen("length").array("point", (point)-> point

                                .field("x")
                                .field("y")
                )
        );

        static PointArray create(Accelerator accelerator, int len) {
            System.out.println(LAYOUT);
            PointArray pointArray= schema.allocate(accelerator,100);
            pointArray.length(100);
            return pointArray;
        }
    }

    public static class Compute {


        @CodeReflection
         public static void testMethodKernel(KernelContext kc, PointArray pointArray) {

            int len = pointArray.length();
            PointArray.Point point = pointArray.point(4);
            point.x(1);


        }

        @CodeReflection
        public static void compute(ComputeContext cc, PointArray pointArray) {
            cc.dispatchKernel(1, kc -> Compute.testMethodKernel(kc, pointArray));
        }

    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), new BackendAdaptor() {
            @Override
            public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
                var highLevelForm = Op.ofMethod(kernelCallGraph.entrypoint.method).orElseThrow();
                System.out.println("Initial code model");
                System.out.println(highLevelForm.toText());
                System.out.println("------------------");
                CoreOp.FuncOp loweredForm = highLevelForm.transform(OpTransformer.LOWERING_TRANSFORMER);
                System.out.println("Lowered form which maintains original invokes and args");
                System.out.println(loweredForm.toText());
                System.out.println("-------------- ----");
                // highLevelForm.lower();
                CoreOp.FuncOp ssaInvokeForm = SSA.transform(loweredForm);
                System.out.println("SSA form which maintains original invokes and args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

                FunctionType functionType = OpsAndTypes.transformTypes(MethodHandles.lookup(), ssaInvokeForm);
                System.out.println("SSA form with types transformed args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

                CoreOp.FuncOp ssaPtrForm = OpsAndTypes.transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
                System.out.println("SSA form with invokes replaced by ptrs");
                System.out.println(ssaPtrForm.toText());
            }
        });
        var pointArray = PointArray.create(accelerator,100);
        accelerator.compute(cc -> Compute.compute(cc, pointArray));
    }
}
