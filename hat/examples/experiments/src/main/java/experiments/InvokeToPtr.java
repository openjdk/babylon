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
import hat.OpsAndTypes;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.java.lang.reflect.code.OpTransformer;
import jdk.incubator.code.java.lang.reflect.code.analysis.SSA;
import jdk.incubator.code.java.lang.reflect.code.op.CoreOp;
import jdk.incubator.code.java.lang.reflect.code.type.FunctionType;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

import hat.backend.DebugBackend;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.SegmentMapper;

public class InvokeToPtr {

    @CodeReflection
    static float testMethod(PointyHat.ColoredWeightedPoint coloredWeightedPoint) {
        // StructOne* s1
        // s1 -> i
        int color = coloredWeightedPoint.color();
        // s1 -> *s2
        PointyHat.ColoredWeightedPoint.WeightedPoint weightedPoint = coloredWeightedPoint.weightedPoint();
        // s2 -> i
        PointyHat.ColoredWeightedPoint.WeightedPoint.Point point = weightedPoint.point();
        color += point.x();
        coloredWeightedPoint.color(color);
        // s2 -> f
        float weight = weightedPoint.weight();
        return color + weight;
    }


    public static void main(String[] args) {

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),new DebugBackend());

        PointyHat.ColoredWeightedPoint p = PointyHat.ColoredWeightedPoint.schema.allocate(accelerator);
        System.out.println(Buffer.getLayout(p));
        Optional<Method> om = Stream.of(InvokeToPtr.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("testMethod"))
                .findFirst();

        Method m = om.orElseThrow();
        CoreOp.FuncOp highLevelForm = m.getCodeModel().orElseThrow();

        System.out.println("Initial code model");
        System.out.println(highLevelForm.toText());
        System.out.println("------------------");

        CoreOp.FuncOp loweredForm = highLevelForm.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println("Lowered form which maintains original invokes and args");
        System.out.println(loweredForm.toText());
        System.out.println("-------------- ----");

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
}
