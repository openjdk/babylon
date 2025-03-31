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

package oracle.code.onnx;

import java.lang.foreign.ValueLayout;
import java.util.Optional;
import jdk.incubator.code.Quotable;

class ExplicitOnnxOperators {

    // Explicit constant operators

    public static Tensor<Long> Constant(
            Long c) {
        return OnnxOperators.Constant(
                Optional.of(c),Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public static Tensor<Long> Constant(
            long[] c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(c), Optional.empty(), Optional.empty());
    }

    public static Tensor<Float> Constant(
            Float c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.empty(), Optional.empty(), Optional.of(c),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public static Tensor<Float> Constant(
            float[] c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.of(c), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public static Tensor<Integer> Constant(
            String c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of(c), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public static Tensor<Integer> Constant(
            String[] c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.empty(), Optional.of(c), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    // @@@ Constants for value - TENSOR and sparse_value - SPARSE_TENSOR


    public interface IfBody<T> extends Quotable {
        T invoke();
    }

    public static <T> T If(Tensor<Boolean> cond, IfBody<T> thenBody, IfBody<T> elseBody) {
        return booleanValue(cond) ? thenBody.invoke() : elseBody.invoke();
    }

    public record LoopReturn<T>(Tensor<Boolean> cond, T output) {}
    public interface LoopBody<T> extends Quotable {
        LoopReturn<T> invoke(Tensor<Long> i, Tensor<Boolean> cond, T input);
    }

    public static <T> T Loop(Tensor<Long> max, Tensor<Boolean> cond, T values, LoopBody<T> loopBody) {
        long m = max.data().get(ValueLayout.JAVA_LONG, 0);
        for (var i = Tensor.ofScalar(0l); longValue(i) < m && booleanValue(cond); set(i, longValue(i) + 1)) {
            LoopReturn<T> ret = loopBody.invoke(i, cond, values);
            cond = ret.cond();
            values = ret.output();
        }
        return values;
    }

    // @@@ move to Tensor API

    private static boolean booleanValue(Tensor<Boolean> t) {
        return t.data().get(ValueLayout.JAVA_BOOLEAN, 0);
    }

    private static long longValue(Tensor<Long> t) {
        return t.data().get(ValueLayout.JAVA_LONG, 0);
    }

    private static void set(Tensor<Long> t, long value) {
        t.data().set(ValueLayout.JAVA_LONG, 0, value);
    }
}
