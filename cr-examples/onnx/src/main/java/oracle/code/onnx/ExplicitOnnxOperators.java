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
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;

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

    public static <T> List<Tensor<T>> If(Tensor<Boolean> cond, Supplier<List<Tensor<T>>> elseBody, Supplier<List<Tensor<T>>> thenBody) {
        return cond.data().get(ValueLayout.JAVA_BOOLEAN, 0) ? thenBody.get() : elseBody.get();
    }

    public record LoopLocals<T>(Tensor<Long> i, Tensor<Boolean> cond, List<Tensor<T>> userValues) {}
    public static <T> List<Tensor<T>> Loop(Tensor<Long> max, Tensor<Boolean> cond, List<Tensor<T>> v_initial, Function<LoopLocals<T>, LoopLocals<T>> body) {
        long m = max.data().get(ValueLayout.JAVA_LONG, 0);
        LoopLocals<T> ll = new LoopLocals<>(Tensor.ofScalar(0), cond, v_initial);
        while (ll.i.data().get(ValueLayout.JAVA_LONG, 0) < m && ll.cond.data().get(ValueLayout.JAVA_BOOLEAN, 0)) {
            ll = body.apply(ll);
            ll.i.data().set(ValueLayout.JAVA_LONG, 0, ll.i.data().get(ValueLayout.JAVA_LONG, 0) + 1); // i++
        }
        return ll.userValues();
    }
}
