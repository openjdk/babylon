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
import java.util.List;
import java.util.Optional;
import jdk.incubator.code.Quotable;
import oracle.code.onnx.ir.OnnxOps;

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

    public record LoopResult<T>(Tensor<Boolean> cond, T output) {}
    public interface LoopBody<T> extends Quotable {
        LoopResult<T> invoke(Tensor<Long> i, Tensor<Boolean> cond, T input);
    }

    public static <T> T Loop(Tensor<Long> max, Tensor<Boolean> cond, T values, LoopBody<T> loopBody) {
        long m = max.data().get(ValueLayout.JAVA_LONG, 0);
        for (var i = Tensor.ofScalar(0l); longValue(i) < m && booleanValue(cond); set(i, longValue(i) + 1)) {
            LoopResult<T> ret = loopBody.invoke(i, cond, values);
            cond = ret.cond();
            values = ret.output();
        }
        return values;
    }

    // @@@ this should be generated from contrib operators

    public record GroupQueryAttention<T>(Tensor<T> output, Tensor<T> present_key, Tensor<T> present_value) { }
    public static <T, M> GroupQueryAttention<T> GroupQueryAttention(Tensor<T> query, java.util.Optional<Tensor<T>> key, java.util.Optional<Tensor<T>> value, java.util.Optional<Tensor<T>> past_key, java.util.Optional<Tensor<T>> past_value, Tensor<M> seqlens_k, Tensor<M> total_sequence_length, java.util.Optional<Tensor<T>> cos_cache, java.util.Optional<Tensor<T>> sin_cache, java.util.Optional<Long> do_rotary, long kv_num_heads, java.util.Optional<Long> local_window_size, long num_heads, java.util.Optional<Long> rotary_interleaved, java.util.Optional<Float> scale) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GroupQueryAttention.class, List.of(query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache), List.of(do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale));
        Object[] resultArray = (Object[]) result;
        return new GroupQueryAttention<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1], (Tensor<T>)resultArray[2]);
    }

    public static <T1, T2, T3, T4> Tensor<T1> MatMulNBits(Tensor<T1> a, Tensor<T2> b, Tensor<T1> scales, java.util.Optional<Tensor<T3>> zero_points, java.util.Optional<Tensor<T4>> g_idx, java.util.Optional<Tensor<T1>> bias, long K, long N, java.util.Optional<Long> accuracy_level, long bits, long block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMulNBits.class, List.of(a, b, scales, zero_points, g_idx, bias), List.of(K, N, accuracy_level, bits, block_size));
        return (Tensor<T1>)result;
    }

    public record SkipSimplifiedLayerNormalization<T>(Tensor<T> output, Tensor<Float> mean, Tensor<Float> inv_std_var, Tensor<Float> input_skip_bias_sum) { }
    public static <T> SkipSimplifiedLayerNormalization<T> SkipSimplifiedLayerNormalization(Tensor<T> input, Tensor<T> skip, Tensor<T> gamma, java.util.Optional<Tensor<T>> bias, java.util.Optional<Float> epsilon) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SkipSimplifiedLayerNormalization.class, List.of(input, skip, gamma, bias), List.of(epsilon));
        Object[] resultArray = (Object[]) result;
        return new SkipSimplifiedLayerNormalization<>((Tensor<T>)resultArray[0], (Tensor<Float>)resultArray[1], (Tensor<Float>)resultArray[2], (Tensor<Float>)resultArray[3]);
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
