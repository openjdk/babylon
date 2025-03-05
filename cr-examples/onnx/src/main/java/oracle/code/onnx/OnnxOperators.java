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
// Auto-generated from ONNX op schema

package oracle.code.onnx;

import java.lang.foreign.ValueLayout;
import oracle.code.onnx.ir.OnnxOps;

import java.util.Optional;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

@SuppressWarnings({"unchecked", "OptionalUsedAsFieldOrParameterType"})
public final class OnnxOperators extends ExplicitOnnxOperators {

    private OnnxOperators() {}

    public static <T> Tensor<T> If(Tensor<Boolean> cond, Supplier<Tensor<T>> elseBody, Supplier<Tensor<T>> thenBody) {
        return cond.data().get(ValueLayout.JAVA_BOOLEAN, 0) ? thenBody.get() : elseBody.get();
    }

    public static <T> Tensor<T> Abs(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Abs.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Acos(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Acos.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Acosh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Acosh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T3> List<Tensor<T3>> Adagrad(Tensor<T1> R, Tensor<Long> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> decay_factor, Optional<Float> norm_coefficient) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Adagrad.class, List.of(R, T, inputs), List.of(epsilon, decay_factor, norm_coefficient));
        return (List<Tensor<T3>>) result;
    }

    public static <T1, T3> List<Tensor<T3>> Adam(Tensor<T1> R, Tensor<Long> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> norm_coefficient_post, Optional<Float> norm_coefficient, Optional<Float> alpha, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Adam.class, List.of(R, T, inputs), List.of(epsilon, norm_coefficient_post, norm_coefficient, alpha, beta));
        return (List<Tensor<T3>>) result;
    }

    public static <T> Tensor<T> Add(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Add.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T1> Tensor<T1> AffineGrid(Tensor<T1> theta, Tensor<Long> size, Optional<Long> align_corners) {
        Object result = OnnxInterpreter.interpret(OnnxOps.AffineGrid.class, List.of(theta, size), List.of(align_corners));
        return (Tensor<T1>) result;
    }

    public static Tensor<Boolean> And(Tensor<Boolean> A, Tensor<Boolean> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.And.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<Long> ArgMax(Tensor<T> data, Optional<Long> keepdims, Optional<Long> select_last_index, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArgMax.class, List.of(data), List.of(keepdims, select_last_index, axis));
        return (Tensor<Long>) result;
    }

    public static <T> Tensor<Long> ArgMin(Tensor<T> data, Optional<Long> keepdims, Optional<Long> select_last_index, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArgMin.class, List.of(data), List.of(keepdims, select_last_index, axis));
        return (Tensor<Long>) result;
    }

    public static <T> Tensor<T> ArrayFeatureExtractor(Tensor<T> X, Tensor<Long> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArrayFeatureExtractor.class, List.of(X, Y), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Asin(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Asin.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Asinh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Asinh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Atan(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Atan.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Atanh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Atanh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> AveragePool(Tensor<T> X, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<Long> count_include_pad, Optional<Long> ceil_mode, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.AveragePool.class, List.of(X), List.of(pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape));
        return (Tensor<T>) result;
    }

    public record BatchNormalizationResult<T, T2>(Tensor<T> Y, Tensor<T2> running_mean, Tensor<T2> running_var) { }
    public static <T, T1, T2> BatchNormalizationResult<T, T2> BatchNormalization(Tensor<T> X, Tensor<T1> scale, Tensor<T1> B, Tensor<T2> input_mean, Tensor<T2> input_var, Optional<Float> epsilon, Optional<Long> training_mode, Optional<Float> momentum) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BatchNormalization.class, List.of(X, scale, B, input_mean, input_var), List.of(epsilon, training_mode, momentum));
        Object[] resultArray = (Object[]) result;
        return new BatchNormalizationResult<>((Tensor<T>)resultArray[0], (Tensor<T2>)resultArray[1], (Tensor<T2>)resultArray[2]);
    }

    public static <T1, T2> Tensor<T2> Bernoulli(Tensor<T1> input, Optional<Float> seed, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Bernoulli.class, List.of(input), List.of(seed, dtype));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Binarizer(Tensor<T> X, Optional<Float> threshold) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Binarizer.class, List.of(X), List.of(threshold));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> BitShift(Tensor<T> X, Tensor<T> Y, String direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitShift.class, List.of(X, Y), List.of(direction));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> BitwiseAnd(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseAnd.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> BitwiseNot(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseNot.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> BitwiseOr(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseOr.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> BitwiseXor(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseXor.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> BlackmanWindow(Tensor<T1> size, Optional<Long> periodic, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BlackmanWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }

    public static <T1, T2> Tensor<T2> Cast(Tensor<T1> input, Optional<Long> saturate, long to) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cast.class, List.of(input), List.of(saturate, to));
        return (Tensor<T2>) result;
    }

    public static <T1, T2> Tensor<T2> CastLike(Tensor<T1> input, Tensor<T2> target_type, Optional<Long> saturate) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CastLike.class, List.of(input, target_type), List.of(saturate));
        return (Tensor<T2>) result;
    }

    public static <T1, T2> Tensor<T2> CastMap(Map<Long, T1> X, Optional<String> map_form, Optional<String> cast_to, Optional<Long> max_map) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CastMap.class, List.of(X), List.of(map_form, cast_to, max_map));
        return (Tensor<T2>) result;
    }

    public static <T1, T2> Tensor<T2> CategoryMapper(Tensor<T1> X, Optional<long[]> cats_int64s, Optional<String[]> cats_strings, Optional<Long> default_int64, Optional<String> default_string) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CategoryMapper.class, List.of(X), List.of(cats_int64s, cats_strings, default_int64, default_string));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Ceil(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Ceil.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static Tensor<Float> Celu(Tensor<Float> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Celu.class, List.of(X), List.of(alpha));
        return (Tensor<Float>) result;
    }

    public static <T, Tind> Tensor<T> CenterCropPad(Tensor<T> input_data, Tensor<Tind> shape, Optional<long[]> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CenterCropPad.class, List.of(input_data, shape), List.of(axes));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Clip(Tensor<T> input, Optional<Tensor<T>> min, Optional<Tensor<T>> max) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Clip.class, List.of(input, min, max), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Col2Im(Tensor<T> input, Tensor<Long> image_shape, Tensor<Long> block_shape, Optional<long[]> pads, Optional<long[]> dilations, Optional<long[]> strides) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Col2Im.class, List.of(input, image_shape, block_shape), List.of(pads, dilations, strides));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Compress(Tensor<T> input, Tensor<Boolean> condition, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Compress.class, List.of(input, condition), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Concat(List<Tensor<T>> inputs, long axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Concat.class, List.of(inputs), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <S, T> Tensor<T> ConcatFromSequence(List<Tensor<S>> input_sequence, long axis, Optional<Long> new_axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConcatFromSequence.class, List.of(input_sequence), List.of(axis, new_axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Constant(Optional<Long> value_int, Optional<float[]> value_floats, Optional<String[]> value_strings, Optional<Float> value_float, Optional<String> value_string, Optional<long[]> value_ints, Optional<byte[]> sparse_value, Optional<byte[]> value) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Constant.class, List.of(), List.of(value_int, value_floats, value_strings, value_float, value_string, value_ints, sparse_value, value));
        return (Tensor<T>) result;
    }

    public static <T2> Tensor<T2> ConstantOfShape(Tensor<Long> input, Optional<byte[]> value) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConstantOfShape.class, List.of(input), List.of(value));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Conv(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Conv.class, List.of(X, W, B), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<Integer> ConvInteger(Tensor<T1> x, Tensor<T2> w, Optional<Tensor<T1>> x_zero_point, Optional<Tensor<T2>> w_zero_point, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConvInteger.class, List.of(x, w, x_zero_point, w_zero_point), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<Integer>) result;
    }

    public static <T> Tensor<T> ConvTranspose(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<long[]> output_shape, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape, Optional<long[]> output_padding) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConvTranspose.class, List.of(X, W, B), List.of(output_shape, pads, dilations, auto_pad, strides, group, kernel_shape, output_padding));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Cos(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cos.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Cosh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cosh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T, T2> Tensor<T> CumSum(Tensor<T> x, Tensor<T2> axis, Optional<Long> exclusive, Optional<Long> reverse) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CumSum.class, List.of(x, axis), List.of(exclusive, reverse));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T1> DFT(Tensor<T1> input, Optional<Tensor<T2>> dft_length, Optional<Tensor<Long>> axis, Optional<Long> inverse, Optional<Long> onesided) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DFT.class, List.of(input, dft_length, axis), List.of(inverse, onesided));
        return (Tensor<T1>) result;
    }

    public static <T> Tensor<T> DeformConv(Tensor<T> X, Tensor<T> W, Tensor<T> offset, Optional<Tensor<T>> B, Optional<Tensor<T>> mask, Optional<long[]> pads, Optional<long[]> dilations, Optional<long[]> strides, Optional<Long> offset_group, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DeformConv.class, List.of(X, W, offset, B, mask), List.of(pads, dilations, strides, offset_group, group, kernel_shape));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> DepthToSpace(Tensor<T> input, Optional<String> mode, long blocksize) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DepthToSpace.class, List.of(input), List.of(mode, blocksize));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> DequantizeLinear(Tensor<T1> x, Tensor<T2> x_scale, Optional<Tensor<T1>> x_zero_point, Optional<Long> axis, Optional<Long> block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DequantizeLinear.class, List.of(x, x_scale, x_zero_point), List.of(axis, block_size));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Det(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Det.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T2> Tensor<T2> DictVectorizer(Map<?, ?> X, Optional<String[]> string_vocabulary, Optional<long[]> int64_vocabulary) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DictVectorizer.class, List.of(X), List.of(string_vocabulary, int64_vocabulary));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Div(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Div.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public record DropoutResult<T>(Tensor<T> output, Tensor<Boolean> mask) { }
    public static <T, T1> DropoutResult<T> Dropout(Tensor<T> data, Optional<Tensor<T1>> ratio, Optional<Tensor<Boolean>> training_mode, Optional<Long> seed) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Dropout.class, List.of(data, ratio, training_mode), List.of(seed));
        Object[] resultArray = (Object[]) result;
        return new DropoutResult<>((Tensor<T>)resultArray[0], (Tensor<Boolean>)resultArray[1]);
    }

    public record DynamicQuantizeLinearResult(Tensor<Byte> y, Tensor<Float> y_scale, Tensor<Byte> y_zero_point) { }
    public static DynamicQuantizeLinearResult DynamicQuantizeLinear(Tensor<Float> x) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DynamicQuantizeLinear.class, List.of(x), List.of());
        Object[] resultArray = (Object[]) result;
        return new DynamicQuantizeLinearResult((Tensor<Byte>)resultArray[0], (Tensor<Float>)resultArray[1], (Tensor<Byte>)resultArray[2]);
    }

    public static <T> Tensor<T> Einsum(List<Tensor<T>> Inputs, String equation) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Einsum.class, List.of(Inputs), List.of(equation));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Elu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Elu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<Boolean> Equal(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Equal.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<T> Erf(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Erf.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Exp(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Exp.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Expand(Tensor<T> input, Tensor<Long> shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Expand.class, List.of(input, shape), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> EyeLike(Tensor<T1> input, Optional<Long> dtype, Optional<Long> k) {
        Object result = OnnxInterpreter.interpret(OnnxOps.EyeLike.class, List.of(input), List.of(dtype, k));
        return (Tensor<T2>) result;
    }

    public static <T1> Tensor<Float> FeatureVectorizer(List<Tensor<T1>> X, Optional<long[]> inputdimensions) {
        Object result = OnnxInterpreter.interpret(OnnxOps.FeatureVectorizer.class, List.of(X), List.of(inputdimensions));
        return (Tensor<Float>) result;
    }

    public static <T> Tensor<T> Flatten(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Flatten.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Floor(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Floor.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public record GRUResult<T>(Tensor<T> Y, Tensor<T> Y_h) { }
    public static <T> GRUResult<T> GRU(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<Integer>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Long> layout, Optional<float[]> activation_alpha, Optional<Long> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Long> linear_before_reset, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GRU.class, List.of(X, W, R, B, sequence_lens, initial_h), List.of(layout, activation_alpha, hidden_size, activation_beta, activations, linear_before_reset, clip, direction));
        Object[] resultArray = (Object[]) result;
        return new GRUResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1]);
    }

    public static <T, Tind> Tensor<T> Gather(Tensor<T> data, Tensor<Tind> indices, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gather.class, List.of(data, indices), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <T, Tind> Tensor<T> GatherElements(Tensor<T> data, Tensor<Tind> indices, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GatherElements.class, List.of(data, indices), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> GatherND(Tensor<T> data, Tensor<Long> indices, Optional<Long> batch_dims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GatherND.class, List.of(data, indices), List.of(batch_dims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Gelu(Tensor<T> X, Optional<String> approximate) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gelu.class, List.of(X), List.of(approximate));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Gemm(Tensor<T> A, Tensor<T> B, Optional<Tensor<T>> C, Optional<Float> alpha, Optional<Long> transB, Optional<Float> beta, Optional<Long> transA) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gemm.class, List.of(A, B, C), List.of(alpha, transB, beta, transA));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> GlobalAveragePool(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalAveragePool.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> GlobalLpPool(Tensor<T> X, Optional<Long> p) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalLpPool.class, List.of(X), List.of(p));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> GlobalMaxPool(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalMaxPool.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2> List<Tensor<T2>> Gradient(List<Tensor<T1>> Inputs, String y, Optional<String[]> zs, String[] xs) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gradient.class, List.of(Inputs), List.of(y, zs, xs));
        return (List<Tensor<T2>>) result;
    }

    public static <T> Tensor<Boolean> Greater(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Greater.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<Boolean> GreaterOrEqual(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GreaterOrEqual.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T1, T2> Tensor<T1> GridSample(Tensor<T1> X, Tensor<T2> grid, Optional<String> mode, Optional<Long> align_corners, Optional<String> padding_mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GridSample.class, List.of(X, grid), List.of(mode, align_corners, padding_mode));
        return (Tensor<T1>) result;
    }

    public static <T> Tensor<T> GroupNormalization(Tensor<T> X, Tensor<T> scale, Tensor<T> bias, Optional<Float> epsilon, Optional<Long> stash_type, long num_groups) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GroupNormalization.class, List.of(X, scale, bias), List.of(epsilon, stash_type, num_groups));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> HammingWindow(Tensor<T1> size, Optional<Long> periodic, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HammingWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }

    public static <T1, T2> Tensor<T2> HannWindow(Tensor<T1> size, Optional<Long> periodic, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HannWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> HardSigmoid(Tensor<T> X, Optional<Float> alpha, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HardSigmoid.class, List.of(X), List.of(alpha, beta));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> HardSwish(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HardSwish.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Hardmax(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Hardmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <V> V Identity(V input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Identity.class, List.of(input), List.of());
        return (V) result;
    }

    public static Tensor<Byte> ImageDecoder(Tensor<Byte> encoded_stream, Optional<String> pixel_format) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ImageDecoder.class, List.of(encoded_stream), List.of(pixel_format));
        return (Tensor<Byte>) result;
    }

    public static <T> Tensor<T> Imputer(Tensor<T> X, Optional<Long> replaced_value_int64, Optional<Float> replaced_value_float, Optional<long[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Imputer.class, List.of(X), List.of(replaced_value_int64, replaced_value_float, imputed_value_int64s, imputed_value_floats));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> InstanceNormalization(Tensor<T> input, Tensor<T> scale, Tensor<T> B, Optional<Float> epsilon) {
        Object result = OnnxInterpreter.interpret(OnnxOps.InstanceNormalization.class, List.of(input, scale, B), List.of(epsilon));
        return (Tensor<T>) result;
    }

    public static <T1> Tensor<Boolean> IsInf(Tensor<T1> X, Optional<Long> detect_negative, Optional<Long> detect_positive) {
        Object result = OnnxInterpreter.interpret(OnnxOps.IsInf.class, List.of(X), List.of(detect_negative, detect_positive));
        return (Tensor<Boolean>) result;
    }

    public static <T1> Tensor<Boolean> IsNaN(Tensor<T1> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.IsNaN.class, List.of(X), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<T> LRN(Tensor<T> X, long size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LRN.class, List.of(X), List.of(size, alpha, bias, beta));
        return (Tensor<T>) result;
    }

    public record LSTMResult<T>(Tensor<T> Y, Tensor<T> Y_h, Tensor<T> Y_c) { }
    public static <T> LSTMResult<T> LSTM(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<Integer>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Tensor<T>> initial_c, Optional<Tensor<T>> P, Optional<Long> layout, Optional<Long> input_forget, Optional<float[]> activation_alpha, Optional<Long> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LSTM.class, List.of(X, W, R, B, sequence_lens, initial_h, initial_c, P), List.of(layout, input_forget, activation_alpha, hidden_size, activation_beta, activations, clip, direction));
        Object[] resultArray = (Object[]) result;
        return new LSTMResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1], (Tensor<T>)resultArray[2]);
    }

    public static <T1, T2> Tensor<T2> LabelEncoder(Tensor<T1> X, Optional<String[]> values_strings, Optional<long[]> keys_int64s, Optional<byte[]> keys_tensor, Optional<String[]> keys_strings, Optional<Float> default_float, Optional<float[]> keys_floats, Optional<byte[]> default_tensor, Optional<Long> default_int64, Optional<byte[]> values_tensor, Optional<long[]> values_int64s, Optional<String> default_string, Optional<float[]> values_floats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LabelEncoder.class, List.of(X), List.of(values_strings, keys_int64s, keys_tensor, keys_strings, default_float, keys_floats, default_tensor, default_int64, values_tensor, values_int64s, default_string, values_floats));
        return (Tensor<T2>) result;
    }

    public record LayerNormalizationResult<T, U>(Tensor<T> Y, Tensor<U> Mean, Tensor<U> InvStdDev) { }
    public static <T, U> LayerNormalizationResult<T, U> LayerNormalization(Tensor<T> X, Tensor<T> Scale, Optional<Tensor<T>> B, Optional<Float> epsilon, Optional<Long> stash_type, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LayerNormalization.class, List.of(X, Scale, B), List.of(epsilon, stash_type, axis));
        Object[] resultArray = (Object[]) result;
        return new LayerNormalizationResult<>((Tensor<T>)resultArray[0], (Tensor<U>)resultArray[1], (Tensor<U>)resultArray[2]);
    }

    public static <T> Tensor<T> LeakyRelu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LeakyRelu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<Boolean> Less(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Less.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<Boolean> LessOrEqual(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LessOrEqual.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public record LinearClassifierResult<T2>(Tensor<T2> Y, Tensor<Float> Z) { }
    public static <T1, T2> LinearClassifierResult<T2> LinearClassifier(Tensor<T1> X, Optional<long[]> classlabels_ints, Optional<String> post_transform, float[] coefficients, Optional<Long> multi_class, Optional<float[]> intercepts, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LinearClassifier.class, List.of(X), List.of(classlabels_ints, post_transform, coefficients, multi_class, intercepts, classlabels_strings));
        Object[] resultArray = (Object[]) result;
        return new LinearClassifierResult<>((Tensor<T2>)resultArray[0], (Tensor<Float>)resultArray[1]);
    }

    public static <T> Tensor<Float> LinearRegressor(Tensor<T> X, Optional<String> post_transform, Optional<float[]> coefficients, Optional<Long> targets, Optional<float[]> intercepts) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LinearRegressor.class, List.of(X), List.of(post_transform, coefficients, targets, intercepts));
        return (Tensor<Float>) result;
    }

    public static <T> Tensor<T> Log(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Log.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> LogSoftmax(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LogSoftmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> LpNormalization(Tensor<T> input, Optional<Long> p, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LpNormalization.class, List.of(input), List.of(p, axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> LpPool(Tensor<T> X, Optional<Long> p, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<Long> ceil_mode, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LpPool.class, List.of(X), List.of(p, pads, dilations, auto_pad, ceil_mode, strides, kernel_shape));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> MatMul(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMul.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<Integer> MatMulInteger(Tensor<T1> A, Tensor<T2> B, Optional<Tensor<T1>> a_zero_point, Optional<Tensor<T2>> b_zero_point) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMulInteger.class, List.of(A, B, a_zero_point, b_zero_point), List.of());
        return (Tensor<Integer>) result;
    }

    public static <T> Tensor<T> Max(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Max.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    public record MaxPoolResult<T>(Tensor<T> Y, Tensor<Long> Indices) { }
    public static <T> MaxPoolResult<T> MaxPool(Tensor<T> X, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<Long> ceil_mode, Optional<Long> storage_order, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxPool.class, List.of(X), List.of(pads, dilations, auto_pad, ceil_mode, storage_order, strides, kernel_shape));
        Object[] resultArray = (Object[]) result;
        return new MaxPoolResult<>((Tensor<T>)resultArray[0], (Tensor<Long>)resultArray[1]);
    }

    public static <T> Tensor<T> MaxRoiPool(Tensor<T> X, Tensor<T> rois, Optional<Float> spatial_scale, long[] pooled_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxRoiPool.class, List.of(X, rois), List.of(spatial_scale, pooled_shape));
        return (Tensor<T>) result;
    }

    public static <T1> Tensor<T1> MaxUnpool(Tensor<T1> X, Tensor<Long> I, Optional<Tensor<Long>> output_shape, Optional<long[]> pads, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxUnpool.class, List.of(X, I, output_shape), List.of(pads, strides, kernel_shape));
        return (Tensor<T1>) result;
    }

    public static <T> Tensor<T> Mean(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mean.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> MeanVarianceNormalization(Tensor<T> X, Optional<long[]> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MeanVarianceNormalization.class, List.of(X), List.of(axes));
        return (Tensor<T>) result;
    }

    public static <T1, T2, T3> Tensor<T3> MelWeightMatrix(Tensor<T1> num_mel_bins, Tensor<T1> dft_length, Tensor<T1> sample_rate, Tensor<T2> lower_edge_hertz, Tensor<T2> upper_edge_hertz, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MelWeightMatrix.class, List.of(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz), List.of(output_datatype));
        return (Tensor<T3>) result;
    }

    public static <T> Tensor<T> Min(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Min.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Mish(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mish.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Mod(Tensor<T> A, Tensor<T> B, Optional<Long> fmod) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mod.class, List.of(A, B), List.of(fmod));
        return (Tensor<T>) result;
    }

    public static <T1, T3> List<Tensor<T3>> Momentum(Tensor<T1> R, Tensor<Long> T, List<Tensor<T3>> inputs, String mode, float norm_coefficient, float alpha, float beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Momentum.class, List.of(R, T, inputs), List.of(mode, norm_coefficient, alpha, beta));
        return (List<Tensor<T3>>) result;
    }

    public static <T> Tensor<T> Mul(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mul.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> Multinomial(Tensor<T1> input, Optional<Float> seed, Optional<Long> sample_size, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Multinomial.class, List.of(input), List.of(seed, sample_size, dtype));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Neg(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Neg.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T, Tind> Tensor<T> NegativeLogLikelihoodLoss(Tensor<T> input, Tensor<Tind> target, Optional<Tensor<T>> weight, Optional<Long> ignore_index, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NegativeLogLikelihoodLoss.class, List.of(input, target, weight), List.of(ignore_index, reduction));
        return (Tensor<T>) result;
    }

    public static Tensor<Long> NonMaxSuppression(Tensor<Float> boxes, Tensor<Float> scores, Optional<Tensor<Long>> max_output_boxes_per_class, Optional<Tensor<Float>> iou_threshold, Optional<Tensor<Float>> score_threshold, Optional<Long> center_point_box) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NonMaxSuppression.class, List.of(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold), List.of(center_point_box));
        return (Tensor<Long>) result;
    }

    public static <T> Tensor<Long> NonZero(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NonZero.class, List.of(X), List.of());
        return (Tensor<Long>) result;
    }

    public static <T> Tensor<Float> Normalizer(Tensor<T> X, Optional<String> norm) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Normalizer.class, List.of(X), List.of(norm));
        return (Tensor<Float>) result;
    }

    public static Tensor<Boolean> Not(Tensor<Boolean> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Not.class, List.of(X), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T1, T2, T3> Tensor<T3> OneHot(Tensor<T1> indices, Tensor<T2> depth, Tensor<T3> values, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OneHot.class, List.of(indices, depth, values), List.of(axis));
        return (Tensor<T3>) result;
    }

    public static <T> Tensor<Float> OneHotEncoder(Tensor<T> X, Optional<String[]> cats_strings, Optional<long[]> cats_int64s, Optional<Long> zeros) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OneHotEncoder.class, List.of(X), List.of(cats_strings, cats_int64s, zeros));
        return (Tensor<Float>) result;
    }

    public static <V, O> Optional<O> Optional(Optional<V> input, Optional<Object> type) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Optional.class, List.of(input), List.of(type));
        return (Optional<O>) result;
    }

    public static <O, V> V OptionalGetElement(O input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OptionalGetElement.class, List.of(input), List.of());
        return (V) result;
    }

    public static <O> Tensor<Boolean> OptionalHasElement(Optional<O> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OptionalHasElement.class, List.of(input), List.of());
        return (Tensor<Boolean>) result;
    }

    public static Tensor<Boolean> Or(Tensor<Boolean> A, Tensor<Boolean> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Or.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<T> PRelu(Tensor<T> X, Tensor<T> slope) {
        Object result = OnnxInterpreter.interpret(OnnxOps.PRelu.class, List.of(X, slope), List.of());
        return (Tensor<T>) result;
    }

    public static <T, Tind> Tensor<T> Pad(Tensor<T> data, Tensor<Long> pads, Optional<Tensor<T>> constant_value, Optional<Tensor<Tind>> axes, Optional<String> mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Pad.class, List.of(data, pads, constant_value, axes), List.of(mode));
        return (Tensor<T>) result;
    }

    public static <T, T1> Tensor<T> Pow(Tensor<T> X, Tensor<T1> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Pow.class, List.of(X, Y), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2, T3> Tensor<T3> QLinearConv(Tensor<T1> x, Tensor<Float> x_scale, Tensor<T1> x_zero_point, Tensor<T2> w, Tensor<Float> w_scale, Tensor<T2> w_zero_point, Tensor<Float> y_scale, Tensor<T3> y_zero_point, Optional<Tensor<Integer>> B, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QLinearConv.class, List.of(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T3>) result;
    }

    public static <TS, T1, T2, T3> Tensor<T3> QLinearMatMul(Tensor<T1> a, Tensor<TS> a_scale, Tensor<T1> a_zero_point, Tensor<T2> b, Tensor<TS> b_scale, Tensor<T2> b_zero_point, Tensor<TS> y_scale, Tensor<T3> y_zero_point) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QLinearMatMul.class, List.of(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point), List.of());
        return (Tensor<T3>) result;
    }

    public static <T1, T2> Tensor<T2> QuantizeLinear(Tensor<T1> x, Tensor<T1> y_scale, Optional<Tensor<T2>> y_zero_point, Optional<Long> output_dtype, Optional<Long> saturate, Optional<Long> axis, Optional<Long> block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QuantizeLinear.class, List.of(x, y_scale, y_zero_point), List.of(output_dtype, saturate, axis, block_size));
        return (Tensor<T2>) result;
    }

    public record RNNResult<T>(Tensor<T> Y, Tensor<T> Y_h) { }
    public static <T> RNNResult<T> RNN(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<Integer>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Long> layout, Optional<float[]> activation_alpha, Optional<Long> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RNN.class, List.of(X, W, R, B, sequence_lens, initial_h), List.of(layout, activation_alpha, hidden_size, activation_beta, activations, clip, direction));
        Object[] resultArray = (Object[]) result;
        return new RNNResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1]);
    }

    public static <T> Tensor<T> RandomNormal(long[] shape, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomNormal.class, List.of(), List.of(shape, seed, mean, scale, dtype));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> RandomNormalLike(Tensor<T1> input, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomNormalLike.class, List.of(input), List.of(seed, mean, scale, dtype));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> RandomUniform(Optional<Float> high, long[] shape, Optional<Float> seed, Optional<Float> low, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomUniform.class, List.of(), List.of(high, shape, seed, low, dtype));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T2> RandomUniformLike(Tensor<T1> input, Optional<Float> high, Optional<Float> seed, Optional<Float> low, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomUniformLike.class, List.of(input), List.of(high, seed, low, dtype));
        return (Tensor<T2>) result;
    }

    public static <T> Tensor<T> Range(Tensor<T> start, Tensor<T> limit, Tensor<T> delta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Range.class, List.of(start, limit, delta), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Reciprocal(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Reciprocal.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceL1(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceL1.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceL2(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceL2.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceLogSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceLogSum.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceLogSumExp(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceLogSumExp.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceMax(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMax.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceMean(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMean.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceMin(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMin.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceProd(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceProd.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceSum.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ReduceSumSquare(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceSumSquare.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    public static Tensor<Boolean> RegexFullMatch(Tensor<String> X, Optional<String> pattern) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RegexFullMatch.class, List.of(X), List.of(pattern));
        return (Tensor<Boolean>) result;
    }

    public static <T> Tensor<T> Relu(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Relu.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Reshape(Tensor<T> data, Tensor<Long> shape, Optional<Long> allowzero) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Reshape.class, List.of(data, shape), List.of(allowzero));
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T1> Resize(Tensor<T1> X, Optional<Tensor<T2>> roi, Optional<Tensor<Float>> scales, Optional<Tensor<Long>> sizes, Optional<String> mode, Optional<Float> extrapolation_value, Optional<String> nearest_mode, Optional<Long> antialias, Optional<Float> cubic_coeff_a, Optional<long[]> axes, Optional<String> coordinate_transformation_mode, Optional<String> keep_aspect_ratio_policy, Optional<Long> exclude_outside) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Resize.class, List.of(X, roi, scales, sizes), List.of(mode, extrapolation_value, nearest_mode, antialias, cubic_coeff_a, axes, coordinate_transformation_mode, keep_aspect_ratio_policy, exclude_outside));
        return (Tensor<T1>) result;
    }

    public static <T> Tensor<T> ReverseSequence(Tensor<T> input, Tensor<Long> sequence_lens, Optional<Long> time_axis, Optional<Long> batch_axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReverseSequence.class, List.of(input, sequence_lens), List.of(time_axis, batch_axis));
        return (Tensor<T>) result;
    }

    public static <T1> Tensor<T1> RoiAlign(Tensor<T1> X, Tensor<T1> rois, Tensor<Long> batch_indices, Optional<String> mode, Optional<Long> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Long> sampling_ratio, Optional<Long> output_height) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RoiAlign.class, List.of(X, rois, batch_indices), List.of(mode, output_width, spatial_scale, coordinate_transformation_mode, sampling_ratio, output_height));
        return (Tensor<T1>) result;
    }

    public static <T> Tensor<T> Round(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Round.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T1, T2> Tensor<T1> STFT(Tensor<T1> signal, Tensor<T2> frame_step, Optional<Tensor<T1>> window, Optional<Tensor<T2>> frame_length, Optional<Long> onesided) {
        Object result = OnnxInterpreter.interpret(OnnxOps.STFT.class, List.of(signal, frame_step, window, frame_length), List.of(onesided));
        return (Tensor<T1>) result;
    }

    public record SVMClassifierResult<T2>(Tensor<T2> Y, Tensor<Float> Z) { }
    public static <T1, T2> SVMClassifierResult<T2> SVMClassifier(Tensor<T1> X, Optional<float[]> prob_b, Optional<float[]> kernel_params, Optional<String> kernel_type, Optional<long[]> classlabels_ints, Optional<String> post_transform, Optional<float[]> rho, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<long[]> vectors_per_class, Optional<float[]> prob_a, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SVMClassifier.class, List.of(X), List.of(prob_b, kernel_params, kernel_type, classlabels_ints, post_transform, rho, coefficients, support_vectors, vectors_per_class, prob_a, classlabels_strings));
        Object[] resultArray = (Object[]) result;
        return new SVMClassifierResult<>((Tensor<T2>)resultArray[0], (Tensor<Float>)resultArray[1]);
    }

    public static <T> Tensor<Float> SVMRegressor(Tensor<T> X, Optional<String> kernel_type, Optional<float[]> kernel_params, Optional<Long> n_supports, Optional<float[]> rho, Optional<String> post_transform, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<Long> one_class) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SVMRegressor.class, List.of(X), List.of(kernel_type, kernel_params, n_supports, rho, post_transform, coefficients, support_vectors, one_class));
        return (Tensor<Float>) result;
    }

    public static <T> Tensor<Float> Scaler(Tensor<T> X, Optional<float[]> offset, Optional<float[]> scale) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Scaler.class, List.of(X), List.of(offset, scale));
        return (Tensor<Float>) result;
    }

    public static <T, Tind> Tensor<T> Scatter(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Scatter.class, List.of(data, indices, updates), List.of(axis));
        return (Tensor<T>) result;
    }

    public static <T, Tind> Tensor<T> ScatterElements(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<String> reduction, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ScatterElements.class, List.of(data, indices, updates), List.of(reduction, axis));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> ScatterND(Tensor<T> data, Tensor<Long> indices, Tensor<T> updates, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ScatterND.class, List.of(data, indices, updates), List.of(reduction));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Selu(Tensor<T> X, Optional<Float> alpha, Optional<Float> gamma) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Selu.class, List.of(X), List.of(alpha, gamma));
        return (Tensor<T>) result;
    }

    public static <S, T, I> Tensor<T> SequenceAt(List<Tensor<S>> input_sequence, Tensor<I> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceAt.class, List.of(input_sequence, position), List.of());
        return (Tensor<T>) result;
    }

    public static <T, S> List<Tensor<S>> SequenceConstruct(List<Tensor<T>> inputs) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceConstruct.class, List.of(inputs), List.of());
        return (List<Tensor<S>>) result;
    }

    public static <S> List<Tensor<S>> SequenceEmpty(Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceEmpty.class, List.of(), List.of(dtype));
        return (List<Tensor<S>>) result;
    }

    public static <S, I> List<Tensor<S>> SequenceErase(List<Tensor<S>> input_sequence, Optional<Tensor<I>> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceErase.class, List.of(input_sequence, position), List.of());
        return (List<Tensor<S>>) result;
    }

    public static <T, S, I> List<Tensor<S>> SequenceInsert(List<Tensor<S>> input_sequence, Tensor<T> tensor, Optional<Tensor<I>> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceInsert.class, List.of(input_sequence, tensor, position), List.of());
        return (List<Tensor<S>>) result;
    }

    public static <S> Tensor<Long> SequenceLength(List<Tensor<S>> input_sequence) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceLength.class, List.of(input_sequence), List.of());
        return (Tensor<Long>) result;
    }

    public static <T> Tensor<Long> Shape(Tensor<T> data, Optional<Long> start, Optional<Long> end) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Shape.class, List.of(data), List.of(start, end));
        return (Tensor<Long>) result;
    }

    public static <T> Tensor<T> Shrink(Tensor<T> input, Optional<Float> lambd, Optional<Float> bias) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Shrink.class, List.of(input), List.of(lambd, bias));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Sigmoid(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sigmoid.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Sign(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sign.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Sin(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sin.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Sinh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sinh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<Long> Size(Tensor<T> data) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Size.class, List.of(data), List.of());
        return (Tensor<Long>) result;
    }

    public static <T, Tind> Tensor<T> Slice(Tensor<T> data, Tensor<Tind> starts, Tensor<Tind> ends, Optional<Tensor<Tind>> axes, Optional<Tensor<Tind>> steps) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Slice.class, List.of(data, starts, ends, axes, steps), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Softmax(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    public record SoftmaxCrossEntropyLossResult<T>(Tensor<T> output, Tensor<T> log_prob) { }
    public static <T, Tind> SoftmaxCrossEntropyLossResult<T> SoftmaxCrossEntropyLoss(Tensor<T> scores, Tensor<Tind> labels, Optional<Tensor<T>> weights, Optional<Long> ignore_index, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SoftmaxCrossEntropyLoss.class, List.of(scores, labels, weights), List.of(ignore_index, reduction));
        Object[] resultArray = (Object[]) result;
        return new SoftmaxCrossEntropyLossResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1]);
    }

    public static <T> Tensor<T> Softplus(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softplus.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Softsign(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softsign.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> SpaceToDepth(Tensor<T> input, long blocksize) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SpaceToDepth.class, List.of(input), List.of(blocksize));
        return (Tensor<T>) result;
    }

    public static <T> List<Tensor<T>> Split(Tensor<T> input, Optional<Tensor<Long>> split, Optional<Long> num_outputs, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Split.class, List.of(input, split), List.of(num_outputs, axis));
        return (List<Tensor<T>>) result;
    }

    public static <T, I, S> List<Tensor<S>> SplitToSequence(Tensor<T> input, Optional<Tensor<I>> split, Optional<Long> keepdims, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SplitToSequence.class, List.of(input, split), List.of(keepdims, axis));
        return (List<Tensor<S>>) result;
    }

    public static <T> Tensor<T> Sqrt(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sqrt.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Squeeze(Tensor<T> data, Optional<Tensor<Long>> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Squeeze.class, List.of(data, axes), List.of());
        return (Tensor<T>) result;
    }

    public static Tensor<String> StringConcat(Tensor<String> X, Tensor<String> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringConcat.class, List.of(X, Y), List.of());
        return (Tensor<String>) result;
    }

    public static Tensor<String> StringNormalizer(Tensor<String> X, Optional<Long> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringNormalizer.class, List.of(X), List.of(is_case_sensitive, locale, stopwords, case_change_action));
        return (Tensor<String>) result;
    }

    public record StringSplitResult(Tensor<String> Y, Tensor<Long> Z) { }
    public static StringSplitResult StringSplit(Tensor<String> X, Optional<String> delimiter, Optional<Long> maxsplit) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringSplit.class, List.of(X), List.of(delimiter, maxsplit));
        Object[] resultArray = (Object[]) result;
        return new StringSplitResult((Tensor<String>)resultArray[0], (Tensor<Long>)resultArray[1]);
    }

    public static <T> Tensor<T> Sub(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sub.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Sum(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sum.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Tan(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tan.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Tanh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tanh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<Float> TfIdfVectorizer(Tensor<T> X, long[] ngram_counts, long min_gram_length, Optional<String[]> pool_strings, String mode, long max_gram_length, long max_skip_count, Optional<long[]> pool_int64s, Optional<float[]> weights, long[] ngram_indexes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TfIdfVectorizer.class, List.of(X), List.of(ngram_counts, min_gram_length, pool_strings, mode, max_gram_length, max_skip_count, pool_int64s, weights, ngram_indexes));
        return (Tensor<Float>) result;
    }

    public static <T> Tensor<T> ThresholdedRelu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ThresholdedRelu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Tile(Tensor<T> input, Tensor<Long> repeats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tile.class, List.of(input, repeats), List.of());
        return (Tensor<T>) result;
    }

    public record TopKResult<T>(Tensor<T> Values, Tensor<Long> Indices) { }
    public static <T> TopKResult<T> TopK(Tensor<T> X, Tensor<Long> K, Optional<Long> largest, Optional<Long> sorted, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TopK.class, List.of(X, K), List.of(largest, sorted, axis));
        Object[] resultArray = (Object[]) result;
        return new TopKResult<>((Tensor<T>)resultArray[0], (Tensor<Long>)resultArray[1]);
    }

    public static <T> Tensor<T> Transpose(Tensor<T> data, Optional<long[]> perm) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Transpose.class, List.of(data), List.of(perm));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> TreeEnsemble(Tensor<T> X, Optional<Long> aggregate_function, Optional<byte[]> nodes_hitrates, long[] nodes_featureids, long[] nodes_falseleafs, Optional<Long> post_transform, long[] nodes_trueleafs, byte[] nodes_modes, long[] nodes_falsenodeids, long[] nodes_truenodeids, byte[] leaf_weights, long[] leaf_targetids, long[] tree_roots, Optional<Long> n_targets, Optional<long[]> nodes_missing_value_tracks_true, Optional<byte[]> membership_values, byte[] nodes_splits) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsemble.class, List.of(X), List.of(aggregate_function, nodes_hitrates, nodes_featureids, nodes_falseleafs, post_transform, nodes_trueleafs, nodes_modes, nodes_falsenodeids, nodes_truenodeids, leaf_weights, leaf_targetids, tree_roots, n_targets, nodes_missing_value_tracks_true, membership_values, nodes_splits));
        return (Tensor<T>) result;
    }

    public record TreeEnsembleClassifierResult<T2>(Tensor<T2> Y, Tensor<Float> Z) { }
    public static <T1, T2> TreeEnsembleClassifierResult<T2> TreeEnsembleClassifier(Tensor<T1> X, Optional<long[]> classlabels_int64s, Optional<long[]> class_ids, Optional<float[]> nodes_hitrates, Optional<long[]> nodes_featureids, Optional<long[]> nodes_treeids, Optional<byte[]> class_weights_as_tensor, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<long[]> nodes_falsenodeids, Optional<String[]> classlabels_strings, Optional<long[]> nodes_truenodeids, Optional<long[]> nodes_nodeids, Optional<byte[]> nodes_hitrates_as_tensor, Optional<float[]> class_weights, Optional<byte[]> base_values_as_tensor, Optional<long[]> nodes_missing_value_tracks_true, Optional<long[]> class_nodeids, Optional<long[]> class_treeids, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<byte[]> nodes_values_as_tensor) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsembleClassifier.class, List.of(X), List.of(classlabels_int64s, class_ids, nodes_hitrates, nodes_featureids, nodes_treeids, class_weights_as_tensor, post_transform, nodes_modes, nodes_falsenodeids, classlabels_strings, nodes_truenodeids, nodes_nodeids, nodes_hitrates_as_tensor, class_weights, base_values_as_tensor, nodes_missing_value_tracks_true, class_nodeids, class_treeids, base_values, nodes_values, nodes_values_as_tensor));
        Object[] resultArray = (Object[]) result;
        return new TreeEnsembleClassifierResult<>((Tensor<T2>)resultArray[0], (Tensor<Float>)resultArray[1]);
    }

    public static <T> Tensor<Float> TreeEnsembleRegressor(Tensor<T> X, Optional<String> aggregate_function, Optional<float[]> nodes_hitrates, Optional<byte[]> target_weights_as_tensor, Optional<long[]> nodes_featureids, Optional<long[]> target_treeids, Optional<long[]> nodes_treeids, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<float[]> target_weights, Optional<long[]> nodes_falsenodeids, Optional<long[]> target_ids, Optional<long[]> nodes_truenodeids, Optional<long[]> target_nodeids, Optional<long[]> nodes_nodeids, Optional<byte[]> nodes_hitrates_as_tensor, Optional<byte[]> base_values_as_tensor, Optional<Long> n_targets, Optional<long[]> nodes_missing_value_tracks_true, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<byte[]> nodes_values_as_tensor) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsembleRegressor.class, List.of(X), List.of(aggregate_function, nodes_hitrates, target_weights_as_tensor, nodes_featureids, target_treeids, nodes_treeids, post_transform, nodes_modes, target_weights, nodes_falsenodeids, target_ids, nodes_truenodeids, target_nodeids, nodes_nodeids, nodes_hitrates_as_tensor, base_values_as_tensor, n_targets, nodes_missing_value_tracks_true, base_values, nodes_values, nodes_values_as_tensor));
        return (Tensor<Float>) result;
    }

    public static <T> Tensor<T> Trilu(Tensor<T> input, Optional<Tensor<Long>> k, Optional<Long> upper) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Trilu.class, List.of(input, k), List.of(upper));
        return (Tensor<T>) result;
    }

    public record UniqueResult<T>(Tensor<T> Y, Tensor<Long> indices, Tensor<Long> inverse_indices, Tensor<Long> counts) { }
    public static <T> UniqueResult<T> Unique(Tensor<T> X, Optional<Long> sorted, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Unique.class, List.of(X), List.of(sorted, axis));
        Object[] resultArray = (Object[]) result;
        return new UniqueResult<>((Tensor<T>)resultArray[0], (Tensor<Long>)resultArray[1], (Tensor<Long>)resultArray[2], (Tensor<Long>)resultArray[3]);
    }

    public static <T> Tensor<T> Unsqueeze(Tensor<T> data, Tensor<Long> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Unsqueeze.class, List.of(data, axes), List.of());
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Upsample(Tensor<T> X, Tensor<Float> scales, Optional<String> mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Upsample.class, List.of(X, scales), List.of(mode));
        return (Tensor<T>) result;
    }

    public static <T> Tensor<T> Where(Tensor<Boolean> condition, Tensor<T> X, Tensor<T> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Where.class, List.of(condition, X, Y), List.of());
        return (Tensor<T>) result;
    }

    public static Tensor<Boolean> Xor(Tensor<Boolean> A, Tensor<Boolean> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Xor.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public static <T> List<Map<T, Float>> ZipMap(Tensor<Float> X, Optional<long[]> classlabels_int64s, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ZipMap.class, List.of(X), List.of(classlabels_int64s, classlabels_strings));
        return (List<Map<T, Float>>) result;
    }

}
