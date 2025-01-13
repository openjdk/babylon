// Auto-generated from ONNX op schema

package oracle.code.onnx;

import oracle.code.onnx.ir.OnnxOps;

import java.util.Optional;
import java.util.List;

@SuppressWarnings({"unchecked", "OptionalUsedAsFieldOrParameterType"})
public final class OnnxOperators {
    
    private OnnxOperators() {}
    
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
    
    public static <T1, T2, T3> List<Tensor<T3>> Adagrad(Tensor<T1> R, Tensor<T2> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> decay_factor, Optional<Float> norm_coefficient) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Adagrad.class, List.of(R, T, inputs), List.of(epsilon, decay_factor, norm_coefficient));
        return (List<Tensor<T3>>) result;
    }
    
    public static <T1, T2, T3> List<Tensor<T3>> Adam(Tensor<T1> R, Tensor<T2> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> norm_coefficient_post, Optional<Float> norm_coefficient, Optional<Float> alpha, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Adam.class, List.of(R, T, inputs), List.of(epsilon, norm_coefficient_post, norm_coefficient, alpha, beta));
        return (List<Tensor<T3>>) result;
    }
    
    public static <T> Tensor<T> Add(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Add.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T1> AffineGrid(Tensor<T1> theta, Tensor<T2> size, Optional<Integer> align_corners) {
        Object result = OnnxInterpreter.interpret(OnnxOps.AffineGrid.class, List.of(theta, size), List.of(align_corners));
        return (Tensor<T1>) result;
    }
    
    public static <T, T1> Tensor<T1> And(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.And.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<Long> ArgMax(Tensor<T> data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArgMax.class, List.of(data), List.of(keepdims, select_last_index, axis));
        return (Tensor<Long>) result;
    }
    
    public static <T> Tensor<Long> ArgMin(Tensor<T> data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
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
    
    public static <T> Tensor<T> AveragePool(Tensor<T> X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> count_include_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.AveragePool.class, List.of(X), List.of(pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape));
        return (Tensor<T>) result;
    }
    
    public static <T, T1, T2> List<Tensor<T>> BatchNormalization(Tensor<T> X, Tensor<T1> scale, Tensor<T1> B, Tensor<T2> input_mean, Tensor<T2> input_var, Optional<Float> epsilon, Optional<Integer> training_mode, Optional<Float> momentum) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BatchNormalization.class, List.of(X, scale, B, input_mean, input_var), List.of(epsilon, training_mode, momentum));
        return (List<Tensor<T>>) result;
    }
    
    public static <T1, T2> Tensor<T2> Bernoulli(Tensor<T1> input, Optional<Float> seed, Optional<Integer> dtype) {
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
    
    public static <T1, T2> Tensor<T2> BlackmanWindow(Tensor<T1> size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BlackmanWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }
    
    public static <T1, T2> Tensor<T2> Cast(Tensor<T1> input, Optional<Integer> saturate, int to) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cast.class, List.of(input), List.of(saturate, to));
        return (Tensor<T2>) result;
    }
    
    public static <T1, T2> Tensor<T2> CastLike(Tensor<T1> input, Tensor<T2> target_type, Optional<Integer> saturate) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CastLike.class, List.of(input, target_type), List.of(saturate));
        return (Tensor<T2>) result;
    }
    
    public static <T1, T2> Tensor<T2> CastMap(Tensor<T1> X, Optional<String> map_form, Optional<String> cast_to, Optional<Integer> max_map) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CastMap.class, List.of(X), List.of(map_form, cast_to, max_map));
        return (Tensor<T2>) result;
    }
    
    public static <T1, T2> Tensor<T2> CategoryMapper(Tensor<T1> X, Optional<int[]> cats_int64s, Optional<String[]> cats_strings, Optional<Integer> default_int64, Optional<String> default_string) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CategoryMapper.class, List.of(X), List.of(cats_int64s, cats_strings, default_int64, default_string));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Ceil(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Ceil.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Celu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Celu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }
    
    public static <T, Tind> Tensor<T> CenterCropPad(Tensor<T> input_data, Tensor<Tind> shape, Optional<int[]> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CenterCropPad.class, List.of(input_data, shape), List.of(axes));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Clip(Tensor<T> input, Optional<Tensor<T>> min, Optional<Tensor<T>> max) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Clip.class, List.of(input, min, max), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Col2Im(Tensor<T> input, Tensor<Long> image_shape, Tensor<Long> block_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Col2Im.class, List.of(input, image_shape, block_shape), List.of(pads, dilations, strides));
        return (Tensor<T>) result;
    }
    
    public static <T, T1> Tensor<T> Compress(Tensor<T> input, Tensor<T1> condition, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Compress.class, List.of(input, condition), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Concat(List<Tensor<T>> inputs, int axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Concat.class, List.of(inputs), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <S, T> Tensor<T> ConcatFromSequence(Tensor<S> input_sequence, int axis, Optional<Integer> new_axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConcatFromSequence.class, List.of(input_sequence), List.of(axis, new_axis));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Constant(Optional<Integer> value_int, Optional<float[]> value_floats, Optional<String[]> value_strings, Optional<Float> value_float, Optional<String> value_string, Optional<int[]> value_ints, Optional<Object> sparse_value, Optional<Tensor<?>> value) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Constant.class, List.of(), List.of(value_int, value_floats, value_strings, value_float, value_string, value_ints, sparse_value, value));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> ConstantOfShape(Tensor<T1> input, Optional<Tensor<?>> value) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConstantOfShape.class, List.of(input), List.of(value));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Conv(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Conv.class, List.of(X, W, B), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2, T3> Tensor<T3> ConvInteger(Tensor<T1> x, Tensor<T2> w, Optional<Tensor<T1>> x_zero_point, Optional<Tensor<T2>> w_zero_point, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConvInteger.class, List.of(x, w, x_zero_point, w_zero_point), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T3>) result;
    }
    
    public static <T> Tensor<T> ConvTranspose(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<int[]> output_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape, Optional<int[]> output_padding) {
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
    
    public static <T, T2> Tensor<T> CumSum(Tensor<T> x, Tensor<T2> axis, Optional<Integer> exclusive, Optional<Integer> reverse) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CumSum.class, List.of(x, axis), List.of(exclusive, reverse));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T1> DFT(Tensor<T1> input, Optional<Tensor<T2>> dft_length, Optional<Tensor<Long>> axis, Optional<Integer> inverse, Optional<Integer> onesided) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DFT.class, List.of(input, dft_length, axis), List.of(inverse, onesided));
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> DeformConv(Tensor<T> X, Tensor<T> W, Tensor<T> offset, Optional<Tensor<T>> B, Optional<Tensor<T>> mask, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides, Optional<Integer> offset_group, Optional<Integer> group, Optional<int[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DeformConv.class, List.of(X, W, offset, B, mask), List.of(pads, dilations, strides, offset_group, group, kernel_shape));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> DepthToSpace(Tensor<T> input, Optional<String> mode, int blocksize) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DepthToSpace.class, List.of(input), List.of(mode, blocksize));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> DequantizeLinear(Tensor<T1> x, Tensor<T2> x_scale, Optional<Tensor<T1>> x_zero_point, Optional<Integer> axis, Optional<Integer> block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DequantizeLinear.class, List.of(x, x_scale, x_zero_point), List.of(axis, block_size));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Det(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Det.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> DictVectorizer(Tensor<T1> X, Optional<String[]> string_vocabulary, Optional<int[]> int64_vocabulary) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DictVectorizer.class, List.of(X), List.of(string_vocabulary, int64_vocabulary));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Div(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Div.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, T1, T2> List<Tensor<T>> Dropout(Tensor<T> data, Optional<Tensor<T1>> ratio, Optional<Tensor<T2>> training_mode, Optional<Integer> seed) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Dropout.class, List.of(data, ratio, training_mode), List.of(seed));
        return (List<Tensor<T>>) result;
    }
    
    public static <T1, T2> List<Tensor<T2>> DynamicQuantizeLinear(Tensor<T1> x) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DynamicQuantizeLinear.class, List.of(x), List.of());
        return (List<Tensor<T2>>) result;
    }
    
    public static <T> Tensor<T> Einsum(List<Tensor<T>> Inputs, String equation) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Einsum.class, List.of(Inputs), List.of(equation));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Elu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Elu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }
    
    public static <T, T1> Tensor<T1> Equal(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Equal.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
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
    
    public static <T1, T2> Tensor<T2> EyeLike(Tensor<T1> input, Optional<Integer> dtype, Optional<Integer> k) {
        Object result = OnnxInterpreter.interpret(OnnxOps.EyeLike.class, List.of(input), List.of(dtype, k));
        return (Tensor<T2>) result;
    }
    
    public static <T1> Tensor<Float> FeatureVectorizer(List<Tensor<T1>> X, Optional<int[]> inputdimensions) {
        Object result = OnnxInterpreter.interpret(OnnxOps.FeatureVectorizer.class, List.of(X), List.of(inputdimensions));
        return (Tensor<Float>) result;
    }
    
    public static <T> Tensor<T> Flatten(Tensor<T> input, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Flatten.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Floor(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Floor.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, T1> List<Tensor<T>> GRU(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<T1>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Integer> layout, Optional<float[]> activation_alpha, Optional<Integer> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Integer> linear_before_reset, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GRU.class, List.of(X, W, R, B, sequence_lens, initial_h), List.of(layout, activation_alpha, hidden_size, activation_beta, activations, linear_before_reset, clip, direction));
        return (List<Tensor<T>>) result;
    }
    
    public static <T, Tind> Tensor<T> Gather(Tensor<T> data, Tensor<Tind> indices, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gather.class, List.of(data, indices), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T, Tind> Tensor<T> GatherElements(Tensor<T> data, Tensor<Tind> indices, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GatherElements.class, List.of(data, indices), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> GatherND(Tensor<T> data, Tensor<Long> indices, Optional<Integer> batch_dims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GatherND.class, List.of(data, indices), List.of(batch_dims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Gelu(Tensor<T> X, Optional<String> approximate) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gelu.class, List.of(X), List.of(approximate));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Gemm(Tensor<T> A, Tensor<T> B, Optional<Tensor<T>> C, Optional<Float> alpha, Optional<Integer> transB, Optional<Float> beta, Optional<Integer> transA) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gemm.class, List.of(A, B, C), List.of(alpha, transB, beta, transA));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> GlobalAveragePool(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalAveragePool.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> GlobalLpPool(Tensor<T> X, Optional<Integer> p) {
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
    
    public static <T, T1> Tensor<T1> Greater(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Greater.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T, T1> Tensor<T1> GreaterOrEqual(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GreaterOrEqual.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T1, T2> Tensor<T1> GridSample(Tensor<T1> X, Tensor<T2> grid, Optional<String> mode, Optional<Integer> align_corners, Optional<String> padding_mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GridSample.class, List.of(X, grid), List.of(mode, align_corners, padding_mode));
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> GroupNormalization(Tensor<T> X, Tensor<T> scale, Tensor<T> bias, Optional<Float> epsilon, Optional<Integer> stash_type, int num_groups) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GroupNormalization.class, List.of(X, scale, bias), List.of(epsilon, stash_type, num_groups));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> HammingWindow(Tensor<T1> size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HammingWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }
    
    public static <T1, T2> Tensor<T2> HannWindow(Tensor<T1> size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
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
    
    public static <T> Tensor<T> Hardmax(Tensor<T> input, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Hardmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <V> Tensor<V> Identity(Tensor<V> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Identity.class, List.of(input), List.of());
        return (Tensor<V>) result;
    }
    
    public static <T1, T2> Tensor<T2> ImageDecoder(Tensor<T1> encoded_stream, Optional<String> pixel_format) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ImageDecoder.class, List.of(encoded_stream), List.of(pixel_format));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Imputer(Tensor<T> X, Optional<Integer> replaced_value_int64, Optional<Float> replaced_value_float, Optional<int[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Imputer.class, List.of(X), List.of(replaced_value_int64, replaced_value_float, imputed_value_int64s, imputed_value_floats));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> InstanceNormalization(Tensor<T> input, Tensor<T> scale, Tensor<T> B, Optional<Float> epsilon) {
        Object result = OnnxInterpreter.interpret(OnnxOps.InstanceNormalization.class, List.of(input, scale, B), List.of(epsilon));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> IsInf(Tensor<T1> X, Optional<Integer> detect_negative, Optional<Integer> detect_positive) {
        Object result = OnnxInterpreter.interpret(OnnxOps.IsInf.class, List.of(X), List.of(detect_negative, detect_positive));
        return (Tensor<T2>) result;
    }
    
    public static <T1, T2> Tensor<T2> IsNaN(Tensor<T1> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.IsNaN.class, List.of(X), List.of());
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> LRN(Tensor<T> X, int size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LRN.class, List.of(X), List.of(size, alpha, bias, beta));
        return (Tensor<T>) result;
    }
    
    public static <T, T1> List<Tensor<T>> LSTM(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<T1>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Tensor<T>> initial_c, Optional<Tensor<T>> P, Optional<Integer> layout, Optional<Integer> input_forget, Optional<float[]> activation_alpha, Optional<Integer> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LSTM.class, List.of(X, W, R, B, sequence_lens, initial_h, initial_c, P), List.of(layout, input_forget, activation_alpha, hidden_size, activation_beta, activations, clip, direction));
        return (List<Tensor<T>>) result;
    }
    
    public static <T1, T2> Tensor<T2> LabelEncoder(Tensor<T1> X, Optional<String[]> values_strings, Optional<int[]> keys_int64s, Optional<Tensor<?>> keys_tensor, Optional<String[]> keys_strings, Optional<Float> default_float, Optional<float[]> keys_floats, Optional<Tensor<?>> default_tensor, Optional<Integer> default_int64, Optional<Tensor<?>> values_tensor, Optional<int[]> values_int64s, Optional<String> default_string, Optional<float[]> values_floats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LabelEncoder.class, List.of(X), List.of(values_strings, keys_int64s, keys_tensor, keys_strings, default_float, keys_floats, default_tensor, default_int64, values_tensor, values_int64s, default_string, values_floats));
        return (Tensor<T2>) result;
    }
    
    public static <T, U> List<Tensor<T>> LayerNormalization(Tensor<T> X, Tensor<T> Scale, Optional<Tensor<T>> B, Optional<Float> epsilon, Optional<Integer> stash_type, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LayerNormalization.class, List.of(X, Scale, B), List.of(epsilon, stash_type, axis));
        return (List<Tensor<T>>) result;
    }
    
    public static <T> Tensor<T> LeakyRelu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LeakyRelu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }
    
    public static <T, T1> Tensor<T1> Less(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Less.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T, T1> Tensor<T1> LessOrEqual(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LessOrEqual.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T1, T2> List<Tensor<T2>> LinearClassifier(Tensor<T1> X, Optional<int[]> classlabels_ints, Optional<String> post_transform, float[] coefficients, Optional<Integer> multi_class, Optional<float[]> intercepts, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LinearClassifier.class, List.of(X), List.of(classlabels_ints, post_transform, coefficients, multi_class, intercepts, classlabels_strings));
        return (List<Tensor<T2>>) result;
    }
    
    public static <T> Tensor<Float> LinearRegressor(Tensor<T> X, Optional<String> post_transform, Optional<float[]> coefficients, Optional<Integer> targets, Optional<float[]> intercepts) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LinearRegressor.class, List.of(X), List.of(post_transform, coefficients, targets, intercepts));
        return (Tensor<Float>) result;
    }
    
    public static <T> Tensor<T> Log(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Log.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> LogSoftmax(Tensor<T> input, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LogSoftmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> LpNormalization(Tensor<T> input, Optional<Integer> p, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LpNormalization.class, List.of(input), List.of(p, axis));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> LpPool(Tensor<T> X, Optional<Integer> p, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LpPool.class, List.of(X), List.of(p, pads, dilations, auto_pad, ceil_mode, strides, kernel_shape));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> MatMul(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMul.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T1, T2, T3> Tensor<T3> MatMulInteger(Tensor<T1> A, Tensor<T2> B, Optional<Tensor<T1>> a_zero_point, Optional<Tensor<T2>> b_zero_point) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMulInteger.class, List.of(A, B, a_zero_point, b_zero_point), List.of());
        return (Tensor<T3>) result;
    }
    
    public static <T> Tensor<T> Max(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Max.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, I> List<Tensor<T>> MaxPool(Tensor<T> X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<Integer> storage_order, Optional<int[]> strides, int[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxPool.class, List.of(X), List.of(pads, dilations, auto_pad, ceil_mode, storage_order, strides, kernel_shape));
        return (List<Tensor<T>>) result;
    }
    
    public static <T> Tensor<T> MaxRoiPool(Tensor<T> X, Tensor<T> rois, Optional<Float> spatial_scale, int[] pooled_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxRoiPool.class, List.of(X, rois), List.of(spatial_scale, pooled_shape));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T1> MaxUnpool(Tensor<T1> X, Tensor<T2> I, Optional<Tensor<T2>> output_shape, Optional<int[]> pads, Optional<int[]> strides, int[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxUnpool.class, List.of(X, I, output_shape), List.of(pads, strides, kernel_shape));
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> Mean(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mean.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> MeanVarianceNormalization(Tensor<T> X, Optional<int[]> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MeanVarianceNormalization.class, List.of(X), List.of(axes));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2, T3> Tensor<T3> MelWeightMatrix(Tensor<T1> num_mel_bins, Tensor<T1> dft_length, Tensor<T1> sample_rate, Tensor<T2> lower_edge_hertz, Tensor<T2> upper_edge_hertz, Optional<Integer> output_datatype) {
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
    
    public static <T> Tensor<T> Mod(Tensor<T> A, Tensor<T> B, Optional<Integer> fmod) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mod.class, List.of(A, B), List.of(fmod));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2, T3> List<Tensor<T3>> Momentum(Tensor<T1> R, Tensor<T2> T, List<Tensor<T3>> inputs, String mode, float norm_coefficient, float alpha, float beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Momentum.class, List.of(R, T, inputs), List.of(mode, norm_coefficient, alpha, beta));
        return (List<Tensor<T3>>) result;
    }
    
    public static <T> Tensor<T> Mul(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mul.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> Multinomial(Tensor<T1> input, Optional<Float> seed, Optional<Integer> sample_size, Optional<Integer> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Multinomial.class, List.of(input), List.of(seed, sample_size, dtype));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Neg(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Neg.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, Tind> Tensor<T> NegativeLogLikelihoodLoss(Tensor<T> input, Tensor<Tind> target, Optional<Tensor<T>> weight, Optional<Integer> ignore_index, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NegativeLogLikelihoodLoss.class, List.of(input, target, weight), List.of(ignore_index, reduction));
        return (Tensor<T>) result;
    }
    
    public static Tensor<Long> NonMaxSuppression(Tensor<Float> boxes, Tensor<Float> scores, Optional<Tensor<Long>> max_output_boxes_per_class, Optional<Tensor<Float>> iou_threshold, Optional<Tensor<Float>> score_threshold, Optional<Integer> center_point_box) {
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
    
    public static <T> Tensor<T> Not(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Not.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T1, T2, T3> Tensor<T3> OneHot(Tensor<T1> indices, Tensor<T2> depth, Tensor<T3> values, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OneHot.class, List.of(indices, depth, values), List.of(axis));
        return (Tensor<T3>) result;
    }
    
    public static <T> Tensor<Float> OneHotEncoder(Tensor<T> X, Optional<String[]> cats_strings, Optional<int[]> cats_int64s, Optional<Integer> zeros) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OneHotEncoder.class, List.of(X), List.of(cats_strings, cats_int64s, zeros));
        return (Tensor<Float>) result;
    }
    
    public static <V, O> Tensor<O> Optional(Optional<Tensor<V>> input, Optional<byte[]> type) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Optional.class, List.of(input), List.of(type));
        return (Tensor<O>) result;
    }
    
    public static <O, V> Tensor<V> OptionalGetElement(Tensor<O> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OptionalGetElement.class, List.of(input), List.of());
        return (Tensor<V>) result;
    }
    
    public static <O, B> Tensor<B> OptionalHasElement(Optional<Tensor<O>> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OptionalHasElement.class, List.of(input), List.of());
        return (Tensor<B>) result;
    }
    
    public static <T, T1> Tensor<T1> Or(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Or.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
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
    
    public static <T1, T2, T3, T4> Tensor<T3> QLinearConv(Tensor<T1> x, Tensor<Float> x_scale, Tensor<T1> x_zero_point, Tensor<T2> w, Tensor<Float> w_scale, Tensor<T2> w_zero_point, Tensor<Float> y_scale, Tensor<T3> y_zero_point, Optional<Tensor<T4>> B, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QLinearConv.class, List.of(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T3>) result;
    }
    
    public static <TS, T1, T2, T3> Tensor<T3> QLinearMatMul(Tensor<T1> a, Tensor<TS> a_scale, Tensor<T1> a_zero_point, Tensor<T2> b, Tensor<TS> b_scale, Tensor<T2> b_zero_point, Tensor<TS> y_scale, Tensor<T3> y_zero_point) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QLinearMatMul.class, List.of(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point), List.of());
        return (Tensor<T3>) result;
    }
    
    public static <T1, T2> Tensor<T2> QuantizeLinear(Tensor<T1> x, Tensor<T1> y_scale, Optional<Tensor<T2>> y_zero_point, Optional<Integer> output_dtype, Optional<Integer> saturate, Optional<Integer> axis, Optional<Integer> block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QuantizeLinear.class, List.of(x, y_scale, y_zero_point), List.of(output_dtype, saturate, axis, block_size));
        return (Tensor<T2>) result;
    }
    
    public static <T, T1> List<Tensor<T>> RNN(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<T1>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Integer> layout, Optional<float[]> activation_alpha, Optional<Integer> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RNN.class, List.of(X, W, R, B, sequence_lens, initial_h), List.of(layout, activation_alpha, hidden_size, activation_beta, activations, clip, direction));
        return (List<Tensor<T>>) result;
    }
    
    public static <T> Tensor<T> RandomNormal(int[] shape, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomNormal.class, List.of(), List.of(shape, seed, mean, scale, dtype));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> RandomNormalLike(Tensor<T1> input, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomNormalLike.class, List.of(input), List.of(seed, mean, scale, dtype));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> RandomUniform(Optional<Float> high, int[] shape, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomUniform.class, List.of(), List.of(high, shape, seed, low, dtype));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> RandomUniformLike(Tensor<T1> input, Optional<Float> high, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
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
    
    public static <T> Tensor<T> ReduceL1(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceL1.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceL2(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceL2.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceLogSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceLogSum.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceLogSumExp(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceLogSumExp.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceMax(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMax.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceMean(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMean.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceMin(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMin.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceProd(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceProd.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceSum.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> ReduceSumSquare(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceSumSquare.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T2> RegexFullMatch(Tensor<T1> X, Optional<String> pattern) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RegexFullMatch.class, List.of(X), List.of(pattern));
        return (Tensor<T2>) result;
    }
    
    public static <T> Tensor<T> Relu(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Relu.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Reshape(Tensor<T> data, Tensor<Long> shape, Optional<Integer> allowzero) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Reshape.class, List.of(data, shape), List.of(allowzero));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T1> Resize(Tensor<T1> X, Optional<Tensor<T2>> roi, Optional<Tensor<Float>> scales, Optional<Tensor<Long>> sizes, Optional<String> mode, Optional<Float> extrapolation_value, Optional<String> nearest_mode, Optional<Integer> antialias, Optional<Float> cubic_coeff_a, Optional<int[]> axes, Optional<String> coordinate_transformation_mode, Optional<String> keep_aspect_ratio_policy, Optional<Integer> exclude_outside) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Resize.class, List.of(X, roi, scales, sizes), List.of(mode, extrapolation_value, nearest_mode, antialias, cubic_coeff_a, axes, coordinate_transformation_mode, keep_aspect_ratio_policy, exclude_outside));
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> ReverseSequence(Tensor<T> input, Tensor<Long> sequence_lens, Optional<Integer> time_axis, Optional<Integer> batch_axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReverseSequence.class, List.of(input, sequence_lens), List.of(time_axis, batch_axis));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T1> RoiAlign(Tensor<T1> X, Tensor<T1> rois, Tensor<T2> batch_indices, Optional<String> mode, Optional<Integer> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Integer> sampling_ratio, Optional<Integer> output_height) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RoiAlign.class, List.of(X, rois, batch_indices), List.of(mode, output_width, spatial_scale, coordinate_transformation_mode, sampling_ratio, output_height));
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> Round(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Round.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> Tensor<T1> STFT(Tensor<T1> signal, Tensor<T2> frame_step, Optional<Tensor<T1>> window, Optional<Tensor<T2>> frame_length, Optional<Integer> onesided) {
        Object result = OnnxInterpreter.interpret(OnnxOps.STFT.class, List.of(signal, frame_step, window, frame_length), List.of(onesided));
        return (Tensor<T1>) result;
    }
    
    public static <T1, T2> List<Tensor<T2>> SVMClassifier(Tensor<T1> X, Optional<float[]> prob_b, Optional<float[]> kernel_params, Optional<String> kernel_type, Optional<int[]> classlabels_ints, Optional<String> post_transform, Optional<float[]> rho, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<int[]> vectors_per_class, Optional<float[]> prob_a, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SVMClassifier.class, List.of(X), List.of(prob_b, kernel_params, kernel_type, classlabels_ints, post_transform, rho, coefficients, support_vectors, vectors_per_class, prob_a, classlabels_strings));
        return (List<Tensor<T2>>) result;
    }
    
    public static <T> Tensor<Float> SVMRegressor(Tensor<T> X, Optional<String> kernel_type, Optional<float[]> kernel_params, Optional<Integer> n_supports, Optional<float[]> rho, Optional<String> post_transform, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<Integer> one_class) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SVMRegressor.class, List.of(X), List.of(kernel_type, kernel_params, n_supports, rho, post_transform, coefficients, support_vectors, one_class));
        return (Tensor<Float>) result;
    }
    
    public static <T> Tensor<Float> Scaler(Tensor<T> X, Optional<float[]> offset, Optional<float[]> scale) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Scaler.class, List.of(X), List.of(offset, scale));
        return (Tensor<Float>) result;
    }
    
    public static <T, Tind> Tensor<T> Scatter(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Scatter.class, List.of(data, indices, updates), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T, Tind> Tensor<T> ScatterElements(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<String> reduction, Optional<Integer> axis) {
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
    
    public static <S, T, I> Tensor<T> SequenceAt(Tensor<S> input_sequence, Tensor<I> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceAt.class, List.of(input_sequence, position), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, S> Tensor<S> SequenceConstruct(List<Tensor<T>> inputs) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceConstruct.class, List.of(inputs), List.of());
        return (Tensor<S>) result;
    }
    
    public static <S> Tensor<S> SequenceEmpty(Optional<Integer> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceEmpty.class, List.of(), List.of(dtype));
        return (Tensor<S>) result;
    }
    
    public static <S, I> Tensor<S> SequenceErase(Tensor<S> input_sequence, Optional<Tensor<I>> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceErase.class, List.of(input_sequence, position), List.of());
        return (Tensor<S>) result;
    }
    
    public static <T, S, I> Tensor<S> SequenceInsert(Tensor<S> input_sequence, Tensor<T> tensor, Optional<Tensor<I>> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceInsert.class, List.of(input_sequence, tensor, position), List.of());
        return (Tensor<S>) result;
    }
    
    public static <S, I> Tensor<I> SequenceLength(Tensor<S> input_sequence) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceLength.class, List.of(input_sequence), List.of());
        return (Tensor<I>) result;
    }
    
    public static <T, T1> Tensor<T1> Shape(Tensor<T> data, Optional<Integer> start, Optional<Integer> end) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Shape.class, List.of(data), List.of(start, end));
        return (Tensor<T1>) result;
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
    
    public static <T, T1> Tensor<T1> Size(Tensor<T> data) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Size.class, List.of(data), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T, Tind> Tensor<T> Slice(Tensor<T> data, Tensor<Tind> starts, Tensor<Tind> ends, Optional<Tensor<Tind>> axes, Optional<Tensor<Tind>> steps) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Slice.class, List.of(data, starts, ends, axes, steps), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Softmax(Tensor<T> input, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }
    
    public static <T, Tind> List<Tensor<T>> SoftmaxCrossEntropyLoss(Tensor<T> scores, Tensor<Tind> labels, Optional<Tensor<T>> weights, Optional<Integer> ignore_index, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SoftmaxCrossEntropyLoss.class, List.of(scores, labels, weights), List.of(ignore_index, reduction));
        return (List<Tensor<T>>) result;
    }
    
    public static <T> Tensor<T> Softplus(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softplus.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Softsign(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softsign.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> SpaceToDepth(Tensor<T> input, int blocksize) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SpaceToDepth.class, List.of(input), List.of(blocksize));
        return (Tensor<T>) result;
    }
    
    public static <T> List<Tensor<T>> Split(Tensor<T> input, Optional<Tensor<Long>> split, Optional<Integer> num_outputs, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Split.class, List.of(input, split), List.of(num_outputs, axis));
        return (List<Tensor<T>>) result;
    }
    
    public static <T, I, S> Tensor<S> SplitToSequence(Tensor<T> input, Optional<Tensor<I>> split, Optional<Integer> keepdims, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SplitToSequence.class, List.of(input, split), List.of(keepdims, axis));
        return (Tensor<S>) result;
    }
    
    public static <T> Tensor<T> Sqrt(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sqrt.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Squeeze(Tensor<T> data, Optional<Tensor<Long>> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Squeeze.class, List.of(data, axes), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> StringConcat(Tensor<T> X, Tensor<T> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringConcat.class, List.of(X, Y), List.of());
        return (Tensor<T>) result;
    }
    
    public static Tensor<String> StringNormalizer(Tensor<String> X, Optional<Integer> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringNormalizer.class, List.of(X), List.of(is_case_sensitive, locale, stopwords, case_change_action));
        return (Tensor<String>) result;
    }
    
    public static <T1, T2, T3> List<Tensor<T2>> StringSplit(Tensor<T1> X, Optional<String> delimiter, Optional<Integer> maxsplit) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringSplit.class, List.of(X), List.of(delimiter, maxsplit));
        return (List<Tensor<T2>>) result;
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
    
    public static <T, T1> Tensor<T1> TfIdfVectorizer(Tensor<T> X, int[] ngram_counts, int min_gram_length, Optional<String[]> pool_strings, String mode, int max_gram_length, int max_skip_count, Optional<int[]> pool_int64s, Optional<float[]> weights, int[] ngram_indexes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TfIdfVectorizer.class, List.of(X), List.of(ngram_counts, min_gram_length, pool_strings, mode, max_gram_length, max_skip_count, pool_int64s, weights, ngram_indexes));
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> ThresholdedRelu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ThresholdedRelu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }
    
    public static <T, T1> Tensor<T> Tile(Tensor<T> input, Tensor<T1> repeats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tile.class, List.of(input, repeats), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, I> List<Tensor<T>> TopK(Tensor<T> X, Tensor<Long> K, Optional<Integer> largest, Optional<Integer> sorted, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TopK.class, List.of(X, K), List.of(largest, sorted, axis));
        return (List<Tensor<T>>) result;
    }
    
    public static <T> Tensor<T> Transpose(Tensor<T> data, Optional<int[]> perm) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Transpose.class, List.of(data), List.of(perm));
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> TreeEnsemble(Tensor<T> X, Optional<Integer> aggregate_function, Optional<Tensor<?>> nodes_hitrates, int[] nodes_featureids, int[] nodes_falseleafs, Optional<Integer> post_transform, int[] nodes_trueleafs, Tensor<?> nodes_modes, int[] nodes_falsenodeids, int[] nodes_truenodeids, Tensor<?> leaf_weights, int[] leaf_targetids, int[] tree_roots, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<Tensor<?>> membership_values, Tensor<?> nodes_splits) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsemble.class, List.of(X), List.of(aggregate_function, nodes_hitrates, nodes_featureids, nodes_falseleafs, post_transform, nodes_trueleafs, nodes_modes, nodes_falsenodeids, nodes_truenodeids, leaf_weights, leaf_targetids, tree_roots, n_targets, nodes_missing_value_tracks_true, membership_values, nodes_splits));
        return (Tensor<T>) result;
    }
    
    public static <T1, T2> List<Tensor<T2>> TreeEnsembleClassifier(Tensor<T1> X, Optional<int[]> classlabels_int64s, Optional<int[]> class_ids, Optional<float[]> nodes_hitrates, Optional<int[]> nodes_featureids, Optional<int[]> nodes_treeids, Optional<Tensor<?>> class_weights_as_tensor, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<int[]> nodes_falsenodeids, Optional<String[]> classlabels_strings, Optional<int[]> nodes_truenodeids, Optional<int[]> nodes_nodeids, Optional<Tensor<?>> nodes_hitrates_as_tensor, Optional<float[]> class_weights, Optional<Tensor<?>> base_values_as_tensor, Optional<int[]> nodes_missing_value_tracks_true, Optional<int[]> class_nodeids, Optional<int[]> class_treeids, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor<?>> nodes_values_as_tensor) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsembleClassifier.class, List.of(X), List.of(classlabels_int64s, class_ids, nodes_hitrates, nodes_featureids, nodes_treeids, class_weights_as_tensor, post_transform, nodes_modes, nodes_falsenodeids, classlabels_strings, nodes_truenodeids, nodes_nodeids, nodes_hitrates_as_tensor, class_weights, base_values_as_tensor, nodes_missing_value_tracks_true, class_nodeids, class_treeids, base_values, nodes_values, nodes_values_as_tensor));
        return (List<Tensor<T2>>) result;
    }
    
    public static <T> Tensor<Float> TreeEnsembleRegressor(Tensor<T> X, Optional<String> aggregate_function, Optional<float[]> nodes_hitrates, Optional<Tensor<?>> target_weights_as_tensor, Optional<int[]> nodes_featureids, Optional<int[]> target_treeids, Optional<int[]> nodes_treeids, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<float[]> target_weights, Optional<int[]> nodes_falsenodeids, Optional<int[]> target_ids, Optional<int[]> nodes_truenodeids, Optional<int[]> target_nodeids, Optional<int[]> nodes_nodeids, Optional<Tensor<?>> nodes_hitrates_as_tensor, Optional<Tensor<?>> base_values_as_tensor, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor<?>> nodes_values_as_tensor) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsembleRegressor.class, List.of(X), List.of(aggregate_function, nodes_hitrates, target_weights_as_tensor, nodes_featureids, target_treeids, nodes_treeids, post_transform, nodes_modes, target_weights, nodes_falsenodeids, target_ids, nodes_truenodeids, target_nodeids, nodes_nodeids, nodes_hitrates_as_tensor, base_values_as_tensor, n_targets, nodes_missing_value_tracks_true, base_values, nodes_values, nodes_values_as_tensor));
        return (Tensor<Float>) result;
    }
    
    public static <T> Tensor<T> Trilu(Tensor<T> input, Optional<Tensor<Long>> k, Optional<Integer> upper) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Trilu.class, List.of(input, k), List.of(upper));
        return (Tensor<T>) result;
    }
    
    public static <T> List<Tensor<T>> Unique(Tensor<T> X, Optional<Integer> sorted, Optional<Integer> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Unique.class, List.of(X), List.of(sorted, axis));
        return (List<Tensor<T>>) result;
    }
    
    public static <T> Tensor<T> Unsqueeze(Tensor<T> data, Tensor<Long> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Unsqueeze.class, List.of(data, axes), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T> Tensor<T> Upsample(Tensor<T> X, Tensor<Float> scales, Optional<String> mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Upsample.class, List.of(X, scales), List.of(mode));
        return (Tensor<T>) result;
    }
    
    public static <B, T> Tensor<T> Where(Tensor<B> condition, Tensor<T> X, Tensor<T> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Where.class, List.of(condition, X, Y), List.of());
        return (Tensor<T>) result;
    }
    
    public static <T, T1> Tensor<T1> Xor(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Xor.class, List.of(A, B), List.of());
        return (Tensor<T1>) result;
    }
    
    public static <T> Tensor<T> ZipMap(Tensor<Float> X, Optional<int[]> classlabels_int64s, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ZipMap.class, List.of(X), List.of(classlabels_int64s, classlabels_strings));
        return (Tensor<T>) result;
    }
    
}
