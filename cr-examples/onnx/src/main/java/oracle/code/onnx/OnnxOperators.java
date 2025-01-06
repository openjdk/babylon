// Auto-generated from ONNX op schema

package oracle.code.onnx;

import java.util.Optional;
import java.util.List;

public final class OnnxOperators {
    
    private OnnxOperators() {}
    
    public static <T> Tensor<T> Abs(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Acos(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Acosh(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> List<Tensor<T3>> Adagrad(Tensor<T1> R, Tensor<T2> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> decay_factor, Optional<Float> norm_coefficient) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> List<Tensor<T3>> Adam(Tensor<T1> R, Tensor<T2> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> norm_coefficient_post, Optional<Float> norm_coefficient, Optional<Float> alpha, Optional<Float> beta) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Add(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> AffineGrid(Tensor<T1> theta, Tensor<T2> size, Optional<Integer> align_corners) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> And(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Long> ArgMax(Tensor<T> data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Long> ArgMin(Tensor<T> data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ArrayFeatureExtractor(Tensor<T> X, Tensor<Long> Y) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Asin(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Asinh(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Atan(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Atanh(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> AveragePool(Tensor<T> X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> count_include_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1, T2> List<Tensor<T>> BatchNormalization(Tensor<T> X, Tensor<T1> scale, Tensor<T1> B, Tensor<T2> input_mean, Tensor<T2> input_var, Optional<Float> epsilon, Optional<Integer> training_mode, Optional<Float> momentum) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> Bernoulli(Tensor<T1> input, Optional<Float> seed, Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Binarizer(Tensor<T> X, Optional<Float> threshold) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> BitShift(Tensor<T> X, Tensor<T> Y, String direction) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> BitwiseAnd(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> BitwiseNot(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> BitwiseOr(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> BitwiseXor(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> BlackmanWindow(Tensor<T1> size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> Cast(Tensor<T1> input, Optional<Integer> saturate, int to) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> CastLike(Tensor<T1> input, Tensor<T2> target_type, Optional<Integer> saturate) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> CastMap(Tensor<T1> X, Optional<String> map_form, Optional<String> cast_to, Optional<Integer> max_map) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> CategoryMapper(Tensor<T1> X, Optional<int[]> cats_int64s, Optional<String[]> cats_strings, Optional<Integer> default_int64, Optional<String> default_string) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Ceil(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Celu(Tensor<T> X, Optional<Float> alpha) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> CenterCropPad(Tensor<T> input_data, Tensor<Tind> shape, Optional<int[]> axes) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Clip(Tensor<T> input, Optional<Tensor<T>> min, Optional<Tensor<T>> max) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Col2Im(Tensor<T> input, Tensor<Long> image_shape, Tensor<Long> block_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T> Compress(Tensor<T> input, Tensor<T1> condition, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Concat(List<Tensor<T>> inputs, int axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <S, T> Tensor<T> ConcatFromSequence(Tensor<S> input_sequence, int axis, Optional<Integer> new_axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Constant(Optional<Integer> value_int, Optional<float[]> value_floats, Optional<String[]> value_strings, Optional<Float> value_float, Optional<String> value_string, Optional<int[]> value_ints, Optional<Object> sparse_value, Optional<Tensor> value) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> ConstantOfShape(Tensor<T1> input, Optional<Tensor> value) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Conv(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> Tensor<T3> ConvInteger(Tensor<T1> x, Tensor<T2> w, Optional<Tensor<T1>> x_zero_point, Optional<Tensor<T2>> w_zero_point, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ConvTranspose(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<int[]> output_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape, Optional<int[]> output_padding) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Cos(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Cosh(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T2> Tensor<T> CumSum(Tensor<T> x, Tensor<T2> axis, Optional<Integer> exclusive, Optional<Integer> reverse) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> DFT(Tensor<T1> input, Optional<Tensor<T2>> dft_length, Optional<Tensor<Long>> axis, Optional<Integer> inverse, Optional<Integer> onesided) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> DeformConv(Tensor<T> X, Tensor<T> W, Tensor<T> offset, Optional<Tensor<T>> B, Optional<Tensor<T>> mask, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides, Optional<Integer> offset_group, Optional<Integer> group, Optional<int[]> kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> DepthToSpace(Tensor<T> input, Optional<String> mode, int blocksize) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> DequantizeLinear(Tensor<T1> x, Tensor<T2> x_scale, Optional<Tensor<T1>> x_zero_point, Optional<Integer> axis, Optional<Integer> block_size) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Det(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> DictVectorizer(Tensor<T1> X, Optional<String[]> string_vocabulary, Optional<int[]> int64_vocabulary) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Div(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1, T2> List<Tensor<T>> Dropout(Tensor<T> data, Optional<Tensor<T1>> ratio, Optional<Tensor<T2>> training_mode, Optional<Integer> seed) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> List<Tensor<T2>> DynamicQuantizeLinear(Tensor<T1> x) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Einsum(List<Tensor<T>> Inputs, String equation) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Elu(Tensor<T> X, Optional<Float> alpha) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Equal(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Erf(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Exp(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Expand(Tensor<T> input, Tensor<Long> shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> EyeLike(Tensor<T1> input, Optional<Integer> dtype, Optional<Integer> k) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1> Tensor<Float> FeatureVectorizer(List<Tensor<T1>> X, Optional<int[]> inputdimensions) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Flatten(Tensor<T> input, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Floor(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> List<Tensor<T>> GRU(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<T1>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Integer> layout, Optional<float[]> activation_alpha, Optional<Integer> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Integer> linear_before_reset, Optional<Float> clip, Optional<String> direction) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> Gather(Tensor<T> data, Tensor<Tind> indices, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> GatherElements(Tensor<T> data, Tensor<Tind> indices, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> GatherND(Tensor<T> data, Tensor<Long> indices, Optional<Integer> batch_dims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Gelu(Tensor<T> X, Optional<String> approximate) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Gemm(Tensor<T> A, Tensor<T> B, Optional<Tensor<T>> C, Optional<Float> alpha, Optional<Integer> transB, Optional<Float> beta, Optional<Integer> transA) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> GlobalAveragePool(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> GlobalLpPool(Tensor<T> X, Optional<Integer> p) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> GlobalMaxPool(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> List<Tensor<T2>> Gradient(List<Tensor<T1>> Inputs, String y, Optional<String[]> zs, String[] xs) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Greater(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> GreaterOrEqual(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> GridSample(Tensor<T1> X, Tensor<T2> grid, Optional<String> mode, Optional<Integer> align_corners, Optional<String> padding_mode) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> GroupNormalization(Tensor<T> X, Tensor<T> scale, Tensor<T> bias, Optional<Float> epsilon, Optional<Integer> stash_type, int num_groups) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> HammingWindow(Tensor<T1> size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> HannWindow(Tensor<T1> size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> HardSigmoid(Tensor<T> X, Optional<Float> alpha, Optional<Float> beta) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> HardSwish(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Hardmax(Tensor<T> input, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <V> Tensor<V> Identity(Tensor<V> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <V, B> List<Tensor<V>> If(Tensor<B> cond, OnnxRunnable else_branch, OnnxRunnable then_branch) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> ImageDecoder(Tensor<T1> encoded_stream, Optional<String> pixel_format) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Imputer(Tensor<T> X, Optional<Integer> replaced_value_int64, Optional<Float> replaced_value_float, Optional<int[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> InstanceNormalization(Tensor<T> input, Tensor<T> scale, Tensor<T> B, Optional<Float> epsilon) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> IsInf(Tensor<T1> X, Optional<Integer> detect_negative, Optional<Integer> detect_positive) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> IsNaN(Tensor<T1> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> LRN(Tensor<T> X, int size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> List<Tensor<T>> LSTM(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<T1>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Tensor<T>> initial_c, Optional<Tensor<T>> P, Optional<Integer> layout, Optional<Integer> input_forget, Optional<float[]> activation_alpha, Optional<Integer> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> LabelEncoder(Tensor<T1> X, Optional<String[]> values_strings, Optional<int[]> keys_int64s, Optional<Tensor> keys_tensor, Optional<String[]> keys_strings, Optional<Float> default_float, Optional<float[]> keys_floats, Optional<Tensor> default_tensor, Optional<Integer> default_int64, Optional<Tensor> values_tensor, Optional<int[]> values_int64s, Optional<String> default_string, Optional<float[]> values_floats) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, U> List<Tensor<T>> LayerNormalization(Tensor<T> X, Tensor<T> Scale, Optional<Tensor<T>> B, Optional<Float> epsilon, Optional<Integer> stash_type, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> LeakyRelu(Tensor<T> X, Optional<Float> alpha) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Less(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> LessOrEqual(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> List<Tensor<T2>> LinearClassifier(Tensor<T1> X, Optional<int[]> classlabels_ints, Optional<String> post_transform, float[] coefficients, Optional<Integer> multi_class, Optional<float[]> intercepts, Optional<String[]> classlabels_strings) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Float> LinearRegressor(Tensor<T> X, Optional<String> post_transform, Optional<float[]> coefficients, Optional<Integer> targets, Optional<float[]> intercepts) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Log(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> LogSoftmax(Tensor<T> input, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <V, I, B> List<Tensor<V>> Loop(Optional<Tensor<I>> M, Optional<Tensor<B>> cond, List<Tensor<V>> v_initial, OnnxRunnable body) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> LpNormalization(Tensor<T> input, Optional<Integer> p, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> LpPool(Tensor<T> X, Optional<Integer> p, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> MatMul(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> Tensor<T3> MatMulInteger(Tensor<T1> A, Tensor<T2> B, Optional<Tensor<T1>> a_zero_point, Optional<Tensor<T2>> b_zero_point) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Max(List<Tensor<T>> data_0) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, I> List<Tensor<T>> MaxPool(Tensor<T> X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<Integer> storage_order, Optional<int[]> strides, int[] kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> MaxRoiPool(Tensor<T> X, Tensor<T> rois, Optional<Float> spatial_scale, int[] pooled_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> MaxUnpool(Tensor<T1> X, Tensor<T2> I, Optional<Tensor<T2>> output_shape, Optional<int[]> pads, Optional<int[]> strides, int[] kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Mean(List<Tensor<T>> data_0) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> MeanVarianceNormalization(Tensor<T> X, Optional<int[]> axes) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> Tensor<T3> MelWeightMatrix(Tensor<T1> num_mel_bins, Tensor<T1> dft_length, Tensor<T1> sample_rate, Tensor<T2> lower_edge_hertz, Tensor<T2> upper_edge_hertz, Optional<Integer> output_datatype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Min(List<Tensor<T>> data_0) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Mish(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Mod(Tensor<T> A, Tensor<T> B, Optional<Integer> fmod) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> List<Tensor<T3>> Momentum(Tensor<T1> R, Tensor<T2> T, List<Tensor<T3>> inputs, String mode, float norm_coefficient, float alpha, float beta) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Mul(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> Multinomial(Tensor<T1> input, Optional<Float> seed, Optional<Integer> sample_size, Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Neg(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> NegativeLogLikelihoodLoss(Tensor<T> input, Tensor<Tind> target, Optional<Tensor<T>> weight, Optional<Integer> ignore_index, Optional<String> reduction) {
         throw new UnsupportedOperationException();
    }
    
    public static Tensor<Long> NonMaxSuppression(Tensor<Float> boxes, Tensor<Float> scores, Optional<Tensor<Long>> max_output_boxes_per_class, Optional<Tensor<Float>> iou_threshold, Optional<Tensor<Float>> score_threshold, Optional<Integer> center_point_box) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Long> NonZero(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Float> Normalizer(Tensor<T> X, Optional<String> norm) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Not(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> Tensor<T3> OneHot(Tensor<T1> indices, Tensor<T2> depth, Tensor<T3> values, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Float> OneHotEncoder(Tensor<T> X, Optional<String[]> cats_strings, Optional<int[]> cats_int64s, Optional<Integer> zeros) {
         throw new UnsupportedOperationException();
    }
    
    public static <V, O> Tensor<O> Optional(Optional<Tensor<V>> input, Optional<byte[]> type) {
         throw new UnsupportedOperationException();
    }
    
    public static <O, V> Tensor<V> OptionalGetElement(Tensor<O> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <O, B> Tensor<B> OptionalHasElement(Optional<Tensor<O>> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Or(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> PRelu(Tensor<T> X, Tensor<T> slope) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> Pad(Tensor<T> data, Tensor<Long> pads, Optional<Tensor<T>> constant_value, Optional<Tensor<Tind>> axes, Optional<String> mode) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T> Pow(Tensor<T> X, Tensor<T1> Y) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3, T4> Tensor<T3> QLinearConv(Tensor<T1> x, Tensor<Float> x_scale, Tensor<T1> x_zero_point, Tensor<T2> w, Tensor<Float> w_scale, Tensor<T2> w_zero_point, Tensor<Float> y_scale, Tensor<T3> y_zero_point, Optional<Tensor<T4>> B, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<int[]> strides, Optional<Integer> group, Optional<int[]> kernel_shape) {
         throw new UnsupportedOperationException();
    }
    
    public static <TS, T1, T2, T3> Tensor<T3> QLinearMatMul(Tensor<T1> a, Tensor<TS> a_scale, Tensor<T1> a_zero_point, Tensor<T2> b, Tensor<TS> b_scale, Tensor<T2> b_zero_point, Tensor<TS> y_scale, Tensor<T3> y_zero_point) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> QuantizeLinear(Tensor<T1> x, Tensor<T1> y_scale, Optional<Tensor<T2>> y_zero_point, Optional<Integer> output_dtype, Optional<Integer> saturate, Optional<Integer> axis, Optional<Integer> block_size) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> List<Tensor<T>> RNN(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<T1>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Integer> layout, Optional<float[]> activation_alpha, Optional<Integer> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> RandomNormal(int[] shape, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> RandomNormalLike(Tensor<T1> input, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> RandomUniform(Optional<Float> high, int[] shape, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> RandomUniformLike(Tensor<T1> input, Optional<Float> high, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Range(Tensor<T> start, Tensor<T> limit, Tensor<T> delta) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Reciprocal(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceL1(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceL2(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceLogSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceLogSumExp(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceMax(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceMean(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceMin(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceProd(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReduceSumSquare(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Integer> noop_with_empty_axes, Optional<Integer> keepdims) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T2> RegexFullMatch(Tensor<T1> X, Optional<String> pattern) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Relu(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Reshape(Tensor<T> data, Tensor<Long> shape, Optional<Integer> allowzero) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> Resize(Tensor<T1> X, Optional<Tensor<T2>> roi, Optional<Tensor<Float>> scales, Optional<Tensor<Long>> sizes, Optional<String> mode, Optional<Float> extrapolation_value, Optional<String> nearest_mode, Optional<Integer> antialias, Optional<Float> cubic_coeff_a, Optional<int[]> axes, Optional<String> coordinate_transformation_mode, Optional<String> keep_aspect_ratio_policy, Optional<Integer> exclude_outside) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ReverseSequence(Tensor<T> input, Tensor<Long> sequence_lens, Optional<Integer> time_axis, Optional<Integer> batch_axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> RoiAlign(Tensor<T1> X, Tensor<T1> rois, Tensor<T2> batch_indices, Optional<String> mode, Optional<Integer> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Integer> sampling_ratio, Optional<Integer> output_height) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Round(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> Tensor<T1> STFT(Tensor<T1> signal, Tensor<T2> frame_step, Optional<Tensor<T1>> window, Optional<Tensor<T2>> frame_length, Optional<Integer> onesided) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> List<Tensor<T2>> SVMClassifier(Tensor<T1> X, Optional<float[]> prob_b, Optional<float[]> kernel_params, Optional<String> kernel_type, Optional<int[]> classlabels_ints, Optional<String> post_transform, Optional<float[]> rho, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<int[]> vectors_per_class, Optional<float[]> prob_a, Optional<String[]> classlabels_strings) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Float> SVMRegressor(Tensor<T> X, Optional<String> kernel_type, Optional<float[]> kernel_params, Optional<Integer> n_supports, Optional<float[]> rho, Optional<String> post_transform, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<Integer> one_class) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Float> Scaler(Tensor<T> X, Optional<float[]> offset, Optional<float[]> scale) {
         throw new UnsupportedOperationException();
    }
    
    public static <V> List<Tensor<V>> Scan(List<Tensor<V>> initial_state_and_scan_inputs, Optional<int[]> scan_input_axes, Optional<int[]> scan_output_axes, Optional<int[]> scan_output_directions, Optional<int[]> scan_input_directions, OnnxRunnable body, int num_scan_inputs) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> Scatter(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> ScatterElements(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<String> reduction, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ScatterND(Tensor<T> data, Tensor<Long> indices, Tensor<T> updates, Optional<String> reduction) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Selu(Tensor<T> X, Optional<Float> alpha, Optional<Float> gamma) {
         throw new UnsupportedOperationException();
    }
    
    public static <S, T, I> Tensor<T> SequenceAt(Tensor<S> input_sequence, Tensor<I> position) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, S> Tensor<S> SequenceConstruct(List<Tensor<T>> inputs) {
         throw new UnsupportedOperationException();
    }
    
    public static <S> Tensor<S> SequenceEmpty(Optional<Integer> dtype) {
         throw new UnsupportedOperationException();
    }
    
    public static <S, I> Tensor<S> SequenceErase(Tensor<S> input_sequence, Optional<Tensor<I>> position) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, S, I> Tensor<S> SequenceInsert(Tensor<S> input_sequence, Tensor<T> tensor, Optional<Tensor<I>> position) {
         throw new UnsupportedOperationException();
    }
    
    public static <S, I> Tensor<I> SequenceLength(Tensor<S> input_sequence) {
         throw new UnsupportedOperationException();
    }
    
    public static <S, V> List<Tensor<S>> SequenceMap(Tensor<S> input_sequence, List<Tensor<V>> additional_inputs, OnnxRunnable body) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Shape(Tensor<T> data, Optional<Integer> start, Optional<Integer> end) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Shrink(Tensor<T> input, Optional<Float> lambd, Optional<Float> bias) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sigmoid(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sign(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sin(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sinh(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Size(Tensor<T> data) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> Tensor<T> Slice(Tensor<T> data, Tensor<Tind> starts, Tensor<Tind> ends, Optional<Tensor<Tind>> axes, Optional<Tensor<Tind>> steps) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Softmax(Tensor<T> input, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, Tind> List<Tensor<T>> SoftmaxCrossEntropyLoss(Tensor<T> scores, Tensor<Tind> labels, Optional<Tensor<T>> weights, Optional<Integer> ignore_index, Optional<String> reduction) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Softplus(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Softsign(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> SpaceToDepth(Tensor<T> input, int blocksize) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> List<Tensor<T>> Split(Tensor<T> input, Optional<Tensor<Long>> split, Optional<Integer> num_outputs, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, I, S> Tensor<S> SplitToSequence(Tensor<T> input, Optional<Tensor<I>> split, Optional<Integer> keepdims, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sqrt(Tensor<T> X) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Squeeze(Tensor<T> data, Optional<Tensor<Long>> axes) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> StringConcat(Tensor<T> X, Tensor<T> Y) {
         throw new UnsupportedOperationException();
    }
    
    public static Tensor<String> StringNormalizer(Tensor<String> X, Optional<Integer> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2, T3> List<Tensor<T2>> StringSplit(Tensor<T1> X, Optional<String> delimiter, Optional<Integer> maxsplit) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sub(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Sum(List<Tensor<T>> data_0) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Tan(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Tanh(Tensor<T> input) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> TfIdfVectorizer(Tensor<T> X, int[] ngram_counts, int min_gram_length, Optional<String[]> pool_strings, String mode, int max_gram_length, int max_skip_count, Optional<int[]> pool_int64s, Optional<float[]> weights, int[] ngram_indexes) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ThresholdedRelu(Tensor<T> X, Optional<Float> alpha) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T> Tile(Tensor<T> input, Tensor<T1> repeats) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, I> List<Tensor<T>> TopK(Tensor<T> X, Tensor<Long> K, Optional<Integer> largest, Optional<Integer> sorted, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Transpose(Tensor<T> data, Optional<int[]> perm) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> TreeEnsemble(Tensor<T> X, Optional<Integer> aggregate_function, Optional<Tensor> nodes_hitrates, int[] nodes_featureids, int[] nodes_falseleafs, Optional<Integer> post_transform, int[] nodes_trueleafs, Tensor nodes_modes, int[] nodes_falsenodeids, int[] nodes_truenodeids, Tensor leaf_weights, int[] leaf_targetids, int[] tree_roots, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<Tensor> membership_values, Tensor nodes_splits) {
         throw new UnsupportedOperationException();
    }
    
    public static <T1, T2> List<Tensor<T2>> TreeEnsembleClassifier(Tensor<T1> X, Optional<int[]> classlabels_int64s, Optional<int[]> class_ids, Optional<float[]> nodes_hitrates, Optional<int[]> nodes_featureids, Optional<int[]> nodes_treeids, Optional<Tensor> class_weights_as_tensor, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<int[]> nodes_falsenodeids, Optional<String[]> classlabels_strings, Optional<int[]> nodes_truenodeids, Optional<int[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<float[]> class_weights, Optional<Tensor> base_values_as_tensor, Optional<int[]> nodes_missing_value_tracks_true, Optional<int[]> class_nodeids, Optional<int[]> class_treeids, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<Float> TreeEnsembleRegressor(Tensor<T> X, Optional<String> aggregate_function, Optional<float[]> nodes_hitrates, Optional<Tensor> target_weights_as_tensor, Optional<int[]> nodes_featureids, Optional<int[]> target_treeids, Optional<int[]> nodes_treeids, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<float[]> target_weights, Optional<int[]> nodes_falsenodeids, Optional<int[]> target_ids, Optional<int[]> nodes_truenodeids, Optional<int[]> target_nodeids, Optional<int[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<Tensor> base_values_as_tensor, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Trilu(Tensor<T> input, Optional<Tensor<Long>> k, Optional<Integer> upper) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> List<Tensor<T>> Unique(Tensor<T> X, Optional<Integer> sorted, Optional<Integer> axis) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Unsqueeze(Tensor<T> data, Tensor<Long> axes) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> Upsample(Tensor<T> X, Tensor<Float> scales, Optional<String> mode) {
         throw new UnsupportedOperationException();
    }
    
    public static <B, T> Tensor<T> Where(Tensor<B> condition, Tensor<T> X, Tensor<T> Y) {
         throw new UnsupportedOperationException();
    }
    
    public static <T, T1> Tensor<T1> Xor(Tensor<T> A, Tensor<T> B) {
         throw new UnsupportedOperationException();
    }
    
    public static <T> Tensor<T> ZipMap(Tensor<Float> X, Optional<int[]> classlabels_int64s, Optional<String[]> classlabels_strings) {
         throw new UnsupportedOperationException();
    }
}
