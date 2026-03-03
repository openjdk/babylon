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

import oracle.code.onnx.ir.OnnxOps;

import java.util.Optional;
import java.util.List;
import java.util.Map;

@SuppressWarnings({"unchecked", "OptionalUsedAsFieldOrParameterType"})
public final class OnnxOperators extends ExplicitOnnxOperators {

    private OnnxOperators() {}

    ///
    /// Absolute takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where absolute value, y = abs(x), is applied to
    /// the tensor elementwise.
    public static <T> Tensor<T> Abs(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Abs.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
    public static <T> Tensor<T> Acos(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Acos.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the hyperbolic arccosine of the given input tensor element-wise.
    public static <T> Tensor<T> Acosh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Acosh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    ///     Compute one iteration of ADAGRAD, a stochastic gradient based optimization
    ///     algorithm. This operator can conduct the optimization of multiple tensor variables.
    ///
    ///     Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    ///     some parameters:
    ///
    ///      - The initial learning-rate "R".
    ///      - The update count "T". That is, the number of training iterations conducted.
    ///      - A L2-norm regularization coefficient "norm_coefficient".
    ///      - A learning-rate decay factor "decay_factor".
    ///      - A small constant "epsilon" to avoid dividing-by-zero.
    ///
    ///     At each ADAGRAD iteration, the optimized tensors are moved along a direction
    ///     computed based on their estimated gradient and accumulated squared gradient. Assume
    ///     that only a single tensor "X" is updated by this operator. We need the value of "X",
    ///     its gradient "G", and its accumulated squared gradient "H". Therefore, variables in
    ///     this operator's input list are sequentially "R", "T", "X", "G", and "H". Other
    ///     parameters are given as attributes because they are usually constants. Also, the
    ///     corresponding output tensors are the new value of "X" (called "X_new"), and then
    ///     the new accumulated squared gradient (called "H_new"). Those outputs are computed
    ///     from the given inputs following the pseudo code below.
    ///
    ///     Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    ///     numpy-style broadcasting support. The pseudo code to compute those outputs is:
    ///
    ///       // Compute a scalar learning-rate factor. At the first update of X, T is generally
    ///       // 0 (0-based update index) or 1 (1-based update index).
    ///       r = R / (1 + T * decay_factor);
    ///
    ///       // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
    ///       G_regularized = norm_coefficient * X + G;
    ///
    ///       // Compute new accumulated squared gradient.
    ///       H_new = H + G_regularized * G_regularized;
    ///
    ///       // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
    ///       // computes element-wise square-root.
    ///       H_adaptive = Sqrt(H_new) + epsilon
    ///
    ///       // Compute the new value of "X".
    ///       X_new = X - r * G_regularized / H_adaptive;
    ///
    ///     If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2", the same
    ///     pseudo code may be extended to handle all tensors jointly. More specifically, we can view "X" as a
    ///     concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
    ///     be concatenated too) and then just reuse the entire pseudo code.
    ///
    ///     Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    ///     In that reference paper, this operator is a special case of the Figure 1's composite mirror
    ///     descent update.
    public static <T1, T3> List<Tensor<T3>> Adagrad(Tensor<T1> R, Tensor<Long> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> decay_factor, Optional<Float> norm_coefficient) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Adagrad.class, List.of(R, T, inputs), List.of(epsilon, decay_factor, norm_coefficient));
        return (List<Tensor<T3>>) result;
    }

    ///
    ///     Compute one iteration of Adam, a stochastic gradient based optimization
    ///     algorithm. This operator can conduct the optimization of multiple tensor variables.
    ///
    ///     Let's define the behavior of this operator. First of all, Adam requires
    ///     some parameters:
    ///
    ///      - The learning-rate "R".
    ///      - The update count "T". That is, the number of training iterations conducted.
    ///      - A L2-norm regularization coefficient "norm_coefficient".
    ///      - A small constant "epsilon" to avoid dividing-by-zero.
    ///      - Two coefficients, "alpha" and "beta".
    ///
    ///     At each Adam iteration, the optimized tensors are moved along a direction
    ///     computed based on their exponentially-averaged historical gradient and
    ///     exponentially-averaged historical squared gradient. Assume that only a tensor
    ///     "X" is being optimized. The rest of required information is
    ///
    ///      - the value of "X",
    ///      - "X"'s gradient (denoted by "G"),
    ///      - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
    ///      - "X"'s exponentially-averaged historical squared gradient (denoted by "H").
    ///
    ///     Some of those parameters are passed into this operator as input tensors and others
    ///     are stored as this operator's attributes. Specifically, this operator's input tensor
    ///     list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
    ///     the second input, and so on. Other parameters are given as attributes because they
    ///     are constants. Moreover, the corresponding output tensors are
    ///
    ///      - the new value of "X" (called "X_new"),
    ///      - the new exponentially-averaged historical gradient (denoted by "V_new"), and
    ///      - the new exponentially-averaged historical squared gradient (denoted by "H_new").
    ///
    ///     Those outputs are computed following the pseudo code below.
    ///
    ///     Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    ///     numpy-style broadcasting support. The pseudo code to compute those outputs is:
    ///
    ///       // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
    ///       G_regularized = norm_coefficient * X + G
    ///
    ///       // Update exponentially-averaged historical gradient.
    ///       V_new = alpha * V + (1 - alpha) * G_regularized
    ///
    ///       // Update exponentially-averaged historical squared gradient.
    ///       H_new = beta * H + (1 - beta) * G_regularized * G_regularized
    ///
    ///       // Compute the element-wise square-root of H_new. V_new will be element-wisely
    ///       // divided by H_sqrt for a better update direction.
    ///       H_sqrt = Sqrt(H_new) + epsilon
    ///
    ///       // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
    ///       R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R
    ///
    ///       // Compute new value of "X".
    ///       X_new = X - R_adjusted * V_new / H_sqrt
    ///
    ///       // Post-update regularization.
    ///       X_final = (1 - norm_coefficient_post) * X_new
    ///
    ///     If there are multiple inputs to be optimized, the pseudo code will be applied
    ///     independently to each of them.
    public static <T1, T3> List<Tensor<T3>> Adam(Tensor<T1> R, Tensor<Long> T, List<Tensor<T3>> inputs, Optional<Float> epsilon, Optional<Float> norm_coefficient_post, Optional<Float> norm_coefficient, Optional<Float> alpha, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Adam.class, List.of(R, T, inputs), List.of(epsilon, norm_coefficient_post, norm_coefficient, alpha, beta));
        return (List<Tensor<T3>>) result;
    }

    ///
    /// Performs element-wise binary addition (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    ///
    /// (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
    public static <T> Tensor<T> Add(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Add.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
    /// (https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
    /// An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
    /// ```
    /// [r00, r01, r02, t0]   [x]   [x']
    /// [r10, r11, r12, t1] * [y] = [y']
    /// [r20, r21, r22, t2]   [z]   [z']
    /// [0,   0,   0,   1 ]   [1]   [1 ]
    /// ```
    /// where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
    /// The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.
    ///
    /// Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
    /// The output `grid` contains positions in the output space.
    ///
    /// When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
    /// ```
    /// v            v            v            v
    /// |-------------------|------------------|
    /// -1                  0                  1
    /// ```
    /// When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
    /// ```
    ///     v        v         v         v
    /// |------------------|-------------------|
    /// -1                 0                   1
    /// ```
    public static <T1> Tensor<T1> AffineGrid(Tensor<T1> theta, Tensor<Long> size, Optional<Long> align_corners) {
        Object result = OnnxInterpreter.interpret(OnnxOps.AffineGrid.class, List.of(theta, size), List.of(align_corners));
        return (Tensor<T1>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `and` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static Tensor<Boolean> And(Tensor<Boolean> A, Tensor<Boolean> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.And.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Computes the indices of the max elements of the input tensor's element along the
    /// provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
    /// If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
    /// If select_last_index is True (default False), the index of the last occurrence of the max
    /// is selected if the max appears more than once in the input. Otherwise the index of the
    /// first occurrence is selected.
    /// The type of the output tensor is integer.
    public static <T> Tensor<Long> ArgMax(Tensor<T> data, Optional<Long> keepdims, Optional<Long> select_last_index, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArgMax.class, List.of(data), List.of(keepdims, select_last_index, axis));
        return (Tensor<Long>) result;
    }

    ///
    /// Computes the indices of the min elements of the input tensor's element along the
    /// provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
    /// If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
    /// If select_last_index is True (default False), the index of the last occurrence of the min
    /// is selected if the min appears more than once in the input. Otherwise the index of the
    /// first occurrence is selected.
    /// The type of the output tensor is integer.
    public static <T> Tensor<Long> ArgMin(Tensor<T> data, Optional<Long> keepdims, Optional<Long> select_last_index, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArgMin.class, List.of(data), List.of(keepdims, select_last_index, axis));
        return (Tensor<Long>) result;
    }

    ///
    ///     Select elements of the input tensor based on the indices passed.<br>
    ///     The indices are applied to the last axes of the tensor.
    public static <T> Tensor<T> ArrayFeatureExtractor(Tensor<T> X, Tensor<Long> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ArrayFeatureExtractor.class, List.of(X, Y), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
    public static <T> Tensor<T> Asin(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Asin.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the hyperbolic arcsine of the given input tensor element-wise.
    public static <T> Tensor<T> Asinh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Asinh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
    public static <T> Tensor<T> Atan(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Atan.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the hyperbolic arctangent of the given input tensor element-wise.
    public static <T> Tensor<T> Atanh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Atanh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    public record AttentionResult<T1, T2>(Tensor<T1> Y, Tensor<T1> present_key, Tensor<T2> present_value, Tensor<T1> qk_matmul_output) { }
    ///
    ///
    /// Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.
    ///
    /// This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.
    ///
    /// For self attention, `kv_sequence_length` equals to `q_sequence_length`.
    ///
    /// For cross attention, query and key might have different lengths.
    ///
    /// This operator also covers the 3 following variants based on the number of heads:
    /// 1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
    /// 2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
    /// 3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.
    ///
    /// Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
    /// 1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
    /// 2) If `is_causal` is set to `1`, attention scores above the diagonal are masked out, regardless of the `attn_mask` input.
    ///
    /// With respect to KV cache update, this operator allows the following two use cases:
    ///
    /// 1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
    /// tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
    /// all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
    /// key and value, respectively. Note that this only works correctly for the special case where the past key and value
    /// do not contain padded tokens.
    /// 2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
    /// case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
    /// present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
    /// provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
    /// Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
    /// as the maximum value of `nonpad_kv_seqlen`.
    ///
    /// Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
    /// The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:
    ///
    /// ```
    ///   The following pattern is applied by this operator:
    ///       Q          K          V
    ///       |          |          |
    /// Q*sqrt(scale) K*sqrt(scale) |
    ///       |          |          |
    ///       |       Transpose     |
    ///       |          |          |
    ///       ---MatMul---          |
    ///             |               |
    ///  at_mask---Add              |
    ///             |               |
    ///   softcap (if provided)     |
    ///             |               |
    ///          Softmax            |
    ///             |               |
    ///             -----MatMul------
    ///                    |
    ///                    Y
    /// ```
    public static <T1, T2, U> AttentionResult<T1, T2> Attention(Tensor<T1> Q, Tensor<T1> K, Tensor<T2> V, Optional<Tensor<U>> attn_mask, Optional<Tensor<T1>> past_key, Optional<Tensor<T2>> past_value, Optional<Tensor<Long>> nonpad_kv_seqlen, Optional<Long> qk_matmul_output_mode, Optional<Float> softcap, Optional<Long> softmax_precision, Optional<Float> scale, Optional<Long> is_causal, Optional<Long> q_num_heads, Optional<Long> kv_num_heads) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Attention.class, List.of(Q, K, V, attn_mask, past_key, past_value, nonpad_kv_seqlen), List.of(qk_matmul_output_mode, softcap, softmax_precision, scale, is_causal, q_num_heads, kv_num_heads));
        Object[] resultArray = (Object[]) result;
        return new AttentionResult<>((Tensor<T1>)resultArray[0], (Tensor<T1>)resultArray[1], (Tensor<T2>)resultArray[2], (Tensor<T1>)resultArray[3]);
    }

    ///
    ///  AveragePool consumes an input tensor X and applies average pooling across
    ///  the tensor according to kernel sizes, stride sizes, and pad lengths.
    ///  average pooling consisting of computing the average on all values of a
    ///  subset of the input tensor according to the kernel size and downsampling the
    ///  data into the output tensor Y for further processing. The output spatial shape is calculated differently
    ///  depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
    ///  With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
    ///  ```
    ///  output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
    ///  ```
    ///  or
    ///  ```
    ///  output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
    ///  ```
    ///  if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.
    ///
    ///  `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
    ///  ```
    ///  VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    ///  SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    ///  ```
    ///  or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
    ///  ```
    ///  VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
    ///  SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
    ///  ```
    ///  And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
    ///  ```
    ///  pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
    ///  ```
    ///  The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
    ///
    public static <T> Tensor<T> AveragePool(Tensor<T> X, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<Long> count_include_pad, Optional<Long> ceil_mode, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.AveragePool.class, List.of(X), List.of(pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape));
        return (Tensor<T>) result;
    }

    public record BatchNormalizationResult<T, T2>(Tensor<T> Y, Tensor<T2> running_mean, Tensor<T2> running_var) { }
    ///
    /// Carries out batch normalization as described in the paper
    /// https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    /// There are five required inputs 'X', 'scale', 'B', 'input_mean' and
    /// 'input_var'.
    /// Note that 'input_mean' and 'input_var' are expected to be the estimated
    /// statistics in inference mode (training_mode=False, default),
    /// and the running statistics in training mode (training_mode=True).
    /// There are multiple cases for the number of outputs, which we list below:
    ///
    /// * Output case #1: Y, running_mean, running_var (training_mode=True)
    /// * Output case #2: Y (training_mode=False)
    ///
    /// When training_mode=False, extra outputs are invalid.
    /// The outputs are updated as follows when training_mode=True:
    /// ```
    /// running_mean = input_mean * momentum + current_mean * (1 - momentum)
    /// running_var = input_var * momentum + current_var * (1 - momentum)
    ///
    /// Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
    /// ```
    /// where:
    /// ```
    /// current_mean = ReduceMean(X, axis=all_except_channel_index)
    /// current_var =  ReduceVar(X, axis=all_except_channel_index)
    /// ```
    /// Notice that `ReduceVar` refers to the population variance, and it equals to
    /// `sum(sqrd(x_i - x_avg)) / N`
    /// where `N` is the population size (this formula does not use sample size `N - 1`).
    ///
    /// The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.
    ///
    /// When training_mode=False:
    /// ```
    /// Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    /// ```
    ///
    /// For previous (depreciated) non-spatial cases, implementors are suggested
    /// to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
    /// This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    public static <T, T1, T2> BatchNormalizationResult<T, T2> BatchNormalization(Tensor<T> X, Tensor<T1> scale, Tensor<T1> B, Tensor<T2> input_mean, Tensor<T2> input_var, Optional<Float> epsilon, Optional<Long> training_mode, Optional<Float> momentum) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BatchNormalization.class, List.of(X, scale, B, input_mean, input_var), List.of(epsilon, training_mode, momentum));
        Object[] resultArray = (Object[]) result;
        return new BatchNormalizationResult<>((Tensor<T>)resultArray[0], (Tensor<T2>)resultArray[1], (Tensor<T2>)resultArray[2]);
    }

    ///
    /// Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
    /// containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
    /// where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).
    ///
    /// This operator is non-deterministic and may not produce the same values in different
    /// implementations (even if a seed is specified).
    public static <T1, T2> Tensor<T2> Bernoulli(Tensor<T1> input, Optional<Float> seed, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Bernoulli.class, List.of(input), List.of(seed, dtype));
        return (Tensor<T2>) result;
    }

    ///
    ///     Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
    public static <T> Tensor<T> Binarizer(Tensor<T> X, Optional<Float> threshold) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Binarizer.class, List.of(X), List.of(threshold));
        return (Tensor<T>) result;
    }

    ///
    /// Bitwise shift operator performs element-wise operation. For each input element, if the
    /// attribute "direction" is "RIGHT", this operator moves its binary representation toward
    /// the right side so that the input value is effectively decreased. If the attribute "direction"
    /// is "LEFT", bits of binary representation moves toward the left side, which results the
    /// increase of its actual value. The input X is the tensor to be shifted and another input
    /// Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
    /// and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
    /// X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].
    ///
    /// Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
    /// not necessarily identical.
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> BitShift(Tensor<T> X, Tensor<T> Y, String direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitShift.class, List.of(X, Y), List.of(direction));
        return (Tensor<T>) result;
    }

    ///
    /// Returns the tensor resulting from performing the bitwise `and` operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> BitwiseAnd(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseAnd.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Returns the bitwise not of the input tensor element-wise.
    public static <T> Tensor<T> BitwiseNot(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseNot.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Returns the tensor resulting from performing the bitwise `or` operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> BitwiseOr(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseOr.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Returns the tensor resulting from performing the bitwise `xor` operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> BitwiseXor(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BitwiseXor.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.
    public static <T1, T2> Tensor<T2> BlackmanWindow(Tensor<T1> size, Optional<Long> periodic, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.BlackmanWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }

    ///
    /// The operator casts the elements of a given input tensor to a data type
    /// specified by the 'to' argument and returns an output tensor of the same size in
    /// the converted type. The 'to' argument must be one of the data types specified
    /// in the 'DataType' enum field in the TensorProto message.
    ///
    /// Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
    /// (e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
    /// yield result 100. There are some string literals reserved for special floating-point values;
    /// "+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
    /// Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
    /// this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
    /// to string tensors, plain floating-point representation (such as "314.15926") would be used.
    /// Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
    /// of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.
    ///
    /// Conversion from a numerical type to any numerical type is always allowed.
    /// User must be aware of precision loss and value change caused by range difference between two types.
    /// For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
    /// an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.
    ///
    /// In more detail, the conversion among numerical types should follow these rules
    /// if the destination type is not a float 8 type.
    ///
    /// * Casting from floating point to:
    ///   * floating point: +/- infinity if OOR (out of range).
    ///   * fixed point: undefined if OOR.
    ///   * bool: +/- 0.0 to False; all else to True.
    /// * Casting from fixed point to:
    ///   * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
    ///   * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    ///     signed types). For example, 200 (int16) -> -56 (int8).
    ///   * bool: zero to False; nonzero to True.
    /// * Casting from bool to:
    ///   * floating point: `{1.0, 0.0}`.
    ///   * fixed point: `{1, 0}`.
    ///   * bool: no change.
    ///
    /// Float 8 types (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ) were introduced to speed up the training of
    /// deep models. By default the conversion of a float *x* obeys
    /// to the following rules. `[x]` means the value rounded to
    /// the target mantissa width.
    ///
    /// | x                 | E4M3FN   | E4M3FNUZ | E5M2     | E5M2FNUZ |
    /// | ----------------- | -------- | -------- | -------- | -------- |
    /// | 0                 | 0        | 0        | 0        | 0        |
    /// | -0                | -0       | 0        | -0       | 0        |
    /// | NaN               | NaN      | NaN      | NaN      | NaN      |
    /// | Inf               | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
    /// | -Inf              | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
    /// | \[x\] > FLT_MAX   | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
    /// | \[x\] \< -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
    /// | else              | RNE      | RNE      | RNE      | RNE      |
    ///
    /// The behavior changes if the parameter 'saturate' is set to False.
    /// The rules then become:
    ///
    /// | x                 | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
    /// | ----------------- | ------ | -------- | ---- | -------- |
    /// | 0                 | 0      | 0        | 0    | 0        |
    /// | -0                | -0     | 0        | -0   | 0        |
    /// | NaN               | NaN    | NaN      | NaN  | NaN      |
    /// | -NaN              | -NaN   | NaN      | -NaN | NaN      |
    /// | Inf               | NaN    | NaN      | Inf  | NaN      |
    /// | -Inf              | -NaN   | NaN      | -Inf | NaN      |
    /// | \[x\] > FLT_MAX   | NaN    | NaN      | Inf  | NaN      |
    /// | \[x\] \< -FLT_MAX | NaN    | NaN      | -Inf | NaN      |
    /// | else              | RNE    | RNE      | RNE  | RNE      |
    ///
    /// FLOAT8E8M0 type was introduced to enable [Microscaling (MX) formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
    /// When casting to FLOAT8E8M0, the rounding behavior can be specified using the `round_mode` and `saturate` attributes.
    /// The current CUDA behavior is to round up and saturate. Casting negative values to FLOAT8E8M0 gives undefined behavior.
    /// The following table describes the casting behavior of special values to FLOAT8E8M0 in the two most common cases.
    ///
    /// | x                 | saturate + up | non-saturate + nearest |
    /// | ----------------- | ------------- | ---------------------  |
    /// | 0                 | 0             | NaN                    |
    /// | -0                | Unspecified   | Unspecified            |
    /// | NaN               | NaN           | NaN                    |
    /// | Inf               | E8M0_MAX      | NaN                    |
    /// | x > E8M0_MAX      | E8M0_MAX      | NaN                    |
    /// | x \< E8M0_MIN     | E8M0_MIN      | NaN                    |
    /// | x \< 0            | Unspecified   | Unspecified            |
    public static <T1, T2> Tensor<T2> Cast(Tensor<T1> input, Optional<Long> saturate, long to, Optional<String> round_mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cast.class, List.of(input), List.of(saturate, to, round_mode));
        return (Tensor<T2>) result;
    }

    ///
    /// The operator casts the elements of a given input tensor (the first input) to
    /// the same data type as the elements of the second input tensor.
    /// See documentation of the Cast operator for further details.
    public static <T1, T2> Tensor<T2> CastLike(Tensor<T1> input, Tensor<T2> target_type, Optional<Long> saturate, Optional<String> round_mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CastLike.class, List.of(input, target_type), List.of(saturate, round_mode));
        return (Tensor<T2>) result;
    }

    ///
    ///     Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    ///     in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    ///     If using sparse packing, the key cannot exceed the max_map-1 value.
    public static <T1, T2> Tensor<T2> CastMap(Map<Long, T1> X, Optional<String> map_form, Optional<String> cast_to, Optional<Long> max_map) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CastMap.class, List.of(X), List.of(map_form, cast_to, max_map));
        return (Tensor<T2>) result;
    }

    ///
    ///     Converts strings to integers and vice versa.<br>
    ///     Two sequences of equal length are used to map between integers and strings,
    ///     with strings and integers at the same index detailing the mapping.<br>
    ///     Each operator converts either integers to strings or strings to integers, depending
    ///     on which default value attribute is provided. Only one default value attribute
    ///     should be defined.<br>
    ///     If the string default value is set, it will convert integers to strings.
    ///     If the int default value is set, it will convert strings to integers.
    public static <T1, T2> Tensor<T2> CategoryMapper(Tensor<T1> X, Optional<long[]> cats_int64s, Optional<String[]> cats_strings, Optional<Long> default_int64, Optional<String> default_string) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CategoryMapper.class, List.of(X), List.of(cats_int64s, cats_strings, default_int64, default_string));
        return (Tensor<T2>) result;
    }

    ///
    /// Ceil takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the ceil is, y = ceil(x), is applied to
    /// the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
    public static <T> Tensor<T> Ceil(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Ceil.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Continuously Differentiable Exponential Linear Units:
    /// Perform the linear unit element-wise on the input tensor X
    /// using formula:
    ///
    /// ```
    /// max(0,x) + min(0,alpha*(exp(x/alpha)-1))
    /// ```
    public static Tensor<Float> Celu(Tensor<Float> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Celu.class, List.of(X), List.of(alpha));
        return (Tensor<Float>) result;
    }

    ///
    /// Center crop or pad an input to given dimensions.
    ///
    /// The crop/pad dimensions can be specified for a subset of the `axes`; unspecified dimensions will remain unchanged.
    ///
    /// If the input dimensions are larger than the target crop dimensions, a centered cropping window will be extracted
    /// from the input. The starting value for the cropping window is rounded down, which means that if the difference
    /// between the input shape and the crop shape is odd, the cropping window will be shifted half a pixel to the left
    /// of the input center.
    ///
    /// If the input dimensions are smaller than the target crop dimensions, the input will be padded equally on both sides
    /// to center it in the output. In cases where the total number of padding pixels is odd, an additional pixel will be
    /// added to the right side.
    ///
    /// The padding value used is zero.
    public static <T, Tind> Tensor<T> CenterCropPad(Tensor<T> input_data, Tensor<Tind> shape, Optional<long[]> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CenterCropPad.class, List.of(input_data, shape), List.of(axes));
        return (Tensor<T>) result;
    }

    ///
    /// Clip operator limits the given input within an interval. The interval is
    /// specified by the inputs 'min' and 'max'. They default to
    /// numeric_limits::lowest() and numeric_limits::max(), respectively.
    /// When 'min' is greater than 'max', the clip operator sets all the 'input' values to
    /// the value of 'max'. Thus, this is equivalent to 'Min(max, Max(input, min))'.
    public static <T> Tensor<T> Clip(Tensor<T> input, Optional<Tensor<T>> min, Optional<Tensor<T>> max) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Clip.class, List.of(input, min, max), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// The operator rearranges column blocks back into a multidimensional image
    ///
    /// Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
    /// but it only supports *batched* multi-dimensional image tensors.
    /// Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.
    ///
    /// NOTE:
    ///   Although specifying image_shape looks redundant because it could be calculated from
    ///   convolution formulas, it is required as input for more advanced scenarios as explained
    ///   at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)
    public static <T> Tensor<T> Col2Im(Tensor<T> input, Tensor<Long> image_shape, Tensor<Long> block_shape, Optional<long[]> pads, Optional<long[]> dilations, Optional<long[]> strides) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Col2Im.class, List.of(input, image_shape, block_shape), List.of(pads, dilations, strides));
        return (Tensor<T>) result;
    }

    ///
    ///     Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    ///     In case axis is not provided, input is flattened before elements are selected.
    ///     Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    ///
    public static <T> Tensor<T> Compress(Tensor<T> input, Tensor<Boolean> condition, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Compress.class, List.of(input, condition), List.of(axis));
        return (Tensor<T>) result;
    }

    /// Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
    public static <T> Tensor<T> Concat(List<Tensor<T>> inputs, long axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Concat.class, List.of(inputs), List.of(axis));
        return (Tensor<T>) result;
    }

    ///
    /// Concatenate a sequence of tensors into a single tensor.
    /// All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
    /// By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
    /// When 'new_axis' is 1, the behavior is similar to numpy.stack.
    public static <S, T> Tensor<T> ConcatFromSequence(List<Tensor<S>> input_sequence, long axis, Optional<Long> new_axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConcatFromSequence.class, List.of(input_sequence), List.of(axis, new_axis));
        return (Tensor<T>) result;
    }

    ///
    /// This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
    /// or value_* must be specified.
    public static <T> Tensor<T> Constant(Optional<Long> value_int, Optional<float[]> value_floats, Optional<String[]> value_strings, Optional<Float> value_float, Optional<String> value_string, Optional<long[]> value_ints, Optional<byte[]> sparse_value, Optional<Tensor> value) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Constant.class, List.of(), List.of(value_int, value_floats, value_strings, value_float, value_string, value_ints, sparse_value, value));
        return (Tensor<T>) result;
    }

    ///
    /// Generate a tensor with given value and shape.
    public static <T2> Tensor<T2> ConstantOfShape(Tensor<Long> input, Optional<Tensor> value) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConstantOfShape.class, List.of(input), List.of(value));
        return (Tensor<T2>) result;
    }

    ///
    /// The convolution operator consumes an input tensor and a filter, and
    /// computes the output.
    public static <T> Tensor<T> Conv(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Conv.class, List.of(X, W, B), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T>) result;
    }

    ///
    /// The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
    /// and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
    public static <T1, T2> Tensor<Integer> ConvInteger(Tensor<T1> x, Tensor<T2> w, Optional<Tensor<T1>> x_zero_point, Optional<Tensor<T2>> w_zero_point, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConvInteger.class, List.of(x, w, x_zero_point, w_zero_point), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<Integer>) result;
    }

    ///
    /// The convolution transpose operator consumes an input tensor and a filter,
    /// and computes the output.
    ///
    /// If the pads parameter is provided the shape of the output is calculated via the following equation:
    ///
    ///   output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
    ///
    /// output_shape can also be explicitly specified in which case pads values are auto generated using these equations:
    ///
    ///   total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
    ///   If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
    ///   Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
    ///
    ///
    public static <T> Tensor<T> ConvTranspose(Tensor<T> X, Tensor<T> W, Optional<Tensor<T>> B, Optional<long[]> output_shape, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape, Optional<long[]> output_padding) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ConvTranspose.class, List.of(X, W, B), List.of(output_shape, pads, dilations, auto_pad, strides, group, kernel_shape, output_padding));
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the cosine of the given input tensor, element-wise.
    public static <T> Tensor<T> Cos(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cos.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the hyperbolic cosine of the given input tensor element-wise.
    public static <T> Tensor<T> Cosh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Cosh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Performs cumulative sum of the input elements along the given axis.
    /// By default, it will do the sum inclusively meaning the first element is copied as is.
    /// Through an `exclusive` attribute, this behavior can change to exclude the first element.
    /// It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.
    ///
    /// Example:
    /// ```
    /// input_x = [1, 2, 3]
    /// axis=0
    /// output = [1, 3, 6]
    /// exclusive=1
    /// output = [0, 1, 3]
    /// exclusive=0
    /// reverse=1
    /// output = [6, 5, 3]
    /// exclusive=1
    /// reverse=1
    /// output = [5, 3, 0]
    /// ```
    ///
    public static <T, T2> Tensor<T> CumSum(Tensor<T> x, Tensor<T2> axis, Optional<Long> exclusive, Optional<Long> reverse) {
        Object result = OnnxInterpreter.interpret(OnnxOps.CumSum.class, List.of(x, axis), List.of(exclusive, reverse));
        return (Tensor<T>) result;
    }

    /// Computes the discrete Fourier Transform (DFT) of the input.
    ///
    /// Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
    /// DFT is computed and `M` denotes the conceptual "all other dimensions,"
    /// the DFT `y[m, k]` of shape `[M, N]` is defined as
    ///
    /// $$y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,$$
    ///
    /// and the inverse transform is defined as
    ///
    /// $$x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,$$
    ///
    /// where $j$ is the imaginary unit.
    ///
    /// The actual shape of the output is specified in the "output" section.
    ///
    /// Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html
    public static <T1, T2> Tensor<T1> DFT(Tensor<T1> input, Optional<Tensor<T2>> dft_length, Optional<Tensor<Long>> axis, Optional<Long> inverse, Optional<Long> onesided) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DFT.class, List.of(input, dft_length, axis), List.of(inverse, onesided));
        return (Tensor<T1>) result;
    }

    ///
    /// Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
    /// This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
    public static <T> Tensor<T> DeformConv(Tensor<T> X, Tensor<T> W, Tensor<T> offset, Optional<Tensor<T>> B, Optional<Tensor<T>> mask, Optional<long[]> pads, Optional<long[]> dilations, Optional<long[]> strides, Optional<Long> offset_group, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DeformConv.class, List.of(X, W, offset, B, mask), List.of(pads, dilations, strides, offset_group, group, kernel_shape));
        return (Tensor<T>) result;
    }

    /// DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
    /// This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
    /// the input tensor where values from the depth dimension are moved in spatial blocks to the height
    /// and width dimensions. By default, `mode` = `DCR`.
    /// In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
    /// following order: depth, column, and then row. The output y is computed from the input x as below:
    ///
    /// ```
    /// b, c, h, w = x.shape
    /// tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
    /// tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    /// y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
    /// ```
    ///
    /// In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
    /// following order: column, row, and the depth. The output y is computed from the input x as below:
    ///
    /// ```
    /// b, c, h, w = x.shape
    /// tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
    /// tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
    /// y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
    /// ```
    public static <T> Tensor<T> DepthToSpace(Tensor<T> input, Optional<String> mode, long blocksize) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DepthToSpace.class, List.of(input), List.of(mode, blocksize));
        return (Tensor<T>) result;
    }

    ///
    /// The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
    /// full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
    /// must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
    /// a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
    /// See QuantizeLinear for details on quantization granularity.
    ///
    /// `x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
    /// `int32`, there's no zero point (zero point is supposed to be 0).
    /// `zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
    /// for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
    /// is the same as `x_scale`. The output type also determines the precision of the multiplication operation.
    public static <T1, T2, T3> Tensor<T3> DequantizeLinear(Tensor<T1> x, Tensor<T2> x_scale, Optional<Tensor<T1>> x_zero_point, Optional<Long> output_dtype, Optional<Long> axis, Optional<Long> block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DequantizeLinear.class, List.of(x, x_scale, x_zero_point), List.of(output_dtype, axis, block_size));
        return (Tensor<T3>) result;
    }

    ///
    /// Det calculates determinant of a square matrix or batches of square matrices.
    /// Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
    /// and the inner-most 2 dimensions form square matrices.
    /// The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
    /// e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
    public static <T> Tensor<T> Det(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Det.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    ///     Uses an index mapping to convert a dictionary to an array.<br>
    ///     Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
    ///     the key type. The index into the vocabulary array at which the key is found is then
    ///     used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
    ///     The key type of the input map must correspond to the element type of the defined vocabulary attribute.
    ///     Therefore, the output array will be equal in length to the index mapping vector parameter.
    ///     All keys in the input dictionary must be present in the index mapping vector.
    ///     For each item in the input dictionary, insert its value in the output array.
    ///     Any keys not present in the input dictionary, will be zero in the output array.<br>
    ///     For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    ///     then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
    ///
    public static <T2> Tensor<T2> DictVectorizer(Map<?, ?> X, Optional<String[]> string_vocabulary, Optional<long[]> int64_vocabulary) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DictVectorizer.class, List.of(X), List.of(string_vocabulary, int64_vocabulary));
        return (Tensor<T2>) result;
    }

    ///
    /// Performs element-wise binary division (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    ///
    /// (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
    public static <T> Tensor<T> Div(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Div.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    public record DropoutResult<T>(Tensor<T> output, Tensor<Boolean> mask) { }
    ///
    /// Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
    /// output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
    /// Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
    /// the user can simply not pass `training_mode` input or set it to false.
    /// ```
    /// output = scale * data * mask,
    /// ```
    /// where
    /// ```
    /// scale = 1. / (1. - ratio).
    /// ```
    /// This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    public static <T, T1> DropoutResult<T> Dropout(Tensor<T> data, Optional<Tensor<T1>> ratio, Optional<Tensor<Boolean>> training_mode, Optional<Long> seed) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Dropout.class, List.of(data, ratio, training_mode), List.of(seed));
        Object[] resultArray = (Object[]) result;
        return new DropoutResult<>((Tensor<T>)resultArray[0], (Tensor<Boolean>)resultArray[1]);
    }

    public record DynamicQuantizeLinearResult(Tensor<Byte> y, Tensor<Float> y_scale, Tensor<Byte> y_zero_point) { }
    ///
    /// A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
    /// Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
    /// Scale is calculated as:
    /// ```
    /// y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
    /// ```
    ///
    /// * where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
    /// * data range is adjusted to include 0.
    ///
    /// Zero point is calculated as:
    /// ```
    /// intermediate_zero_point = qmin - min(x)/y_scale
    /// y_zero_point = cast(round(saturate(itermediate_zero_point)))
    /// ```
    ///
    /// * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
    /// * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
    /// * rounding to nearest ties to even.
    ///
    /// Data quantization formula is:
    /// ```
    /// y = saturate (round (x / y_scale) + y_zero_point)
    /// ```
    ///
    /// * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
    /// * rounding to nearest ties to even.
    public static DynamicQuantizeLinearResult DynamicQuantizeLinear(Tensor<Float> x) {
        Object result = OnnxInterpreter.interpret(OnnxOps.DynamicQuantizeLinear.class, List.of(x), List.of());
        Object[] resultArray = (Object[]) result;
        return new DynamicQuantizeLinearResult((Tensor<Byte>)resultArray[0], (Tensor<Float>)resultArray[1], (Tensor<Byte>)resultArray[2]);
    }

    ///
    /// An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation
    ///
    /// ```
    /// output[output-term] = reduce-sum( input1[term1] * input2[term2] )
    /// ```
    ///
    /// where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
    /// that do not occur in the output-term.
    ///
    /// The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
    /// convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
    /// an operand tensor, and the characters within the terms correspond to operands dimensions.
    ///
    /// This sequence may be followed by "->" to separate the left and right hand side of the equation.
    /// If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
    /// summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
    /// output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
    /// equation.
    ///
    /// When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
    ///
    /// The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
    /// Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
    /// The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
    /// beginning of the output. The equation string may contain space (U+0020) character.
    public static <T> Tensor<T> Einsum(List<Tensor<T>> Inputs, String equation) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Einsum.class, List.of(Inputs), List.of(equation));
        return (Tensor<T>) result;
    }

    ///
    /// Elu takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
    /// 0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
    public static <T> Tensor<T> Elu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Elu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `equal` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<Boolean> Equal(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Equal.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Computes the error function of the given input tensor element-wise.
    public static <T> Tensor<T> Erf(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Erf.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the exponential of the given input tensor, element-wise.
    public static <T> Tensor<T> Exp(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Exp.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Broadcast the input tensor following the given shape and the broadcast rule.
    /// The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
    /// Dimensions are right alignment;
    /// Two corresponding dimensions must have the same value, or one of them is equal to 1.
    /// Also, this operator is similar to numpy.broadcast_to(input, shape),
    /// but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
    /// It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
    /// or the shape.ndim < input.shape.ndim.
    public static <T> Tensor<T> Expand(Tensor<T> input, Tensor<Long> shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Expand.class, List.of(input, shape), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
    /// tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
    /// same as the input tensor. The data type can be specified by the 'dtype' argument. If
    /// 'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
    /// is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
    /// The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
    /// TensorProto message and be valid as an output type.
    public static <T1, T2> Tensor<T2> EyeLike(Tensor<T1> input, Optional<Long> dtype, Optional<Long> k) {
        Object result = OnnxInterpreter.interpret(OnnxOps.EyeLike.class, List.of(input), List.of(dtype, k));
        return (Tensor<T2>) result;
    }

    ///
    ///     Concatenates input tensors into one continuous output.<br>
    ///     All input shapes are 2-D and are concatenated along the second dimension. 1-D tensors are treated as [1,C].
    ///     Inputs are copied to the output maintaining the order of the input arguments.<br>
    ///     All inputs must be integers or floats, while the output will be all floating point values.
    public static <T1> Tensor<Float> FeatureVectorizer(List<Tensor<T1>> X, Optional<long[]> inputdimensions) {
        Object result = OnnxInterpreter.interpret(OnnxOps.FeatureVectorizer.class, List.of(X), List.of(inputdimensions));
        return (Tensor<Float>) result;
    }

    ///
    /// Flattens the input tensor into a 2D matrix. If input tensor has shape
    /// (d_0, d_1, ... d_n) then the output will have shape
    /// (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
    public static <T> Tensor<T> Flatten(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Flatten.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    ///
    /// Floor takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the floor is, y = floor(x), is applied to
    /// the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
    public static <T> Tensor<T> Floor(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Floor.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    public record GRUResult<T>(Tensor<T> Y, Tensor<T> Y_h) { }
    ///
    /// Computes an one-layer GRU. This operator is usually supported via some custom
    /// implementation such as CuDNN.
    ///
    /// Notations:
    ///
    /// * `X` - input tensor
    /// * `z` - update gate
    /// * `r` - reset gate
    /// * `h` - hidden gate
    /// * `t` - time step (t-1 means previous time step)
    /// * `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
    /// * `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
    /// * `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
    /// * `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
    /// * `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
    /// * `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
    /// * `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
    /// * `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
    /// * `H` - Hidden state
    /// * `num_directions` - 2 if direction == bidirectional else 1
    ///
    /// Activation functions:
    ///
    /// * Relu(x)                - max(0, x)
    /// * Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
    /// * Sigmoid(x)             - 1/(1 + e^{-x})
    ///
    /// NOTE:
    ///   Below are optional
    ///
    /// * Affine(x)              - alpha * x + beta
    /// * LeakyRelu(x)           - x if x >= 0 else alpha * x
    /// * ThresholdedRelu(x)     - x if x >= alpha else 0
    /// * ScaledTanh(x)          - alpha * Tanh(beta * x)
    /// * HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
    /// * Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
    /// * Softsign(x)            - x/(1 + |x|)
    /// * Softplus(x)            - log(1 + e^x)
    ///
    /// Equations (Default: f=Sigmoid, g=Tanh):
    ///
    /// * zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    /// * rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    /// * ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
    /// * ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
    /// * Ht = (1 - zt) (.) ht + zt (.) Ht-1
    /// This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    public static <T> GRUResult<T> GRU(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<Integer>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Long> layout, Optional<float[]> activation_alpha, Optional<Long> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Long> linear_before_reset, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GRU.class, List.of(X, W, R, B, sequence_lens, initial_h), List.of(layout, activation_alpha, hidden_size, activation_beta, activations, linear_before_reset, clip, direction));
        Object[] resultArray = (Object[]) result;
        return new GRUResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1]);
    }

    ///
    /// Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
    /// entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
    /// them in an output tensor of rank q + (r - 1).
    ///
    /// It is an indexing operation that indexes into the input `data` along a single (specified) axis.
    /// Each entry in `indices` produces a `r-1` dimensional slice of the input tensor.
    /// The entire operation produces, conceptually, a `q`-dimensional tensor of `r-1` dimensional slices,
    /// which is arranged into a `q + (r-1)`-dimensional tensor, with the `q` dimensions taking the
    /// place of the original `axis` that is being indexed into.
    ///
    /// The following few examples illustrate how `Gather` works for specific shapes of `data`,
    /// `indices`, and given value of `axis`:
    /// | data shape | indices shape | axis | output shape | output equation |
    /// | --- | --- | --- | --- | --- |
    /// | (P, Q) | ( )  (a scalar)   | 0 | (Q)       | output[q] = data[indices, q] |
    /// | (P, Q, R) | ( )  (a scalar)   | 1 | (P, R)       | output[p, r] = data[p, indices, r] |
    /// | (P, Q) | (R, S) | 0 | (R, S, Q) | output[r, s, q] = data[ [indices[r, s], q] |
    /// | (P, Q) | (R, S) | 1 | (P, R, S) | output[p, r, s] = data[ p, indices[r, s]] |
    ///
    /// More generally, if `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]`
    /// then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:
    ///
    /// ```
    /// data = [
    ///     [1.0, 1.2],
    ///     [2.3, 3.4],
    ///     [4.5, 5.7],
    /// ]
    /// indices = [
    ///     [0, 1],
    ///     [1, 2],
    /// ]
    /// output = [
    ///     [
    ///         [1.0, 1.2],
    ///         [2.3, 3.4],
    ///     ],
    ///     [
    ///         [2.3, 3.4],
    ///         [4.5, 5.7],
    ///     ],
    /// ]
    /// ```
    ///
    /// If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]`
    /// then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:
    ///
    /// ```
    /// data = [
    ///     [1.0, 1.2, 1.9],
    ///     [2.3, 3.4, 3.9],
    ///     [4.5, 5.7, 5.9],
    /// ]
    /// indices = [
    ///     [0, 2],
    /// ]
    /// axis = 1,
    /// output = [
    ///         [[1.0, 1.9]],
    ///         [[2.3, 3.9]],
    ///         [[4.5, 5.9]],
    /// ]
    /// ```
    public static <T, Tind> Tensor<T> Gather(Tensor<T> data, Tensor<Tind> indices, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gather.class, List.of(data, indices), List.of(axis));
        return (Tensor<T>) result;
    }

    ///
    ///
    /// GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
    /// and an optional attribute `axis` that identifies an axis of `data`
    /// (by default, the outer-most axis, that is axis 0). It is an indexing operation
    /// that produces its output by indexing into the input data tensor at index
    /// positions determined by elements of the `indices` tensor.
    /// Its output shape is the same as the shape of `indices` and consists of one value
    /// (gathered from the `data`) for each element in `indices`.
    ///
    /// For instance, in the 3-D case (r = 3), the output produced is determined
    /// by the following equations:
    /// ```
    /// out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
    /// out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
    /// out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
    /// ```
    ///
    /// This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.
    ///
    /// Example 1:
    /// ```
    /// data = [
    ///     [1, 2],
    ///     [3, 4],
    /// ]
    /// indices = [
    ///     [0, 0],
    ///     [1, 0],
    /// ]
    /// axis = 1
    /// output = [
    ///     [1, 1],
    ///     [4, 3],
    /// ]
    /// ```
    /// Example 2:
    /// ```
    /// data = [
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]
    /// indices = [
    ///     [1, 2, 0],
    ///     [2, 0, 0],
    /// ]
    /// axis = 0
    /// output = [
    ///     [4, 8, 3],
    ///     [7, 2, 3],
    /// ]
    /// ```
    public static <T, Tind> Tensor<T> GatherElements(Tensor<T> data, Tensor<Tind> indices, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GatherElements.class, List.of(data, indices), List.of(axis));
        return (Tensor<T>) result;
    }

    ///
    /// Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
    /// slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.
    ///
    /// `indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
    /// where each element defines a slice of `data`
    ///
    /// `batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
    /// `data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.
    ///
    /// Some salient points about the inputs' rank and shape:
    ///
    /// 1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`
    ///
    /// 2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.
    ///
    /// 3) b < min(q, r) is to be honored.
    ///
    /// 4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)
    ///
    /// 5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
    ///    It is an error if any of the index values are out of bounds.
    ///
    /// The output is computed as follows:
    ///
    /// The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.
    ///
    /// 1) If `indices_shape[-1] > r-b` => error condition
    ///
    /// 2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
    ///    containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
    ///    of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
    ///    is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)
    ///
    /// 3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
    ///    containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
    ///    to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
    ///    to form the `output` tensor (Examples 2, 3, 4 and 5 below)
    ///
    /// This operator is the inverse of `ScatterND`.
    ///
    /// **Example 1**
    ///
    /// ```
    /// batch_dims = 0
    /// data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
    /// indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
    /// output  = [0,3]           # output_shape  = [2]
    /// ```
    ///
    /// **Example 2**
    ///
    /// ```
    /// batch_dims = 0
    /// data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
    /// indices = [[1],[0]]      # indices_shape = [2, 1]
    /// output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
    /// ```
    ///
    /// **Example 3**
    ///
    /// ```
    /// batch_dims = 0
    /// data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
    /// indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
    /// output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
    /// ```
    ///
    /// **Example 4**
    ///
    /// ```
    /// batch_dims = 0
    /// data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
    /// indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
    /// output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
    /// ```
    ///
    /// **Example 5**
    ///
    /// ```
    /// batch_dims = 1
    /// data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
    /// indices = [[1],[0]]                     # indices_shape = [2, 1]
    /// output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
    /// ```
    public static <T> Tensor<T> GatherND(Tensor<T> data, Tensor<Long> indices, Optional<Long> batch_dims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GatherND.class, List.of(data, indices), List.of(batch_dims));
        return (Tensor<T>) result;
    }

    ///
    /// Gelu takes one input data (Tensor<T>) and produces one
    /// output data (Tensor<T>) where the gaussian error linear units function,
    /// $y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
    /// If the attribute "approximate" is set to "tanh", the function estimation,
    /// $y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
    /// to the tensor elementwise.
    public static <T> Tensor<T> Gelu(Tensor<T> X, Optional<String> approximate) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gelu.class, List.of(X), List.of(approximate));
        return (Tensor<T>) result;
    }

    /// General Matrix multiplication:
    /// https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
    ///
    /// * A' = transpose(A) if transA else A
    /// * B' = transpose(B) if transB else B
    ///
    /// Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
    /// input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
    /// and output tensor Y has shape (M, N). A will be transposed before doing the
    /// computation if attribute transA is non-zero, same for B and transB.
    /// This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
    /// This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    public static <T> Tensor<T> Gemm(Tensor<T> A, Tensor<T> B, Optional<Tensor<T>> C, Optional<Float> alpha, Optional<Long> transB, Optional<Float> beta, Optional<Long> transA) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gemm.class, List.of(A, B, C), List.of(alpha, transB, beta, transA));
        return (Tensor<T>) result;
    }

    ///
    ///  GlobalAveragePool consumes an input tensor X and applies average pooling across
    ///  the values in the same channel. This is equivalent to AveragePool with kernel size
    ///  equal to the spatial dimension of input tensor.
    public static <T> Tensor<T> GlobalAveragePool(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalAveragePool.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    ///  GlobalLpPool consumes an input tensor X and applies lp pool pooling across
    ///  the values in the same channel. This is equivalent to LpPool with kernel size
    ///  equal to the spatial dimension of input tensor.
    public static <T> Tensor<T> GlobalLpPool(Tensor<T> X, Optional<Long> p) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalLpPool.class, List.of(X), List.of(p));
        return (Tensor<T>) result;
    }

    ///
    ///  GlobalMaxPool consumes an input tensor X and applies max pooling across
    ///  the values in the same channel. This is equivalent to MaxPool with kernel size
    ///  equal to the spatial dimension of input tensor.
    public static <T> Tensor<T> GlobalMaxPool(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GlobalMaxPool.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Gradient operator computes the partial derivatives of a specific tensor w.r.t.
    /// some other tensors. This operator is widely used in gradient-based training
    /// algorithms. To illustrate its use, let's consider a computation graph,
    ///
    /// ```
    /// X -----.
    ///        |
    ///        v
    /// W --> Conv --> H --> Gemm --> Y
    ///                       ^
    ///                       |
    ///                       Z
    /// ```
    ///
    /// , where W and Z are trainable tensors. Note that operators' attributes are
    /// omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
    /// Y with respect to W (Z). The user can compute gradient by inserting Gradient
    /// operator to form another graph shown below.
    ///
    /// ```
    /// W --> Conv --> H --> Gemm --> Y
    /// |      ^              ^
    /// |      |              |
    /// |      X              Z
    /// |      |              |
    /// |      |   .----------'
    /// |      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
    /// |      |   |   "xs" followed by "zs")
    /// |      v   v
    /// '---> Gradient(xs=["W", "Z"], zs=["X"], y="Y")
    ///        |   |
    ///        |   '-----------------------------------> dY/dW (1st output of Gradient)
    ///        |
    ///        '---------------------------------------> dY/dZ (2nd output of Gradient)
    /// ```
    ///
    /// By definition, the tensor "y" is a function of independent variables in "xs"
    /// and "zs". Since we only compute the gradient of "y" w.r.t. the differentiable
    /// variables in "xs", this Gradient only outputs dY/dW and dY/dZ. Note that "H"
    /// cannot appear in "xs" and "zs". The reason is that "H" can be determined by
    /// tensors "W" and "X" and therefore "H" is not an independent variable.
    ///
    /// All outputs are optional. If needed, for example, user can assign an empty
    /// string to the 1st output name of that Gradient to skip the generation of dY/dW.
    /// Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
    /// and LSTM.
    ///
    /// Gradient operator can compute derivative against intermediate tensors. For
    /// example, the gradient of Y with respect to H can be done via
    ///
    /// ```
    /// W --> Conv --> H --> Gemm --> Y
    ///        ^       |      ^
    ///        |       |      |
    ///        X       |      Z
    ///        .-------'      |
    ///        |   .----------'
    ///        |   | (H/Z is the 1st/2nd input of Gradient as shown in "xs")
    ///        v   v
    ///       Gradient(xs=["H", "Z"], y="Y")
    ///        |   |
    ///        |   '-----------------------------------> dY/dH (1st output of Gradient)
    ///        |
    ///        '---------------------------------------> dY/dZ (2nd output of Gradient)
    /// ```
    ///
    /// It is possible to represent high-order differentiation using Gradient operators.
    /// For example, given the following linear model:
    ///
    /// ```
    /// W --> Gemm --> Y --> Loss --> O
    ///        ^              ^
    ///        |              |
    ///        X              L
    /// ```
    ///
    /// To compute the 2nd order derivative of O with respect to W (denoted by
    /// d^2O/dW^2), one can do
    ///
    /// ```
    /// W --> Gemm --> Y --> Loss --> O
    /// |      ^              ^
    /// |      |              |
    /// |      X .------------L
    /// |      | |            |
    /// |      | |            v
    /// +------+-+> Gradient(xs=["X", "W"], zs=["L"], y="O") ---> dO/dX (1st output of Gradient)
    /// |      | |    |
    /// |      | |    '---> dO/dW (2nd output of Gradient)
    /// |      v v
    /// '---> Gradient(xs=["X", "W"], zs=["L"], y="dO/dW") ---> d(dO/dW)dX (1st output of
    ///        |                                                  Gradient)
    ///        |
    ///        |
    ///        '---> d^2O/dW^2 (2nd output of Gradient)
    /// ```
    ///
    /// The tensors named in attributes "xs", "zs", and "y" define the differentiated
    /// computation graph, and the inputs to Gradient node define the values at
    /// which the gradient is computed. We can feed different tensors to the identified
    /// graph. For example, one can compute the gradient of Y with respect to H at
    /// a specific value of H, H_1, by providing that value as an input to the Gradient
    /// node.
    ///
    /// ```
    /// W --> Conv --> H --> Gemm --> Y
    ///        ^              ^
    ///        |              |
    ///        X              Z
    ///
    ///           Z_1 (2nd input of Gradient)
    ///            |
    ///            v
    /// H_1 --> Gradient(xs=["H", "Z"], y="Y") ---> dY/dH when H = H_1 and Y = Y_1.
    ///            |
    ///            '------------------------------> dY/dZ (2nd output of Gradient)
    /// ```
    ///
    /// When the inputs of Gradient are the tensors named in "xs" and "zs", the
    /// computation can be optimized. More specifically, intermediate variables in
    /// forward pass can be reused if the gradient is computed via reverse-mode
    /// auto-differentiation.
    public static <T1, T2> List<Tensor<T2>> Gradient(List<Tensor<T1>> Inputs, String y, Optional<String[]> zs, String[] xs) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Gradient.class, List.of(Inputs), List.of(y, zs, xs));
        return (List<Tensor<T2>>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `greater` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<Boolean> Greater(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Greater.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `greater_equal` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<Boolean> GreaterOrEqual(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GreaterOrEqual.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
    /// For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
    /// the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
    /// the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
    /// More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
    /// the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).
    ///
    /// The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
    /// The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
    /// at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
    /// and a padding mode (for `grid` positions falling outside the 2-dimensional image).
    ///
    /// For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
    /// They are used to interpolate output values of `Y[n, c, h_out, w_out]`.
    ///
    /// The GridSample operator is often used in doing grid generator and sampler in the
    /// [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
    /// See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
    public static <T1, T2> Tensor<T1> GridSample(Tensor<T1> X, Tensor<T2> grid, Optional<String> mode, Optional<Long> align_corners, Optional<String> padding_mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GridSample.class, List.of(X, grid), List.of(mode, align_corners, padding_mode));
        return (Tensor<T1>) result;
    }

    ///
    /// A GroupNormalization function. Carries out group normalization as described in
    /// the paper https://arxiv.org/abs/1803.08494
    ///
    /// This operator transforms input according to
    /// ```
    /// y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
    /// ```
    /// where the mean and variance are computed per instance per group of channels, and
    /// `scale` and `bias` should be specified for each channel. The number of
    /// groups `num_groups` should be divisible by the number of channels so that there are
    /// an equal number of channels per group.
    ///
    /// The overall computation has two stages: the first stage normalizes the elements to
    /// have zero mean and unit variance for each instance in each group, and the second
    /// stage scales and shifts the results of the first stage. The floating-point precision
    /// used in the first stage is determined by the `stash_type` attribute. For example,
    /// if `stash_type` is 1, the operator casts all input variables to 32-bit float,
    /// performs the computation, and finally casts the normalized results back to the
    /// original type of `X`. The second stage does not depend on `stash_type`.
    ///
    /// When the number of groups is the same as the number of channels, this operator is
    /// equivalent to InstanceNormalization. When there is only one group, this operator
    /// is equivalent to LayerNormalization.
    public static <T> Tensor<T> GroupNormalization(Tensor<T> X, Tensor<T> scale, Tensor<T> bias, Optional<Float> epsilon, Optional<Long> stash_type, long num_groups) {
        Object result = OnnxInterpreter.interpret(OnnxOps.GroupNormalization.class, List.of(X, scale, bias), List.of(epsilon, stash_type, num_groups));
        return (Tensor<T>) result;
    }

    ///
    /// Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.
    public static <T1, T2> Tensor<T2> HammingWindow(Tensor<T1> size, Optional<Long> periodic, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HammingWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }

    ///
    /// Generates a Hann window as described in the paper https://ieeexplore.ieee.org/document/1455106.
    public static <T1, T2> Tensor<T2> HannWindow(Tensor<T1> size, Optional<Long> periodic, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HannWindow.class, List.of(size), List.of(periodic, output_datatype));
        return (Tensor<T2>) result;
    }

    ///
    /// HardSigmoid takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
    /// is applied to the tensor elementwise.
    public static <T> Tensor<T> HardSigmoid(Tensor<T> X, Optional<Float> alpha, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HardSigmoid.class, List.of(X), List.of(alpha, beta));
        return (Tensor<T>) result;
    }

    ///
    /// HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
    /// the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
    /// where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
    public static <T> Tensor<T> HardSwish(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.HardSwish.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// The operator computes the hardmax values for the given input:
    ///
    ///  Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise
    ///
    /// The "axis" attribute indicates the dimension along which Hardmax
    /// will be performed. The output tensor has the same shape
    /// and contains the Hardmax values of the corresponding input.
    public static <T> Tensor<T> Hardmax(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Hardmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    /// Identity operator
    public static <V> V Identity(V input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Identity.class, List.of(input), List.of());
        return (V) result;
    }

    /// Loads and decodes and image from a file. If it can't decode for any reason (e.g. corrupted encoded
    /// stream, invalid format, it will return an empty matrix).
    /// The following image formats are supported:
    /// * BMP
    /// * JPEG (note: Lossless JPEG support is optional)
    /// * JPEG2000
    /// * TIFF
    /// * PNG
    /// * WebP
    /// * Portable image format (PBM, PGM, PPM, PXM, PNM)
    /// Decoded images follow a channel-last layout: (Height, Width, Channels).
    /// **JPEG chroma upsampling method:**
    /// When upsampling the chroma components by a factor of 2, the pixels are linearly interpolated so that the
    /// centers of the output pixels are 1/4 and 3/4 of the way between input pixel centers.
    /// When rounding, 0.5 is rounded down and up at alternative pixels locations to prevent bias towards
    /// larger values (ordered dither pattern).
    /// Considering adjacent input pixels A, B, and C, B is upsampled to pixels B0 and B1 so that
    /// ```
    /// B0 = round_half_down((1/4) * A + (3/4) * B)
    /// B1 = round_half_up((3/4) * B + (1/4) * C)
    /// ```
    /// This method,  is the default chroma upsampling method in the well-established libjpeg-turbo library,
    /// also referred as "smooth" or "fancy" upsampling.
    public static Tensor<Byte> ImageDecoder(Tensor<Byte> encoded_stream, Optional<String> pixel_format) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ImageDecoder.class, List.of(encoded_stream), List.of(pixel_format));
        return (Tensor<Byte>) result;
    }

    ///
    ///     Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    ///     This operator is typically used to replace missing values in situations where they have a canonical
    ///     representation, such as -1, 0, NaN, or some extreme value.<br>
    ///     One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    ///     holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    ///     width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    ///     which one depends on whether floats or integers are being processed.<br>
    ///     The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
    public static <T> Tensor<T> Imputer(Tensor<T> X, Optional<Long> replaced_value_int64, Optional<Float> replaced_value_float, Optional<long[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Imputer.class, List.of(X), List.of(replaced_value_int64, replaced_value_float, imputed_value_int64s, imputed_value_floats));
        return (Tensor<T>) result;
    }

    ///
    /// Carries out instance normalization as described in the paper
    /// https://arxiv.org/abs/1607.08022.
    ///
    /// y = scale * (x - mean) / sqrt(variance + epsilon) + B,
    /// where mean and variance are computed per instance per channel.
    public static <T> Tensor<T> InstanceNormalization(Tensor<T> input, Tensor<T> scale, Tensor<T> B, Optional<Float> epsilon) {
        Object result = OnnxInterpreter.interpret(OnnxOps.InstanceNormalization.class, List.of(input, scale, B), List.of(epsilon));
        return (Tensor<T>) result;
    }

    /// Map infinity to true and other values to false.
    public static <T1> Tensor<Boolean> IsInf(Tensor<T1> X, Optional<Long> detect_negative, Optional<Long> detect_positive) {
        Object result = OnnxInterpreter.interpret(OnnxOps.IsInf.class, List.of(X), List.of(detect_negative, detect_positive));
        return (Tensor<Boolean>) result;
    }

    /// Returns which elements of the input are NaN.
    public static <T1> Tensor<Boolean> IsNaN(Tensor<T1> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.IsNaN.class, List.of(X), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
    /// It normalizes over local input regions.
    /// The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
    /// of shape `(N x C x D1 x D2, ..., Dk)`, its region is
    /// `{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.
    ///
    /// `square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
    /// where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.
    ///
    /// `Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
    public static <T> Tensor<T> LRN(Tensor<T> X, long size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LRN.class, List.of(X), List.of(size, alpha, bias, beta));
        return (Tensor<T>) result;
    }

    public record LSTMResult<T>(Tensor<T> Y, Tensor<T> Y_h, Tensor<T> Y_c) { }
    ///
    /// Computes an one-layer LSTM. This operator is usually supported via some
    /// custom implementation such as CuDNN.
    ///
    /// Notations:
    ///
    /// * `X` - input tensor
    /// * `i` - input gate
    /// * `o` - output gate
    /// * `f` - forget gate
    /// * `c` - cell gate
    /// * `t` - time step (t-1 means previous time step)
    /// * `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
    /// * `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
    /// * `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
    /// * `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
    /// * `P[iof]`  - P peephole weight vector for input, output, and forget gates
    /// * `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
    /// * `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
    /// * `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
    /// * `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
    /// * `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
    /// * `H` - Hidden state
    /// * `num_directions` - 2 if direction == bidirectional else 1
    ///
    /// Activation functions:
    ///
    /// * Relu(x)                - max(0, x)
    /// * Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
    /// * Sigmoid(x)             - 1/(1 + e^{-x})
    ///
    /// NOTE: Below are optional
    ///
    /// * Affine(x)              - alpha*x + beta
    /// * LeakyRelu(x)           - x if x >= 0 else alpha * x
    /// * ThresholdedRelu(x)     - x if x >= alpha else 0
    /// * ScaledTanh(x)          - alpha*Tanh(beta*x)
    /// * HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
    /// * Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
    /// * Softsign(x)            - x/(1 + |x|)
    /// * Softplus(x)            - log(1 + e^x)
    ///
    /// Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
    ///
    /// * it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    /// * ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    /// * ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    /// * Ct = ft (.) Ct-1 + it (.) ct
    /// * ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    /// * Ht = ot (.) h(Ct)
    /// This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    public static <T> LSTMResult<T> LSTM(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<Integer>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Tensor<T>> initial_c, Optional<Tensor<T>> P, Optional<Long> layout, Optional<Long> input_forget, Optional<float[]> activation_alpha, Optional<Long> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LSTM.class, List.of(X, W, R, B, sequence_lens, initial_h, initial_c, P), List.of(layout, input_forget, activation_alpha, hidden_size, activation_beta, activations, clip, direction));
        Object[] resultArray = (Object[]) result;
        return new LSTMResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1], (Tensor<T>)resultArray[2]);
    }

    ///
    ///     Maps each element in the input tensor to another value.<br>
    ///     The mapping is determined by the two parallel attributes, 'keys_*' and
    ///     'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    ///     would be mapped to the i-th value in the specified 'values_*' attribute. It
    ///     implies that input's element type and the element type of the specified
    ///     'keys_*' should be identical while the output type is identical to the
    ///     specified 'values_*' attribute. Note that the 'keys_*' and 'values_*' attributes
    ///     must have the same length. If an input element can not be found in the
    ///     specified 'keys_*' attribute, the 'default_*' that matches the specified
    ///     'values_*' attribute may be used as its output value. The type of the 'default_*'
    ///     attribute must match the 'values_*' attribute chosen. <br>
    ///     Let's consider an example which maps a string tensor to an integer tensor.
    ///     Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
    ///     and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
    ///     "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
    ///     Since this operator is an one-to-one mapping, its input and output shapes
    ///     are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    ///     Float keys with value 'NaN' match any input 'NaN' value regardless of bit
    ///     value. If a key is repeated, the last key takes precedence.
    public static <T1, T2> Tensor<T2> LabelEncoder(Tensor<T1> X, Optional<String[]> values_strings, Optional<long[]> keys_int64s, Optional<Tensor> keys_tensor, Optional<String[]> keys_strings, Optional<Float> default_float, Optional<float[]> keys_floats, Optional<Tensor> default_tensor, Optional<Long> default_int64, Optional<Tensor> values_tensor, Optional<long[]> values_int64s, Optional<String> default_string, Optional<float[]> values_floats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LabelEncoder.class, List.of(X), List.of(values_strings, keys_int64s, keys_tensor, keys_strings, default_float, keys_floats, default_tensor, default_int64, values_tensor, values_int64s, default_string, values_floats));
        return (Tensor<T2>) result;
    }

    public record LayerNormalizationResult<T, U>(Tensor<T> Y, Tensor<U> Mean, Tensor<U> InvStdDev) { }
    ///
    ///       This is layer normalization defined in ONNX as function.
    ///       The overall computation can be split into two stages.
    ///       The first stage is standardization, which makes the
    ///       normalized elements have zero mean and unit variances.
    ///       The computation required by standardization can be
    ///       described by the following equations.
    ///       ```
    ///       Mean = ReduceMean<axes=normalized_axes>(X)
    ///       D = Sub(X, Mean)
    ///       DD = Mul(D, D)
    ///       Var = ReduceMean<axes=normalized_axes>(DD)
    ///       VarEps = Add(Var, epsilon)
    ///       StdDev = Sqrt(VarEps)
    ///       InvStdDev = Reciprocal(StdDev)
    ///       Normalized = Mul(D, InvStdDev)
    ///       ```
    ///       where `normalized_axes` is `[axis, ..., rank of X - 1]`.
    ///       The variables `Var` and `StdDev` stand for variance and
    ///       standard deviation, respectively. The second output is
    ///       `Mean` and the last one is `InvStdDev`.
    ///       Depending on `stash_type` attribute, the actual computation
    ///       must happen in different floating-point precision.
    ///       For example, if `stash_type` is 1, this operator casts
    ///       all input variables to 32-bit float, perform the computation, and
    ///       finally cast `Normalized` back to the original type of `X`.
    ///       The second stage then scales and shifts the outcome of the
    ///       first stage using
    ///       ```
    ///       NormalizedScaled = Mul(Normalized, Scale)
    ///       Y = Add(NormalizedScaled, B)
    ///       ```
    ///       The second stage doesn't depends on `stash_type`.
    ///       All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
    ///       The same variable (i.e., input, output, and attribute) uses
    ///       the same name in the equations above and this operator's definition.
    ///       Let `d[i]` indicate the i-th dimension of `X`.
    ///       If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
    ///       the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
    ///       `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
    ///       (tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
    ///       for more details please check [the doc](Broadcasting.md).
    public static <T, U> LayerNormalizationResult<T, U> LayerNormalization(Tensor<T> X, Tensor<T> Scale, Optional<Tensor<T>> B, Optional<Float> epsilon, Optional<Long> stash_type, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LayerNormalization.class, List.of(X, Scale, B), List.of(epsilon, stash_type, axis));
        Object[] resultArray = (Object[]) result;
        return new LayerNormalizationResult<>((Tensor<T>)resultArray[0], (Tensor<U>)resultArray[1], (Tensor<U>)resultArray[2]);
    }

    ///
    /// LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
    /// output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
    /// `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
    public static <T> Tensor<T> LeakyRelu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LeakyRelu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `less` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<Boolean> Less(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Less.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `less_equal` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<Boolean> LessOrEqual(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LessOrEqual.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    public record LinearClassifierResult<T2>(Tensor<T2> Y, Tensor<Float> Z) { }
    ///
    ///     Linear classifier
    public static <T1, T2> LinearClassifierResult<T2> LinearClassifier(Tensor<T1> X, Optional<long[]> classlabels_ints, Optional<String> post_transform, float[] coefficients, Optional<Long> multi_class, Optional<float[]> intercepts, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LinearClassifier.class, List.of(X), List.of(classlabels_ints, post_transform, coefficients, multi_class, intercepts, classlabels_strings));
        Object[] resultArray = (Object[]) result;
        return new LinearClassifierResult<>((Tensor<T2>)resultArray[0], (Tensor<Float>)resultArray[1]);
    }

    ///
    ///     Generalized linear regression evaluation.<br>
    ///     If targets is set to 1 (default) then univariate regression is performed.<br>
    ///     If targets is set to M then M sets of coefficients must be passed in as a sequence
    ///     and M results will be output for each input n in N.<br>
    ///     The coefficients array is of length n, and the coefficients for each target are contiguous.
    ///     Intercepts are optional but if provided must match the number of targets.
    public static <T> Tensor<Float> LinearRegressor(Tensor<T> X, Optional<String> post_transform, Optional<float[]> coefficients, Optional<Long> targets, Optional<float[]> intercepts) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LinearRegressor.class, List.of(X), List.of(post_transform, coefficients, targets, intercepts));
        return (Tensor<Float>) result;
    }

    ///
    /// Calculates the natural log of the given input tensor, element-wise.
    public static <T> Tensor<T> Log(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Log.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// The operator computes the log of softmax values for the given input:
    ///
    ///  LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
    ///
    /// The "axis" attribute indicates the dimension along which LogSoftmax
    /// will be performed. The output tensor has the same shape
    /// and contains the LogSoftmax values of the corresponding input.
    public static <T> Tensor<T> LogSoftmax(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LogSoftmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    ///
    /// Given a matrix, apply Lp-normalization along the provided axis.
    public static <T> Tensor<T> LpNormalization(Tensor<T> input, Optional<Long> p, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LpNormalization.class, List.of(input), List.of(p, axis));
        return (Tensor<T>) result;
    }

    ///
    ///  LpPool consumes an input tensor X and applies Lp pooling across
    ///  the tensor according to kernel sizes, stride sizes, and pad lengths.
    ///  Lp pooling consisting of computing the Lp norm on all values of a subset
    ///  of the input tensor according to the kernel size and downsampling the
    ///  data into the output tensor Y for further processing. The output spatial shape will be following:
    ///  ```
    ///  output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
    ///  ```
    ///  or
    ///  ```
    ///  output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
    ///  ```
    ///  if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.
    ///
    ///  `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
    ///  ```
    ///  VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
    ///  SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    ///  ```
    ///  And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
    ///  ```
    ///  pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
    ///  ```
    public static <T> Tensor<T> LpPool(Tensor<T> X, Optional<Long> p, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<Long> ceil_mode, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.LpPool.class, List.of(X), List.of(p, pads, dilations, auto_pad, ceil_mode, strides, kernel_shape));
        return (Tensor<T>) result;
    }

    ///
    /// Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
    public static <T> Tensor<T> MatMul(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMul.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
    /// The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
    public static <T1, T2> Tensor<Integer> MatMulInteger(Tensor<T1> A, Tensor<T2> B, Optional<Tensor<T1>> a_zero_point, Optional<Tensor<T2>> b_zero_point) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MatMulInteger.class, List.of(A, B, a_zero_point, b_zero_point), List.of());
        return (Tensor<Integer>) result;
    }

    ///
    /// Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
    /// All inputs and outputs must have the same data type.
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> Max(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Max.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    public record MaxPoolResult<T>(Tensor<T> Y, Tensor<Long> Indices) { }
    ///
    ///  MaxPool consumes an input tensor X and applies max pooling across
    ///  the tensor according to kernel sizes, stride sizes, and pad lengths.
    ///  max pooling consisting of computing the max on all values of a
    ///  subset of the input tensor according to the kernel size and downsampling the
    ///  data into the output tensor Y for further processing. The output spatial shape is calculated differently
    ///  depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
    ///  With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
    ///  ```
    ///  output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
    ///  ```
    ///  or
    ///  ```
    ///  output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
    ///  ```
    ///  if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.
    ///
    ///  `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
    ///  ```
    ///  VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    ///  SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    ///  ```
    ///  or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
    ///  ```
    ///  VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
    ///  SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
    ///  ```
    ///  And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
    ///  ```
    ///  pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
    ///  ```
    ///  The output of each pooling window is maximum number of elements exclude pad.
    ///
    public static <T> MaxPoolResult<T> MaxPool(Tensor<T> X, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<Long> ceil_mode, Optional<Long> storage_order, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxPool.class, List.of(X), List.of(pads, dilations, auto_pad, ceil_mode, storage_order, strides, kernel_shape));
        Object[] resultArray = (Object[]) result;
        return new MaxPoolResult<>((Tensor<T>)resultArray[0], (Tensor<Long>)resultArray[1]);
    }

    ///
    ///  ROI max pool consumes an input tensor X and region of interests (RoIs) to
    ///  apply max pooling across each RoI, to produce output 4-D tensor of shape
    ///  (num_rois, channels, pooled_shape[0], pooled_shape[1]).
    public static <T> Tensor<T> MaxRoiPool(Tensor<T> X, Tensor<T> rois, Optional<Float> spatial_scale, long[] pooled_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxRoiPool.class, List.of(X, rois), List.of(spatial_scale, pooled_shape));
        return (Tensor<T>) result;
    }

    ///
    /// MaxUnpool essentially computes the partial inverse of the MaxPool op.
    ///  The input information to this op is typically the output information from a MaxPool op. The first
    ///  input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
    ///  from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
    ///  to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
    ///  The third (optional) input is a tensor that specifies the output size of the unpooling operation.
    ///
    /// MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
    ///  values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
    ///  the result of an unpooling operation should give back the original input to the unpooling op.
    ///
    /// MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
    ///  The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
    ///  known/predictable size.
    ///
    /// In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
    ///  which define the exact unpooling op. The attributes typically have the same values as the corresponding
    ///  pooling op that the unpooling op is trying to invert.
    public static <T1> Tensor<T1> MaxUnpool(Tensor<T1> X, Tensor<Long> I, Optional<Tensor<Long>> output_shape, Optional<long[]> pads, Optional<long[]> strides, long[] kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MaxUnpool.class, List.of(X, I, output_shape), List.of(pads, strides, kernel_shape));
        return (Tensor<T1>) result;
    }

    ///
    /// Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
    /// All inputs and outputs must have the same data type.
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> Mean(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mean.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    ///
    ///       A MeanVarianceNormalization Function: Perform mean variance normalization
    ///       on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
    public static <T> Tensor<T> MeanVarianceNormalization(Tensor<T> X, Optional<long[]> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MeanVarianceNormalization.class, List.of(X), List.of(axes));
        return (Tensor<T>) result;
    }

    ///
    /// Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
    /// This function defines the mel scale in terms of a frequency in hertz according to the following formula:
    ///
    ///     mel(f) = 2595 * log10(1 + f/700)
    ///
    /// In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.
    ///
    /// The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
    public static <T1, T2, T3> Tensor<T3> MelWeightMatrix(Tensor<T1> num_mel_bins, Tensor<T1> dft_length, Tensor<T1> sample_rate, Tensor<T2> lower_edge_hertz, Tensor<T2> upper_edge_hertz, Optional<Long> output_datatype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.MelWeightMatrix.class, List.of(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz), List.of(output_datatype));
        return (Tensor<T3>) result;
    }

    ///
    /// Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
    /// All inputs and outputs must have the same data type.
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> Min(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Min.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    ///
    /// Perform the linear unit element-wise on the input tensor X using formula:
    ///
    /// ```
    /// mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    /// ```
    public static <T> Tensor<T> Mish(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mish.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Performs an element-wise binary modulo operation.
    /// The semantics and supported data types depend on the value of the `fmod` attribute which must be `0` (default), or `1`.
    ///
    /// If the `fmod` attribute is set to `0`, `T` is constrained to integer data types and the semantics follow that of the Python `%`-operator.
    /// The sign of the result is that of the divisor.
    ///
    /// If `fmod` is set to `1`, the behavior of this operator follows that of the `fmod` function in C and `T` is constrained to floating point data types.
    /// The result of this operator is the remainder of the division operation `x / y` where `x` and `y` are respective elements of `A` and `B`. The result is exactly the value `x - n * y`, where `n` is `x / y` with its fractional part truncated.
    /// The returned value has the same sign as `x` (except if `x` is `-0`) and is less or equal to `|y|` in magnitude.
    /// The following special cases apply when `fmod` is set to `1`:
    /// - If `x` is `-0` and `y` is greater than zero, either `+0` or `-0` may be returned.
    /// - If `x` is `±∞` and `y` is not `NaN`, `NaN` is returned.
    /// - If `y` is `±0` and `x` is not `NaN`, `NaN` should be returned.
    /// - If `y` is `±∞` and `x` is finite, `x` is returned.
    /// - If either argument is `NaN`, `NaN` is returned.
    ///
    /// This operator supports **multidirectional (i.e., NumPy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> Mod(Tensor<T> A, Tensor<T> B, Optional<Long> fmod) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mod.class, List.of(A, B), List.of(fmod));
        return (Tensor<T>) result;
    }

    ///
    ///     Compute one iteration of stochastic gradient update with momentum.
    ///     This operator can conduct the optimization of multiple tensor variables.
    ///
    ///     Let's define the behavior of this operator. As you can imagine, SG with momentum requires
    ///     several parameters:
    ///
    ///      - The learning-rate "R".
    ///      - The update count "T". That is, the number of conducted training iterations. It should
    ///        be zero in the first training iteration.
    ///      - A L2-norm regularization coefficient "norm_coefficient".
    ///      - A decay coefficient of previous accumulated gradient (i.e., momentum) "alpha".
    ///      - The scaling coefficient of current gradient "beta".
    ///      - An attribute to choose either standard momentum or Nesterov's momentum "mode" should
    ///        be used.
    ///
    ///     For the sake of simplicity, assume that there is only one tensor (called "X") to be optimized.
    ///     Other necessary inputs are "X"'s gradient (called "G") and "X"'s momentum (called "V"). This
    ///     Momentum operator maps all these inputs to the new value of "X" (called "X_new") and its new
    ///     momentum (called "V_new").
    ///
    ///     This operator supports two different momentum algorithms. Set the attribute "mode" to
    ///     "nesterov" if Nesterov's momentum is desired. Otherwise, set the attribute "model" to
    ///     "standard" to use standard momentum. Computation details are described subsequently.
    ///
    ///     Let "+", "-", "*", and "/" are all element-wise operations with numpy-style broadcasting.
    ///
    ///     Pseudo code for SG with standard momentum:
    ///
    ///       // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
    ///       // values of all elements in X.
    ///       G_regularized = norm_coefficient * X + G
    ///
    ///       // In the first training iteration, beta should always be 1.
    ///       beta_adjusted = T > 0 ? beta : 1
    ///
    ///       // Compute the current momentum based on previous momentum and the current gradient.
    ///       V_new = alpha * V + beta_adjusted * G_regularized
    ///
    ///       // Update X.
    ///       X_new = X - R * V_new
    ///
    ///     Pseudo code for SG with Nesterov's momentum:
    ///
    ///       // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
    ///       // values of all elements in X.
    ///       G_regularized = norm_coefficient * X + G;
    ///
    ///       // In the first training iteration, beta should always be 1.
    ///       beta_adjusted = T > 0 ? beta : 1
    ///
    ///       // Compute the current momentum based on previous momentum and the current gradient.
    ///       V_new = alpha * V + beta_adjusted * G_regularized;
    ///
    ///       // Compute final update direction and then update X.
    ///       X_new = X - R * (G_regularized + alpha * V_new)
    ///
    ///     If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2". The same
    ///     pseudo code would be extended to handle all tensors jointly. More specifically, we can view "X" as a
    ///     concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
    ///     be concatenated too) and then our pseudo code becomes applicable.
    public static <T1, T3> List<Tensor<T3>> Momentum(Tensor<T1> R, Tensor<Long> T, List<Tensor<T3>> inputs, String mode, float norm_coefficient, float alpha, float beta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Momentum.class, List.of(R, T, inputs), List.of(mode, norm_coefficient, alpha, beta));
        return (List<Tensor<T3>>) result;
    }

    ///
    /// Performs element-wise binary multiplication (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    ///
    /// (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
    public static <T> Tensor<T> Mul(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Mul.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Generate a tensor of samples from a multinomial distribution according to the probabilities
    /// of each of the possible outcomes.
    public static <T1, T2> Tensor<T2> Multinomial(Tensor<T1> input, Optional<Float> seed, Optional<Long> sample_size, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Multinomial.class, List.of(input), List.of(seed, sample_size, dtype));
        return (Tensor<T2>) result;
    }

    ///
    /// Neg takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where each element flipped sign, y = -x, is applied to
    /// the tensor elementwise.
    public static <T> Tensor<T> Neg(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Neg.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
    /// Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
    /// The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
    /// The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
    /// or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
    /// The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:
    ///
    /// ```
    /// loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
    /// ```
    ///
    /// When an optional "weight" is provided, the sample loss is calculated as:
    ///
    /// ```
    /// loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
    /// ```
    ///
    /// loss is zero for the case when target-value equals ignore_index.
    ///
    /// ```
    /// loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
    /// ```
    ///
    /// If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
    /// If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:
    ///
    /// ```
    /// mean(loss), if "weight" is not provided,
    /// ```
    ///
    /// or if weight is provided,
    ///
    /// ```
    /// sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
    /// ```
    ///
    /// If "reduction" attribute is set to "sum", the output is a scalar: `sum(loss)`.
    ///
    /// See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.
    ///
    /// Example 1:
    ///
    /// ```
    /// // negative log likelihood loss, "none" reduction
    /// N, C, d1 = 2, 3, 2
    /// input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
    ///           [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    /// target = [[2, 1], [0, 2]]
    ///
    /// loss = np.zeros((N, d1))
    /// for n in range(N):
    ///     for d_1 in range(d1):
    ///         c = target[n][d_1]
    ///         loss[n][d_1] = -input[n][c][d_1]
    ///
    /// // print(loss)
    /// // [[-3. -2.]
    /// //  [-0. -2.]]
    /// ```
    ///
    /// Example 2:
    ///
    /// ```
    /// // weighted negative log likelihood loss, sum reduction
    /// N, C, d1 = 2, 3, 2
    /// input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
    ///         [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    /// target = [[2, 1], [0, 2]]
    /// weight = [0.2, 0.3, 0.1]
    /// loss = np.zeros((N, d1))
    /// for n in range(N):
    ///     for d_1 in range(d1):
    ///         c = target[n][d_1]
    ///         loss[n][d_1] = -input[n][c][d_1] * weight[c]
    ///
    /// loss = np.sum(loss)
    /// // print(loss)
    /// // -1.1
    /// ```
    ///
    /// Example 3:
    ///
    /// ```
    /// // weighted negative log likelihood loss, mean reduction
    /// N, C, d1 = 2, 3, 2
    /// input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
    ///         [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    /// target = [[2, 1], [0, 2]]
    /// weight = [0.2, 0.3, 0.1]
    /// loss = np.zeros((N, d1))
    /// weight_total = 0
    /// for n in range(N):
    ///     for d_1 in range(d1):
    ///         c = target[n][d_1]
    ///         loss[n][d_1] = -input[n][c][d_1] * weight[c]
    ///         weight_total = weight_total + weight[c]
    ///
    /// loss = np.sum(loss) / weight_total
    /// // print(loss)
    /// // -1.57
    /// ```
    public static <T, Tind> Tensor<T> NegativeLogLikelihoodLoss(Tensor<T> input, Tensor<Tind> target, Optional<Tensor<T>> weight, Optional<Long> ignore_index, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NegativeLogLikelihoodLoss.class, List.of(input, target, weight), List.of(ignore_index, reduction));
        return (Tensor<T>) result;
    }

    ///
    /// Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
    /// Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
    /// Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
    /// orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
    /// result in the same boxes being selected by the algorithm.
    /// The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
    /// The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
    public static Tensor<Long> NonMaxSuppression(Tensor<Float> boxes, Tensor<Float> scores, Optional<Tensor<Long>> max_output_boxes_per_class, Optional<Tensor<Float>> iou_threshold, Optional<Tensor<Float>> score_threshold, Optional<Long> center_point_box) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NonMaxSuppression.class, List.of(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold), List.of(center_point_box));
        return (Tensor<Long>) result;
    }

    ///
    ///     Returns the indices of the elements that are non-zero
    ///     (in row-major order - by dimension).
    ///     NonZero behaves similar to numpy.nonzero:
    ///     https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
    ///     but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
    public static <T> Tensor<Long> NonZero(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.NonZero.class, List.of(X), List.of());
        return (Tensor<Long>) result;
    }

    ///
    ///     Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    ///     defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
    /// <br>
    ///     Max: Y = X / max(X)<br>
    ///     L1:  Y = X / sum(X)<br>
    ///     L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    ///     In all modes, if the divisor is zero, Y == X.
    /// <br>
    ///     For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    ///     of the batch is normalized independently.
    public static <T> Tensor<Float> Normalizer(Tensor<T> X, Optional<String> norm) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Normalizer.class, List.of(X), List.of(norm));
        return (Tensor<Float>) result;
    }

    ///
    /// Returns the negation of the input tensor element-wise.
    public static Tensor<Boolean> Not(Tensor<Boolean> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Not.class, List.of(X), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    ///     Produces a one-hot tensor based on inputs.
    ///     The locations represented by the index values in the 'indices' input tensor will have 'on_value'
    ///     and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
    ///     are specified as part of required input argument 'values', which is a two-element tensor of format
    ///     [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
    ///     input tensor. The additional dimension is for one-hot representation. The additional dimension will
    ///     be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
    ///     dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
    ///     dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
    ///     as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
    ///     the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
    ///     output tensor.
    ///
    ///     when axis = 0:
    ///     output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.
    ///
    ///     when axis = -1:
    ///     output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
    public static <T1, T2, T3> Tensor<T3> OneHot(Tensor<T1> indices, Tensor<T2> depth, Tensor<T3> values, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OneHot.class, List.of(indices, depth, values), List.of(axis));
        return (Tensor<T3>) result;
    }

    ///
    ///     Replace each input element with an array of ones and zeros, where a single
    ///     one is placed at the index of the category that was passed in. The total category count
    ///     will determine the size of the extra dimension of the output array Y.<br>
    ///     For example, if we pass a tensor with a single value of 4, and a category count of 8,
    ///     the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    ///     This operator assumes every input feature is from the same set of categories.<br>
    ///     If the input is a tensor of float, int32, or double, the data will be cast
    ///     to integers and the cats_int64s category list will be used for the lookups.
    public static <T> Tensor<Float> OneHotEncoder(Tensor<T> X, Optional<String[]> cats_strings, Optional<long[]> cats_int64s, Optional<Long> zeros) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OneHotEncoder.class, List.of(X), List.of(cats_strings, cats_int64s, zeros));
        return (Tensor<Float>) result;
    }

    ///
    /// Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
    /// or a non-empty value containing the input element.
    public static <V, O> Optional<O> Optional(Optional<V> input, Optional<Object> type) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Optional.class, List.of(input), List.of(type));
        return (Optional<O>) result;
    }

    ///
    /// If the input is a tensor or sequence type, it returns the input.
    /// If the input is an optional type, it outputs the element in the input.
    /// It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
    public static <O, V> V OptionalGetElement(O input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OptionalGetElement.class, List.of(input), List.of());
        return (V) result;
    }

    ///
    /// Returns true if (1) the input is an optional-type and contains an element,
    /// or, (2) the input is a tensor or sequence type.
    /// If the input is not provided or is an empty optional-type, this op returns false.
    public static <O> Tensor<Boolean> OptionalHasElement(Optional<O> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.OptionalHasElement.class, List.of(input), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `or` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static Tensor<Boolean> Or(Tensor<Boolean> A, Tensor<Boolean> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Or.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    /// PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
    /// output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
    /// `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
    /// This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> PRelu(Tensor<T> X, Tensor<T> slope) {
        Object result = OnnxInterpreter.interpret(OnnxOps.PRelu.class, List.of(X, slope), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
    /// a padded tensor (`output`) is generated.
    ///
    /// The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):
    ///
    /// 1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)
    ///
    /// 2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
    ///
    /// 3) `edge` - pads with the edge values of array
    ///
    /// 4) `wrap` - wrap-around padding as if the data tensor forms a torus
    ///
    ///
    /// Example 1 (`constant` mode):
    ///
    /// Insert 0 pads to the beginning of the second dimension.
    ///
    /// ```
    /// data = [
    ///     [1.0, 1.2],
    ///     [2.3, 3.4],
    ///     [4.5, 5.7],
    /// ]
    ///
    /// pads = [0, 2, 0, 0]
    ///
    /// mode = 'constant'
    ///
    /// constant_value = 0.0
    ///
    /// output = [
    ///     [0.0, 0.0, 1.0, 1.2],
    ///     [0.0, 0.0, 2.3, 3.4],
    ///     [0.0, 0.0, 4.5, 5.7],
    /// ]
    /// ```
    ///
    /// Example 2 (`reflect` mode):
    ///
    /// ```
    /// data = [
    ///     [1.0, 1.2],
    ///     [2.3, 3.4],
    ///     [4.5, 5.7],
    /// ]
    ///
    /// pads = [0, 2, 0, 0]
    ///
    /// mode = 'reflect'
    ///
    /// output = [
    ///     [1.0, 1.2, 1.0, 1.2],
    ///     [2.3, 3.4, 2.3, 3.4],
    ///     [4.5, 5.7, 4.5, 5.7],
    /// ]
    /// ```
    ///
    /// Example 3 (`edge` mode):
    ///
    /// ```
    /// data = [
    ///     [1.0, 1.2],
    ///     [2.3, 3.4],
    ///     [4.5, 5.7],
    /// ]
    ///
    /// pads = [0, 2, 0, 0]
    ///
    /// mode = 'edge'
    ///
    /// output = [
    ///     [1.0, 1.0, 1.0, 1.2],
    ///     [2.3, 2.3, 2.3, 3.4],
    ///     [4.5, 4.5, 4.5, 5.7],
    /// ]
    /// ```
    ///
    /// Example 4 (`wrap` mode):
    ///
    /// ```
    /// data = [
    ///     [1.0, 1.2],
    ///     [2.3, 3.4],
    ///     [4.5, 5.7],
    /// ]
    ///
    /// pads = [2, 1, 1, 1]
    ///
    /// mode = 'wrap'
    ///
    /// output = [
    ///     [3.4, 2.3, 3.4, 2.3],
    ///     [5.7, 4.5, 5.7, 4.5],
    ///     [1.2, 1.0, 1.2, 1.0],
    ///     [3.4, 2.3, 3.4, 2.3],
    ///     [5.7, 4.5, 5.7, 4.5],
    ///     [1.2, 1.0, 1.2, 1.0],
    /// ]
    /// ```
    public static <T, Tind> Tensor<T> Pad(Tensor<T> data, Tensor<Long> pads, Optional<Tensor<T>> constant_value, Optional<Tensor<Tind>> axes, Optional<String> mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Pad.class, List.of(data, pads, constant_value, axes), List.of(mode));
        return (Tensor<T>) result;
    }

    ///
    /// Pow takes input data (Tensor<T>) and exponent Tensor, and
    /// produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
    /// is applied to the data tensor elementwise.
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T, T1> Tensor<T> Pow(Tensor<T> X, Tensor<T1> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Pow.class, List.of(X, Y), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// The convolution operator consumes a quantized input tensor, its scale and zero point,
    /// a quantized filter, its scale and zero point, and output's scale and zero point,
    /// and computes the quantized output. Each scale and zero-point pair must have same shape.
    /// It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
    /// Each input or output and its related zero point must have same type.
    /// When bias is present it must be quantized using scale = input scale * weight scale and
    /// zero point as 0.
    public static <T1, T2, T3> Tensor<T3> QLinearConv(Tensor<T1> x, Tensor<Float> x_scale, Tensor<T1> x_zero_point, Tensor<T2> w, Tensor<Float> w_scale, Tensor<T2> w_zero_point, Tensor<Float> y_scale, Tensor<T3> y_zero_point, Optional<Tensor<Integer>> B, Optional<long[]> pads, Optional<long[]> dilations, Optional<String> auto_pad, Optional<long[]> strides, Optional<Long> group, Optional<long[]> kernel_shape) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QLinearConv.class, List.of(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B), List.of(pads, dilations, auto_pad, strides, group, kernel_shape));
        return (Tensor<T3>) result;
    }

    ///
    /// Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
    /// It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
    /// and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
    /// For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
    /// Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
    /// (per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
    /// or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
    /// an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
    /// for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
    /// have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
    /// Production must never overflow, and accumulation may overflow if and only if in 32 bits.
    public static <TS, T1, T2, T3> Tensor<T3> QLinearMatMul(Tensor<T1> a, Tensor<TS> a_scale, Tensor<T1> a_zero_point, Tensor<T2> b, Tensor<TS> b_scale, Tensor<T2> b_zero_point, Tensor<TS> y_scale, Tensor<T3> y_zero_point) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QLinearMatMul.class, List.of(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point), List.of());
        return (Tensor<T3>) result;
    }

    ///
    /// The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
    /// low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
    /// granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.
    ///
    /// Saturation is done according to:
    /// - uint16: [0, 65535]
    /// - int16: [-32768, 32767]
    /// - uint8: [0, 255]
    /// - int8: [-128, 127]
    /// - uint4: [0, 15]
    /// - int4: [-8, 7]
    ///
    /// For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
    ///
    /// `y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
    /// formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
    /// `x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
    /// `y_scale`, unless the `precision` attribute is specified.
    ///
    /// There are three supported quantization granularities, determined by the shape of `y_scale`.
    /// In all cases, `y_zero_point` must have the same shape as `y_scale`.
    /// - Per-tensor (per-layer) quantization: `y_scale` is a scalar.
    /// - Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
    ///  `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
    /// - Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
    ///   blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
    ///   `(D0, ..., ceil(Di/B), ..., Dn)`.
    public static <T1, T2, T3> Tensor<T3> QuantizeLinear(Tensor<T1> x, Tensor<T2> y_scale, Optional<Tensor<T3>> y_zero_point, Optional<Long> output_dtype, Optional<Long> saturate, Optional<Long> precision, Optional<Long> axis, Optional<Long> block_size) {
        Object result = OnnxInterpreter.interpret(OnnxOps.QuantizeLinear.class, List.of(x, y_scale, y_zero_point), List.of(output_dtype, saturate, precision, axis, block_size));
        return (Tensor<T3>) result;
    }

    ///
    ///       This is RMS normalization defined in ONNX as function as described in the paper https://arxiv.org/pdf/1910.07467.
    ///       The overall computation can be split into two stages. The root mean squared norm is taken over the last D dimensions,
    ///       where D is the dimension of normalized_shape. For example, if normalized_shape is (3, 5) (a 2-dimensional shape),
    ///       the rms norm is computed over the last 2 dimensions of the input. The computation required by standardization can be
    ///       described by the following equations.
    ///       ```
    ///       XSquared = Mul(X, X)
    ///       XSquaredMean = ReduceMean<axes=normalized_axes>(XSquared)
    ///       MeanSquareEpsilon = Add(XSquaredMean, epsilon)
    ///       RMS = Sqrt(MeanSquareEpsilon)
    ///       Normalized = Div(X, RMS)
    ///       ```
    ///       where `normalized_axes` is `[axis, ..., rank of X - 1]`. The variables `RMS` stand for root mean square,
    ///       Depending on `stash_type` attribute, the actual computation
    ///       must happen in different floating-point precision.
    ///       For example, if `stash_type` is 1, this operator casts
    ///       all input variables to 32-bit float, perform the computation, and
    ///       finally cast `Normalized` back to the original type of `X`.
    ///       The second stage then scales the outcome of the first stage using:
    ///       ```
    ///       Y= Mul(Normalized, Scale)
    ///       ```
    ///       Let `d[i]` indicate the i-th dimension of `X`.
    ///       If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
    ///       the shape of `RMS` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
    ///       `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
    ///       (`Scale` should be unidirectional broadcastable to tensor `X`);
    ///       for more details please check [the doc](Broadcasting.md).
    public static <T, V> Tensor<V> RMSNormalization(Tensor<T> X, Tensor<V> scale, Optional<Float> epsilon, Optional<Long> stash_type, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RMSNormalization.class, List.of(X, scale), List.of(epsilon, stash_type, axis));
        return (Tensor<V>) result;
    }

    public record RNNResult<T>(Tensor<T> Y, Tensor<T> Y_h) { }
    ///
    /// Computes an one-layer simple RNN. This operator is usually supported
    /// via some custom implementation such as CuDNN.
    ///
    /// Notations:
    ///
    /// * `X` - input tensor
    /// * `i` - input gate
    /// * `t` - time step (t-1 means previous time step)
    /// * `Wi` - W parameter weight matrix for input gate
    /// * `Ri` - R recurrence weight matrix for input gate
    /// * `Wbi` - W parameter bias vector for input gate
    /// * `Rbi` - R parameter bias vector for input gate
    /// * `WBi` - W parameter weight matrix for backward input gate
    /// * `RBi` - R recurrence weight matrix for backward input gate
    /// * `WBbi` - WR bias vectors for backward input gate
    /// * `RBbi` - RR bias vectors for backward input gate
    /// * `H` - Hidden state
    /// * `num_directions` - 2 if direction == bidirectional else 1
    ///
    /// Activation functions:
    ///
    /// * Relu(x)                - max(0, x)
    /// * Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
    /// * Sigmoid(x)             - 1/(1 + e^{-x})
    ///
    /// NOTE: Below are optional
    ///
    /// * Affine(x)              - alpha*x + beta
    /// * LeakyRelu(x)           - x if x >= 0 else alpha * x
    /// * ThresholdedRelu(x)     - x if x >= alpha else 0
    /// * ScaledTanh(x)          - alpha*Tanh(beta*x)
    /// * HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
    /// * Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
    /// * Softsign(x)            - x/(1 + |x|)
    /// * Softplus(x)            - log(1 + e^x)
    ///
    /// Equations (Default: f=Tanh):
    ///
    /// * Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    /// This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    public static <T> RNNResult<T> RNN(Tensor<T> X, Tensor<T> W, Tensor<T> R, Optional<Tensor<T>> B, Optional<Tensor<Integer>> sequence_lens, Optional<Tensor<T>> initial_h, Optional<Long> layout, Optional<float[]> activation_alpha, Optional<Long> hidden_size, Optional<float[]> activation_beta, Optional<String[]> activations, Optional<Float> clip, Optional<String> direction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RNN.class, List.of(X, W, R, B, sequence_lens, initial_h), List.of(layout, activation_alpha, hidden_size, activation_beta, activations, clip, direction));
        Object[] resultArray = (Object[]) result;
        return new RNNResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1]);
    }

    ///
    /// Generate a tensor with random values drawn from a normal distribution. The shape
    /// of the tensor is specified by the `shape` argument and the parameter of the normal distribution
    /// specified by `mean` and `scale`.
    ///
    /// The data type is specified by the 'dtype' argument. The 'dtype' argument must
    /// be one of the data types specified in the 'DataType' enum field in the
    /// TensorProto message.
    public static <T> Tensor<T> RandomNormal(long[] shape, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomNormal.class, List.of(), List.of(shape, seed, mean, scale, dtype));
        return (Tensor<T>) result;
    }

    ///
    /// Generate a tensor with random values drawn from a normal distribution.
    /// The shape of the output tensor is copied from the shape of the input tensor,
    /// and the parameters of the normal distribution are specified by `mean` and `scale`.
    ///
    /// The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
    /// The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
    /// TensorProto message, and be valid as an output type.
    public static <T1, T2> Tensor<T2> RandomNormalLike(Tensor<T1> input, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomNormalLike.class, List.of(input), List.of(seed, mean, scale, dtype));
        return (Tensor<T2>) result;
    }

    ///
    /// Generate a tensor with random values drawn from a uniform distribution. The shape
    /// of the tensor is specified by the `shape` argument and the range by `low` and `high`.
    ///
    /// The data type is specified by the 'dtype' argument. The 'dtype' argument must
    /// be one of the data types specified in the 'DataType' enum field in the
    /// TensorProto message.
    public static <T> Tensor<T> RandomUniform(Optional<Float> high, long[] shape, Optional<Float> seed, Optional<Float> low, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomUniform.class, List.of(), List.of(high, shape, seed, low, dtype));
        return (Tensor<T>) result;
    }

    ///
    /// Generate a tensor with random values drawn from a uniform distribution.
    /// The shape of the output tensor is copied from the shape of the input tensor,
    /// and the parameters of the uniform distribution are specified by `low` and `high`.
    ///
    /// The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
    /// The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
    /// TensorProto message and be valid as an output type.
    public static <T1, T2> Tensor<T2> RandomUniformLike(Tensor<T1> input, Optional<Float> high, Optional<Float> seed, Optional<Float> low, Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RandomUniformLike.class, List.of(input), List.of(high, seed, low, dtype));
        return (Tensor<T2>) result;
    }

    ///
    /// Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
    /// up to `limit` (exclusive).
    ///
    /// The number of elements in the output of range is computed as below:
    ///
    /// ```
    /// number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
    /// ```
    ///
    /// The pseudocode determining the contents of the output is shown below:
    ///
    /// ```
    /// for(int i=0; i<number_of_elements; ++i) {
    ///   output[i] =  start + (i * delta);
    /// }
    /// ```
    ///
    /// Example 1
    ///
    /// ```
    /// Inputs: start = 3, limit = 9, delta = 3
    /// Output: [3, 6]
    /// ```
    ///
    /// Example 2
    ///
    /// ```
    /// Inputs: start = 10, limit = 4, delta = -2
    /// Output: [10, 8, 6]
    /// ```
    public static <T> Tensor<T> Range(Tensor<T> start, Tensor<T> limit, Tensor<T> delta) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Range.class, List.of(start, limit, delta), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Reciprocal takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
    /// the tensor elementwise.
    public static <T> Tensor<T> Reciprocal(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Reciprocal.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields 0.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceL1(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceL1.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields 0.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceL2(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceL2.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the log sum of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceLogSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceLogSum.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceLogSumExp(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceLogSumExp.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the max of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.
    ///
    ///
    /// If the input data type is Boolean, the comparison should consider `False < True`.
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceMax(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMax.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the mean of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields undefined.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceMean(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMean.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the min of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.
    ///
    ///
    /// If the input data type is Boolean, the comparison should consider `False < True`.
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceMin(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceMin.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the product of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields 1.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceProd(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceProd.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the sum of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields 0.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceSum(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceSum.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    ///
    /// Computes the sum square of the input tensor's elements along the provided axes. The resulting
    /// tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
    /// the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
    /// valid. Reduction over an empty set of values yields 0.
    ///
    ///
    /// The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
    /// to `False` instead of `True`.
    public static <T> Tensor<T> ReduceSumSquare(Tensor<T> data, Optional<Tensor<Long>> axes, Optional<Long> noop_with_empty_axes, Optional<Long> keepdims) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReduceSumSquare.class, List.of(data, axes), List.of(noop_with_empty_axes, keepdims));
        return (Tensor<T>) result;
    }

    /// RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.
    public static Tensor<Boolean> RegexFullMatch(Tensor<String> X, Optional<String> pattern) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RegexFullMatch.class, List.of(X), List.of(pattern));
        return (Tensor<Boolean>) result;
    }

    ///
    /// Relu takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
    /// the tensor elementwise.
    public static <T> Tensor<T> Relu(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Relu.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Reshape the input tensor similar to numpy.reshape.
    /// First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
    /// At most one dimension of the new shape can be -1. In this case, the value is
    /// inferred from the size of the tensor and the remaining dimensions. A dimension
    /// could also be 0, in which case the actual dimension value is unchanged (i.e. taken
    /// from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
    /// dimension will be set explicitly to zero (i.e. not taken from input tensor).
    /// Shape (second input) could be an empty shape, which means converting to a scalar.
    /// The input tensor's shape and the output tensor's shape are required to have the same number of elements.
    ///
    /// If the attribute 'allowzero' is set, it is invalid for the specified shape to
    /// contain both a zero value and -1, as the value of the dimension corresponding
    /// to -1 cannot be determined uniquely.
    public static <T> Tensor<T> Reshape(Tensor<T> data, Tensor<Long> shape, Optional<Long> allowzero) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Reshape.class, List.of(data, shape), List.of(allowzero));
        return (Tensor<T>) result;
    }

    ///
    /// Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
    /// Each dimension value of the output tensor is:
    /// ```
    /// output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
    /// ```
    /// if input \"sizes\" is not specified.
    public static <T1, T2> Tensor<T1> Resize(Tensor<T1> X, Optional<Tensor<T2>> roi, Optional<Tensor<Float>> scales, Optional<Tensor<Long>> sizes, Optional<String> mode, Optional<Float> extrapolation_value, Optional<String> nearest_mode, Optional<Long> antialias, Optional<Float> cubic_coeff_a, Optional<long[]> axes, Optional<String> coordinate_transformation_mode, Optional<String> keep_aspect_ratio_policy, Optional<Long> exclude_outside) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Resize.class, List.of(X, roi, scales, sizes), List.of(mode, extrapolation_value, nearest_mode, antialias, cubic_coeff_a, axes, coordinate_transformation_mode, keep_aspect_ratio_policy, exclude_outside));
        return (Tensor<T1>) result;
    }

    ///
    /// Reverse batch of sequences having different lengths specified by `sequence_lens`.
    ///
    /// For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
    /// and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
    /// sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.
    ///
    /// Example 1:
    ///   input = [[0.0, 4.0, 8.0,  12.0],
    ///            [1.0, 5.0, 9.0,  13.0],
    ///            [2.0, 6.0, 10.0, 14.0],
    ///            [3.0, 7.0, 11.0, 15.0]]
    ///   sequence_lens = [4, 3, 2, 1]
    ///   time_axis = 0
    ///   batch_axis = 1
    ///
    ///   output = [[3.0, 6.0, 9.0,  12.0],
    ///             [2.0, 5.0, 8.0,  13.0],
    ///             [1.0, 4.0, 10.0, 14.0],
    ///             [0.0, 7.0, 11.0, 15.0]]
    ///
    /// Example 2:
    ///   input = [[0.0,  1.0,  2.0,  3.0 ],
    ///            [4.0,  5.0,  6.0,  7.0 ],
    ///            [8.0,  9.0,  10.0, 11.0],
    ///            [12.0, 13.0, 14.0, 15.0]]
    ///   sequence_lens = [1, 2, 3, 4]
    ///   time_axis = 1
    ///   batch_axis = 0
    ///
    ///   output = [[0.0,  1.0,  2.0,  3.0 ],
    ///             [5.0,  4.0,  6.0,  7.0 ],
    ///             [10.0, 9.0,  8.0,  11.0],
    ///             [15.0, 14.0, 13.0, 12.0]]
    public static <T> Tensor<T> ReverseSequence(Tensor<T> input, Tensor<Long> sequence_lens, Optional<Long> time_axis, Optional<Long> batch_axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ReverseSequence.class, List.of(input, sequence_lens), List.of(time_axis, batch_axis));
        return (Tensor<T>) result;
    }

    ///
    /// Region of Interest (RoI) align operation described in the
    /// [Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
    /// RoiAlign consumes an input tensor X and region of interests (rois)
    /// to apply pooling across each RoI; it produces a 4-D tensor of shape
    /// (num_rois, C, output_height, output_width).
    ///
    /// RoiAlign is proposed to avoid the misalignment by removing
    /// quantizations while converting from original image into feature
    /// map and from feature map into RoI feature; in each ROI bin,
    /// the value of the sampled locations are computed directly
    /// through bilinear interpolation.
    public static <T1> Tensor<T1> RoiAlign(Tensor<T1> X, Tensor<T1> rois, Tensor<Long> batch_indices, Optional<String> mode, Optional<Long> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Long> sampling_ratio, Optional<Long> output_height) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RoiAlign.class, List.of(X, rois, batch_indices), List.of(mode, output_width, spatial_scale, coordinate_transformation_mode, sampling_ratio, output_height));
        return (Tensor<T1>) result;
    }

    ///
    /// RotaryEmbedding is the implementation of rotary positional embeddings (RoPE) based on the paper https://arxiv.org/pdf/2104.09864.
    /// The key advantage of RoPE is that it allows the model to understand both the absolute position of a token and the relative distances
    /// between tokens. This is achieved through a rotational mechanism where the extent of rotation is computed based on the token's absolute position (position_ids).
    ///
    /// The rotational mechanism is defined by sine and cosine functions that are used to represent the rotation angles.
    /// For each token in the sequence, its positional embedding is computed by rotating its embedding vector. This is done by splitting the
    /// embedding vector either into two halves or interleaving every alternate token and applying the rotation matrix to each half of the embedding vector.
    /// The rotation matrix is parameterized by the token's position in the sequence. The rotated halves of the embedding vector are concatenated
    /// to form the final positional embedding for each token. The rotated positional embeddings are used in the self-attention mechanism.
    /// The rotation ensures that the model captures both absolute and relative positional information.
    ///
    /// Rotary embeddings are defined using the following algorithm:
    ///
    /// ```python
    /// def rotary_embedding(
    ///     input: np.ndarray,
    ///     cos_cache: np.ndarray,
    ///     sin_cache: np.ndarray,
    ///     position_ids: np.ndarray | None = None,
    ///     interleaved=None,
    ///     rotary_embedding_dim=None,
    ///     num_heads=None,
    /// ) -> np.ndarray:
    ///     original_input_shape = input.shape
    ///     # First ensure input to be processed has shape [batch_size, seq_len, num_heads, head_size]
    ///     if len(input.shape) == 4:
    ///         input = np.transpose(input, (0, 2, 1, 3))
    ///     batch_size = input.shape[0]
    ///     sequence_length = input.shape[1]
    ///     if len(input.shape) == 3:
    ///         hidden_size = input.shape[2]
    ///         assert num_heads != 0
    ///         head_size = int(hidden_size / num_heads)
    ///         new_shape = [batch_size, sequence_length, num_heads, head_size]
    ///         input = np.reshape(input, new_shape)
    ///     assert len(input.shape) == 4
    ///     head_size = input.shape[3]
    ///
    ///     # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
    ///     if rotary_embedding_dim is None or rotary_embedding_dim == 0:
    ///         # If rotary_embedding_dim not provided, perform full rotation by using head_size
    ///         rotary_embedding_dim = head_size
    ///     x_rotate = input[:, :, :, :rotary_embedding_dim]
    ///     x_not_rotate = input[:, :, :, rotary_embedding_dim:]
    ///     rotary_embedding_dim_half = int(rotary_embedding_dim / 2)
    ///
    ///     # Retrieve sin and cos caches using position ids
    ///     if position_ids is not None:
    ///         cos_cache = cos_cache[
    ///             position_ids
    ///         ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    ///         sin_cache = sin_cache[
    ///             position_ids
    ///         ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    ///
    ///     # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    ///     if cos_cache.shape[-1] != rotary_embedding_dim_half:
    ///         raise ValueError(
    ///             f"Last dimension of cos cache ({cos_cache.shape[-1]}) does not match rotary_embedding_dim/2 ({rotary_embedding_dim_half})."
    ///         )
    ///     if sin_cache.shape[-1] != rotary_embedding_dim_half:
    ///         raise ValueError(
    ///             f"Last dimension of sin cache ({sin_cache.shape[-1]}) does not match rotary_embedding_dim/2 ({rotary_embedding_dim_half})."
    ///         )
    ///
    ///     cos_cache = np.expand_dims(
    ///         cos_cache, axis=2
    ///     )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    ///     sin_cache = np.expand_dims(
    ///         sin_cache, axis=2
    ///     )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    ///
    ///     # Either divide the input in halves or interleave (based on interleaved attribute)
    ///     if interleaved:
    ///         x1 = x_rotate[:, :, :, 0::2]
    ///         x2 = x_rotate[:, :, :, 1::2]
    ///     else:
    ///         x1, x2 = np.split(x_rotate, 2, axis=-1)
    ///
    ///     # Calculate real and imaginary values
    ///     real = (cos_cache * x1) - (sin_cache * x2)
    ///     imag = (sin_cache * x1) + (cos_cache * x2)
    ///
    ///     # Inserted rotated embeddings back to the original input
    ///     if interleaved:
    ///         # x_rotate[:, :, :, 0::2] = real
    ///         # x_rotate[:, :, :, 1::2] = imag
    ///         real = np.expand_dims(real, axis=-1)
    ///         imag = np.expand_dims(imag, axis=-1)
    ///         x_rotate_concat = np.concatenate((real, imag), axis=-1)
    ///         x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
    ///     else:
    ///         x_rotate = np.concatenate((real, imag), axis=-1)
    ///     output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    ///     if len(original_input_shape) == 3:
    ///         output = np.reshape(output, original_input_shape)
    ///     else:
    ///         output = np.transpose(output, (0, 2, 1, 3))
    ///     return output
    /// ```
    public static <T> Tensor<T> RotaryEmbedding(Tensor<T> X, Tensor<T> cos_cache, Tensor<T> sin_cache, Optional<Tensor<Long>> position_ids, Optional<Long> num_heads, Optional<Long> rotary_embedding_dim, Optional<Long> interleaved) {
        Object result = OnnxInterpreter.interpret(OnnxOps.RotaryEmbedding.class, List.of(X, cos_cache, sin_cache, position_ids), List.of(num_heads, rotary_embedding_dim, interleaved));
        return (Tensor<T>) result;
    }

    ///
    /// Round takes one input Tensor and rounds the values, element-wise, meaning
    /// it finds the nearest integer for each value.
    /// In case of halves, the rule is to round them to the nearest even integer.
    /// If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
    /// The output tensor has the same shape and type as the input.
    ///
    /// Examples:
    /// ```
    /// round([0.9]) = [1.0]
    /// round([2.5]) = [2.0]
    /// round([2.3]) = [2.0]
    /// round([1.5]) = [2.0]
    /// round([-4.5]) = [-4.0]
    /// ```
    public static <T> Tensor<T> Round(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Round.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    /// Computes the Short-time Fourier Transform of the signal.
    public static <T1, T2> Tensor<T1> STFT(Tensor<T1> signal, Tensor<T2> frame_step, Optional<Tensor<T1>> window, Optional<Tensor<T2>> frame_length, Optional<Long> onesided) {
        Object result = OnnxInterpreter.interpret(OnnxOps.STFT.class, List.of(signal, frame_step, window, frame_length), List.of(onesided));
        return (Tensor<T1>) result;
    }

    public record SVMClassifierResult<T2>(Tensor<T2> Y, Tensor<Float> Z) { }
    ///
    ///     Support Vector Machine classifier
    public static <T1, T2> SVMClassifierResult<T2> SVMClassifier(Tensor<T1> X, Optional<float[]> prob_b, Optional<float[]> kernel_params, Optional<String> kernel_type, Optional<long[]> classlabels_ints, Optional<String> post_transform, Optional<float[]> rho, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<long[]> vectors_per_class, Optional<float[]> prob_a, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SVMClassifier.class, List.of(X), List.of(prob_b, kernel_params, kernel_type, classlabels_ints, post_transform, rho, coefficients, support_vectors, vectors_per_class, prob_a, classlabels_strings));
        Object[] resultArray = (Object[]) result;
        return new SVMClassifierResult<>((Tensor<T2>)resultArray[0], (Tensor<Float>)resultArray[1]);
    }

    ///
    ///     Support Vector Machine regression prediction and one-class SVM anomaly detection.
    public static <T> Tensor<Float> SVMRegressor(Tensor<T> X, Optional<String> kernel_type, Optional<float[]> kernel_params, Optional<Long> n_supports, Optional<float[]> rho, Optional<String> post_transform, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<Long> one_class) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SVMRegressor.class, List.of(X), List.of(kernel_type, kernel_params, n_supports, rho, post_transform, coefficients, support_vectors, one_class));
        return (Tensor<Float>) result;
    }

    ///
    ///     Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
    public static <T> Tensor<Float> Scaler(Tensor<T> X, Optional<float[]> offset, Optional<float[]> scale) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Scaler.class, List.of(X), List.of(offset, scale));
        return (Tensor<Float>) result;
    }

    ///
    /// This operator is deprecated. Please use ScatterElements, which provides the same functionality.
    ///
    /// Scatter takes three inputs `data`, `updates`, and `indices` of the same
    /// rank r >= 1 and an optional attribute axis that identifies an axis of `data`
    /// (by default, the outer-most axis, that is axis 0). The output of the operation
    /// is produced by creating a copy of the input `data`, and then updating its value
    /// to values specified by `updates` at specific index positions specified by
    /// `indices`. Its output shape is the same as the shape of `data`.
    ///
    /// For each entry in `updates`, the target index in `data` is obtained by combining
    /// the corresponding entry in `indices` with the index of the entry itself: the
    /// index-value for dimension = axis is obtained from the value of the corresponding
    /// entry in `indices` and the index-value for dimension != axis is obtained from the
    /// index of the entry itself.
    ///
    /// For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
    /// is performed as below:
    /// ```
    ///   output[indices[i][j]][j] = updates[i][j] if axis = 0,
    ///   output[i][indices[i][j]] = updates[i][j] if axis = 1,
    /// ```
    ///
    /// This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.
    ///
    /// Example 1:
    /// ```
    ///   data = [
    ///       [0.0, 0.0, 0.0],
    ///       [0.0, 0.0, 0.0],
    ///       [0.0, 0.0, 0.0],
    ///   ]
    ///   indices = [
    ///       [1, 0, 2],
    ///       [0, 2, 1],
    ///   ]
    ///   updates = [
    ///       [1.0, 1.1, 1.2],
    ///       [2.0, 2.1, 2.2],
    ///   ]
    ///   output = [
    ///       [2.0, 1.1, 0.0]
    ///       [1.0, 0.0, 2.2]
    ///       [0.0, 2.1, 1.2]
    ///   ]
    /// ```
    /// Example 2:
    /// ```
    ///   data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    ///   indices = [[1, 3]]
    ///   updates = [[1.1, 2.1]]
    ///   axis = 1
    ///   output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
    /// ```
    public static <T, Tind> Tensor<T> Scatter(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Scatter.class, List.of(data, indices, updates), List.of(axis));
        return (Tensor<T>) result;
    }

    ///
    /// ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
    /// rank r >= 1 and an optional attribute axis that identifies an axis of `data`
    /// (by default, the outer-most axis, that is axis 0). The output of the operation
    /// is produced by creating a copy of the input `data`, and then updating its value
    /// to values specified by `updates` at specific index positions specified by
    /// `indices`. Its output shape is the same as the shape of `data`.
    ///
    /// For each entry in `updates`, the target index in `data` is obtained by combining
    /// the corresponding entry in `indices` with the index of the entry itself: the
    /// index-value for dimension = axis is obtained from the value of the corresponding
    /// entry in `indices` and the index-value for dimension != axis is obtained from the
    /// index of the entry itself.
    ///
    /// `reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
    /// tensor into `output` at the specified `indices`.
    /// In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
    /// then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case, the update
    /// corresponding to the [i][j] entry is performed as below:
    /// ```
    /// output[indices[i][j]][j] = updates[i][j] if axis = 0,
    /// output[i][indices[i][j]] = updates[i][j] if axis = 1,
    /// ```
    /// When `reduction` is set to some reduction function `f`, the update corresponding to the [i][j] entry is performed as below:
    /// ```
    /// output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
    /// output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    /// ```
    /// where the `f` is `+`, `*`, `max` or `min` as specified.
    ///
    /// This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.
    ///
    /// (Opset 18 change): Adds max/min to the set of allowed reduction ops.
    ///
    /// Example 1:
    /// ```
    /// data = [
    ///     [0.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0],
    ///     [0.0, 0.0, 0.0],
    /// ]
    /// indices = [
    ///     [1, 0, 2],
    ///     [0, 2, 1],
    /// ]
    /// updates = [
    ///     [1.0, 1.1, 1.2],
    ///     [2.0, 2.1, 2.2],
    /// ]
    /// output = [
    ///     [2.0, 1.1, 0.0]
    ///     [1.0, 0.0, 2.2]
    ///     [0.0, 2.1, 1.2]
    /// ]
    /// ```
    /// Example 2:
    /// ```
    /// data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    /// indices = [[1, 3]]
    /// updates = [[1.1, 2.1]]
    /// axis = 1
    /// output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
    /// ```
    public static <T, Tind> Tensor<T> ScatterElements(Tensor<T> data, Tensor<Tind> indices, Tensor<T> updates, Optional<String> reduction, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ScatterElements.class, List.of(data, indices, updates), List.of(reduction, axis));
        return (Tensor<T>) result;
    }

    ///
    /// ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
    /// and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
    /// is produced by creating a copy of the input `data`, and then updating its value to values
    /// specified by `updates` at specific index positions specified by `indices`. Its output shape
    /// is the same as the shape of `data`.
    ///
    /// `indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
    /// `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
    /// Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
    /// update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
    /// update to a slice of the tensor. Index values are allowed to be negative, as per the usual
    /// convention for counting backwards from the end, but are expected in the valid range.
    ///
    /// `updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
    /// first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
    /// The remaining dimensions of `updates` correspond to the dimensions of the
    /// replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
    /// corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
    /// must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
    /// of shapes.
    ///
    /// The `output` is calculated via the following equation:
    ///
    /// ```
    /// output = np.copy(data)
    /// update_indices = indices.shape[:-1]
    /// for idx in np.ndindex(update_indices):
    ///     output[indices[idx]] = updates[idx]
    /// ```
    ///
    /// The order of iteration in the above loop is not specified.
    /// In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
    /// This ensures that the output value does not depend on the iteration order.
    ///
    /// `reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
    /// tensor into `output` at the specified `indices`.
    /// In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
    /// then indices[idx1] != indices[idx2]. This ensures that the output value does not depend on the iteration order.
    /// When `reduction` is set to some reduction function `f`, `output` is calculated as follows:
    ///
    /// ```
    /// output = np.copy(data)
    /// update_indices = indices.shape[:-1]
    /// for idx in np.ndindex(update_indices):
    ///     output[indices[idx]] = f(output[indices[idx]], updates[idx])
    /// ```
    ///
    /// where the `f` is `+`, `*`, `max` or `min` as specified.
    ///
    /// This operator is the inverse of GatherND.
    ///
    /// (Opset 18 change): Adds max/min to the set of allowed reduction ops.
    ///
    /// Example 1:
    /// ```
    /// data    = [1, 2, 3, 4, 5, 6, 7, 8]
    /// indices = [[4], [3], [1], [7]]
    /// updates = [9, 10, 11, 12]
    /// output  = [1, 11, 3, 10, 9, 6, 7, 12]
    /// ```
    ///
    /// Example 2:
    /// ```
    /// data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
    ///             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
    ///             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
    ///             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
    /// indices = [[0], [2]]
    /// updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    ///             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
    /// output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    ///             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
    ///             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
    ///             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
    /// ```
    public static <T> Tensor<T> ScatterND(Tensor<T> data, Tensor<Long> indices, Tensor<T> updates, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ScatterND.class, List.of(data, indices, updates), List.of(reduction));
        return (Tensor<T>) result;
    }

    ///
    /// Selu takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the scaled exponential linear unit function,
    /// `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
    /// is applied to the tensor elementwise.
    public static <T> Tensor<T> Selu(Tensor<T> X, Optional<Float> alpha, Optional<Float> gamma) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Selu.class, List.of(X), List.of(alpha, gamma));
        return (Tensor<T>) result;
    }

    ///
    /// Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
    /// Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
    /// Negative value means counting positions from the back.
    public static <S, T, I> Tensor<T> SequenceAt(List<Tensor<S>> input_sequence, Tensor<I> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceAt.class, List.of(input_sequence, position), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Construct a tensor sequence containing 'inputs' tensors.
    /// All tensors in 'inputs' must have the same data type.
    public static <T, S> List<Tensor<S>> SequenceConstruct(List<Tensor<T>> inputs) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceConstruct.class, List.of(inputs), List.of());
        return (List<Tensor<S>>) result;
    }

    ///
    /// Construct an empty tensor sequence, with given data type.
    public static <S> List<Tensor<S>> SequenceEmpty(Optional<Long> dtype) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceEmpty.class, List.of(), List.of(dtype));
        return (List<Tensor<S>>) result;
    }

    ///
    /// Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
    /// Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
    /// Negative value means counting positions from the back.
    /// 'position' is optional, by default it erases the last tensor from 'input_sequence'.
    public static <S, I> List<Tensor<S>> SequenceErase(List<Tensor<S>> input_sequence, Optional<Tensor<I>> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceErase.class, List.of(input_sequence, position), List.of());
        return (List<Tensor<S>>) result;
    }

    ///
    /// Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
    /// 'tensor' must have the same data type as 'input_sequence'.
    /// Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
    /// Negative value means counting positions from the back.
    /// 'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
    public static <T, S, I> List<Tensor<S>> SequenceInsert(List<Tensor<S>> input_sequence, Tensor<T> tensor, Optional<Tensor<I>> position) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceInsert.class, List.of(input_sequence, tensor, position), List.of());
        return (List<Tensor<S>>) result;
    }

    ///
    /// Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
    public static <S> Tensor<Long> SequenceLength(List<Tensor<S>> input_sequence) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SequenceLength.class, List.of(input_sequence), List.of());
        return (Tensor<Long>) result;
    }

    ///
    /// Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
    /// Optional attributes start and end can be used to compute a slice of the input tensor's shape.
    /// If start axis is omitted, the slice starts from axis 0.
    /// The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
    /// If the end axis is omitted, the axes upto the last one will be included.
    /// Negative axes indicate counting back from the last axis.
    /// Note that axes will be clamped to the range [0, r], where r is the
    /// rank of the input tensor if they are out-of-range (after adding r in the case of
    /// negative axis). Thus, specifying any end value > r is equivalent to specifying an end
    /// value of r, and specifying any start value < -r is equivalent to specifying a start
    /// value of 0. If start > end, the result will be an empty shape.
    ///
    /// Examples:
    ///
    /// ```
    /// Input tensor with shape: [2, 3, 4]
    /// No attributes specified.
    /// Output: [2, 3, 4]
    /// ```
    ///
    /// ```
    /// Input tensor with shape: [2, 3, 4]
    /// start: -1
    /// Output: [4]
    /// ```
    ///
    /// ```
    /// Input tensor with shape: [2, 3, 4]
    /// end: -1
    /// Output: [2, 3]
    /// ```
    ///
    /// ```
    /// Input tensor with shape: [2, 3, 4]
    /// start: 1
    /// end: 2
    /// Output: [3]
    /// ```
    public static <T> Tensor<Long> Shape(Tensor<T> data, Optional<Long> start, Optional<Long> end) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Shape.class, List.of(data), List.of(start, end));
        return (Tensor<Long>) result;
    }

    ///
    /// Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
    /// having same datatype and shape with input. It has two attributes, lambd and
    /// bias. The formula of this operator is: If x < -lambd, y = x + bias;
    /// If x > lambd, y = x - bias; Otherwise, y = 0.
    public static <T> Tensor<T> Shrink(Tensor<T> input, Optional<Float> lambd, Optional<Float> bias) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Shrink.class, List.of(input), List.of(lambd, bias));
        return (Tensor<T>) result;
    }

    ///
    /// Sigmoid takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
    /// tensor elementwise.
    public static <T> Tensor<T> Sigmoid(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sigmoid.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculate the sign of the given input tensor element-wise.
    /// If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
    public static <T> Tensor<T> Sign(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sign.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the sine of the given input tensor, element-wise.
    public static <T> Tensor<T> Sin(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sin.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the hyperbolic sine of the given input tensor element-wise.
    public static <T> Tensor<T> Sinh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sinh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
    public static <T> Tensor<Long> Size(Tensor<T> data) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Size.class, List.of(data), List.of());
        return (Tensor<Long>) result;
    }

    ///
    /// Produces a slice of the input tensor along multiple axes. Similar to numpy:
    /// https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding
    ///
    /// Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
    /// of its input `data` tensor.
    ///
    /// An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
    /// in `[0, ... r-1]` where `r = rank(input)` as follows:
    ///
    /// If `axes` are omitted, they are set to `[0, ..., r-1]`.
    /// If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
    ///
    /// The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
    /// `dims` are the dimensions of `input` and `steps[i] = 1`.
    ///
    /// All negative elements of `axes` are made non-negative by adding `r` to them, where
    /// `r =rank(input)`.
    ///
    /// All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
    /// where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
    /// `starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
    /// and `[0, dims[axes[i]]-1]` for negative stepping.
    ///
    /// The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
    /// accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
    /// `ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
    /// is clamped to `[-1, dims[axes[i]]-1]`.
    ///
    /// Finally, `steps[axes[i]] = steps[i]`.
    ///
    /// For slicing to the end of a dimension with unknown size, it is recommended to pass
    /// in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.
    ///
    /// Example 1:
    ///
    /// ```
    /// data = [
    ///     [1, 2, 3, 4],
    ///     [5, 6, 7, 8],
    /// ]
    /// axes = [0, 1]
    /// starts = [1, 0]
    /// ends = [2, 3]
    /// steps = [1, 2]
    /// result = [
    ///     [5, 7],
    /// ]
    /// ```
    ///
    /// Example 2:
    ///
    /// ```
    /// data = [
    ///     [1, 2, 3, 4],
    ///     [5, 6, 7, 8],
    /// ]
    /// starts = [0, 1]
    /// ends = [-1, 1000]
    /// result = [
    ///     [2, 3, 4],
    /// ]
    /// ```
    public static <T, Tind> Tensor<T> Slice(Tensor<T> data, Tensor<Tind> starts, Tensor<Tind> ends, Optional<Tensor<Tind>> axes, Optional<Tensor<Tind>> steps) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Slice.class, List.of(data, starts, ends, axes, steps), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// The operator computes the normalized exponential values for the given input:
    ///
    ///  Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
    ///
    /// The "axis" attribute indicates the dimension along which Softmax
    /// will be performed. The output tensor has the same shape
    /// and contains the Softmax values of the corresponding input.
    public static <T> Tensor<T> Softmax(Tensor<T> input, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softmax.class, List.of(input), List.of(axis));
        return (Tensor<T>) result;
    }

    public record SoftmaxCrossEntropyLossResult<T>(Tensor<T> output, Tensor<T> log_prob) { }
    /// Loss function that measures the softmax cross entropy
    /// between 'scores' and 'labels'.
    /// This operator first computes a loss tensor whose shape is identical to the labels input.
    /// If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
    /// If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
    /// the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
    /// After L is available, this operator can optionally do a reduction operator.
    ///
    /// * shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
    ///   with K >= 1 in case of K-dimensional loss.
    /// * shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
    ///   with K >= 1 in case of K-dimensional loss.
    ///
    /// The loss for one sample, l_i, can calculated as follows:
    /// ```
    /// l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
    /// ```
    /// or
    /// ```
    /// l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
    /// ```
    ///
    /// loss is zero for the case when label-value equals ignore_index.
    /// ```
    /// l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
    /// ```
    ///
    /// where:
    /// ```
    /// p = Softmax(scores)
    /// y = Log(p)
    /// c = labels[i][d1][d2]...[dk]
    /// ```
    ///
    /// Finally, L is optionally reduced:
    ///
    /// * If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
    /// * If reduction = 'sum', the output is scalar: Sum(L).
    /// * If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
    ///   where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.
    public static <T, Tind> SoftmaxCrossEntropyLossResult<T> SoftmaxCrossEntropyLoss(Tensor<T> scores, Tensor<Tind> labels, Optional<Tensor<T>> weights, Optional<Long> ignore_index, Optional<String> reduction) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SoftmaxCrossEntropyLoss.class, List.of(scores, labels, weights), List.of(ignore_index, reduction));
        Object[] resultArray = (Object[]) result;
        return new SoftmaxCrossEntropyLossResult<>((Tensor<T>)resultArray[0], (Tensor<T>)resultArray[1]);
    }

    ///
    /// Softplus takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
    /// the tensor elementwise.
    public static <T> Tensor<T> Softplus(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softplus.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
    public static <T> Tensor<T> Softsign(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Softsign.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    /// SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
    /// this op outputs a copy of the input tensor where values from the height and width dimensions
    /// are moved to the depth dimension.
    public static <T> Tensor<T> SpaceToDepth(Tensor<T> input, long blocksize) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SpaceToDepth.class, List.of(input), List.of(blocksize));
        return (Tensor<T>) result;
    }

    /// Split a tensor into a list of tensors, along the specified 'axis'.
    /// Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
    /// If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
    /// If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
    /// If the input 'split' is specified, it indicates the sizes of each output in the split.
    public static <T> List<Tensor<T>> Split(Tensor<T> input, Optional<Tensor<Long>> split, Optional<Long> num_outputs, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Split.class, List.of(input, split), List.of(num_outputs, axis));
        return (List<Tensor<T>>) result;
    }

    ///
    /// Split a tensor into a sequence of tensors, along the specified 'axis'.
    /// Lengths of the parts can be specified using the optional argument 'split'.
    /// If the argument `split' is not specified, a default scalar value of 1
    /// is used as the value of `split'.
    /// 'split' must contain only positive numbers.
    /// 'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
    /// If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
    /// if possible. The last chunk alone may be smaller than 'split' if the 'input' size
    /// along the given axis 'axis' is not divisible by 'split'.
    /// If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
    /// with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
    /// in 'split' must be equal to the dimension size of input tensor on 'axis'.
    public static <T, I, S> List<Tensor<S>> SplitToSequence(Tensor<T> input, Optional<Tensor<I>> split, Optional<Long> keepdims, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.SplitToSequence.class, List.of(input, split), List.of(keepdims, axis));
        return (List<Tensor<S>>) result;
    }

    ///
    /// Square root takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the square root is, y = x^0.5, is applied to
    /// the tensor elementwise. If x is negative, then it will return NaN.
    public static <T> Tensor<T> Sqrt(Tensor<T> X) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sqrt.class, List.of(X), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Remove single-dimensional entries from the shape of a tensor.
    /// Takes an input `axes` with a list of axes to squeeze.
    /// If `axes` is not provided, all the single dimensions will be removed from
    /// the shape. If an axis is selected with shape entry not equal to one, an error is raised.
    public static <T> Tensor<T> Squeeze(Tensor<T> data, Optional<Tensor<Long>> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Squeeze.class, List.of(data, axes), List.of());
        return (Tensor<T>) result;
    }

    /// StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support)
    public static Tensor<String> StringConcat(Tensor<String> X, Tensor<String> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringConcat.class, List.of(X, Y), List.of());
        return (Tensor<String>) result;
    }

    ///
    /// StringNormalization performs string operations for basic cleaning.
    /// This operator has only one input (denoted by X) and only one output
    /// (denoted by Y). This operator first examines the elements in the X,
    /// and removes elements specified in "stopwords" attribute.
    /// After removing stop words, the intermediate result can be further lowercased,
    /// uppercased, or just returned depending the "case_change_action" attribute.
    /// This operator only accepts [C]- and [1, C]-tensor.
    /// If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
    /// if input shape is [C] and shape [1, 1] if input shape is [1, C].
    public static Tensor<String> StringNormalizer(Tensor<String> X, Optional<Long> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringNormalizer.class, List.of(X), List.of(is_case_sensitive, locale, stopwords, case_change_action));
        return (Tensor<String>) result;
    }

    public record StringSplitResult(Tensor<String> Y, Tensor<Long> Z) { }
    /// StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute.
    ///
    /// The first output of this operator is a tensor of strings representing the substrings from splitting each input string on the `delimiter` substring. This tensor has one additional rank compared to the input tensor in order to store the substrings for each input element (where the input tensor is not empty). Note that, in order to ensure the same number of elements are present in the final dimension, this tensor will pad empty strings as illustrated in the examples below. Consecutive delimiters are not grouped together and are deemed to delimit empty strings, except if the `delimiter` is unspecified or is the empty string (""). In the case where the `delimiter` is unspecified or the empty string, consecutive whitespace characters are regarded as a single separator and leading or trailing whitespace is removed in the output.
    ///
    /// The second output tensor represents the number of substrings generated. `maxsplit` can be used to limit the number of splits performed - after the `maxsplit`th split if the string is not fully split, the trailing suffix of input string after the final split point is also added. For elements where fewer splits are possible than specified in `maxsplit`, it has no effect.
    public static StringSplitResult StringSplit(Tensor<String> X, Optional<String> delimiter, Optional<Long> maxsplit) {
        Object result = OnnxInterpreter.interpret(OnnxOps.StringSplit.class, List.of(X), List.of(delimiter, maxsplit));
        Object[] resultArray = (Object[]) result;
        return new StringSplitResult((Tensor<String>)resultArray[0], (Tensor<Long>)resultArray[1]);
    }

    ///
    /// Performs element-wise binary subtraction (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    ///
    /// (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
    public static <T> Tensor<T> Sub(Tensor<T> A, Tensor<T> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sub.class, List.of(A, B), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
    /// All inputs and outputs must have the same data type.
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> Sum(List<Tensor<T>> data_0) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Sum.class, List.of(data_0), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Swish function takes one input data (Tensor<T>) and produces one output data (Tensor<T>) of the same shape,
    /// where $Swish(x) = x * sigmoid(alpha * x)$.
    public static <T> Tensor<T> Swish(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Swish.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the tangent of the given input tensor, element-wise.
    public static <T> Tensor<T> Tan(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tan.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Calculates the hyperbolic tangent of the given input tensor element-wise.
    public static <T> Tensor<T> Tanh(Tensor<T> input) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tanh.class, List.of(input), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// TensorScatter is a generic tensor update operation, motivated by the requirements for KV cache updates for Attention
    /// ops commonly found in LLMs. It is a functional operation that models an in-place update to a KV cache buffer.
    ///
    /// The past and present cache tensors have the same shape (batch_size, D1, D2, ..., max_sequence_length, ..., Dn), with
    /// the sequence dimension (indicated by the `axis` attribute) being max_sequence_length, so the sizes of these tensors do
    /// not need to grow between iterations. The `update` tensor's shape only differs from the cache tensors in the sequence
    /// dimension: (batch_size, D1, D2, ..., sequence_length, ..., Dn), where sequence_length <= max_sequence_length.
    ///
    /// The optional `write_indices` input indicates the write index for each sample in the batch, assumed to be zero
    /// if not provided. When the `mode` attribute is set to "circular", the write index is modulo max_sequence_length.
    /// The operation can be described using the following pseudocode:
    ///
    /// ```
    /// for prefix_idx in np.ndindex(past_cache.shape[:axis]):
    ///     batch_idx = prefix_idx[0]
    ///     for sequence_idx in range(sequence_length):
    ///         cache_idx = (*prefix_idx, write_indices[batch_idx] + sequence_idx)
    ///         if mode == "circular":
    ///             cache_idx = tuple(np.mod(np.asarray(cache_idx), max_sequence_length))
    ///         update_idx = (*prefix_idx, sequence_idx)
    ///         present_cache[cache_idx] = update[update_idx]
    /// ```
    ///
    /// During the prefill phase of attention, only the first two inputs are needed. During the decode phase, `write_indices`
    /// is also needed so that the incoming key or value update can be appended after the last valid token for each sample
    /// in the batch.
    public static <T> Tensor<T> TensorScatter(Tensor<T> past_cache, Tensor<T> update, Optional<Tensor<Long>> write_indices, Optional<String> mode, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TensorScatter.class, List.of(past_cache, update, write_indices), List.of(mode, axis));
        return (Tensor<T>) result;
    }

    ///
    /// This transform extracts n-grams from the input sequence and save them as a vector. Input can
    /// be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
    /// For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
    /// More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
    /// If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.
    ///
    /// In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
    /// sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
    /// If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
    /// Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
    /// The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
    /// If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
    /// indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.
    ///
    /// The output vector (denoted by Y) stores the count of each n-gram;
    /// Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
    /// between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
    /// ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
    /// respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
    /// Note that we may consider all skips up to S when generating the n-grams.
    ///
    /// The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
    /// the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
    /// this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.
    ///
    /// Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
    /// If pool_strings is set, the input must be a string tensor.
    public static <T> Tensor<Float> TfIdfVectorizer(Tensor<T> X, long[] ngram_counts, long min_gram_length, Optional<String[]> pool_strings, String mode, long max_gram_length, long max_skip_count, Optional<long[]> pool_int64s, Optional<float[]> weights, long[] ngram_indexes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TfIdfVectorizer.class, List.of(X), List.of(ngram_counts, min_gram_length, pool_strings, mode, max_gram_length, max_skip_count, pool_int64s, weights, ngram_indexes));
        return (Tensor<Float>) result;
    }

    ///
    /// ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
    /// (Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
    /// is applied to the tensor elementwise.
    public static <T> Tensor<T> ThresholdedRelu(Tensor<T> X, Optional<Float> alpha) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ThresholdedRelu.class, List.of(X), List.of(alpha));
        return (Tensor<T>) result;
    }

    /// Constructs a tensor by tiling a given tensor.
    /// This is the same as function `tile` in Numpy, but no broadcast.
    /// For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
    public static <T> Tensor<T> Tile(Tensor<T> input, Tensor<Long> repeats) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Tile.class, List.of(input, repeats), List.of());
        return (Tensor<T>) result;
    }

    public record TopKResult<T>(Tensor<T> Values, Tensor<Long> Indices) { }
    ///
    /// Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
    /// shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:
    ///
    /// * Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
    ///   which contains the values of the top k elements along the specified axis
    /// * Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
    ///   contains the indices of the top k elements (original indices from the input
    ///   tensor).
    ///
    /// * If "largest" is 1 (the default value) then the k largest elements are returned.
    /// * If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
    /// * If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.
    ///
    /// Given two equivalent values, this operator uses the indices along the axis as
    /// a tiebreaker. That is, the element with the lower index will appear first.
    public static <T> TopKResult<T> TopK(Tensor<T> X, Tensor<Long> K, Optional<Long> largest, Optional<Long> sorted, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TopK.class, List.of(X, K), List.of(largest, sorted, axis));
        Object[] resultArray = (Object[]) result;
        return new TopKResult<>((Tensor<T>)resultArray[0], (Tensor<Long>)resultArray[1]);
    }

    ///
    /// Transpose the input tensor similar to numpy.transpose. For example, when
    /// perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
    /// will be (2, 1, 3).
    public static <T> Tensor<T> Transpose(Tensor<T> data, Optional<long[]> perm) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Transpose.class, List.of(data), List.of(perm));
        return (Tensor<T>) result;
    }

    ///
    ///     Tree Ensemble operator.  Returns the regressed values for each input in a batch.
    ///     Inputs have dimensions `[N, F]` where `N` is the input batch size and `F` is the number of input features.
    ///     Outputs have dimensions `[N, num_targets]` where `N` is the batch size and `num_targets` is the number of targets, which is a configurable attribute.
    ///
    ///     The encoding of this attribute is split along interior nodes and the leaves of the trees. Notably, attributes with the prefix `nodes_*` are associated with interior nodes, and attributes with the prefix `leaf_*` are associated with leaves.
    ///     The attributes `nodes_*` must all have the same length and encode a sequence of tuples, as defined by taking all the `nodes_*` fields at a given position.
    ///
    ///     All fields prefixed with `leaf_*` represent tree leaves, and similarly define tuples of leaves and must have identical length.
    ///
    ///     This operator can be used to implement both the previous `TreeEnsembleRegressor` and `TreeEnsembleClassifier` nodes.
    ///     The `TreeEnsembleRegressor` node maps directly to this node and requires changing how the nodes are represented.
    ///     The `TreeEnsembleClassifier` node can be implemented by adding a `ArgMax` node after this node to determine the top class.
    ///     To encode class labels, a `LabelEncoder` or `GatherND` operator may be used.
    public static <T> Tensor<T> TreeEnsemble(Tensor<T> X, Optional<Long> aggregate_function, Optional<Tensor> nodes_hitrates, long[] nodes_featureids, long[] nodes_falseleafs, Optional<Long> post_transform, long[] nodes_trueleafs, Tensor nodes_modes, long[] nodes_falsenodeids, long[] nodes_truenodeids, Tensor leaf_weights, long[] leaf_targetids, long[] tree_roots, Optional<Long> n_targets, Optional<long[]> nodes_missing_value_tracks_true, Optional<Tensor> membership_values, Tensor nodes_splits) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsemble.class, List.of(X), List.of(aggregate_function, nodes_hitrates, nodes_featureids, nodes_falseleafs, post_transform, nodes_trueleafs, nodes_modes, nodes_falsenodeids, nodes_truenodeids, leaf_weights, leaf_targetids, tree_roots, n_targets, nodes_missing_value_tracks_true, membership_values, nodes_splits));
        return (Tensor<T>) result;
    }

    public record TreeEnsembleClassifierResult<T2>(Tensor<T2> Y, Tensor<Float> Z) { }
    ///
    ///     This operator is DEPRECATED. Please use TreeEnsemble with provides similar functionality.
    ///     In order to determine the top class, the ArgMax node can be applied to the output of TreeEnsemble.
    ///     To encode class labels, use a LabelEncoder operator.
    ///     Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
    ///     The attributes named 'nodes_X' form a sequence of tuples, associated by
    ///     index into the sequences, which must all be of equal length. These tuples
    ///     define the nodes.<br>
    ///     Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    ///     A leaf may have multiple votes, where each vote is weighted by
    ///     the associated class_weights index.<br>
    ///     One and only one of classlabels_strings or classlabels_int64s
    ///     will be defined. The class_ids are indices into this list.
    ///     All fields ending with <i>_as_tensor</i> can be used instead of the
    ///     same parameter without the suffix if the element type is double and not float.
    public static <T1, T2> TreeEnsembleClassifierResult<T2> TreeEnsembleClassifier(Tensor<T1> X, Optional<long[]> classlabels_int64s, Optional<long[]> class_ids, Optional<float[]> nodes_hitrates, Optional<long[]> nodes_featureids, Optional<long[]> nodes_treeids, Optional<Tensor> class_weights_as_tensor, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<long[]> nodes_falsenodeids, Optional<String[]> classlabels_strings, Optional<long[]> nodes_truenodeids, Optional<long[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<float[]> class_weights, Optional<Tensor> base_values_as_tensor, Optional<long[]> nodes_missing_value_tracks_true, Optional<long[]> class_nodeids, Optional<long[]> class_treeids, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsembleClassifier.class, List.of(X), List.of(classlabels_int64s, class_ids, nodes_hitrates, nodes_featureids, nodes_treeids, class_weights_as_tensor, post_transform, nodes_modes, nodes_falsenodeids, classlabels_strings, nodes_truenodeids, nodes_nodeids, nodes_hitrates_as_tensor, class_weights, base_values_as_tensor, nodes_missing_value_tracks_true, class_nodeids, class_treeids, base_values, nodes_values, nodes_values_as_tensor));
        Object[] resultArray = (Object[]) result;
        return new TreeEnsembleClassifierResult<>((Tensor<T2>)resultArray[0], (Tensor<Float>)resultArray[1]);
    }

    ///
    ///     This operator is DEPRECATED. Please use TreeEnsemble instead which provides the same
    ///     functionality.<br>
    ///     Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    ///     All args with nodes_ are fields of a tuple of tree nodes, and
    ///     it is assumed they are the same length, and an index i will decode the
    ///     tuple across these inputs.  Each node id can appear only once
    ///     for each tree id.<br>
    ///     All fields prefixed with target_ are tuples of votes at the leaves.<br>
    ///     A leaf may have multiple votes, where each vote is weighted by
    ///     the associated target_weights index.<br>
    ///     All fields ending with <i>_as_tensor</i> can be used instead of the
    ///     same parameter without the suffix if the element type is double and not float.
    ///     All trees must have their node ids start at 0 and increment by 1.<br>
    ///     Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
    public static <T> Tensor<Float> TreeEnsembleRegressor(Tensor<T> X, Optional<String> aggregate_function, Optional<float[]> nodes_hitrates, Optional<Tensor> target_weights_as_tensor, Optional<long[]> nodes_featureids, Optional<long[]> target_treeids, Optional<long[]> nodes_treeids, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<float[]> target_weights, Optional<long[]> nodes_falsenodeids, Optional<long[]> target_ids, Optional<long[]> nodes_truenodeids, Optional<long[]> target_nodeids, Optional<long[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<Tensor> base_values_as_tensor, Optional<Long> n_targets, Optional<long[]> nodes_missing_value_tracks_true, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
        Object result = OnnxInterpreter.interpret(OnnxOps.TreeEnsembleRegressor.class, List.of(X), List.of(aggregate_function, nodes_hitrates, target_weights_as_tensor, nodes_featureids, target_treeids, nodes_treeids, post_transform, nodes_modes, target_weights, nodes_falsenodeids, target_ids, nodes_truenodeids, target_nodeids, nodes_nodeids, nodes_hitrates_as_tensor, base_values_as_tensor, n_targets, nodes_missing_value_tracks_true, base_values, nodes_values, nodes_values_as_tensor));
        return (Tensor<Float>) result;
    }

    ///
    /// Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
    /// The attribute "upper" determines whether the upper or lower part is retained. If set to true,
    /// the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
    /// Default value for the "upper" attribute is true.
    /// Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
    /// of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
    /// All other elements in the matrix are set to zero.
    /// If k = 0, the triangular part on and above/below the main diagonal is retained.
    /// If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
    /// A negative k value retains the main diagonal and |k| diagonals below it.
    /// If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
    /// A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
    public static <T> Tensor<T> Trilu(Tensor<T> input, Optional<Tensor<Long>> k, Optional<Long> upper) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Trilu.class, List.of(input, k), List.of(upper));
        return (Tensor<T>) result;
    }

    public record UniqueResult<T>(Tensor<T> Y, Tensor<Long> indices, Tensor<Long> inverse_indices, Tensor<Long> counts) { }
    ///
    /// Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
    /// Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.
    ///
    /// This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
    /// The first output tensor 'Y' contains all unique values or subtensors of the input.
    /// The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
    /// The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
    /// The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.
    ///
    /// Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.
    ///
    /// https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    ///
    /// Example 1:
    /// ```
    /// input_X = [2, 1, 1, 3, 4, 3]
    /// attribute_sorted = 0
    /// attribute_axis = None
    /// output_Y = [2, 1, 3, 4]
    /// output_indices = [0, 1, 3, 4]
    /// output_inverse_indices = [0, 1, 1, 2, 3, 2]
    /// output_counts = [1, 2, 2, 1]
    /// ```
    ///
    /// Example 2:
    /// ```
    /// input_X = [[1, 3], [2, 3]]
    /// attribute_sorted = 1
    /// attribute_axis = None
    /// output_Y = [1, 2, 3]
    /// output_indices = [0, 2, 1]
    /// output_inverse_indices = [0, 2, 1, 2]
    /// output_counts = [1, 1, 2]
    /// ```
    ///
    /// Example 3:
    /// ```
    /// input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
    /// attribute_sorted = 1
    /// attribute_axis = 0
    /// output_Y = [[1, 0, 0], [2, 3, 4]]
    /// output_indices = [0, 2]
    /// output_inverse_indices = [0, 0, 1]
    /// output_counts = [2, 1]
    /// ```
    ///
    /// Example 4:
    /// ```
    /// input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
    ///             [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
    /// attribute_sorted = 1
    /// attribute_axis = 1
    /// ```
    ///
    /// intermediate data are presented below for better understanding:
    /// there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
    /// ```
    /// A: [[1, 1], [1, 1]],
    ///    [[0, 1], [0, 1]],
    ///    [[2, 1], [2, 1]],
    ///    [[0, 1], [0, 1]].
    /// ```
    ///
    /// there are 3 unique subtensors:
    /// ```
    /// [[1, 1], [1, 1]],
    /// [[0, 1], [0, 1]],
    /// [[2, 1], [2, 1]].
    /// ```
    ///
    /// sorted unique subtensors:
    /// ```
    /// B: [[0, 1], [0, 1]],
    ///    [[1, 1], [1, 1]],
    ///    [[2, 1], [2, 1]].
    /// ```
    ///
    /// output_Y is constructed from B:
    /// ```
    /// [[[0. 1.], [1. 1.], [2. 1.]],
    ///  [[0. 1.], [1. 1.], [2. 1.]]]
    /// ```
    ///
    /// output_indices is to map from B to A:
    /// ```
    /// [1, 0, 2]
    /// ```
    ///
    /// output_inverse_indices is to map from A to B:
    /// ```
    /// [1, 0, 2, 0]
    /// ```
    ///
    /// output_counts:
    /// ```
    /// [2, 1, 1]
    /// ```
    public static <T> UniqueResult<T> Unique(Tensor<T> X, Optional<Long> sorted, Optional<Long> axis) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Unique.class, List.of(X), List.of(sorted, axis));
        Object[] resultArray = (Object[]) result;
        return new UniqueResult<>((Tensor<T>)resultArray[0], (Tensor<Long>)resultArray[1], (Tensor<Long>)resultArray[2], (Tensor<Long>)resultArray[3]);
    }

    ///
    /// Insert single-dimensional entries to the shape of an input tensor (`data`).
    /// Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).
    ///
    /// For example, given an input tensor (`data`) of shape [3, 4, 5], then
    /// Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].
    ///
    /// The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
    /// The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
    /// Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
    /// The order of values in `axes` does not matter and can come in any order.
    public static <T> Tensor<T> Unsqueeze(Tensor<T> data, Tensor<Long> axes) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Unsqueeze.class, List.of(data, axes), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Upsample the input tensor.
    /// Each dimension value of the output tensor is:
    ///   output_dimension = floor(input_dimension * scale).
    public static <T> Tensor<T> Upsample(Tensor<T> X, Tensor<Float> scales, Optional<String> mode) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Upsample.class, List.of(X, scales), List.of(mode));
        return (Tensor<T>) result;
    }

    ///
    /// Return elements, either from X or Y, depending on condition.
    /// Where behaves like
    /// [numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
    /// with three parameters.
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static <T> Tensor<T> Where(Tensor<Boolean> condition, Tensor<T> X, Tensor<T> Y) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Where.class, List.of(condition, X, Y), List.of());
        return (Tensor<T>) result;
    }

    ///
    /// Returns the tensor resulted from performing the `xor` logical operation
    /// elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
    ///
    /// This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
    public static Tensor<Boolean> Xor(Tensor<Boolean> A, Tensor<Boolean> B) {
        Object result = OnnxInterpreter.interpret(OnnxOps.Xor.class, List.of(A, B), List.of());
        return (Tensor<Boolean>) result;
    }

    ///
    ///     Creates a map from the input and the attributes.<br>
    ///     The values are provided by the input tensor, while the keys are specified by the attributes.
    ///     Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    ///     The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
    public static <T> List<Map<T, Float>> ZipMap(Tensor<Float> X, Optional<long[]> classlabels_int64s, Optional<String[]> classlabels_strings) {
        Object result = OnnxInterpreter.interpret(OnnxOps.ZipMap.class, List.of(X), List.of(classlabels_int64s, classlabels_strings));
        return (List<Map<T, Float>>) result;
    }

}
