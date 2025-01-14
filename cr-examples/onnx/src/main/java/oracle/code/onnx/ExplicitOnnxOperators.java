package oracle.code.onnx;

import java.util.Optional;

class ExplicitOnnxOperators {

    // Explicit constant operators

    public static Tensor<Long> Constant(
            Integer c) {
        return OnnxOperators.Constant(
                Optional.of(c),Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
    }

    public static Tensor<Long> Constant(
            int[] c) {
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
}
