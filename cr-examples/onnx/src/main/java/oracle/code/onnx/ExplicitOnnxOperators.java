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

    public static Tensor<?> Constant(
            Object c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.of(c), Optional.empty());
    }

    public static <T> Tensor<T> Constant(
            Tensor<T> c) {
        return OnnxOperators.Constant(
                Optional.empty(),Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(c));
    }
}
