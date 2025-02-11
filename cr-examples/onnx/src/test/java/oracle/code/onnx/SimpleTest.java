package oracle.code.onnx;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Test;

public class SimpleTest {

    // Java code model -> ONNX code model -> ONNX runtime instance -> execute via ORT
    // Run directly, each operation reflectively executes via ORT
    @CodeReflection
    public Tensor<Float> add(Tensor<Float> a, Tensor<Float> b) {
        return OnnxOperators.Add(a, b);
    }

    @Test
    public void testAdd() {
        assertEquals(add(new Tensor(1f, 2, 3), new Tensor(6f, 5, 4)), 7f, 7, 7);
    }

    @CodeReflection
    public Tensor<Float> reshape(Tensor<Float> data, Tensor<Long> shape) {
        return OnnxOperators.Reshape(data, shape, Optional.empty());
    }

    @CodeReflection
    public Tensor<Long> shape(Tensor<Float> data) {
        return OnnxOperators.Shape(data, Optional.empty(), Optional.empty());
    }

    @Test
    public void testReshapeAndShape() {
        var reshaped = reshape(new Tensor(1f, 2, 3, 4, 5, 6, 7, 8), new Tensor(2l, 2, 2));
        assertEquals(reshaped, 1f, 2, 3, 4, 5, 6, 7, 8);
        var shape = shape(reshaped);
        assertEquals(shape, 2l, 2, 2);
    }

    static void assertEquals(Tensor actual, float... expected) {
        RuntimeTest.assertEqualData(FloatBuffer.wrap(expected), actual.rtTensor.asByteBuffer().asFloatBuffer());
    }

    static void assertEquals(Tensor actual, long... expected) {
        RuntimeTest.assertEqualData(LongBuffer.wrap(expected), actual.rtTensor.asByteBuffer().asLongBuffer());
    }
}
