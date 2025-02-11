package oracle.code.onnx;

import java.nio.FloatBuffer;
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
        assertEquals(add(new Tensor(1f, 2, 3), new Tensor(6f, 5, 4)), 7, 7, 7);
    }

    @CodeReflection
    public Tensor<Float> reshape(Tensor<Float> a, Tensor<Long> b) {
        return OnnxOperators.Reshape(a, b, Optional.empty());
    }

    @Test
    public void testReshape() {
        var reshaped = reshape(new Tensor(1f, 2, 3, 4, 5, 6, 7, 8), new Tensor(2, 2, 2));
        assertEquals(reshaped, 1f, 2, 3, 4, 5, 6, 7, 8);

    }

    static void assertEquals(Tensor actual, float... expected) {
        RuntimeTest.assertEqualData(FloatBuffer.wrap(expected), actual.rtTensor.asByteBuffer().asFloatBuffer());
    }
}
