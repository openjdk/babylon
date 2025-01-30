package oracle.code.onnx;

import java.nio.FloatBuffer;
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
    public void test() {
        assertEquals(add(new Tensor(1, 2, 3), new Tensor(6, 5, 4)), 7, 7, 7);
    }

    static void assertEquals(Tensor actual, float... expected) {
        RuntimeTest.assertEqualData(FloatBuffer.wrap(expected), actual.rtTensor.asByteBuffer().asFloatBuffer());
    }
}
