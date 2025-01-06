package oracle.code.onnx;

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

    }
}
