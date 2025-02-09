package oracle.code.onnx;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import oracle.code.onnx.ir.OnnxOps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class RuntimeTest {

    @Test
    public void test() throws Exception {
        var env = new OnnxRuntime().createEnv();
        try (var absOp = env.createSession(OnnxProtoBuilder.op(OnnxOps.Abs.SCHEMA, Tensor.ElementType.FLOAT));
             var addOp = env.createSession(OnnxProtoBuilder.op(OnnxOps.Add.SCHEMA, Tensor.ElementType.FLOAT))) {

            assertEquals(1, absOp.getNumberOfInputs());
            assertEquals(1, absOp.getNumberOfOutputs());

            assertEquals(2, addOp.getNumberOfInputs());
            assertEquals(1, addOp.getNumberOfOutputs());

            var inputShape = ((OnnxRuntime.OrtTensorTypeAndShapeInfo)absOp.getInputTypeInfo(0)).getShape();
            var inputTensor = env.createTensor(inputShape, -1, 2, -3, 4, -5, 6);

            var absExpectedShape = ((OnnxRuntime.OrtTensorTypeAndShapeInfo)absOp.getOutputTypeInfo(0)).getShape();
            var absExpectedTensor = env.createTensor(absExpectedShape, 1, 2, 3, 4, 5, 6);

            var absResult = absOp.run(Map.of(absOp.getInputName(0), inputTensor), List.of(absOp.getOutputName(0)));

            assertEquals(1, absResult.size());

            var absOutputTensor = (OnnxRuntime.OrtTensor)absResult.getFirst();

            assertTensorEquals(absExpectedTensor, absOutputTensor);

            var addResult = addOp.run(Map.of(addOp.getInputName(0), inputTensor, addOp.getInputName(1), absOutputTensor), List.of(addOp.getOutputName(0)));

            assertEquals(1, addResult.size());

            var addOutputTensor = (OnnxRuntime.OrtTensor)addResult.getFirst();

            var addExpectedShape = ((OnnxRuntime.OrtTensorTypeAndShapeInfo)absOp.getOutputTypeInfo(0)).getShape();
            var addExpectedTensor = env.createTensor(addExpectedShape, 0, 4, 0, 8, 0, 12);

            assertTensorEquals(addExpectedTensor, addOutputTensor);
        }
    }

    static void assertTensorEquals(OnnxRuntime.OrtTensor expectedTensor, OnnxRuntime.OrtTensor actualTensor) {
        var expectedType = expectedTensor.getTensorTypeAndShape();
        var expectedShape = expectedType.getShape();

        var actualType = actualTensor.getTensorTypeAndShape();
        var actualShape = actualType.getShape();

        assertEquals(expectedShape.getDimensionsCount(), actualShape.getDimensionsCount());
        for (int i = 0; i < expectedShape.getDimensionsCount(); i++) {
            assertEquals(expectedShape.getDimension(i), actualShape.getDimension(i));
        }

        assertEquals(expectedType.getTensorElementType(), actualType.getTensorElementType());
        assertEquals(expectedType.getTensorShapeElementCount(), actualType.getTensorShapeElementCount());

        assertEqualData(expectedTensor.asByteBuffer().asFloatBuffer(), actualTensor.asByteBuffer().asFloatBuffer());
    }

    static void assertEqualData(FloatBuffer expectedData, FloatBuffer actualData) {
        assertEquals(expectedData.capacity(), actualData.capacity());
        for (int i = 0; i < expectedData.capacity(); i++) {
            assertEquals(expectedData.get(i), actualData.get(i), 1e-6f);
        }
    }
}
