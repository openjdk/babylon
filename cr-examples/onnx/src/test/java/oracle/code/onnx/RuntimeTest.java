package oracle.code.onnx;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

import static oracle.code.onnx.Tensor.ElementType.*;

import static org.junit.jupiter.api.Assertions.*;

public class RuntimeTest {

    @Test
    public void test() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (var absOp = ort.createSession(OnnxProtoBuilder.buildOpModel("Abs", List.of(FLOAT), 1, Map.of()));
             var addOp = ort.createSession(OnnxProtoBuilder.buildOpModel("Add", List.of(FLOAT, FLOAT), 1, Map.of()))) {

            assertEquals(1, absOp.getNumberOfInputs());
            assertEquals(1, absOp.getNumberOfOutputs());

            assertEquals(2, addOp.getNumberOfInputs());
            assertEquals(1, addOp.getNumberOfOutputs());

            var inputTensor = Tensor.ofFlat(-1f, 2, -3, 4, -5, 6);

            var absExpectedTensor = Tensor.ofFlat(1f, 2, 3, 4, 5, 6);

            var absResult = absOp.run(List.of(inputTensor.tensorAddr));

            assertEquals(1, absResult.size());

            var absOutputTensor = new Tensor(absResult.getFirst());

            SimpleTest.assertEquals(absExpectedTensor, absOutputTensor);

            var addResult = addOp.run(List.of(inputTensor.tensorAddr, absOutputTensor.tensorAddr));

            assertEquals(1, addResult.size());

            var addOutputTensor = new Tensor(addResult.getFirst());

            var addExpectedTensor = Tensor.ofFlat(0f, 4, 0, 8, 0, 12);

            SimpleTest.assertEquals(addExpectedTensor, addOutputTensor);
        }
    }
}
