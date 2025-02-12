package oracle.code.onnx;

import java.util.List;
import oracle.code.onnx.ir.OnnxOps;
import org.junit.jupiter.api.Test;

import static oracle.code.onnx.Tensor.ElementType.*;

import static org.junit.jupiter.api.Assertions.*;

public class RuntimeTest {

    @Test
    public void test() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (var absOp = ort.createSession(OnnxProtoBuilder.buildOpModel(OnnxOps.Abs.SCHEMA, List.of(FLOAT, FLOAT), List.of()));
             var addOp = ort.createSession(OnnxProtoBuilder.buildOpModel(OnnxOps.Add.SCHEMA, List.of(FLOAT, FLOAT), List.of()))) {

            assertEquals(1, absOp.getNumberOfInputs());
            assertEquals(1, absOp.getNumberOfOutputs());

            assertEquals(2, addOp.getNumberOfInputs());
            assertEquals(1, addOp.getNumberOfOutputs());

            var inputTensor = ort.createFlatTensor(-1f, 2, -3, 4, -5, 6);

            var absExpectedTensor = ort.createFlatTensor(1f, 2, 3, 4, 5, 6);

            var absResult = absOp.run(List.of(inputTensor));

            assertEquals(1, absResult.size());

            var absOutputTensor = (OnnxRuntime.OrtTensor)absResult.getFirst();

            SimpleTest.assertEquals(absExpectedTensor, absOutputTensor);

            var addResult = addOp.run(List.of(inputTensor, absOutputTensor));

            assertEquals(1, addResult.size());

            var addOutputTensor = (OnnxRuntime.OrtTensor)addResult.getFirst();

            var addExpectedTensor = ort.createFlatTensor(0f, 4, 0, 8, 0, 12);

            SimpleTest.assertEquals(addExpectedTensor, addOutputTensor);
        }
    }
}
