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
        try (var absOp = ort.createSession(OnnxProtoBuilder.buildModel(
                List.of(new OnnxProtoBuilder.Input("x", FLOAT.id)),
                List.of(new OnnxProtoBuilder.OpNode("Abs", List.of("x"), List.of("y"), Map.of())),
                List.of("y")));
             var addOp = ort.createSession(OnnxProtoBuilder.buildModel(
                List.of(new OnnxProtoBuilder.Input("a", FLOAT.id), new OnnxProtoBuilder.Input("b", FLOAT.id)),
                List.of(new OnnxProtoBuilder.OpNode("Add", List.of("a", "b"), List.of("y"), Map.of())),
                List.of("y")))) {

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

    @Test
    public void testIf() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (var ifOp = ort.createSession(OnnxProtoBuilder.buildModel(
                List.of(new OnnxProtoBuilder.Input("cond", BOOL.id)),
                List.of(new OnnxProtoBuilder.OpNode("If", List.of("cond"), List.of("y"), Map.of(
                        "then_branch", new OnnxProtoBuilder.Subgraph(
                                List.of(),
                                List.of(new OnnxProtoBuilder.OpNode("Constant",
                                        List.of(),
                                        List.of("y"),
                                        Map.of("value_int", 1l))),
                                List.of("y")),
                        "else_branch", new OnnxProtoBuilder.Subgraph(
                                List.of(),
                                List.of(new OnnxProtoBuilder.OpNode("Constant",
                                        List.of(),
                                        List.of("y"),
                                        Map.of("value_int", 2l))),
                                List.of("y"))))),
                List.of("y")))) {

            SimpleTest.assertEquals(Tensor.ofScalar(1l), new Tensor(ifOp.run(List.of(Tensor.ofScalar(true).tensorAddr)).getFirst()));
            SimpleTest.assertEquals(Tensor.ofScalar(2l), new Tensor(ifOp.run(List.of(Tensor.ofScalar(false).tensorAddr)).getFirst()));
        }
    }
}
