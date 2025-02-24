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
                List.of(new OnnxProtoBuilder.Input("cond", BOOL.id), new OnnxProtoBuilder.Input("a", INT64.id), new OnnxProtoBuilder.Input("b", INT64.id)),
                List.of(new OnnxProtoBuilder.OpNode("If", List.of("cond"), List.of("y"), Map.of(
                        "then_branch", new OnnxProtoBuilder.Subgraph(
                                List.of(),
                                List.of(new OnnxProtoBuilder.OpNode("Identity", List.of("a"), List.of("y"), Map.of())),
                                List.of("y")),
                        "else_branch", new OnnxProtoBuilder.Subgraph(
                                List.of(),
                                List.of(new OnnxProtoBuilder.OpNode("Identity", List.of("b"), List.of("y"), Map.of())),
                                List.of("y"))))),
                List.of("y")))) {

            var a = Tensor.ofScalar(1l);
            var b = Tensor.ofScalar(2l);
            SimpleTest.assertEquals(a, new Tensor(ifOp.run(List.of(Tensor.ofScalar(true).tensorAddr, a.tensorAddr, b.tensorAddr)).getFirst()));
            SimpleTest.assertEquals(b, new Tensor(ifOp.run(List.of(Tensor.ofScalar(false).tensorAddr, a.tensorAddr, b.tensorAddr)).getFirst()));
        }
    }
}
