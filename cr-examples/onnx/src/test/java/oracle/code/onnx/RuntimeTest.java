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
        try (var absOp = ort.createSession(OnnxProtoBuilder.build(
                List.of(OnnxProtoBuilder.valueInfo("x", FLOAT.id)),
                List.of(OnnxProtoBuilder.node("Abs", List.of("x"), List.of("y"), Map.of())),
                List.of("y")));
             var addOp = ort.createSession(OnnxProtoBuilder.build(
                List.of(OnnxProtoBuilder.valueInfo("a", FLOAT.id), OnnxProtoBuilder.valueInfo("b", FLOAT.id)),
                List.of(OnnxProtoBuilder.node("Add", List.of("a", "b"), List.of("y"), Map.of())),
                List.of("y")))) {

            assertEquals(1, absOp.getNumberOfInputs());
            assertEquals(1, absOp.getNumberOfOutputs());

            assertEquals(2, addOp.getNumberOfInputs());
            assertEquals(1, addOp.getNumberOfOutputs());

            var inputTensor = Tensor.ofFlat(-1f, 2, -3, 4, -5, 6);

            var absExpectedTensor = Tensor.ofFlat(1f, 2, 3, 4, 5, 6);

            var absResult = absOp.run(List.of(inputTensor));

            assertEquals(1, absResult.size());

            var absOutputTensor = absResult.getFirst();

            SimpleTest.assertEquals(absExpectedTensor, absOutputTensor);

            var addResult = addOp.run(List.of(inputTensor, absOutputTensor));

            assertEquals(1, addResult.size());

            var addOutputTensor = addResult.getFirst();

            var addExpectedTensor = Tensor.ofFlat(0f, 4, 0, 8, 0, 12);

            SimpleTest.assertEquals(addExpectedTensor, addOutputTensor);
        }
    }

    @Test
    public void testIf() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (var ifOp = ort.createSession(OnnxProtoBuilder.build(
                List.of(OnnxProtoBuilder.valueInfo("cond", BOOL.id), OnnxProtoBuilder.valueInfo("a", INT64.id), OnnxProtoBuilder.valueInfo("b", INT64.id)),
                List.of(OnnxProtoBuilder.node("If", List.of("cond"), List.of("y"), Map.of(
                        "then_branch", OnnxProtoBuilder.graph(
                                List.of(),
                                List.of(OnnxProtoBuilder.node("Identity", List.of("a"), List.of("y"), Map.of())),
                                List.of("y")),
                        "else_branch", OnnxProtoBuilder.graph(
                                List.of(),
                                List.of(OnnxProtoBuilder.node("Identity", List.of("b"), List.of("y"), Map.of())),
                                List.of("y"))))),
                List.of("y")))) {

            var a = Tensor.ofScalar(1l);
            var b = Tensor.ofScalar(2l);
            SimpleTest.assertEquals(a, ifOp.run(List.of(Tensor.ofScalar(true), a, b)).getFirst());
            SimpleTest.assertEquals(b, ifOp.run(List.of(Tensor.ofScalar(false), a, b)).getFirst());
        }
    }

//    @Test
//    public void testFor() throws Exception {
//        var ort = OnnxRuntime.getInstance();
//        try (var forOp = ort.createSession(OnnxProtoBuilder.build(
//                List.of(new OnnxProtoBuilder.Input("max", INT64.id), new OnnxProtoBuilder.Input("cond", BOOL.id), new OnnxProtoBuilder.Input("a", INT64.id)),
//                List.of(new OnnxProtoBuilder.OpNode("Loop", List.of("max", "cond", "a"), List.of("y"), Map.of(
//                        "body", new OnnxProtoBuilder.Subgraph(
//                                List.of(new OnnxProtoBuilder.Input("i", INT64.id), new OnnxProtoBuilder.Input("cond_in", BOOL.id), new OnnxProtoBuilder.Input("a_in", INT64.id)),
//                                List.of(new OnnxProtoBuilder.OpNode("Mul", List.of("a_in", "a_in"), List.of("a_out"), Map.of())),
//                                List.of("cond_in", "a_out"))))),
//                List.of("y")))) {
//
//            var a = Tensor.ofScalar(2l);
//            SimpleTest.assertEquals(Tensor.ofScalar(65536l), new Tensor(forOp.run(List.of(Tensor.ofScalar(16l).tensorAddr, Tensor.ofScalar(true).tensorAddr, Tensor.ofScalar(16l).tensorAddr)).getFirst()));
//        }
//    }
}
