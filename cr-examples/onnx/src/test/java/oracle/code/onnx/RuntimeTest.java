package oracle.code.onnx;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

import static oracle.code.onnx.OnnxProtoBuilder.*;
import static oracle.code.onnx.Tensor.ElementType.*;

import static org.junit.jupiter.api.Assertions.*;

public class RuntimeTest {

    @Test
    public void test() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (Arena sessionArena = Arena.ofConfined()) {
            var absOp = ort.createSession(build(
                    List.of(tensorInfo("x", FLOAT.id)),
                    List.of(node("Abs", List.of("x"), List.of("y"), Map.of())),
                    List.of("y")), sessionArena);

             var addOp = ort.createSession(build(
                List.of(tensorInfo("a", FLOAT.id), tensorInfo("b", FLOAT.id)),
                List.of(node("Add", List.of("a", "b"), List.of("y"), Map.of())),
                List.of("y")), sessionArena);

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
        try (Arena sessionArena = Arena.ofConfined()) {
        var ifOp = ort.createSession(build(
                List.of(tensorInfo("cond", BOOL.id), tensorInfo("a", INT64.id), tensorInfo("b", INT64.id)),
                List.of(node("If", List.of("cond"), List.of("y"), Map.of(
                        "then_branch", graph(
                                List.of(),
                                List.of(node("Identity", List.of("a"), List.of("y"), Map.of())),
                                List.of("y")),
                        "else_branch", graph(
                                List.of(),
                                List.of(node("Identity", List.of("b"), List.of("y"), Map.of())),
                                List.of("y"))))),
                List.of("y")), sessionArena);

            var a = Tensor.ofScalar(1l);
            var b = Tensor.ofScalar(2l);
            SimpleTest.assertEquals(a, ifOp.run(List.of(Tensor.ofScalar(true), a, b)).getFirst());
            SimpleTest.assertEquals(b, ifOp.run(List.of(Tensor.ofScalar(false), a, b)).getFirst());
        }
    }

    @Test
    public void testLoop() throws Exception {
        var ort = OnnxRuntime.getInstance();
        try (Arena sessionArena = Arena.ofConfined()) {
            var forOp = ort.createSession(build(
                    List.of(tensorInfo("max", INT64.id), tensorInfo("cond", BOOL.id), tensorInfo("a", INT64.id)),
                    List.of(node("Loop", List.of("max", "cond", "a"), List.of("a_out"), Map.of(
                            "body", graph(
                                    List.of(scalarInfo("i", INT64.id), scalarInfo("cond_in", BOOL.id), tensorInfo("a_in", INT64.id)),
                                    List.of(node("Identity", List.of("cond_in"), List.of("cond_out"), Map.of()),
                                            node("Add", List.of("a_in", "a_in"), List.of("a_out"), Map.of())),
                                    List.of("cond_out", "a_out"))))),
                    List.of("a_out")), sessionArena);

            SimpleTest.assertEquals(Tensor.ofScalar(65536l), forOp.run(List.of(Tensor.ofScalar(15l), Tensor.ofScalar(true), Tensor.ofScalar(2l))).getFirst());
        }
    }
}
