package oracle.code.onnx;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import oracle.code.onnx.compiler.OnnxTransformer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SimpleTest {

    // Java code model -> ONNX code model -> ONNX runtime instance -> execute via ORT
    // Run directly, each operation reflectively executes via ORT
    @CodeReflection
    public static Tensor<Float> add(Tensor<Float> a, Tensor<Float> b) {
        return OnnxOperators.Add(a, b);
    }

    @Test
    public void testAdd() throws Exception {
        var a = new Tensor(1f, 2, 3);
        var b = new Tensor(6f, 5, 4);
        assertEquals(
                add(a, b),
                new Tensor(OnnxRuntime.getInstance().runFunc(getOnnxModel("add", Tensor.class, Tensor.class),
                        List.of(a.rtTensor, b.rtTensor)).getFirst()));
    }

    @CodeReflection
    public static Tensor<Long> reshapeAndShape(Tensor<Float> data, Tensor<Long> shape) {
        return OnnxOperators.Shape(OnnxOperators.Reshape(data, shape, Optional.empty()), Optional.empty(), Optional.empty());
    }

    @Test
    public void testReshapeAndShape() throws Exception {
        var data = new Tensor(1f, 2, 3, 4, 5, 6, 7, 8);
        var shape = new Tensor(2l, 2, 2);
        assertEquals(shape, reshapeAndShape(data, shape));
        assertEquals(shape, new Tensor(OnnxRuntime.getInstance().runFunc(getOnnxModel("reshapeAndShape", Tensor.class, Tensor.class),
                        List.of(data.rtTensor, shape.rtTensor)).getFirst()));
    }

    private static CoreOp.FuncOp getOnnxModel(String name, Class... params) throws NoSuchMethodException {
        return OnnxTransformer.transform(MethodHandles.publicLookup(),
                Op.ofMethod(SimpleTest.class.getDeclaredMethod(name, params)).get());
    }

    static void assertEquals(Tensor actual, Tensor expected) {
        var expectedTS = expected.rtTensor.getTensorTypeAndShape();
        var actualTS = actual.rtTensor.getTensorTypeAndShape();
        assertSame(expectedTS.getTensorElementType(), actualTS.getTensorElementType());

        // @@@ assert equal shapes

        switch (actualTS.getTensorElementType()) {
            case FLOAT ->
                RuntimeTest.assertEqualData(expected.rtTensor.asByteBuffer().asFloatBuffer(), actual.rtTensor.asByteBuffer().asFloatBuffer());
            case INT64 ->
                RuntimeTest.assertEqualData(expected.rtTensor.asByteBuffer().asLongBuffer(), actual.rtTensor.asByteBuffer().asLongBuffer());
            default ->
                throw new UnsupportedOperationException(); // @@@ ToDo
        }
    }
}
