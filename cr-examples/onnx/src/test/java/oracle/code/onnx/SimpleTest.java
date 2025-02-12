package oracle.code.onnx;

import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import oracle.code.onnx.compiler.OnnxTransformer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

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

    static void assertEquals(Tensor expected, Tensor actual) {
        assertEquals(expected.rtTensor, actual.rtTensor);
    }

    static void assertEquals(OnnxRuntime.OrtTensor expected, OnnxRuntime.OrtTensor actual) {

        var expectedType = expected.getTensorTypeAndShape();
        var expectedShape = expectedType.getShape();

        var actualType = actual.getTensorTypeAndShape();
        var actualShape = actualType.getShape();

        Assertions.assertSame(expectedType.getTensorElementType(), actualType.getTensorElementType());

        Assertions.assertEquals(expectedShape.getDimensionsCount(), actualShape.getDimensionsCount());
        for (int i = 0; i < expectedShape.getDimensionsCount(); i++) {
            Assertions.assertEquals(expectedShape.getDimension(i), actualShape.getDimension(i));
        }

        switch (actualType.getTensorElementType()) {
            case UINT8, INT8, UINT16, INT16, INT32, INT64, STRING, BOOL, UINT32, UINT64, UINT4, INT4 ->
                assertEquals(expected.asByteBuffer(), actual.asByteBuffer());
            case FLOAT ->
                assertEquals(expected.asByteBuffer().asFloatBuffer(), actual.asByteBuffer().asFloatBuffer());
            case DOUBLE ->
                assertEquals(expected.asByteBuffer().asDoubleBuffer(), actual.asByteBuffer().asDoubleBuffer());
            default ->
                throw new UnsupportedOperationException("Unsupported tensor element type " + actualType.getTensorElementType());
        }
    }

    static void assertEquals(ByteBuffer expectedData, ByteBuffer actualData) {
        Assertions.assertEquals(expectedData.capacity(), actualData.capacity());
        for (int i = 0; i < expectedData.capacity(); i++) {
            Assertions.assertEquals(expectedData.get(i), actualData.get(i));
        }
    }

    static void assertEquals(FloatBuffer expectedData, FloatBuffer actualData) {
        Assertions.assertEquals(expectedData.capacity(), actualData.capacity());
        for (int i = 0; i < expectedData.capacity(); i++) {
            Assertions.assertEquals(expectedData.get(i), actualData.get(i), 1e-6f);
        }
    }

    static void assertEquals(DoubleBuffer expectedData, DoubleBuffer actualData) {
        Assertions.assertEquals(expectedData.capacity(), actualData.capacity());
        for (int i = 0; i < expectedData.capacity(); i++) {
            Assertions.assertEquals(expectedData.get(i), actualData.get(i), 1e-6f);
        }
    }
}
