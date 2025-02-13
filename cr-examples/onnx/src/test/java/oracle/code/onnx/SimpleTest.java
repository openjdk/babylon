package oracle.code.onnx;

import java.lang.invoke.MethodHandles;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Stream;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import oracle.code.onnx.compiler.OnnxTransformer;
import oracle.code.onnx.ir.OnnxOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class SimpleTest {

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
                runModel("add", a, b));
    }

    @CodeReflection
    public static Tensor<Float> fconstant() {
        return OnnxOperators.Constant(-1f);
    }

    @Test
    public void testFconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = new Tensor(-1f);
        System.out.println(expected.rtTensor.getTensorTypeAndShape().getDimensionsCount());
        assertEquals(expected, fconstant());
        assertEquals(expected, runModel("fconstant"));
    }

    @CodeReflection
    public static Tensor<Float> fconstants() {
        return OnnxOperators.Constant(new float[]{-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE});
    }

    @Test
    public void testFconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = new Tensor(-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE);
        assertEquals(expected, fconstants());
        assertEquals(expected, runModel("fconstants"));
    }

    @CodeReflection
    public static Tensor<Long> lconstant() {
        return OnnxOperators.Constant(-1l);
    }

    @Test
    public void testLconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = new Tensor(-1l);
        System.out.println(expected.rtTensor.getTensorTypeAndShape().getDimensionsCount());
        assertEquals(expected, lconstant());
        assertEquals(expected, runModel("lconstant"));
    }

    @CodeReflection
    public static Tensor<Long> lconstants() {
        return OnnxOperators.Constant(new long[]{-1, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE});
    }

    @Test
    public void testLconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = new Tensor(-1l, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE);
        assertEquals(expected, lconstants());
        assertEquals(expected, runModel("lconstants"));
    }

    @CodeReflection
    public static Tensor<Long> reshapeAndShape(Tensor<Float> data, Tensor<Long> shape) {
        return OnnxOperators.Shape(OnnxOperators.Reshape(data, shape, Optional.empty()), Optional.empty(), Optional.empty());
    }

    @Test
    public void testReshapeAndShape() throws Exception {
        var data = new Tensor(1f, 2, 3, 4, 5, 6, 7, 8);
        var shape = new Tensor(2l, 2, 2);
        assertEquals(
                reshapeAndShape(data, shape),
                runModel("reshapeAndShape", data, shape));
    }

    private static Tensor runModel(String name, Tensor... params) throws NoSuchMethodException {
        return new Tensor(OnnxRuntime.getInstance().runFunc(
                getOnnxModel(name),
                Stream.of(params).map(t -> t.rtTensor).toList()).getFirst());
    }

    private static CoreOp.FuncOp getOnnxModel(String name) throws NoSuchMethodException {
        return OnnxTransformer.transform(MethodHandles.publicLookup(),
                Op.ofMethod(Stream.of(SimpleTest.class.getDeclaredMethods()).filter(m -> m.getName().equals(name)).findFirst().get()).get());
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
