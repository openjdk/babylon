package oracle.code.onnx;

import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class SimpleTest {

    @CodeReflection
    public static Tensor<Float> add(Tensor<Float> a, Tensor<Float> b) {
        return OnnxOperators.Add(a, b);
    }

    @Test
    public void testAdd() throws Exception {
        var a = Tensor.ofFlat(1f, 2, 3);
        assertEquals(
                add(a, a),
                OnnxRuntime.execute(MethodHandles.lookup(), () -> add(a, a)));
    }

    @CodeReflection
    public static Tensor<Float> sub(Tensor<Float> a, Tensor<Float> b) {
        return OnnxOperators.Sub(a, b);
    }

    @Test
    public void testSub() throws Exception {
        var b = Tensor.ofFlat(6f, 5, 4);
        var a = Tensor.ofFlat(1f, 2, 3);
        assertEquals(
                sub(a, b),
                OnnxRuntime.execute(MethodHandles.lookup(), () -> sub(a, b)));
    }

    @CodeReflection
    public static Tensor<Float> fconstant() {
        return OnnxOperators.Constant(-1f);
    }

    @Test
    public void testFconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofScalar(-1f);
        assertEquals(expected, fconstant());
        assertEquals(expected, OnnxRuntime.execute(MethodHandles.lookup(), () -> fconstant()));
    }

    @CodeReflection
    public static Tensor<Float> fconstants() {
        return OnnxOperators.Constant(new float[]{-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE});
    }

    @Test
    public void testFconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofFlat(-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE);
        assertEquals(expected, fconstants());
        assertEquals(expected, OnnxRuntime.execute(MethodHandles.lookup(), () -> fconstants()));
    }

    @CodeReflection
    public static Tensor<Long> lconstant() {
        return OnnxOperators.Constant(-1l);
    }

    @Test
    public void testLconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofScalar(-1l);
        assertEquals(expected, lconstant());
        assertEquals(expected, OnnxRuntime.execute(MethodHandles.lookup(), () -> lconstant()));
    }

    @CodeReflection
    public static Tensor<Long> lconstants() {
        return OnnxOperators.Constant(new long[]{-1, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE});
    }

    @Test
    public void testLconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofFlat(-1l, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE);
        assertEquals(expected, lconstants());
        assertEquals(expected, OnnxRuntime.execute(MethodHandles.lookup(), () -> lconstants()));
    }

    @CodeReflection
    public static Tensor<Long> reshapeAndShape(Tensor<Float> data, Tensor<Long> shape) {
        return OnnxOperators.Shape(OnnxOperators.Reshape(data, shape, Optional.empty()), Optional.empty(), Optional.empty());
    }

    @Test
    public void testReshapeAndShape() throws Exception {
        var data = Tensor.ofFlat(1f, 2, 3, 4, 5, 6, 7, 8);
        var shape = Tensor.ofFlat(2l, 2, 2);
        assertEquals(
                reshapeAndShape(data, shape),
                OnnxRuntime.execute(MethodHandles.lookup(), () -> reshapeAndShape(data, shape)));
    }

    @CodeReflection
    public static Tensor<Long> indicesOfMaxPool(Tensor<Float> x) {
        // testing secondary output
        return OnnxOperators.MaxPool(x, Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),  new long[]{2}).Indices();
    }

    @Test
    public void testIndicesOfMaxPool() throws Exception {
        var x = Tensor.ofShape(new long[]{2, 2, 2}, 1f, 2, 3, 4, 5, 6, 7, 8);
        assertEquals(
                indicesOfMaxPool(x),
                OnnxRuntime.execute(MethodHandles.lookup(), () -> indicesOfMaxPool(x)));
    }

    @CodeReflection
    public static Tensor<Float> ifConst(Tensor<Boolean> cond) {
        return OnnxOperators.If(cond, () -> OnnxOperators.Constant(-1f), () -> OnnxOperators.Constant(1f));
    }

    @Test
    public void testIfConst() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var expFalse = Tensor.ofScalar(-1f);
        var condTrue = Tensor.ofScalar(true);
        var expTrue = Tensor.ofScalar(1f);

        assertEquals(expFalse, ifConst(condFalse));
        assertEquals(expFalse, OnnxRuntime.execute(MethodHandles.lookup(), () -> ifConst(condFalse)));

        assertEquals(expTrue, ifConst(condTrue));
        assertEquals(expTrue, OnnxRuntime.execute(MethodHandles.lookup(), () -> ifConst(condTrue)));
    }

    @CodeReflection
    public static Tensor<Float> ifCapture(Tensor<Boolean> cond, Tensor<Float> trueValue) {
        var falseValue = OnnxOperators.Constant(-1f);
        return OnnxOperators.If(cond, () -> OnnxOperators.Identity(falseValue), () -> trueValue);
    }

    @Test
    public void testIfCapture() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var expFalse = Tensor.ofScalar(-1f);
        var condTrue = Tensor.ofScalar(true);
        var expTrue = Tensor.ofScalar(1f);

        assertEquals(expFalse, ifCapture(condFalse, expTrue));
        assertEquals(expFalse, OnnxRuntime.execute(MethodHandles.lookup(), () -> ifCapture(condFalse, expTrue)));

        assertEquals(expTrue, ifCapture(condTrue, expTrue));
        assertEquals(expTrue, OnnxRuntime.execute(MethodHandles.lookup(), () -> ifCapture(condTrue, expTrue)));
    }

    static void assertEquals(Tensor expected, Tensor actual) {

        var expectedType = expected.elementType();
        Assertions.assertSame(expectedType, actual.elementType());

        Assertions.assertArrayEquals(expected.shape(), actual.shape());

        switch (expectedType) {
            case UINT8, INT8, BOOL, UINT4, INT4 ->
                Assertions.assertArrayEquals(expected.data().toArray(ValueLayout.JAVA_BYTE),
                                             actual.data().toArray(ValueLayout.JAVA_BYTE));
            case UINT16, INT16 ->
                Assertions.assertArrayEquals(expected.data().toArray(ValueLayout.JAVA_SHORT),
                                             actual.data().toArray(ValueLayout.JAVA_SHORT));
            case INT32, UINT32 ->
                Assertions.assertArrayEquals(expected.data().toArray(ValueLayout.JAVA_INT),
                                             actual.data().toArray(ValueLayout.JAVA_INT));
            case INT64, UINT64 ->
                Assertions.assertArrayEquals(expected.data().toArray(ValueLayout.JAVA_LONG),
                                             actual.data().toArray(ValueLayout.JAVA_LONG));
            case STRING ->
                Assertions.assertEquals(expected.data().getString(0), actual.data().getString(0));
            case FLOAT ->
                Assertions.assertArrayEquals(expected.data().toArray(ValueLayout.JAVA_FLOAT),
                                             actual.data().toArray(ValueLayout.JAVA_FLOAT));
            case DOUBLE ->
                Assertions.assertArrayEquals(expected.data().toArray(ValueLayout.JAVA_DOUBLE),
                                             actual.data().toArray(ValueLayout.JAVA_DOUBLE));
            default ->
                throw new UnsupportedOperationException("Unsupported tensor element type " + expectedType);
        }
    }
}
