package oracle.code.onnx;

import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class SimpleTest {

    @CodeReflection
    public Tensor<Float> add(Tensor<Float> a, Tensor<Float> b) {
        return OnnxOperators.Add(a, b);
    }

    @Test
    public void testAdd() throws Exception {
        var a = Tensor.ofFlat(1f, 2, 3);
        assertEquals(
                add(a, a),
                OnnxRuntime.execute(() -> add(a, a)));
    }

    @CodeReflection
    public Tensor<Float> sub(Tensor<Float> a, Tensor<Float> b) {
        return OnnxOperators.Sub(a, b);
    }

    @Test
    public void testSub() throws Exception {
        var b = Tensor.ofFlat(6f, 5, 4);
        var a = Tensor.ofFlat(1f, 2, 3);
        assertEquals(
                sub(a, b),
                OnnxRuntime.execute(() -> sub(a, b)));
    }

    @CodeReflection
    public Tensor<Float> fconstant() {
        return OnnxOperators.Constant(-1f);
    }

    @Test
    public void testFconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofScalar(-1f);
        assertEquals(expected, fconstant());
        assertEquals(expected, OnnxRuntime.execute(() -> fconstant()));
    }

    @CodeReflection
    public Tensor<Float> fconstants() {
        return OnnxOperators.Constant(new float[]{-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE});
    }

    @Test
    public void testFconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofFlat(-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE);
        assertEquals(expected, fconstants());
        assertEquals(expected, OnnxRuntime.execute(() -> fconstants()));
    }

    @CodeReflection
    public Tensor<Long> lconstant() {
        return OnnxOperators.Constant(-1l);
    }

    @Test
    public void testLconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofScalar(-1l);
        assertEquals(expected, lconstant());
        assertEquals(expected, OnnxRuntime.execute(() -> lconstant()));
    }

    @CodeReflection
    public Tensor<Long> lconstants() {
        return OnnxOperators.Constant(new long[]{-1, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE});
    }

    @Test
    public void testLconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofFlat(-1l, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE);
        assertEquals(expected, lconstants());
        assertEquals(expected, OnnxRuntime.execute(() -> lconstants()));
    }

    @CodeReflection
    public Tensor<Long> reshapeAndShape(Tensor<Float> data, Tensor<Long> shape) {
        return OnnxOperators.Shape(OnnxOperators.Reshape(data, shape, Optional.empty()), Optional.empty(), Optional.empty());
    }

    @Test
    public void testReshapeAndShape() throws Exception {
        var data = Tensor.ofFlat(1f, 2, 3, 4, 5, 6, 7, 8);
        var shape = Tensor.ofFlat(2l, 2, 2);
        assertEquals(
                reshapeAndShape(data, shape),
                OnnxRuntime.execute(() -> reshapeAndShape(data, shape)));
    }

    @CodeReflection
    public Tensor<Long> indicesOfMaxPool(Tensor<Float> x) {
        // testing secondary output
        return OnnxOperators.MaxPool(x, Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),  new long[]{2}).Indices();
    }

    @Test
    public void testIndicesOfMaxPool() throws Exception {
        var x = Tensor.ofShape(new long[]{2, 2, 2}, 1f, 2, 3, 4, 5, 6, 7, 8);
        assertEquals(
                indicesOfMaxPool(x),
                OnnxRuntime.execute(() -> indicesOfMaxPool(x)));
    }

    @CodeReflection
    public Tensor<Float> concat(Tensor<Float> input1, Tensor<Float> input2, long axis) {
        return OnnxOperators.Concat(List.of(input1, input2), axis);
    }

    @Test
    public void testConcat() throws Exception {
        var input1 = Tensor.ofFlat(1f, 2, 3);
        var input2 = Tensor.ofFlat(4f, 5);
        assertEquals(
                concat(input1, input2, 0),
                OnnxRuntime.execute(()-> concat(input1, input2, 0)));
    }

    @CodeReflection
    public Tensor<Float> ifConst(Tensor<Boolean> cond) {
        return OnnxOperators.If(cond, () -> OnnxOperators.Constant(-1f), () -> OnnxOperators.Constant(1f));
    }

    @Test
    public void testIfConst() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var expFalse = Tensor.ofScalar(-1f);
        var condTrue = Tensor.ofScalar(true);
        var expTrue = Tensor.ofScalar(1f);

        assertEquals(expFalse, ifConst(condFalse));
        assertEquals(expFalse, OnnxRuntime.execute(() -> ifConst(condFalse)));

        assertEquals(expTrue, ifConst(condTrue));
        assertEquals(expTrue, OnnxRuntime.execute(() -> ifConst(condTrue)));
    }

    @CodeReflection
    public Tensor<Float> ifCapture(Tensor<Boolean> cond, Tensor<Float> trueValue) {
        var falseValue = OnnxOperators.Constant(-1f);
        return OnnxOperators.If(cond, () -> OnnxOperators.Identity(falseValue), () -> OnnxOperators.Identity(trueValue));
    }

    @Test
    public void testIfCapture() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var expFalse = Tensor.ofScalar(-1f);
        var condTrue = Tensor.ofScalar(true);
        var expTrue = Tensor.ofScalar(1f);

        assertEquals(expFalse, ifCapture(condFalse, expTrue));
        assertEquals(expFalse, OnnxRuntime.execute(() -> ifCapture(condFalse, expTrue)));

        assertEquals(expTrue, ifCapture(condTrue, expTrue));
        assertEquals(expTrue, OnnxRuntime.execute(() -> ifCapture(condTrue, expTrue)));
    }

    final Tensor<Float> initialized = Tensor.ofFlat(42f);

    @CodeReflection
    public Tensor<Float> initialized() {
        return OnnxOperators.Identity(initialized);
    }

    @Test
    public void testInitialized() throws Exception {

        assertEquals(initialized(),
                     OnnxRuntime.execute(() -> initialized()));
    }

    final Tensor<Float> initialized2 = Tensor.ofFlat(33f);
    final Tensor<Float> initialized3 = Tensor.ofFlat(-1f);
    final Tensor<Float> initialized4 = Tensor.ofFlat(-99f);

    @CodeReflection
    public Tensor<Float> ifInitialized(Tensor<Boolean> cond1, Tensor<Boolean> cond2) {
        return OnnxOperators.If(cond1,
                () -> OnnxOperators.If(cond2,
                        () -> OnnxOperators.Identity(initialized4),
                        () -> OnnxOperators.Identity(initialized3)),
                () -> OnnxOperators.If(cond2,
                        () -> OnnxOperators.Identity(initialized2),
                        () -> OnnxOperators.Identity(initialized)));
    }

    @Test
    public void testIfInitialized() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var condTrue = Tensor.ofScalar(true);

        assertEquals(initialized, ifInitialized(condTrue, condTrue));
        assertEquals(initialized, OnnxRuntime.execute(() -> ifInitialized(condTrue, condTrue)));
        assertEquals(initialized2, ifInitialized(condTrue, condFalse));
        assertEquals(initialized2, OnnxRuntime.execute(() -> ifInitialized(condTrue, condFalse)));
        assertEquals(initialized3, ifInitialized(condFalse, condTrue));
        assertEquals(initialized3, OnnxRuntime.execute(() -> ifInitialized(condFalse, condTrue)));
        assertEquals(initialized4, ifInitialized(condFalse, condFalse));
        assertEquals(initialized4, OnnxRuntime.execute(() -> ifInitialized(condFalse, condFalse)));

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
