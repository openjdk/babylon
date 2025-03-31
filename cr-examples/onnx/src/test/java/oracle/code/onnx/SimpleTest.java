package oracle.code.onnx;

import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static java.util.Optional.empty;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.OnnxRuntime.execute;

public class SimpleTest {

    @CodeReflection
    public Tensor<Float> add(Tensor<Float> a, Tensor<Float> b) {
        return Add(a, b);
    }

    @Test
    public void testAdd() throws Exception {
        var a = Tensor.ofFlat(1f, 2, 3);
        assertEquals(
                add(a, a),
                execute(() -> add(a, a)));
    }

    @CodeReflection
    public Tensor<Float> sub(Tensor<Float> a, Tensor<Float> b) {
        return Sub(a, b);
    }

    @Test
    public void testSub() throws Exception {
        var b = Tensor.ofFlat(6f, 5, 4);
        var a = Tensor.ofFlat(1f, 2, 3);
        assertEquals(
                sub(a, b),
                execute(() -> sub(a, b)));
    }

    @CodeReflection
    public Tensor<Float> fconstant() {
        return Constant(-1f);
    }

    @Test
    public void testFconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofScalar(-1f);
        assertEquals(expected, fconstant());
        assertEquals(expected, execute(() -> fconstant()));
    }

    @CodeReflection
    public Tensor<Float> fconstants() {
        return Constant(new float[]{-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE});
    }

    @Test
    public void testFconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofFlat(-1f, 0, 1, Float.MIN_VALUE, Float.MAX_VALUE);
        assertEquals(expected, fconstants());
        assertEquals(expected, execute(() -> fconstants()));
    }

    @CodeReflection
    public Tensor<Long> lconstant() {
        return Constant(-1l);
    }

    @Test
    public void testLconstant() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofScalar(-1l);
        assertEquals(expected, lconstant());
        assertEquals(expected, execute(() -> lconstant()));
    }

    @CodeReflection
    public Tensor<Long> lconstants() {
        return Constant(new long[]{-1, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE});
    }

    @Test
    public void testLconstants() throws Exception {
        // tests the numbers are encoded correctly
        var expected = Tensor.ofFlat(-1l, 0, 1, Long.MIN_VALUE, Long.MAX_VALUE);
        assertEquals(expected, lconstants());
        assertEquals(expected, execute(() -> lconstants()));
    }

    @CodeReflection
    public Tensor<Long> reshapeAndShape(Tensor<Float> data, Tensor<Long> shape) {
        return Shape(Reshape(data, shape, empty()), empty(), empty());
    }

    @Test
    public void testReshapeAndShape() throws Exception {
        var data = Tensor.ofFlat(1f, 2, 3, 4, 5, 6, 7, 8);
        var shape = Tensor.ofFlat(2l, 2, 2);
        assertEquals(
                reshapeAndShape(data, shape),
                execute(() -> reshapeAndShape(data, shape)));
    }

    @CodeReflection
    public Tensor<Long> indicesOfMaxPool(Tensor<Float> x) {
        // testing secondary output
        return MaxPool(x, empty(), empty(), empty(), empty(), empty(), empty(),  new long[]{2}).Indices();
    }

    @Test
    public void testIndicesOfMaxPool() throws Exception {
        var x = Tensor.ofShape(new long[]{2, 2, 2}, 1f, 2, 3, 4, 5, 6, 7, 8);
        assertEquals(
                indicesOfMaxPool(x),
                execute(() -> indicesOfMaxPool(x)));
    }

    @CodeReflection
    public Tensor<Float> concat(Tensor<Float> input1, Tensor<Float> input2, long axis) {
        return Concat(List.of(input1, input2), axis);
    }

    @Test
    public void testConcat() throws Exception {
        var input1 = Tensor.ofFlat(1f, 2, 3);
        var input2 = Tensor.ofFlat(4f, 5);
        assertEquals(
                concat(input1, input2, 0),
                execute(()-> concat(input1, input2, 0)));
    }

    @CodeReflection
    public Tensor<Float> split(Tensor<Float> input, Tensor<Long> split) {
        return Split(input, Optional.of(split), empty(), empty()).get(0);
    }

    @Test
    public void testSplit() throws Exception {
        var input = Tensor.ofFlat(1f, 2, 3, 4, 5);
        var split = Tensor.ofFlat(5l);
        assertEquals(
                split(input, split),
                execute(()-> split(input, split)));
    }

    @CodeReflection
    public Tensor<Float> ifConst(Tensor<Boolean> cond) {
        return If(cond, () -> List.of(Constant(1f)), () -> List.of(Constant(-1f))).get(0);
    }

    @CodeReflection
    public List<Tensor<Float>> ifConstList(Tensor<Boolean> cond) {
        return If(cond, () -> List.of(Constant(1f)), () -> List.of(Constant(-1f)));
    }

    public record SingleValueTuple<T>(T val) {}

    @CodeReflection
    public SingleValueTuple<Tensor<Float>> ifConstRecord(Tensor<Boolean> cond) {
        return If(cond, () -> new SingleValueTuple(Constant(1f)), () -> new SingleValueTuple(Constant(-1f)));
    }

//    @Test
    public void testIfConst() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var expFalse = Tensor.ofScalar(-1f);
        var condTrue = Tensor.ofScalar(true);
        var expTrue = Tensor.ofScalar(1f);

        assertEquals(expFalse, ifConst(condFalse));
        assertEquals(expFalse, execute(() -> ifConst(condFalse)));

        assertEquals(expTrue, ifConst(condTrue));
        assertEquals(expTrue, execute(() -> ifConst(condTrue)));

        assertEquals(expFalse, execute(() -> ifConstList(condFalse)).get(0));
        assertEquals(expTrue, execute(() -> ifConstList(condTrue)).get(0));

        assertEquals(expFalse, execute(() -> ifConstRecord(condFalse)).val());
        assertEquals(expTrue, execute(() -> ifConstRecord(condTrue)).val());
    }

    @CodeReflection
    public Tensor<Float> ifCapture(Tensor<Boolean> cond, Tensor<Float> trueValue) {
        var falseValue = Constant(-1f);
        return If(cond, () -> Identity(trueValue), () -> Identity(falseValue));
    }

    @Test
    public void testIfCapture() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var expFalse = Tensor.ofScalar(-1f);
        var condTrue = Tensor.ofScalar(true);
        var expTrue = Tensor.ofScalar(1f);

        assertEquals(expFalse, ifCapture(condFalse, expTrue));
        assertEquals(expFalse, execute(() -> ifCapture(condFalse, expTrue)));

        assertEquals(expTrue, ifCapture(condTrue, expTrue));
        assertEquals(expTrue, execute(() -> ifCapture(condTrue, expTrue)));
    }

    final Tensor<Float> initialized = Tensor.ofFlat(42f);

    @CodeReflection
    public Tensor<Float> initialized() {
        return Identity(initialized);
    }

    @Test
    public void testInitialized() throws Exception {

        assertEquals(initialized(),
                     execute(() -> initialized()));
    }

    final Tensor<Float> initialized2 = Tensor.ofFlat(33f);
    final Tensor<Float> initialized3 = Tensor.ofFlat(-1f);
    final Tensor<Float> initialized4 = Tensor.ofFlat(-99f);

    @CodeReflection
    public Tensor<Float> ifInitialized(Tensor<Boolean> cond1, Tensor<Boolean> cond2) {
        return If(cond1,
                () -> If(cond2,
                        () -> List.of(Identity(initialized)),
                        () -> List.of(Identity(initialized2))),
                () -> If(cond2,
                        () -> List.of(Identity(initialized3)),
                        () -> List.of(Identity(initialized4)))).get(0);
    }

    @Test
    public void testIfInitialized() throws Exception {
        var condFalse = Tensor.ofScalar(false);
        var condTrue = Tensor.ofScalar(true);

        assertEquals(initialized, ifInitialized(condTrue, condTrue));
        assertEquals(initialized, execute(() -> ifInitialized(condTrue, condTrue)));
        assertEquals(initialized2, ifInitialized(condTrue, condFalse));
        assertEquals(initialized2, execute(() -> ifInitialized(condTrue, condFalse)));
        assertEquals(initialized3, ifInitialized(condFalse, condTrue));
        assertEquals(initialized3, execute(() -> ifInitialized(condFalse, condTrue)));
        assertEquals(initialized4, ifInitialized(condFalse, condFalse));
        assertEquals(initialized4, execute(() -> ifInitialized(condFalse, condFalse)));

    }

    static final Tensor<Boolean> TRUE = Tensor.ofScalar(true);

    @CodeReflection
    public Tensor<Float> forLoopAdd(Tensor<Long> max, Tensor<Float> initialValue) {
        return Loop(max, TRUE, initialValue, (i, cond, v) -> new LoopReturn<>(cond, Add(v, v)));
    }

    @CodeReflection
    public SingleValueTuple<Tensor<Float>> forLoopAddRecord(Tensor<Long> max, Tensor<Float> initialValue) {
        return Loop(max, TRUE, new SingleValueTuple<>(initialValue), (i, cond, v) -> new LoopReturn<>(cond, new SingleValueTuple<>(Add(v.val(), v.val()))));
    }

    @Test
    public void testForLoopAdd() throws Exception {
        var expected = Tensor.ofFlat(0f, 8, 16, 24);
        var value = Tensor.ofFlat(0f, 1, 2, 3);
        var max = Tensor.ofScalar(3l);
        assertEquals(expected, forLoopAdd(max, value));
        assertEquals(expected, execute(() -> forLoopAdd(max, value)));
//        assertEquals(expected, execute(() -> forLoopAddRecord(max, value)).val());
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
