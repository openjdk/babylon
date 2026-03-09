package hat.types;

import hat.buffer.F16Array;
import hat.buffer.F32Array;
import optkl.IfaceValue;

// Tensors are immutable
public record Tensor(int first, Shape shape, Class<?> klass) implements IfaceValue {

    public static final int FIRST = 0;
    public static final int SECOND = 1;
    public static final int ACC = 2;

    public static Shape Shape(int dim1, int dim2, int dim3) {
        return new Shape(dim1, dim2, dim3);
    }

    public static Tensor create(int first, Shape shape, Class<?> klass) {
        return new Tensor(first, shape, klass);
    }

    // Do we do a = fill(a, v)? or void fill(a, v)?
    public static void fill(Tensor acc, float v) {
    }

    public static void mma(Tensor result, Tensor tensorA, Tensor tensorB, Tensor acc) {
    }

    public static Tensor load(F16Array matrix, int index, int ld) {
        return null;
    }

    public static void store(F32Array matrix, int index, Tensor resultTensor, int ld) {}

    public record Shape(int x, int y, int z) {}

}
