package hat.types;

import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.buffer.F32ArrayPadded;
import optkl.IfaceValue;

// Tensors are immutable
public record Tensor(int first, Shape shape, Class<?> klass, Access tensorAccess) implements IfaceValue {

    public static final int FIRST = 0;
    public static final int SECOND = 1;
    public static final int ACC = 2;

    public static Shape shape(int dim1, int dim2, int dim3) {
        return new Shape(dim1, dim2, dim3);
    }

    public static Tensor create(int first, Shape shape, Class<?> klass, final Access tensorAccess) {
        return new Tensor(first, shape, klass, tensorAccess);
    }

    public static Tensor create(int first, Shape shape, Class<?> klass) {
        return new Tensor(first, shape, klass, null);
    }

    // Do we do a = fill(a, v)? or void fill(a, v)?
    public static void fill(Tensor acc, float value) {
    }

    public static void mma(Tensor result, Tensor tensorA, Tensor tensorB, Tensor acc) {
    }

    public static Tensor load(F16Array matrix, int i, int j, int ld) {
        return null;
    }

    public static void store(F32Array matrix, int i, int j, Tensor resultTensor, int ld, Access tensorAccess) {
    }

    public static void store(F32ArrayPadded matrix, int i, int j, Tensor resultTensor, int ld, Access tensorAccess) {
    }

    public record Shape(int x, int y, int z) {
    }

    public static class Accessor {
        public static final int ROW_MAJOR = 0;
        public static final int COL_MAJOR = 1;
        public static final int NOT_DEFINED = -1;

        private Accessor() {
        }
    }

    public interface Access {

    }

    public record ColumMajor() implements Access {
    }

    public record RowMajor() implements Access {
    }

    public static ColumMajor ofColumnMajor() {
        return new ColumMajor();
    }

    public static RowMajor ofRowMajor() {
        return new RowMajor();
    }

}
