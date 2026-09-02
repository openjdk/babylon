package hat;

import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF16;
import hat.buffer.TensorF32;
import hat.types.F16;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.java.JavaType;

public class DType {
    private DType() {}

    // Float 16 (bits) type
    public static final CodeType F16_TYPE = JavaType.type(F16.class);

    public static final CodeType F32_TYPE = JavaType.type(Float.class);

    // 1D
    public static final CodeType TENSOR_F32_TYPE = JavaType.type(TensorF32.class);
    public static final CodeType TENSOR_F16_TYPE = JavaType.type(TensorF16.class);

    // 2D
    public static final CodeType TENSOR_2D_F32_TYPE = JavaType.type(Tensor2DF32.class);
    public static final CodeType TENSOR_2D_F16_TYPE = JavaType.type(Tensor2DF16.class);

    // DType for primitives
    public static final CodeType Float = JavaType.FLOAT;
    public static final CodeType Double = JavaType.DOUBLE;
    public static final CodeType Boolean = JavaType.BOOLEAN;
    public static final CodeType Byte = JavaType.BYTE;
    public static final CodeType Char = JavaType.CHAR;
    public static final CodeType Short = JavaType.SHORT;
    public static final CodeType Int = JavaType.INT;
    public static final CodeType Long = JavaType.LONG;

}
