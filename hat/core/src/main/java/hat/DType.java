package hat;


import hat.types.F16;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.java.JavaType;

public class DType {
    private DType() {}

    // Float 16 (bits) type
    public static final CodeType F16_TYPE = JavaType.type(F16.class);

    public static final CodeType F32_TYPE = JavaType.type(Float.class);

    // DType for primitive types
    public static final CodeType Float = JavaType.FLOAT;
    public static final CodeType Double = JavaType.DOUBLE;
    public static final CodeType Boolean = JavaType.BOOLEAN;
    public static final CodeType Byte = JavaType.BYTE;
    public static final CodeType Char = JavaType.CHAR;
    public static final CodeType Short = JavaType.SHORT;
    public static final CodeType Int = JavaType.INT;
    public static final CodeType Long = JavaType.LONG;

}
