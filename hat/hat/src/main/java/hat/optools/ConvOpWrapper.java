package hat.optools;

import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;

public class ConvOpWrapper extends UnaryOpWrapper<CoreOp.ConvOp> {
    public ConvOpWrapper(CoreOp.ConvOp op) {
        super(op);
    }

    public JavaType resultJavaType() {
        return (JavaType) op().resultType();
    }
}
