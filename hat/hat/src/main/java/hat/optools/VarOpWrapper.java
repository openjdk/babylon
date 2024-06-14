package hat.optools;

import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;

public abstract class VarOpWrapper extends OpWrapper<CoreOp.VarOp> {
    public VarOpWrapper(CoreOp.VarOp op) {
        super(op);
    }

    public JavaType javaType() {
        return (JavaType) op().varType();
    }

    public String varName() {
        return op().varName();
    }
}
