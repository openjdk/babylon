package hat.optools;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.VarType;

public abstract class VarOpWrapper extends OpWrapper<CoreOp.VarOp> {
    public VarOpWrapper(CoreOp.VarOp op) {
        super(op);
    }

    public JavaType javaType() {
        TypeElement typeElement = op().resultType();

        if (typeElement instanceof VarType varType) {
            return (JavaType) varType.valueType();
        }
        return (JavaType) typeElement;
    }

    public String varName() {
        return op().varName();
    }
}
