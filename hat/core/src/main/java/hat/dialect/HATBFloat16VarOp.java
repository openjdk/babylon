package hat.dialect;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.VarType;

import java.util.List;
import java.util.Map;

public class HATBFloat16VarOp extends HATBFloat16Op {

    private final VarType typeElement;

    public HATBFloat16VarOp(String varName, VarType typeElement, List<Value> operands) {
        super(varName, operands);
        this.typeElement = typeElement;
    }

    public HATBFloat16VarOp(HATBFloat16VarOp op, CodeContext copyContext) {
        super(op, copyContext);
        this.typeElement = op.typeElement;
    }

    @Override
    public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
        return new HATBFloat16VarOp(this, copyContext);
    }

    @Override
    public TypeElement resultType() {
        return typeElement;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.bfloat16,varop." + varName(), typeElement);
    }

}
