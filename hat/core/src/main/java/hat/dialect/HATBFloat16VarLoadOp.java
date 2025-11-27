package hat.dialect;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.VarType;

import java.util.List;
import java.util.Map;

public class HATBFloat16VarLoadOp extends HATBFloat16Op {

    private final VarType typeElement;

    public HATBFloat16VarLoadOp(String varName, VarType typeElement, List<Value> operands) {
        super(varName, operands);
        this.typeElement = typeElement;
    }

    public HATBFloat16VarLoadOp(HATBFloat16VarLoadOp op, CodeContext copyContext) {
        super(op, copyContext);
        this.typeElement = op.typeElement;
    }

    @Override
    public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
        return new HATBFloat16VarLoadOp(this, copyContext);
    }

    @Override
    public TypeElement resultType() {
        return typeElement;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.bfloat16.VarOp." + varName(), typeElement);
    }
}
