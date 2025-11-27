package hat.dialect;

import jdk.incubator.code.*;

import java.util.List;

public class HATBFloat16AddOp extends HATBFLOATBinaryOp {

    public HATBFloat16AddOp(TypeElement typeElement, List<Boolean> references, byte f32, List<Value> operands) {
        super(typeElement, BinaryOpType.ADD, references, f32, operands);
    }

    public HATBFloat16AddOp(HATBFloat16AddOp op, CodeContext copyContext) {
        super(op, copyContext);
    }

    @Override
    public Op transform(CodeContext codeContext, CodeTransformer codeTransformer) {
        return new HATBFloat16AddOp(this, codeContext);
    }
}
