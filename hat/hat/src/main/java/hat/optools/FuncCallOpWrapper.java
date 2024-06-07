package hat.optools;

import java.lang.reflect.code.op.CoreOp;

public class FuncCallOpWrapper extends OpWrapper<CoreOp.FuncCallOp> {
    public FuncCallOpWrapper(CoreOp.FuncCallOp op) {
        super(op);
    }


    public String funcName() {
        return op().funcName();
    }
}
