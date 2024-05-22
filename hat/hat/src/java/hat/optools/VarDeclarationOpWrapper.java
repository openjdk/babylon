package hat.optools;

import java.lang.reflect.code.op.CoreOp;

public class VarDeclarationOpWrapper extends VarOpWrapper implements StoreOpWrapper {
    public VarDeclarationOpWrapper(CoreOp.VarOp op) {
        super(op);
    }
}
