package hat.optools;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOp;

public class VarFuncDeclarationOpWrapper extends VarOpWrapper {
    final CoreOp.FuncOp funcOp;
    final Block.Parameter blockParameter;
    final int idx;

    public VarFuncDeclarationOpWrapper(CoreOp.VarOp op, CoreOp.FuncOp funcOp, Block.Parameter blockParameter, int idx) {
        super(op);
        this.funcOp = funcOp;
        this.blockParameter = blockParameter;
        this.idx = idx;
    }


}
