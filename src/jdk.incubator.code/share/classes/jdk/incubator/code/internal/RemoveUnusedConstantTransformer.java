package jdk.incubator.code.internal;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;

/**
 * A transformer that removes unused {@link jdk.incubator.code.dialect.core.CoreOp.ConstantOp}.
 */
public class RemoveUnusedConstantTransformer implements CodeTransformer {
    private RemoveUnusedConstantTransformer() {}

    private static final RemoveUnusedConstantTransformer instance = new RemoveUnusedConstantTransformer();

    public static RemoveUnusedConstantTransformer getInstance() {
        return instance;
    }

    @Override
    public Block.Builder acceptOp(Block.Builder builder, Op op) {
        if (op instanceof CoreOp.ConstantOp && op.result() != null && op.result().uses().isEmpty()) {
            return builder;
        }
        builder.op(op);
        return builder;
    }
}
