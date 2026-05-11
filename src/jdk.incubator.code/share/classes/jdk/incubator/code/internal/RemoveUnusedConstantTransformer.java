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

    public static final RemoveUnusedConstantTransformer INSTANCE = new RemoveUnusedConstantTransformer();

    @Override
    public Block.Builder acceptOp(Block.Builder builder, Op op) {
        if (op instanceof CoreOp.ConstantOp && op.result() != null && op.result().uses().isEmpty()) {
            return builder;
        }
        builder.op(op);
        return builder;
    }
}
