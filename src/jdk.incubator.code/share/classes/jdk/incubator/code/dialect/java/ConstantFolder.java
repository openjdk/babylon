package jdk.incubator.code.dialect.java;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;

import java.lang.invoke.MethodHandles;
import java.util.Optional;

import static jdk.incubator.code.dialect.core.CoreOp.constant;

public class ConstantFolder implements CodeTransformer {
    private final JavaOp.JavaExpression.Evaluator evaluator;

    private ConstantFolder(JavaOp.JavaExpression.Evaluator evaluator) {
        this.evaluator = evaluator;
    }

    public static ConstantFolder getInstance(MethodHandles.Lookup l) {
        return new ConstantFolder(new JavaOp.JavaExpression.Evaluator(l));
    }

    @Override
    public Block.Builder acceptOp(Block.Builder b, Op op) {
        Optional<Object> v = evaluator.evaluate(op.result());
        if (v.isPresent()) {
            Op.Result c = b.op(constant(op.resultType(), v.get()));
            b.context().mapValue(op.result(), c);
        } else {
            b.op(op);
        }
        return b;
    }
}
