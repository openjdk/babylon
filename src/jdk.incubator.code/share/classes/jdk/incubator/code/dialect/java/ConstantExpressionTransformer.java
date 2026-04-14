package jdk.incubator.code.dialect.java;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;

import java.lang.invoke.MethodHandles;
import java.util.Optional;

import static jdk.incubator.code.dialect.core.CoreOp.constant;
import static jdk.incubator.code.dialect.java.JavaOp.JavaExpression.ConstantExpresssionEvaluator;

/**
 * A transformer that replace every operation that model a constant expression with its value.
 */
public class ConstantExpressionTransformer implements CodeTransformer {
    private final ConstantExpresssionEvaluator constantExpresssionEvaluator;

    private ConstantExpressionTransformer(ConstantExpresssionEvaluator constantExpresssionEvaluator) {
        this.constantExpresssionEvaluator = constantExpresssionEvaluator;
    }

    @Override
    public Block.Builder acceptOp(Block.Builder b, Op op) {
        Optional<Object> v = constantExpresssionEvaluator.evaluate(op.result());
        if (v.isPresent()) {
            Op.Result c = b.op(constant(op.resultType(), v.get()));
            b.context().mapValue(op.result(), c);
        } else {
            b.op(op);
        }
        return b;
    }

    public static Op transform(MethodHandles.Lookup l, Op op) {
        return op.transform(CodeContext.create(), new ConstantExpressionTransformer(new ConstantExpresssionEvaluator(l)));
    }
}
