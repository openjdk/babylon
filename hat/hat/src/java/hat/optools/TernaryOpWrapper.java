package hat.optools;


import java.lang.reflect.code.op.ExtendedOp;
import java.util.stream.Stream;

public class TernaryOpWrapper extends OpWrapper<ExtendedOp.JavaConditionalExpressionOp> {
    public TernaryOpWrapper(ExtendedOp.JavaConditionalExpressionOp op) {
        super(op);
    }
    public Stream<OpWrapper<?>> conditionWrappedYieldOpStream() {
        return wrappedYieldOpStream(firstBlockOfBodyN(0));
    }
    public Stream<OpWrapper<?>> thenWrappedYieldOpStream() {
        return wrappedYieldOpStream(firstBlockOfBodyN(1));
    }

    public Stream<OpWrapper<?>> elseWrappedYieldOpStream() {
        return wrappedYieldOpStream(firstBlockOfBodyN(2));
    }
}
