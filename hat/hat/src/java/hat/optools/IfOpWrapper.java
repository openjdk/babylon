package hat.optools;

import java.lang.reflect.code.op.ExtendedOp;
import java.util.stream.Stream;

public class IfOpWrapper extends StructuralOpWrapper<ExtendedOp.JavaIfOp> {
    public IfOpWrapper(ExtendedOp.JavaIfOp op) {
        super(op);
    }

    public boolean hasElseN(int idx) {
        return hasBodyN(idx) && firstBlockOfBodyN(idx).ops().size()>1;
    }

    public Stream<OpWrapper<?>> conditionWrappedYieldOpStream() {
        return wrappedYieldOpStream(bodyN(0).entryBlock());
    }

    public Stream<OpWrapper<?>> thenWrappedRootOpStream() {
       return  wrappedRootOpStream(bodyN(1).entryBlock());
    }
    public Stream<OpWrapper<?>> elseWrappedRootOpStream() {
        return  wrappedRootOpStream(bodyN(2).entryBlock());
    }
}
