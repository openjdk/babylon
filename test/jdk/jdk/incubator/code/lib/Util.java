import jdk.incubator.code.Op;
import jdk.incubator.code.behavior.JavaLowInterpreter;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;

public class Util {
    static <T extends Op & Op.Invokable> Object interpretOp(MethodHandles.Lookup l, T op, Object... args) {
        return interpretOp(l, op, Arrays.asList(args));
    }

    static <T extends Op & Op.Invokable> Object interpretOp(MethodHandles.Lookup l, T op, List<Object> args) {
        return switch (op) {
            // exec -> execution exception | interpreter exception
            case CoreOp.FuncOp fop -> {
                try {
                    yield new JavaLowInterpreter().executeFuncOp(fop, args, l);
                } catch (Throwable t) {
                    eraseAndThrow(t);
                    throw new InternalError(); // @@@ shouldn't reach here
                }
            }
            case JavaOp.LambdaOp lop -> {
                try {
                    yield new JavaLowInterpreter().executeLambdaOp(lop, args, l);
                } catch (Throwable e) {
                    eraseAndThrow(e); // @@@ we can do this trick in JavaLowInterpreter to avoid handling
                    throw new InternalError();
                }
            }
            case null, default -> throw new IllegalArgumentException("Can't interpret " + op);
        };
    }

    @SuppressWarnings("unchecked")
    static <E extends Throwable> void eraseAndThrow(Throwable e) throws E {
        throw (E) e;
    }
}
