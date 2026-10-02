import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

class OpUtil {
    static boolean isOpSideEffectFree(Op op) {
        return switch (op) {
            case JavaOp.ConvOp _, JavaOp.InstanceOfOp _, JavaOp.ConcatOp _, JavaOp.PatternOps.PatternOp _,
                    CoreOp.ConstantOp _ -> true;
            case JavaOp.ArithmeticOperation aop -> !(aop instanceof JavaOp.DivOp);
            // instance field load is side effect free
            // static field load is not, it may trigger class initialization
            case JavaOp.FieldAccessOp.FieldLoadOp flop -> flop.receiverOperand() != null;
            case JavaOp.InvokeOp invop -> invop.invokeReference().refType().equals(JavaType.type(Math.class));
            default -> false;
        };
    }
}
