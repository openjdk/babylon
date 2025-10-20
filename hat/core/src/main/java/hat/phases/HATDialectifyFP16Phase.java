package hat.phases;

import hat.Accelerator;
import hat.ComputeRange;
import hat.buffer.F16Array;
import hat.dialect.HATF16BinaryOp;
import hat.dialect.HATF16VarLoadOp;
import hat.dialect.HATF16VarOp;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyFP16Phase extends HATDialectAbstractPhase implements HATDialectifyPhase {

    public HATDialectifyFP16Phase(Accelerator accelerator) {
        super(accelerator);
    }

    private boolean isFP16Operation(JavaOp.InvokeOp invokeOp, String methodName) {
        String invokeClassName = invokeOp.invokeDescriptor().refType().toString();
        boolean isFP16Operation = invokeClassName.replace("$", ".").startsWith(F16Array.F16.class.getCanonicalName());
        return isFP16Operation
                && OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                && invokeOp.invokeDescriptor().name().equals(methodName);
    }

    private boolean findReference(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findReference(varLoadOp.operands().get(0));
    }

    private boolean findReference(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findReference(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && r.op() instanceof CoreOp.VarOp varOp) {
                Value first = varOp.operands().getFirst();
                return first instanceof Op.Result r2 && r2.op() instanceof JavaOp.InvokeOp invokeOp && invokeOp.invokeDescriptor().name().equals("array");
            }
            return false;
        }
    }

    private CoreOp.FuncOp dialectifyF16Ops(CoreOp.FuncOp funcOp, String methodName) {
        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[BEFORE] FP16 Phase: " + funcOp.toText());

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isFP16Operation(invokeOp, methodName) && invokeOp.resultType() != JavaType.VOID) {
                            Set<Op.Result> uses = invokeOp.result().uses();
                            consumer.accept(invokeOp);
                            for (Op.Result result : uses) {
                                if (result.op() instanceof CoreOp.VarOp varOp) {
                                    consumer.accept(varOp);
                                    break;
                                }
                            }
                        }
                    }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());

        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> operands = invokeOp.operands();
                List<Value> outputOperands = context.getValues(operands);
                // Obtain the memory mapping for each operand
                // if it comes from global memory, HAT replaces with a global* pointer to the inner struct,
                // then, we will need to operate half using a->value, instead of ha directly.
                boolean isFirstOperandReference = findReference(invokeOp.operands().getFirst());
                boolean isSecondOperandReference = findReference(invokeOp.operands().get(1));
                HATF16BinaryOp binaryOp = new HATF16BinaryOp(invokeOp.resultType(),
                        HATF16BinaryOp.OpType.ADD,
                        List.of(isFirstOperandReference, isSecondOperandReference),
                        outputOperands);
                Op.Result op1 = blockBuilder.op(binaryOp);
                binaryOp.setLocation(invokeOp.location());
                context.mapValue(invokeOp.result(), op1);
            } else if (op instanceof CoreOp.VarOp varOp) {
                List<Value> operands = varOp.operands();
                List<Value> outputOperands = context.getValues(operands);
                HATF16VarOp hatf16VarOp = new HATF16VarOp(varOp.varName(), varOp.resultType(), outputOperands);
                Op.Result op1 = blockBuilder.op(hatf16VarOp);
                hatf16VarOp.setLocation(varOp.location());
                context.mapValue(varOp.result(), op1);
            }
            return blockBuilder;
        });

        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[AFTER] FP16 Phase: " + funcOp.toText());
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyF16Stores(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[BEFORE] dialectifyF16Stores Phase: " + funcOp.toText());

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isFP16Operation(invokeOp, "value") && invokeOp.resultType() == JavaType.SHORT) {
                            Value value = invokeOp.operands().getFirst();
                            if (value instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                Value second = varLoadOp.operands().getFirst();
                                if (second instanceof Op.Result r1 && r1.op() instanceof HATF16VarOp hatf16VarOp) {
                                    consumer.accept(invokeOp);
                                    consumer.accept(varLoadOp);
                                }
                            }
                        }
                    }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());

        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                context.mapValue(invokeOp.result(), context.getValue(invokeOp.operands().getFirst()));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                List<Value> operands = varLoadOp.operands();
                List<Value> outputOperands = context.getValues(operands);
                String nameVar = findName(varLoadOp);
                HATF16VarLoadOp hatf16VarLoadOp = new HATF16VarLoadOp(nameVar, varLoadOp.varType(), outputOperands);
                Op.Result op1 = blockBuilder.op(hatf16VarLoadOp);
                hatf16VarLoadOp.setLocation(varLoadOp.location());
                context.mapValue(varLoadOp.result(), op1);
            }
            return blockBuilder;
        });

        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[AFTER] dialectifyF16Stores Phase: " + funcOp.toText());
        return funcOp;
    }

    private String findName(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findName(varLoadOp.operands().get(0));
    }

    private String findName(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findName(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && r.op() instanceof HATF16VarOp hatf16VarOp) {
                return hatf16VarOp.varName();
            }
            return null;
        }
    }

    @Override
    public CoreOp.FuncOp run(CoreOp.FuncOp funcOp) {
        funcOp = dialectifyF16Ops(funcOp, "add");
        funcOp = dialectifyF16Ops(funcOp, "sub");
        funcOp = dialectifyF16Ops(funcOp, "mul");
        funcOp = dialectifyF16Ops(funcOp, "div");
        funcOp = dialectifyF16Stores(funcOp);
        return funcOp;
    }


}
