package hat.phases;

import hat.Accelerator;
import hat.buffer.BF16;
import hat.dialect.*;
import hat.optools.OpTk;
import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyBFloat16Phase implements HATDialect {

    private final Accelerator accelerator;

    private enum BinaryOpMethod {
        ADD("add"),
        SUB("sub"),
        MUL("mul"),
        DIV("div");

        final String methodName;
        BinaryOpMethod(String name) {
            this.methodName = name;
        }
    }

    public HATDialectifyBFloat16Phase(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    @Override
    public Accelerator accelerator() {
        return accelerator;
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

    private boolean isOperandF32(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isOperandF32(varLoadOp.operands().get(0));
    }

    private boolean isOperandF32(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return isOperandF32(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && r.op() instanceof CoreOp.VarOp varOp) {
                VarType varType = varOp.resultType();
                TypeElement typeElement = varType.valueType();
                return typeElement == JavaType.FLOAT;
            }
            return false;
        }
    }

    private String findName(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findName(varLoadOp.operands().get(0));
    }

    private String findName(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findName(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && r.op() instanceof HATBFloat16VarOp hatbFloat16VarOp) {
                return hatbFloat16VarOp.varName();
            }
            return null;
        }
    }

    private boolean isBFloat16Operation(JavaOp.InvokeOp invokeOp, String methodName) {
        String invokeClassName = invokeOp.invokeDescriptor().refType().toString();
        boolean isBFloatOperation = invokeClassName.replace("$", ".").startsWith(BF16.class.getCanonicalName());
        return isBFloatOperation && invokeOp.invokeDescriptor().name().equals(methodName);
    }

    private void createBFloat16VarOp(CoreOp.VarOp varOp, Block.Builder blockBuilder) {
        List<Value> operands = varOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        HATBFloat16VarOp bfloat16VarOp = new HATBFloat16VarOp(varOp.varName(), varOp.resultType(), outputOperands);
        Op.Result opResult = blockBuilder.op(bfloat16VarOp);
        bfloat16VarOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(varOp.result(), opResult);
    }

    private void createBFloat16BinaryOp(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, BinaryOpMethod method) {
        List<Value> operands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);

        // Obtain the memory mapping for each operand
        // if it comes from global memory, HAT replaces with a global* pointer to the inner struct,
        // then, we will need to operate half using a->value, instead of half value directly.
        boolean isFirstOperandReference = findReference(invokeOp.operands().getFirst());
        boolean isSecondOperandReference = findReference(invokeOp.operands().get(1));

        byte valF32Conversion = 0x00;
        if (!isFirstOperandReference && isOperandF32(invokeOp.operands().getFirst())) {
            valF32Conversion = HATF16BinaryOp.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(invokeOp.operands().get(1))) {
            valF32Conversion = HATF16BinaryOp.LAST_OP;
        }

        TypeElement typeElement = invokeOp.resultType();
        List<Boolean> refList = List.of(isFirstOperandReference, isSecondOperandReference);

        HATBFLOATBinaryOp binaryOp = switch (method) {
            case ADD -> new HATBFloat16AddOp(typeElement, refList, valF32Conversion, outputOperands);
            default -> throw new IllegalStateException("Unexpected value: " + method);
        };

        Op.Result opResult = blockBuilder.op(binaryOp);
        binaryOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), opResult);
    }

    private CoreOp.FuncOp dialectifyBFloat16Operations(CoreOp.FuncOp funcOp, BinaryOpMethod operationMethod) {
        var here = OpTk.CallSite.of(this.getClass(), "dialectifyBFloat16Operations" );
        before(here,funcOp);

        IO.println("BEFORE: " + funcOp.toText());

        Stream<CodeElement<?, ?>> bfloats = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isBFloat16Operation(invokeOp, operationMethod.methodName) && invokeOp.resultType() != JavaType.VOID) {
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

        Set<CodeElement<?, ?>> nodesInvolved = bfloats.collect(Collectors.toSet());
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createBFloat16BinaryOp(invokeOp, blockBuilder, operationMethod);
            } else if (op instanceof CoreOp.VarOp varOp) {
                createBFloat16VarOp(varOp, blockBuilder);
            }
            return blockBuilder;
        });
        IO.println("AFTER: " + funcOp.toText());
        after(here,funcOp);
        return funcOp;
    }

    private void createHATBFloat16VarLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp, Block.Builder blockBuilder) {
        List<Value> operands = varLoadOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        String nameVar = findName(varLoadOp);
        HATBFloat16VarLoadOp hatbFloat16VarLoadOp = new HATBFloat16VarLoadOp(nameVar, varLoadOp.varType(), outputOperands);
        Op.Result opResult = blockBuilder.op(hatbFloat16VarLoadOp);
        hatbFloat16VarLoadOp.setLocation(varLoadOp.location());
        blockBuilder.context().mapValue(varLoadOp.result(), opResult);
    }

    private CoreOp.FuncOp dialectifyBFloatStores(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(this.getClass(), "dialectifyBFloatStores");
        before(here,funcOp);

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isBFloat16Operation(invokeOp, "value") && invokeOp.resultType() == JavaType.CHAR) {
                            // This invoke only has one argument: the value to store
                            Value value = invokeOp.operands().getFirst();
                            if (value instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                Value valLoad = varLoadOp.operands().getFirst();
                                if (valLoad instanceof Op.Result r1 && r1.op() instanceof HATBFloat16VarOp) {
                                    consumer.accept(invokeOp);
                                    consumer.accept(varLoadOp);
                                }
                            }
                        }
                    }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());

        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // pass the value
                blockBuilder.context().mapValue(
                        invokeOp.result(), //
                        blockBuilder.context().getValue(invokeOp.operands().getFirst()) //
                );
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                createHATBFloat16VarLoadOp(varLoadOp, blockBuilder);
            }
            return blockBuilder;
        });

        IO.println("AFTER STORES: " + funcOp.toText());
        after(here, funcOp);
        return funcOp;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        for (BinaryOpMethod method : BinaryOpMethod.values())
            funcOp = dialectifyBFloat16Operations(funcOp, method);

        funcOp = dialectifyBFloatStores(funcOp);
        return funcOp;
    }
}
