/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package hat.phases;

import hat.Accelerator;
import hat.buffer.F16;
import hat.dialect.HATF16AddOp;
import hat.dialect.HATF16BinaryOp;
import hat.dialect.HATF16ConvOp;
import hat.dialect.HATF16DivOp;
import hat.dialect.HATF16MulOp;
import hat.dialect.HATF16SubOp;
import hat.dialect.HATF16ToFloatConvOp;
import hat.dialect.HATF16VarLoadOp;
import hat.dialect.HATF16VarOp;
import hat.dialect.HATPhaseUtils;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static hat.dialect.HATPhaseUtils.*;

public class HATDialectifyFP16Phase implements HATDialect {

    public enum OpMethod {
        ADD("add"),
        SUB("sub"),
        MUL("mul"),
        DIV("div");

        final String methodName;
        OpMethod(String name) {
            this.methodName = name;
        }

        public String methodName() {
            return this.methodName;
        }
    }

    private final Accelerator accelerator;
    public HATDialectifyFP16Phase(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    private boolean isFP16Operation(JavaOp.InvokeOp invokeOp, String methodName) {
        String invokeClassName = invokeOp.invokeDescriptor().refType().toString();
        boolean isFP16Operation = invokeClassName.replace("$", ".").startsWith(F16.class.getCanonicalName());
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

    private void createF16VarOp(CoreOp.VarOp varOp, Block.Builder blockBuilder) {
        List<Value> operands = varOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        HATF16VarOp hatf16VarOp = new HATF16VarOp(varOp.varName(), varOp.resultType(), outputOperands);
        Op.Result op1 = blockBuilder.op(hatf16VarOp);
        hatf16VarOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(varOp.result(), op1);
    }

    private void createF16ConvOP(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder) {
        List<Value> operands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        HATF16ConvOp convOp1 = new HATF16ConvOp(JavaType.VOID, outputOperands);
        Op.Result op1 = blockBuilder.op(convOp1);
        convOp1.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), op1);
    }

    private void createFloatFromF16(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder) {
        List<Value> operands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        boolean isLocal = findF16IsLocal(operands.getFirst());
        HATF16ToFloatConvOp convOp1 = new HATF16ToFloatConvOp(JavaType.FLOAT, isLocal, outputOperands);
        Op.Result op1 = blockBuilder.op(convOp1);
        convOp1.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), op1);
    }

    private void createF16VarLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp, Block.Builder blockBuilder) {
        List<Value> operands = varLoadOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        String nameVar = findName(varLoadOp);
        HATF16VarLoadOp hatf16VarLoadOp = new HATF16VarLoadOp(nameVar, varLoadOp.varType(), outputOperands);
        Op.Result op1 = blockBuilder.op(hatf16VarLoadOp);
        hatf16VarLoadOp.setLocation(varLoadOp.location());
        blockBuilder.context().mapValue(varLoadOp.result(), op1);
    }

    private void createF16BinaryOp(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, OpMethod method) {
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

        HATF16BinaryOp binaryOp = switch (method) {
            case ADD -> new HATF16AddOp(typeElement, refList, valF32Conversion, outputOperands);
            case SUB -> new HATF16SubOp(typeElement, refList, valF32Conversion, outputOperands);
            case MUL -> new HATF16MulOp(typeElement, refList, valF32Conversion, outputOperands);
            case DIV -> new HATF16DivOp(typeElement, refList, valF32Conversion, outputOperands);
        };

        Op.Result op1 = blockBuilder.op(binaryOp);
        binaryOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), op1);
    }

    private CoreOp.FuncOp dialectifyF16Ops(CoreOp.FuncOp funcOp, OpMethod method) {
        var here = OpTk.CallSite.of(this.getClass(), "dialectifyF16Ops" );
        before(here,funcOp);

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isFP16Operation(invokeOp, method.methodName) && invokeOp.resultType() != JavaType.VOID) {
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
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16BinaryOp(invokeOp, blockBuilder, method);
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder);
            }
            return blockBuilder;
        });
        after(here,funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyF16Stores(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(this.getClass(), "dialectifyF16Stores");
        before(here,funcOp);

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isFP16Operation(invokeOp, "value") && invokeOp.resultType() == JavaType.SHORT) {
                            // This invoke only has one argument: the value to store
                            Value value = invokeOp.operands().getFirst();
                            if (value instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                Value valLoad = varLoadOp.operands().getFirst();
                                if (valLoad instanceof Op.Result r1 && r1.op() instanceof HATF16VarOp) {
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
                blockBuilder.context().mapValue(
                        invokeOp.result(), //
                        blockBuilder.context().getValue(invokeOp.operands().getFirst()) //
                );
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                createF16VarLoadOp(varLoadOp, blockBuilder);
            }
            return blockBuilder;
        });

        after(here, funcOp);
        return funcOp;
    }

    private boolean isInitMethodForF16(JavaOp.InvokeOp invokeOp) {
        return (isFP16Operation(invokeOp, "of")
                || isFP16Operation(invokeOp, "floatToF16"));
    }

    private CoreOp.FuncOp dialectifyF16Init(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(this.getClass(), "dialectifyF16Init");
        before(here,funcOp);

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isInitMethodForF16(invokeOp) && invokeOp.resultType() != JavaType.VOID) {
                            Set<Op.Result> uses = invokeOp.result().uses();
                            for (Op.Result result : uses) {
                                if (result.op() instanceof CoreOp.VarOp varOp) {
                                    consumer.accept(varOp);
                                    consumer.accept(invokeOp);
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
                createF16ConvOP(invokeOp, blockBuilder);
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder);
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyF16ToFloat(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(this.getClass(), "dialectifyF16ToFloat");
        before(here,funcOp);
        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isMethod(invokeOp, "f16ToFloat")
                                && invokeOp.resultType() == JavaType.FLOAT) {
                            consumer.accept(invokeOp);
                        }
                    }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createFloatFromF16(invokeOp, blockBuilder);
            }
            return blockBuilder;
        });
        after(here, funcOp);
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
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        for (OpMethod method : OpMethod.values())
            // F16 Operations
            funcOp = dialectifyF16Ops(funcOp, method);

        // Init analysis before the store
        funcOp = dialectifyF16Init(funcOp);
        funcOp = dialectifyF16ToFloat(funcOp);
        // Store analysis
        funcOp = dialectifyF16Stores(funcOp);
        return funcOp;
    }

    @Override
    public Accelerator accelerator() {
        return accelerator;
    }
}
