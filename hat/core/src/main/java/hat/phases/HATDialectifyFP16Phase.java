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
import hat.buffer.F16Array;
import hat.dialect.HATF16BinaryOp;
import hat.dialect.HATF16ConvOp;
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
import jdk.incubator.code.extern.OpWriter;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyFP16Phase implements HATDialect {

    private final String[] methodOps = new String[] {"add", "sub", "mul", "div"};

    private final Accelerator accelerator;
    public HATDialectifyFP16Phase(Accelerator accelerator) {
        this.accelerator = accelerator;
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

                // Todo: subclassing to get this
                HATF16BinaryOp.OpType opType = switch (methodName) {
                    case "add" -> HATF16BinaryOp.OpType.ADD;
                    case "sub" -> HATF16BinaryOp.OpType.SUB;
                    case "mul" -> HATF16BinaryOp.OpType.MUL;
                    case "div" -> HATF16BinaryOp.OpType.DIV;
                    default -> throw new IllegalStateException("Unexpected value: " + methodName);
                };

                HATF16BinaryOp binaryOp = new HATF16BinaryOp(invokeOp.resultType(),
                        opType,
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

    private CoreOp.FuncOp dialectifyF16Init(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[BEFORE] dialectifyF16Init Phase: " + funcOp.toText());

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isFP16Operation(invokeOp, "of") && invokeOp.resultType() != JavaType.VOID) {
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
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // Insert a conversion Op
                List<Value> operands = invokeOp.operands();
                List<Value> outputOperands = context.getValues(operands);
                HATF16ConvOp convOp1 = new HATF16ConvOp(JavaType.VOID, outputOperands);
                Op.Result op1 = blockBuilder.op(convOp1);
                convOp1.setLocation(invokeOp.location());
                context.mapValue(invokeOp.result(), op1);
            } else if (op instanceof CoreOp.VarOp varOp) {
                List<Value> operands2 = varOp.operands();
                List<Value> outputOperands2 = context.getValues(operands2);
                HATF16VarOp hatf16VarOp = new HATF16VarOp(varOp.varName(), varOp.resultType(), outputOperands2);
                Op.Result op2 = blockBuilder.op(hatf16VarOp);
                hatf16VarOp.setLocation(varOp.location());
                context.mapValue(varOp.result(), op2);
            }
            return blockBuilder;
        });

        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[AFTER] dialectifyF16Init Phase: " + funcOp.toText());
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
        for (String methodName : methodOps)
            // Operations and loads
            funcOp = dialectifyF16Ops(funcOp, methodName);
        // Init analysis before the store
        funcOp = dialectifyF16Init(funcOp);
        // Store analysis
        funcOp = dialectifyF16Stores(funcOp);
        return funcOp;
    }

    @Override
    public Accelerator accelerator() {
        return accelerator;
    }
}
