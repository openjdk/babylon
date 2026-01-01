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

import hat.callgraph.KernelCallGraph;
import hat.dialect.BinaryOpEnum;
import hat.dialect.HATF16Op;
import hat.types.BF16;
import hat.types.F16;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.Invoke;
import optkl.util.CallSite;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static optkl.Invoke.invokeOpHelper;
import static optkl.Trxfmr.copyLocation;

public record HATFP16Phase(KernelCallGraph kernelCallGraph) implements HATPhase {

    //recursive
    public static boolean findF16IsLocal(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findF16IsLocal(varLoadOp.operands().getFirst());
    }

    //recursive
    public static boolean findF16IsLocal(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findF16IsLocal(varLoadOp); //recurse
            case HATF16Op.HATF16VarOp hatf16VarOp -> true;
            default -> false;
        };
    }

//recursive
    private boolean findReference(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findReference(varLoadOp.operands().getFirst());
    }
//recursive
    private boolean findReference(Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findReference(varLoadOp); // recurse
            case CoreOp.VarOp varOp ->
                    varOp.operands().getFirst() instanceof Op.Result varOpResult
                            && invokeOpHelper(lookup(),varOpResult.op()) instanceof Invoke invoke && invoke.named("array");
            default -> false;
        };
    }
    private ReducedFloatType categorizeReducedFloat(JavaOp.InvokeOp invokeOp) {
        String invokeClassName = invokeOp.invokeDescriptor().refType().toString();
        invokeClassName = invokeClassName.replace("$", ".");
        if (invokeClassName.equals(F16.class.getName())) { // lets not compare strings here
            return ReducedFloatType.HalfFloat.of();
        } else if (invokeClassName.equals(BF16.class.getName())) { // lets not compare strings here
            return ReducedFloatType.BFloat16.of();
        }
        return null;
    }

    private boolean is16BitFloatOperation(JavaOp.InvokeOp invokeOp, String methodName) {
        String invokeClassName = invokeOp.invokeDescriptor().refType().toString();
        invokeClassName = invokeClassName.replace("$", "."); // lets not compare strings here
        boolean is16BitFloatOperation = invokeClassName.startsWith(F16.class.getCanonicalName()) || invokeClassName.startsWith(BF16.class.getCanonicalName());
        return is16BitFloatOperation
                // No need because F16 element is not a Buffer type at the moment
                // && OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                && invokeOp.invokeDescriptor().name().equals(methodName);// lets not compare strings here
    }

    //recursive
    private boolean isOperandF32(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isOperandF32(varLoadOp.operands().getFirst());
    }

    //recursive
    private boolean isOperandF32(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isOperandF32(varLoadOp); //recurse
            case CoreOp.VarOp varOp -> varOp.resultType().valueType() == JavaType.FLOAT;
            default -> false;
        };
    }

    private void createF16VarOp(CoreOp.VarOp varOp, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        var hatf16VarOp = new HATF16Op.HATF16VarOp(varOp.varName(), reducedFloatType, varOp.resultType(), blockBuilder.context().getValues(varOp.operands()));
        blockBuilder.context().mapValue(varOp.result(), blockBuilder.op(copyLocation(varOp,hatf16VarOp)));
    }

    private void createF16ConvOP(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        var convOp = new HATF16Op.HATF16ConvOp(JavaType.VOID, reducedFloatType, blockBuilder.context().getValues(invokeOp.operands()));
        blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp,convOp)));
    }

    private void createFloatFromF16(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        List<Value> operands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        boolean wasFloat = false;
        Value first = operands.getFirst();
        if (first instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            if  (varLoadOp.resultType().equals(JavaType.FLOAT)) {
                wasFloat = true;
            }
        }
        HATF16Op.HATF16ToFloatConvOp convOp1 = new HATF16Op.HATF16ToFloatConvOp(JavaType.FLOAT, reducedFloatType, findF16IsLocal(operands.getFirst()), wasFloat, outputOperands);
        Op.Result result = blockBuilder.op(convOp1);
        convOp1.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), result);
    }

    private void createF16VarLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp, Block.Builder blockBuilder) {
         // what if findNameOrNull is ?
        var hatf16VarLoadOp = new HATF16Op.HATF16VarLoadOp(findNameOrNull(varLoadOp), varLoadOp.varType(), blockBuilder.context().getValues(varLoadOp.operands()));
        blockBuilder.context().mapValue(varLoadOp.result(), blockBuilder.op(copyLocation(varLoadOp,hatf16VarLoadOp)));
    }

    private void createF16BinaryOp(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, BinaryOpEnum binaryOpEnum, ReducedFloatType reducedFloatType) {
        List<Value> operands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);

        // Obtain the memory mapping for each operand
        // if it comes from global memory, HAT replaces with a global* pointer to the inner struct,
        // then, we will need to operate half using a->value, instead of half value directly.
        boolean isFirstOperandReference = findReference(invokeOp.operands().getFirst());
        boolean isSecondOperandReference = findReference(invokeOp.operands().get(1));

        byte valF32Conversion = 0x00;
        if (!isFirstOperandReference && isOperandF32(invokeOp.operands().getFirst())) {
            valF32Conversion = HATF16Op.HATF16BinaryOp.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(invokeOp.operands().get(1))) {
            valF32Conversion = HATF16Op.HATF16BinaryOp.LAST_OP;
        }

        TypeElement typeElement = invokeOp.resultType();
        List<Boolean> refList = List.of(isFirstOperandReference, isSecondOperandReference);

        HATF16Op.HATF16BinaryOp binaryOp = switch (binaryOpEnum) {
            case ADD -> new HATF16Op.HATF16BinaryOp.HATF16AddOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
            case SUB -> new HATF16Op.HATF16BinaryOp.HATF16SubOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
            case MUL -> new HATF16Op.HATF16BinaryOp.HATF16MulOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
            case DIV -> new HATF16Op.HATF16BinaryOp.HATF16DivOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
        };

        Op.Result result = blockBuilder.op(binaryOp);
        binaryOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), result);
    }

    private CoreOp.FuncOp dialectifyF16Ops(CoreOp.FuncOp funcOp, BinaryOpEnum binaryOpEnum) {
        var here = CallSite.of(this.getClass(), "dialectifyF16Ops");
        before(here, funcOp);
        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();
        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke
                         && is16BitFloatOperation(invoke.op(), binaryOpEnum.name().toLowerCase()) && !invoke.returnsVoid()) {
                            consumer.accept(invoke.op());
                            ReducedFloatType category = categorizeReducedFloat(invoke.op());
                            reducedFloatsType.put(invoke.op(), category);
                            // Looks like a find first to me
                            for (Op.Result result : invoke.op().result().uses()) {
                                if (result.op() instanceof CoreOp.VarOp varOp) {
                                    consumer.accept(varOp);
                                    reducedFloatsType.put(varOp, category);
                                    // The variable is created only once for a usage in the same scope
                                    break;
                                }
                            }

                    }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16BinaryOp(invokeOp, blockBuilder, binaryOpEnum, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyF16Stores(CoreOp.FuncOp funcOp) {
        var here = CallSite.of(this.getClass(), "dialectifyF16Stores");
        before(here, funcOp);

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    // This invoke only has one argument: the value to store
                    if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke
                        && is16BitFloatOperation(invoke.op(), "value")
                                && (invoke.returnsShort()||invoke.returnsChar())
                                && invoke.op().operands().getFirst() instanceof Op.Result r
                                && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                                && varLoadOp.operands().getFirst() instanceof Op.Result r1
                                && r1.op() instanceof HATF16Op.HATF16VarOp) {
                                    consumer.accept(invoke.op());
                                    consumer.accept(varLoadOp);
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
        return (is16BitFloatOperation(invokeOp, "of")
                || is16BitFloatOperation(invokeOp, "floatToF16")
                || is16BitFloatOperation(invokeOp, "float2bfloat16"));
    }

    private CoreOp.FuncOp dialectifyF16Init(CoreOp.FuncOp funcOp) {
        var here = CallSite.of(this.getClass(), "dialectifyF16Init");
        before(here, funcOp);
        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();
        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isInitMethodForF16(invokeOp) && invokeOp.resultType() != JavaType.VOID) {
                            invokeOp.result().uses().stream()
                                    .filter(result -> result.op() instanceof CoreOp.VarOp varOp)
                                    .map(result -> (CoreOp.VarOp) result.op() )
                                    .forEach(varOp -> {
                                consumer.accept(varOp);
                                consumer.accept(invokeOp);
                                ReducedFloatType reducedFloatType = categorizeReducedFloat(invokeOp);
                                reducedFloatsType.put(varOp, reducedFloatType);
                                reducedFloatsType.put(invokeOp, reducedFloatType);
                            });
                        }
                    }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16ConvOP(invokeOp, blockBuilder, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyF16ToFloat(CoreOp.FuncOp funcOp) {
        var here = CallSite.of(this.getClass(), "dialectifyF16ToFloat");
        before(here, funcOp);

        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();

        Stream<CodeElement<?, ?>> halfOps = funcOp.elements()
                .mapMulti(((codeElement, consumer) -> {
                    if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke
                        && (invoke.named("f16ToFloat")||invoke.named("bfloat162float")) && invoke.returnsFloat()) {
                            consumer.accept(invoke.op());
                            reducedFloatsType.put(invoke.op(), categorizeReducedFloat(invoke.op()));
                        }
                }));

        Set<CodeElement<?, ?>> nodesInvolved = halfOps.collect(Collectors.toSet());
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createFloatFromF16(invokeOp, blockBuilder, reducedFloatsType.get(invokeOp));
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    // recursive
    private String findNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameOrNull(varLoadOp.operands().getFirst());
    }

    // recursive
    private String findNameOrNull(Value v) {
        return  (v instanceof Op.Result r) ? switch (r.op()){
                 case CoreOp.VarAccessOp.VarLoadOp varLoadOp->findNameOrNull(varLoadOp); //recurse
                 case HATF16Op.HATF16VarOp hatf16VarOp -> hatf16VarOp.varName();
                 default -> null;
            }:null;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        for (BinaryOpEnum binaryOpEnum : BinaryOpEnum.values()) {
            // F16 Operations
            funcOp = dialectifyF16Ops(funcOp, binaryOpEnum);
        }
        // Init analysis before the store
        funcOp = dialectifyF16Init(funcOp);
        funcOp = dialectifyF16ToFloat(funcOp);
        // Store analysis
        funcOp = dialectifyF16Stores(funcOp);
        return funcOp;
    }
}
