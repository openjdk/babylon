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
import optkl.Trxfmr;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
    private static boolean findReference(MethodHandles.Lookup lookup,CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findReference(lookup,varLoadOp.operands().getFirst());
    }
//recursive
    private static boolean findReference(MethodHandles.Lookup lookup,Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findReference(lookup,varLoadOp); // recurse
            case CoreOp.VarOp varOp ->
                    varOp.operands().getFirst() instanceof Op.Result varOpResult
                            && invokeOpHelper(lookup,varOpResult.op()) instanceof Invoke invoke && invoke.named("array");
            default -> false;
        };
    }

    private static boolean isOperandF32(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp varLoadOp -> varLoadOp.varType().valueType() ==JavaType.FLOAT; //recurse
            case CoreOp.VarOp varOp -> varOp.resultType().valueType() == JavaType.FLOAT;
            default -> false;
        };
    }
    // recursive
    private static String findNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameOrNull(varLoadOp.operands().getFirst());
    }

    // recursive
    private static String findNameOrNull(Value v) {
        return  (v instanceof Op.Result r) ? switch (r.op()){
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp->findNameOrNull(varLoadOp); //recurse
            case HATF16Op.HATF16VarOp hatf16VarOp -> hatf16VarOp.varName();
            default -> null;
        }:null;
    }
    private static ReducedFloatType categorizeReducedFloat(JavaOp.InvokeOp invokeOp) {
        String invokeClassName = invokeOp.invokeDescriptor().refType().toString();
        invokeClassName = invokeClassName.replace("$", ".");
        if (invokeClassName.equals(F16.class.getName())) { // lets not compare strings here
            return ReducedFloatType.HalfFloat.of();
        } else if (invokeClassName.equals(BF16.class.getName())) { // lets not compare strings here
            return ReducedFloatType.BFloat16.of();
        }
        return null;
    }

    private boolean is16BitFloat(Invoke invoke, Regex methodName) {
        String invokeClassName = invoke.refType().toString();
        invokeClassName = invokeClassName.replace("$", "."); // lets not compare strings here
        boolean is16BitFloatOperation = invokeClassName.startsWith(F16.class.getCanonicalName()) || invokeClassName.startsWith(BF16.class.getCanonicalName());
        return is16BitFloatOperation
                // No need because F16 element is not a Buffer type at the moment
                // && OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                && invoke.named(methodName);// lets not compare strings here
    }

    private void createF16VarOp(CoreOp.VarOp varOp, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
           blockBuilder.context().mapValue(varOp.result(),
                blockBuilder.op(copyLocation(varOp,
                       new HATF16Op.HATF16VarOp(
                               varOp.varName(),
                               reducedFloatType, varOp.resultType(),
                               blockBuilder.context().getValues(varOp.operands()))
                        )
                )
           );
    }

    private void createF16ConvOP(Invoke invoke, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        blockBuilder.context().mapValue(invoke.op().result(),
                blockBuilder.op(copyLocation(invoke.op(),
                        new HATF16Op.HATF16ConvOp(
                                JavaType.VOID,
                                reducedFloatType,
                                blockBuilder.context().getValues(invoke.op().operands()))
                        )
                )
        );
    }
    private void createF16VarLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp, Block.Builder blockBuilder) {
        blockBuilder.context().mapValue(varLoadOp.result(),
                blockBuilder.op(copyLocation(varLoadOp,
                                new HATF16Op.HATF16VarLoadOp(
                                        findNameOrNull(varLoadOp),
                                        varLoadOp.varType(),
                                        blockBuilder.context().getValues(varLoadOp.operands()))
                        )
                )
        );
    }
    private void createFloatFromF16(Invoke invoke, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        boolean wasFloat = invoke.op().operands().getFirst() instanceof Op.Result r
                && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                && varLoadOp.resultType().equals(JavaType.FLOAT);

        blockBuilder.context().mapValue(invoke.op().result(),
                blockBuilder.op(copyLocation(invoke.op(),
                        new HATF16Op.HATF16ToFloatConvOp(
                                JavaType.FLOAT,
                                reducedFloatType,
                                findF16IsLocal(invoke.op().operands().getFirst()),
                                wasFloat,
                                blockBuilder.context().getValues(invoke.op().operands()))
                        )
                )
        );
    }

    private void createF16BinaryOp(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, BinaryOpEnum binaryOpEnum, ReducedFloatType reducedFloatType) {
        List<Value> operands = invokeOp.operands();

        // Obtain the memory mapping for each operand
        // if it comes from global memory, HAT replaces with a global* pointer to the inner struct,
        // then, we will need to operate half using a->value, instead of half value directly.
        boolean isFirstOperandReference = findReference(lookup(),invokeOp.operands().get(0));
        boolean isSecondOperandReference = findReference(lookup(),invokeOp.operands().get(1));

        byte valF32Conversion = 0x00;
        if (!isFirstOperandReference && isOperandF32(invokeOp.operands().get(0))) {
            valF32Conversion = HATF16Op.HATF16BinaryOp.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(invokeOp.operands().get(1))) {
            valF32Conversion = HATF16Op.HATF16BinaryOp.LAST_OP;
        }

        TypeElement typeElement = invokeOp.resultType();
        List<Boolean> refList = List.of(isFirstOperandReference, isSecondOperandReference);


        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        HATF16Op.HATF16BinaryOp binaryOp = switch (binaryOpEnum) {
            case ADD -> new HATF16Op.HATF16BinaryOp.HATF16AddOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
            case SUB -> new HATF16Op.HATF16BinaryOp.HATF16SubOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
            case MUL -> new HATF16Op.HATF16BinaryOp.HATF16MulOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
            case DIV -> new HATF16Op.HATF16BinaryOp.HATF16DivOp(typeElement, reducedFloatType, refList, valF32Conversion, outputOperands);
        };

        blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp,binaryOp)));
    }

    private CoreOp.FuncOp dialectifyF16Ops(CoreOp.FuncOp funcOp, BinaryOpEnum binaryOpEnum) {
        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();

        Invoke.stream(lookup(),funcOp)
                .filter(invoke -> is16BitFloat(invoke, Regex.of(binaryOpEnum.name().toLowerCase())) && !invoke.returnsVoid())
                .forEach(invoke ->  {
                        ReducedFloatType category = categorizeReducedFloat(invoke.op());
                        reducedFloatsType.put(invoke.op(), category);
                        invoke.op().result().uses().stream()
                                .filter(result -> result.op() instanceof CoreOp.VarOp)
                                .map(result -> (CoreOp.VarOp)result.op())
                                .findFirst()// we expect one
                                .ifPresent(varOp->reducedFloatsType.put(varOp,category));
                });

        return new Trxfmr(funcOp).transform(reducedFloatsType::containsKey,(blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16BinaryOp(invokeOp, blockBuilder, binaryOpEnum, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyF16Stores(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?,?>> nodesInvolved = new HashSet<>();
        Invoke.stream(lookup(),funcOp)
                .filter(invoke-> is16BitFloat(invoke,Regex.of("value")) && invoke.returns16BitValue())
                .forEach(invoke -> {
                    if(invoke.opFromFirstOperandAsResultOrNull() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                         && varLoadOp.operands().getFirst() instanceof Op.Result firstOperandsOpResult
                         && firstOperandsOpResult.op() instanceof HATF16Op.HATF16VarOp) {
                             nodesInvolved.addAll(Set.of(invoke.op(),varLoadOp));
                        }
                });

        return  new Trxfmr(funcOp).transform(ce->nodesInvolved.contains(ce),(blockBuilder, op) -> {
           if (op instanceof JavaOp.InvokeOp invokeOp) {
               blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.context().getValue(invokeOp.operands().getFirst()));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                createF16VarLoadOp(varLoadOp, blockBuilder);
            }
            return blockBuilder;
        }).funcOp();
    }


    private CoreOp.FuncOp dialectifyF16Init(CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();

        Invoke.stream(lookup(),funcOp)
                .filter(invoke -> !invoke.returnsVoid() && is16BitFloat(invoke, Regex.of("(of|floatToF16|float2bfloat16)")))
                .forEach(invoke ->
                    invoke.op().result().uses().stream()
                            .filter(result -> result.op() instanceof CoreOp.VarOp)
                            .map(result -> (CoreOp.VarOp) result.op() )
                            .findFirst()
                            .ifPresent(varOp -> { // is there only one?
                                ReducedFloatType reducedFloatType = categorizeReducedFloat(invoke.op());
                                reducedFloatsType.put(varOp, reducedFloatType);
                                reducedFloatsType.put(invoke.op(), reducedFloatType);
                    })
                );

        return new Trxfmr(funcOp).transform(reducedFloatsType::containsKey,(blockBuilder, op) -> {
            if (!reducedFloatsType.containsKey(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16ConvOP(invokeOpHelper(lookup(),invokeOp), blockBuilder, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyF16ToFloat(CoreOp.FuncOp funcOp) {
        Map<JavaOp.InvokeOp, ReducedFloatType> reducedFloatsType = new HashMap<>();
        funcOp.elements()
                .filter(ce->ce instanceof JavaOp.InvokeOp)
                .map(ce->invokeOpHelper(lookup(),ce))
                .filter(invoke->(invoke.named("f16ToFloat")||invoke.named("bfloat162float")) && invoke.returnsFloat())
                .findFirst() // only one?
                .ifPresent(invoke -> reducedFloatsType.put(invoke.op(), categorizeReducedFloat(invoke.op())));


        return new Trxfmr(funcOp).transform(reducedFloatsType::containsKey,(blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp $ && invokeOpHelper(lookup(),$) instanceof Invoke invoke) {
                createFloatFromF16(invoke, blockBuilder, reducedFloatsType.get(invoke.op()));
            }
            return blockBuilder;
        }).funcOp();
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
