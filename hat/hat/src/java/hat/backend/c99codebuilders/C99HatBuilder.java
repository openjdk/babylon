/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
package hat.backend.c99codebuilders;


import hat.buffer.Buffer;
import hat.optools.BinaryArithmeticOrLogicOperation;
import hat.optools.BinaryTestOpWrapper;
import hat.optools.ConstantOpWrapper;
import hat.optools.ConvOpWrapper;
import hat.optools.FieldLoadOpWrapper;
import hat.optools.FieldStoreOpWrapper;
import hat.optools.ForOpWrapper;
import hat.optools.FuncCallOpWrapper;
import hat.optools.FuncOpWrapper;
import hat.optools.IfOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.JavaBreakOpWrapper;
import hat.optools.JavaContinueOpWrapper;
import hat.optools.JavaLabeledOpWrapper;
import hat.optools.LambdaOpWrapper;
import hat.optools.LogicalOpWrapper;
import hat.optools.OpWrapper;
import hat.optools.ReturnOpWrapper;
import hat.optools.StructuralOpWrapper;
import hat.optools.TernaryOpWrapper;
import hat.optools.TupleOpWrapper;
import hat.optools.VarDeclarationOpWrapper;
import hat.optools.VarFuncDeclarationOpWrapper;
import hat.optools.VarLoadOpWrapper;
import hat.optools.VarStoreOpWrapper;
import hat.optools.WhileOpWrapper;
import hat.optools.YieldOpWrapper;
import hat.util.StreamCounter;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.ClassType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.Map;

public abstract class C99HatBuilder<T extends C99HatBuilder<T>> extends C99CodeBuilder<T> implements C99HatBuilderInterface<T> {
    /*
   0 =  ()[ ] . -> ++ --
   1 = ++ --+ -! ~ (type) *(deref) &(addressof) sizeof
   2 = * / %
   3 = + -
   4 = << >>
   5 = < <= > >=
   6 = == !=
   7 = &
   8 = ^
   9 = |
   10 = &&
   11 = ||
   12 = ()?:
   13 = += -= *= /= %= &= ^= |= <<= >>=
   14 = ,
     */
    public int precedenceOf(Op op) {
        return switch (op) {
            case CoreOp.YieldOp o -> 0;
            case CoreOp.InvokeOp o -> 0;
            case CoreOp.FuncCallOp o -> 0;
            case CoreOp.VarOp o -> 13;
            case CoreOp.VarAccessOp.VarStoreOp o -> 13;
            case CoreOp.FieldAccessOp o -> 0;
            case CoreOp.VarAccessOp.VarLoadOp o -> 0;
            case CoreOp.ConstantOp o -> 0;
            case CoreOp.LambdaOp o -> 0;
            case CoreOp.TupleOp o -> 0;
            case ExtendedOp.JavaWhileOp o -> 0;
            case CoreOp.ConvOp o -> 1;
            case CoreOp.ModOp o -> 2;
            case CoreOp.MulOp o -> 2;
            case CoreOp.DivOp o -> 2;
            case CoreOp.AddOp o -> 3;
            case CoreOp.SubOp o -> 3;
            case CoreOp.LtOp o -> 5;
            case CoreOp.GtOp o -> 5;
            case CoreOp.LeOp o -> 5;
            case CoreOp.GeOp o -> 5;
            case CoreOp.EqOp o -> 6;
            case CoreOp.NeqOp o -> 6;
            case ExtendedOp.JavaConditionalAndOp o -> 10;
            case ExtendedOp.JavaConditionalOrOp o -> 11;
            case ExtendedOp.JavaConditionalExpressionOp o -> 12;
            case CoreOp.ReturnOp o -> 12;

            default -> throw new IllegalStateException("precedence ");
        };
    }

    public T typedefStructOrUnion(MemoryLayout memoryLayout, String name) {
        return typedefKeyword().space().structOrUnion(memoryLayout).space().suffix_s(name);
    }

    T structOrUnion(MemoryLayout memoryLayout) {
        return structOrUnion(memoryLayout instanceof StructLayout);
    }


    public T type(JavaType javaType) {
        if (FuncOpWrapper.ParamTable.Info.isIfaceBuffer(javaType) && javaType instanceof ClassType classType) {
            String name = classType.toClassName();
            int dotIdx = name.lastIndexOf('.');
            int dollarIdx = name.lastIndexOf('$');
            int idx = Math.max(dotIdx, dollarIdx);
            if (idx > 0) {
                name = name.substring(idx + 1);
            }
            suffix_t(name).asterisk();
        } else {
            typeName(javaType.toBasicType().toString());
        }
        return self();
    }

    public T varName(CoreOp.VarOp varOp) {
        identifier(varOp.varName());
        return self();
    }

    @Override
    public T varLoad(C99HatBuildContext buildContext, VarLoadOpWrapper varAccessOpWrapper) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varAccessOpWrapper.operandNAsValue(0));
        varName(varOp);
        return self();
    }

    @Override
    public T varStore(C99HatBuildContext buildContext, VarStoreOpWrapper varAccessOpWrapper) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varAccessOpWrapper.operandNAsValue(0));
        varName(varOp).equals();
        parencedence(buildContext, varAccessOpWrapper, varAccessOpWrapper.operandNAsResult(1).op());
        return self();
    }

    @Override
    public T varDeclaration(C99HatBuildContext buildContext, VarDeclarationOpWrapper varDeclarationOpWrapper) {
        type(varDeclarationOpWrapper.javaType()).space().identifier(varDeclarationOpWrapper.varName()).space().equals().space();
        parencedence(buildContext, varDeclarationOpWrapper, varDeclarationOpWrapper.operandNAsResult(0).op());
        return self();
    }

    @Override
    public T varFuncDeclaration(C99HatBuildContext buildContext, VarFuncDeclarationOpWrapper varFuncDeclarationOpWrapper) {
        // append("/* skipping ").type(varFuncDeclarationOpWrapper.javaType()).append(" param declaration  */");
        return self();
    }

    @Override
    public T fieldLoad(C99HatBuildContext buildContext, FieldLoadOpWrapper fieldLoadOpWrapper) {
        if (fieldLoadOpWrapper.isKernelContextAccess()) {
            identifier("kc").dot().identifier(fieldLoadOpWrapper.fieldName());
        } else {
            // throw new IllegalStateException("What is this field load ?" + fieldLoadOpWrapper.fieldRef());
        }
        return self();
    }

    @Override
    public T fieldStore(C99HatBuildContext buildContext, FieldStoreOpWrapper fieldStoreOpWrapper) {
        //throw new IllegalStateException("What is this field store ?" + fieldStoreOpWrapper.fieldRef());
        return self();
    }


    T symbol(Op op) {
        return switch (op) {
            case CoreOp.ModOp o -> percent();
            case CoreOp.MulOp o -> mul();
            case CoreOp.DivOp o -> div();
            case CoreOp.AddOp o -> plus();
            case CoreOp.SubOp o -> minus();
            case CoreOp.LtOp o -> lt();
            case CoreOp.GtOp o -> gt();
            case CoreOp.LeOp o -> lte();
            case CoreOp.GeOp o -> gte();
            case CoreOp.NeqOp o -> pling().equals();
            case CoreOp.EqOp o -> equals().equals();
            case ExtendedOp.JavaConditionalAndOp o -> condAnd();
            case ExtendedOp.JavaConditionalOrOp o -> condOr();
            default -> throw new IllegalStateException("Unexpected value: " + op);
        };
    }

    @Override
    public T binaryOperation(C99HatBuildContext buildContext, BinaryArithmeticOrLogicOperation binaryOperatorOpWrapper) {
        parencedence(buildContext, binaryOperatorOpWrapper.op(), binaryOperatorOpWrapper.lhsAsOp());
        symbol(binaryOperatorOpWrapper.op());
        parencedence(buildContext, binaryOperatorOpWrapper.op(), binaryOperatorOpWrapper.rhsAsOp());
        return self();
    }

    @Override
    public T logical(C99HatBuildContext buildContext, LogicalOpWrapper logicalOpWrapper) {
        logicalOpWrapper.lhsWrappedYieldOpStream().forEach((wrapped) -> {
            recurse(buildContext, wrapped);
        });
        space().symbol(logicalOpWrapper.op()).space();
        logicalOpWrapper.rhsWrappedYieldOpStream().forEach((wrapped) -> {
            recurse(buildContext, wrapped);
        });
        return self();
    }

    @Override
    public T binaryTest(C99HatBuildContext buildContext, BinaryTestOpWrapper binaryTestOpWrapper) {
        parencedence(buildContext, binaryTestOpWrapper.op(), binaryTestOpWrapper.lhsAsOp());
        symbol(binaryTestOpWrapper.op());
        parencedence(buildContext, binaryTestOpWrapper.op(), binaryTestOpWrapper.rhsAsOp());
        return self();
    }

    @Override

    public T conv(C99HatBuildContext buildContext, ConvOpWrapper convOpWrapper) {
        if (convOpWrapper.resultJavaType() == JavaType.DOUBLE) {
            paren(_ -> type(JavaType.FLOAT));
        } else {
            paren(_ -> type(convOpWrapper.resultJavaType()));
        }
        //paren(() -> type(convOpWrapper.resultJavaType()));
        parencedence(buildContext, convOpWrapper, convOpWrapper.operandNAsResult(0).op());
        return self();
    }

    @Override
    public T constant(C99HatBuildContext buildContext, ConstantOpWrapper constantOpWrapper) {
        Object object = constantOpWrapper.op().value();
        if (object == null) {
            nullKeyword();
        } else {
            literal(constantOpWrapper.op().value().toString());
        }
        return self();
    }

    @Override
    public T javaYield(C99HatBuildContext buildContext, YieldOpWrapper yieldOpWrapper) {
        var operand0 = yieldOpWrapper.operandNAsValue(0);
        if (operand0 instanceof Op.Result result) {
            recurse(buildContext, OpWrapper.wrap(result.op()));
        } else {
            // append("/*nothing to yield*/");
        }
        return self();
    }

    @Override
    public T lambda(C99HatBuildContext buildContext, LambdaOpWrapper lambdaOpWrapper) {
        return commented("/*LAMBDA*/");
    }

    @Override
    public T tuple(C99HatBuildContext buildContext, TupleOpWrapper tupleOpWrapper) {
        StreamCounter.of(tupleOpWrapper.operands(), (c, operand) -> {
            if (c.isNotFirst()) {
                comma().space();
            }
            if (operand instanceof Op.Result result) {
                recurse(buildContext, OpWrapper.wrap(result.op()));
            } else {
                commented("/*nothing to tuple*/");
            }
        });
        return self();
    }

    @Override
    public T funcCall(C99HatBuildContext buildContext, FuncCallOpWrapper funcCallOpWrapper) {
        var functionCallName = funcCallOpWrapper.funcName();


        identifier(funcCallOpWrapper.funcName());
        paren(_ -> {
            commaSeparated(funcCallOpWrapper.operands(), (e) -> {
                if (e instanceof Op.Result r) {
                    parencedence(buildContext, funcCallOpWrapper, r.op());
                } else {
                    throw new IllegalStateException("Value?");
                }
            });
        });
        return self();
    }

    @Override
    public T javaLabeled(C99HatBuildContext buildContext, JavaLabeledOpWrapper javaLabeledOpWrapper) {
        var labelNameOp = OpWrapper.wrap(javaLabeledOpWrapper.firstBlockOfFirstBody().ops().get(0));
        CoreOp.ConstantOp constantOp = (CoreOp.ConstantOp) labelNameOp.op();
        literal(constantOp.value().toString()).colon().nl();
        var forLoopOp = javaLabeledOpWrapper.firstBlockOfFirstBody().ops().get(1);
        recurse(buildContext, OpWrapper.wrap(forLoopOp));
        // var yieldOp = javaLabeledOpWrapper.firstBlockOfFirstBody().ops().get(2);
        return self();
    }

    public T javaBreak(C99HatBuildContext buildContext, JavaBreakOpWrapper javaBreakOpWrapper) {
        breakKeyword();
        if (javaBreakOpWrapper.hasOperands() && javaBreakOpWrapper.operandNAsResult(0) instanceof Op.Result result) {
            space();
            if (result.op() instanceof CoreOp.ConstantOp c) {
                literal(c.value().toString());
            }
        }
        return self();
    }

    public T javaContinue(C99HatBuildContext buildContext, JavaContinueOpWrapper javaContinueOpWrapper) {
        if (javaContinueOpWrapper.hasOperands()
                && javaContinueOpWrapper.operandNAsResult(0) instanceof Op.Result result
                && result.op() instanceof CoreOp.ConstantOp c
        ) {
            continueKeyword().space().literal(c.value().toString());
        } else if (buildContext.scope.parent instanceof C99HatBuildContext.LoopScope<?>) {
            // nope
        } else {
            continueKeyword();
        }

        return self();
    }

    @Override
    public T javaIf(C99HatBuildContext buildContext, IfOpWrapper ifOpWrapper) {
        buildContext.scope(ifOpWrapper, () -> {
            boolean[] lastWasBody = new boolean[]{false};
            StreamCounter.of(ifOpWrapper.bodies(), (c, b) -> {
                if (b.yieldType() instanceof JavaType javaType && javaType == JavaType.VOID) {
                    if (ifOpWrapper.hasElseN(c.value())) {
                        if (lastWasBody[0]) {
                            elseKeyword();
                        }
                        braceNlIndented(_ ->
                                StreamCounter.of(ifOpWrapper.wrappedRootOpStream(ifOpWrapper.firstBlockOfBodyN(c.value())), (innerc, root) ->
                                        nlIf(innerc.isNotFirst())
                                                .recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                                )
                        );
                    }
                    lastWasBody[0] = true;
                } else {
                    if (c.isNotFirst()) {
                        elseKeyword().space();
                    }
                    ifKeyword().paren(_ ->
                            ifOpWrapper.wrappedYieldOpStream(ifOpWrapper.firstBlockOfBodyN(c.value())).forEach((wrapped) ->
                                    recurse(buildContext, wrapped))
                    );
                    lastWasBody[0] = false;
                }

            });
        });
        return self();
    }

    @Override
    public T javaWhile(C99HatBuildContext buildContext, WhileOpWrapper whileOpWrapper) {
        whileKeyword().paren(_ ->
                whileOpWrapper.conditionWrappedYieldOpStream().forEach((wrapped) -> recurse(buildContext, wrapped))
        ).braceNlIndented(_ ->
                StreamCounter.of(whileOpWrapper.loopWrappedRootOpStream(), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                )
        );
        return self();
    }

    @Override
    public T javaFor(C99HatBuildContext buildContext, ForOpWrapper forOpWrapper) {
        buildContext.scope(forOpWrapper, () ->
                forKeyword().paren(_ -> {
                    forOpWrapper.initWrappedYieldOpStream().forEach((wrapped) -> recurse(buildContext, wrapped));
                    semicolon().space();
                    forOpWrapper.conditionWrappedYieldOpStream().forEach((wrapped) -> recurse(buildContext, wrapped));
                    semicolon().space();
                    StreamCounter.of(forOpWrapper.mutateRootWrappedOpStream(), (c, wrapped) ->
                            commaSpaceIf(c.isNotFirst()).recurse(buildContext, wrapped)
                    );
                }).braceNlIndented(_ ->
                        StreamCounter.of(forOpWrapper.loopWrappedRootOpStream(), (c, root) ->
                                nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                        )
                )
        );
        return self();
    }

    public T typedef(Map<String, Typedef> scope, Buffer instance) {
        return typedef(scope, new Typedef(instance));
    }

    public T typedef(Map<String, Typedef> scope, Typedef typeDef) {
        if (!scope.containsKey(typeDef.name())) {
            // Do the dependencies first, so we get them in the right order
            typeDef.nameAndTypes.stream().filter(nameAndType -> nameAndType.typeDef != null).forEach(nameAndType -> {
                typedef(scope, nameAndType.typeDef).nl();
            });
            typedefKeyword().space().structOrUnion(typeDef.isStruct)
                    .space().suffix_s(typeDef.iface.getSimpleName()).braceNlIndented(_ -> {
                        StreamCounter.of(typeDef.nameAndTypes, (c, nameAndType) -> {
                            nlIf(c.isNotFirst());
                            if (nameAndType.type.isPrimitive()) {
                                typeName(nameAndType.type.getSimpleName());
                            } else {
                                suffix_t(nameAndType.type.getSimpleName());
                            }
                            space().typeName(nameAndType.name);
                            if (nameAndType instanceof Typedef.NameAndArrayOfType nameAndArrayOfType) {
                                sbrace(_ -> {
                                    if (nameAndArrayOfType.arraySize >= 0) {
                                        literal(nameAndArrayOfType.arraySize);
                                    }
                                });
                            }
                            semicolon();
                        });
                    }).suffix_t(typeDef.iface.getSimpleName()).semicolon().nl().nl();
            scope.put(typeDef.name(), typeDef);
        }
        return self();
    }

    public T atomicInc(C99HatBuildContext buildContext, Op.Result instanceResult, String name){
         throw new IllegalStateException("atimicInc not implemented");
    }

    @Override
    public T methodCall(C99HatBuildContext buildContext, InvokeOpWrapper invokeOpWrapper) {
        var name = invokeOpWrapper.name();

        if (invokeOpWrapper.isIfaceBufferMethod()) {
            var operandCount = invokeOpWrapper.operandCount();
            var returnType = invokeOpWrapper.javaReturnType();

            if (operandCount == 1 && name.startsWith("atomic") && name.endsWith("Inc")
                    && returnType instanceof PrimitiveType primitiveType && primitiveType.equals(JavaType.INT)) {
                // this is a bit of a hack for atomics.
                if (invokeOpWrapper.operandNAsResult(0) instanceof Op.Result instanceResult) {
                    atomicInc(buildContext, instanceResult, name.substring(0, name.length() - 3));
                    //identifier("atomic_inc").paren(_ -> {
                    //    ampersand().recurse(buildContext, OpWrapper.wrap(instanceResult.op()));
                    //    rarrow().identifier(name.substring(0, name.length() - 3));
                    //});
                } else {
                    throw new IllegalStateException("bad atomic");
                }
            } else {
                if (invokeOpWrapper.operandNAsResult(0) instanceof Op.Result instanceResult) {
                /*
                We have three types of returned values from an ifaceBuffer
                A primitive
                    int id = stage.firstTreeId(); -> stage->firstTreeId;

                Or a sub interface from an array
                     Tree tree = cascade.tree(treeIdx); -> Tree_t * tree = &cascade->tree[treeIdx]
                                                        ->               = cascade->tree + treeIdx;

                Or a sub interface from a field

                var left = feature.left();              ->  LinkOrValue_t * left= &feature->left

                                -
                    if (left.hasValue()) {                  left->hasValue
                        sum += left.anon().value();         left->anon.value;
                        feature = null; // loop ends
                    } else {
                        feature = cascade.feature(tree.firstFeatureId() + left.anon().featureId());
                    }
                 sumOfThisStage += left.anon().value();


                For a primitive we know that the accessor refers to a field so we just  map
                         stage.firstTreeId() -> stage->firstTreeId;

                For the sub interface we need to treat the call
                          cascade.tree(treeIdx);

                As an array index into cascade->tree[] that returns a typedef of Tree_t
                so we need to prefix with an & to return a Tree_t ptr
                          &cascade->tree[treeIdx]

                 of course we could return
                          cascade->tree + treeIdx;
                 */

                    if (returnType instanceof ClassType classType) {
                        ampersand();
                    /* This is way more complicated I think we need to determine the expression type.


                     sumOfThisStage=sumOfThisStage+&left->anon->value; from    sumOfThisStage += left.anon().value();

                     */
                    }
                    recurse(buildContext, OpWrapper.wrap(instanceResult.op()));
                    rarrow().identifier(name);
                    //if (invokeOpWrapper.name().equals("value") || invokeOpWrapper.name().equals("anon")){
                    //System.out.println("value|anon");
                    // }
                    if (returnType instanceof PrimitiveType primitiveType && primitiveType.isVoid()) {
                        //   setter
                        switch (operandCount) {
                            case 2: {
                                if (invokeOpWrapper.operandNAsResult(1) instanceof Op.Result result1) {
                                    equals().recurse(buildContext, OpWrapper.wrap(result1.op()));
                                } else {
                                    throw new IllegalStateException("How ");
                                }
                                break;
                            }
                            case 3: {
                                if (invokeOpWrapper.operandNAsResult(1) instanceof Op.Result result1
                                        && invokeOpWrapper.operandNAsResult(2) instanceof Op.Result result2) {
                                    sbrace(_ -> recurse(buildContext, OpWrapper.wrap(result1.op())));
                                    equals().recurse(buildContext, OpWrapper.wrap(result2.op()));
                                } else {
                                    throw new IllegalStateException("How ");
                                }
                                break;
                            }
                            default: {
                                throw new IllegalStateException("How ");
                            }
                        }
                    } else {
                        if (invokeOpWrapper.operandNAsResult(1) instanceof Op.Result result1) {
                            var rhs = OpWrapper.wrap(result1.op());
                            sbrace(_ -> recurse(buildContext, rhs));
                        } else {
                            // This is a simple usage.   So scaleTable->multiScaleAccumRange
                        }
                    }
                } else {
                    throw new IllegalStateException("arr");
                }

            }
        } else {
            identifier(name).paren(_ ->
                    commaSeparated(invokeOpWrapper.operands(), (op) -> {
                        if (op instanceof Op.Result result) {
                            recurse(buildContext, OpWrapper.wrap(result.op()));
                        } else {
                            throw new IllegalStateException("wtf?");
                        }
                    })
            );
        }
        return self();
    }

    @Override
    public T ternary(C99HatBuildContext buildContext, TernaryOpWrapper ternaryOpWrapper) {
        ternaryOpWrapper.conditionWrappedYieldOpStream().forEach((wrapped) -> recurse(buildContext, wrapped));
        questionMark();
        ternaryOpWrapper.thenWrappedYieldOpStream().forEach((wrapped) -> recurse(buildContext, wrapped));
        colon();
        ternaryOpWrapper.elseWrappedYieldOpStream().forEach((wrapped) -> recurse(buildContext, wrapped));
        return self();
    }

    /**
     * Wrap paren() of precedence of op is higher than parent.
     * Parencedence is just a great name for this ;)
     *
     * @param buildContext
     * @param parent
     * @param child
     */
    @Override
    public T parencedence(C99HatBuildContext buildContext, Op parent, OpWrapper<?> child) {
        return parenWhen(precedenceOf(parent) < precedenceOf(child.op()), _ -> recurse(buildContext, child));
    }

    public T parencedence(C99HatBuildContext buildContext, OpWrapper<?> parent, OpWrapper<?> child) {
        return parenWhen(precedenceOf(parent.op()) < precedenceOf(child.op()), _ -> recurse(buildContext, child));
    }

    public T parencedence(C99HatBuildContext buildContext, Op parent, Op child) {
        return parenWhen(precedenceOf(parent) < precedenceOf(child), _ -> recurse(buildContext, OpWrapper.wrap(child)));
    }

    public T parencedence(C99HatBuildContext buildContext, OpWrapper<?> parent, Op child) {
        return parenWhen(precedenceOf(parent.op()) < precedenceOf(child), _ -> recurse(buildContext, OpWrapper.wrap(child)));
    }


    @Override
    public T ret(C99HatBuildContext buildContext, ReturnOpWrapper returnOpWrapper) {
        returnKeyword();
        if (returnOpWrapper.hasOperands()) {
            space().parencedence(buildContext, returnOpWrapper, returnOpWrapper.operandNAsResult(0).op());
        }
        return self();
    }
}
