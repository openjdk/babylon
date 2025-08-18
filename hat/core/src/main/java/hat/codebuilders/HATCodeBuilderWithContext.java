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
package hat.codebuilders;


import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.Schema;
import hat.optools.BinaryArithmeticOrLogicOperation;
import hat.optools.BinaryTestOpWrapper;
import hat.optools.ConstantOpWrapper;
import hat.optools.ConvOpWrapper;
import hat.optools.FieldLoadOpWrapper;
import hat.optools.FieldStoreOpWrapper;
import hat.optools.ForOpWrapper;
import hat.optools.FuncCallOpWrapper;
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
import hat.optools.UnaryArithmeticOrLogicOpWrapper;
import hat.optools.VarDeclarationOpWrapper;
import hat.optools.VarFuncDeclarationOpWrapper;
import hat.optools.VarLoadOpWrapper;
import hat.optools.VarStoreOpWrapper;
import hat.optools.WhileOpWrapper;
import hat.optools.YieldOpWrapper;
import hat.util.StreamCounter;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

public abstract class HATCodeBuilderWithContext<T extends HATCodeBuilderWithContext<T>> extends HATCodeBuilder<T> implements HATCodeBuilder.CodeBuilderInterface<T> {
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
            case JavaOp.InvokeOp o -> 0;
            case CoreOp.FuncCallOp o -> 0;
            case CoreOp.VarOp o -> 13;
            case CoreOp.VarAccessOp.VarStoreOp o -> 13;
            case JavaOp.FieldAccessOp o -> 0;
            case CoreOp.VarAccessOp.VarLoadOp o -> 0;
            case CoreOp.ConstantOp o -> 0;
            case JavaOp.LambdaOp o -> 0;
            case CoreOp.TupleOp o -> 0;
            case JavaOp.WhileOp o -> 0;
            case JavaOp.ConvOp o -> 1;
            case JavaOp.NegOp  o-> 1;
            case JavaOp.ModOp o -> 2;
            case JavaOp.MulOp o -> 2;
            case JavaOp.DivOp o -> 2;
            case JavaOp.NotOp o -> 2;
            case JavaOp.AddOp o -> 3;
            case JavaOp.SubOp o -> 3;
            case JavaOp.AshrOp o -> 4;
            case JavaOp.LshlOp o -> 4;
            case JavaOp.LshrOp o -> 4;
            case JavaOp.LtOp o -> 5;
            case JavaOp.GtOp o -> 5;
            case JavaOp.LeOp o -> 5;
            case JavaOp.GeOp o -> 5;
            case JavaOp.EqOp o -> 6;
            case JavaOp.NeqOp o -> 6;

            case JavaOp.AndOp o -> 11;
            case JavaOp.XorOp o -> 12;
            case JavaOp.OrOp o -> 13;
            case JavaOp.ConditionalAndOp o -> 14;
            case JavaOp.ConditionalOrOp o -> 15;
            case JavaOp.ConditionalExpressionOp o -> 18;
            case CoreOp.ReturnOp o -> 19;

            default -> throw new IllegalStateException("precedence ");
        };
    }

    public T typedefStructOrUnion(MemoryLayout memoryLayout, String name) {
        return typedefKeyword().space().structOrUnion(memoryLayout).space().suffix_s(name);
    }

    T structOrUnion(MemoryLayout memoryLayout) {
        return structOrUnion(memoryLayout instanceof StructLayout);
    }


    public T type(CodeBuilderContext buildContext,JavaType javaType) {
        if (InvokeOpWrapper.isIfaceUsingLookup(buildContext.lookup(),javaType) && javaType instanceof ClassType classType) {
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



    @Override
    public T varLoad(CodeBuilderContext buildContext, VarLoadOpWrapper varAccessOpWrapper) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varAccessOpWrapper.operandNAsValue(0));
        varName(varOp);
        return self();
    }

    @Override
    public T varStore(CodeBuilderContext buildContext, VarStoreOpWrapper varAccessOpWrapper) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varAccessOpWrapper.operandNAsValue(0));
        varName(varOp).equals();
        parencedence(buildContext, varAccessOpWrapper, varAccessOpWrapper.operandNAsResult(1).op());
        return self();
    }

    @Override
    public T varDeclaration(CodeBuilderContext buildContext, VarDeclarationOpWrapper varDeclarationOpWrapper) {
        if (varDeclarationOpWrapper.op().isUninitialized()) {
            // Variable is uninitialized
            type(buildContext,varDeclarationOpWrapper.javaType()).space().identifier(varDeclarationOpWrapper.varName());
        } else {
            type(buildContext,varDeclarationOpWrapper.javaType()).space().identifier(varDeclarationOpWrapper.varName()).space().equals().space();
            parencedence(buildContext, varDeclarationOpWrapper, varDeclarationOpWrapper.operandNAsResult(0).op());
        }
        return self();
    }

    @Override
    public T varFuncDeclaration(CodeBuilderContext buildContext, VarFuncDeclarationOpWrapper varFuncDeclarationOpWrapper) {
        // append("/* skipping ").type(varFuncDeclarationOpWrapper.javaType()).append(" param declaration  */");
        return self();
    }

    @Override
    public T fieldLoad(CodeBuilderContext buildContext, FieldLoadOpWrapper fieldLoadOpWrapper) {
        if (fieldLoadOpWrapper.isKernelContextAccess()) {
            identifier("kc").rarrow().identifier(fieldLoadOpWrapper.fieldName());
        } else if (fieldLoadOpWrapper.isStaticFinalPrimitive()) {    Object value = fieldLoadOpWrapper.getStaticFinalPrimitiveValue();
            literal(value.toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOpWrapper.fieldRef());
        }

        return self();
    }

    @Override
    public T fieldStore(CodeBuilderContext buildContext, FieldStoreOpWrapper fieldStoreOpWrapper) {
        //throw new IllegalStateException("What is this field store ?" + fieldStoreOpWrapper.fieldRef());
        return self();
    }


    T symbol(Op op) {
        return switch (op) {
            case JavaOp.ModOp o -> percent();
            case JavaOp.MulOp o -> mul();
            case JavaOp.DivOp o -> div();
            case JavaOp.AddOp o -> plus();
            case JavaOp.SubOp o -> minus();
            case JavaOp.LtOp o -> lt();
            case JavaOp.GtOp o -> gt();
            case JavaOp.LeOp o -> lte();
            case JavaOp.GeOp o -> gte();
            case JavaOp.AshrOp o -> cchevron().cchevron();
            case JavaOp.LshlOp o -> ochevron().ochevron();
            case JavaOp.LshrOp o -> cchevron().cchevron();
            case JavaOp.NeqOp o -> pling().equals();
            case JavaOp.NegOp o -> minus();
            case JavaOp.EqOp o -> equals().equals();
            case JavaOp.NotOp o -> pling();
            case JavaOp.AndOp o -> ampersand();
            case JavaOp.OrOp o -> bar();
            case JavaOp.XorOp o -> hat();
            case JavaOp.ConditionalAndOp o -> condAnd();
            case JavaOp.ConditionalOrOp o -> condOr();
            default -> throw new IllegalStateException("Unexpected value: " + op);
        };
    }

    @Override
    public T unaryOperation(CodeBuilderContext buildContext, UnaryArithmeticOrLogicOpWrapper unaryOperatorOpWrapper) {
      //  parencedence(buildContext, binaryOperatorOpWrapper.op(), binaryOperatorOpWrapper.lhsAsOp());
        symbol(unaryOperatorOpWrapper.op());
        parencedence(buildContext, unaryOperatorOpWrapper.op(), unaryOperatorOpWrapper.operandNAsResult(0).op());
        return self();
    }

    @Override
    public T binaryOperation(CodeBuilderContext buildContext, BinaryArithmeticOrLogicOperation binaryOperatorOpWrapper) {
        parencedence(buildContext, binaryOperatorOpWrapper.op(), binaryOperatorOpWrapper.lhsAsOp());
        symbol(binaryOperatorOpWrapper.op());
        parencedence(buildContext, binaryOperatorOpWrapper.op(), binaryOperatorOpWrapper.rhsAsOp());
        return self();
    }

    @Override
    public T logical(CodeBuilderContext buildContext, LogicalOpWrapper logicalOpWrapper) {
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
    public T binaryTest(CodeBuilderContext buildContext, BinaryTestOpWrapper binaryTestOpWrapper) {
        parencedence(buildContext, binaryTestOpWrapper.op(), binaryTestOpWrapper.lhsAsOp());
        symbol(binaryTestOpWrapper.op());
        parencedence(buildContext, binaryTestOpWrapper.op(), binaryTestOpWrapper.rhsAsOp());
        return self();
    }

    @Override

    public T conv(CodeBuilderContext buildContext, ConvOpWrapper convOpWrapper) {
        if (convOpWrapper.resultJavaType() == JavaType.DOUBLE) {
            paren(_ -> type(buildContext,JavaType.FLOAT));
        } else {
            paren(_ -> type(buildContext,convOpWrapper.resultJavaType()));
        }
        //paren(() -> type(convOpWrapper.resultJavaType()));
        parencedence(buildContext, convOpWrapper, convOpWrapper.operandNAsResult(0).op());
        return self();
    }

    @Override
    public T constant(CodeBuilderContext buildContext, ConstantOpWrapper constantOpWrapper) {
        Object object = constantOpWrapper.op().value();
        if (object == null) {
            nullKeyword();
        } else {
            literal(constantOpWrapper.op().value().toString());
        }
        return self();
    }

    @Override
    public T javaYield(CodeBuilderContext buildContext, YieldOpWrapper yieldOpWrapper) {
        var operand0 = yieldOpWrapper.operandNAsValue(0);
        if (operand0 instanceof Op.Result result) {
            recurse(buildContext, OpWrapper.wrap(buildContext.lookup(), result.op()));
        } else {
            // append("/*nothing to yield*/");
        }
        return self();
    }

    @Override
    public T lambda(CodeBuilderContext buildContext, LambdaOpWrapper lambdaOpWrapper) {
        return commented("/*LAMBDA*/");
    }

    @Override
    public T tuple(CodeBuilderContext buildContext, TupleOpWrapper tupleOpWrapper) {
        StreamCounter.of(tupleOpWrapper.operands(), (c, operand) -> {
            if (c.isNotFirst()) {
                comma().space();
            }
            if (operand instanceof Op.Result result) {
                recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),result.op()));
            } else {
                commented("/*nothing to tuple*/");
            }
        });
        return self();
    }

    @Override
    public T funcCall(CodeBuilderContext buildContext, FuncCallOpWrapper funcCallOpWrapper) {
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
    public T javaLabeled(CodeBuilderContext buildContext, JavaLabeledOpWrapper javaLabeledOpWrapper) {
        var labelNameOp = OpWrapper.wrap(buildContext.lookup(),javaLabeledOpWrapper.firstBlockOfFirstBody().ops().get(0));
        CoreOp.ConstantOp constantOp = (CoreOp.ConstantOp) labelNameOp.op();
        literal(constantOp.value().toString()).colon().nl();
        var forLoopOp = javaLabeledOpWrapper.firstBlockOfFirstBody().ops().get(1);
        recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),forLoopOp));
        // var yieldOp = javaLabeledOpWrapper.firstBlockOfFirstBody().ops().get(2);
        return self();
    }

    public T javaBreak(CodeBuilderContext buildContext, JavaBreakOpWrapper javaBreakOpWrapper) {
        breakKeyword();
        if (javaBreakOpWrapper.hasOperands() && javaBreakOpWrapper.operandNAsResult(0) instanceof Op.Result result) {
            space();
            if (result.op() instanceof CoreOp.ConstantOp c) {
                literal(c.value().toString());
            }
        }
        return self();
    }

    public T javaContinue(CodeBuilderContext buildContext, JavaContinueOpWrapper javaContinueOpWrapper) {
        if (javaContinueOpWrapper.hasOperands()
                && javaContinueOpWrapper.operandNAsResult(0) instanceof Op.Result result
                && result.op() instanceof CoreOp.ConstantOp c
        ) {
            continueKeyword().space().literal(c.value().toString());
        } else if (buildContext.scope.parent instanceof CodeBuilderContext.LoopScope<?>) {
            // nope
        } else {
            continueKeyword();
        }

        return self();
    }

    @Override
    public T javaIf(CodeBuilderContext buildContext, IfOpWrapper ifOpWrapper) {
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
    public T javaWhile(CodeBuilderContext buildContext, WhileOpWrapper whileOpWrapper) {
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
    public T javaFor(CodeBuilderContext buildContext, ForOpWrapper forOpWrapper) {
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


    public T typedef(BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        typedefKeyword().space().structOrUnion(ifaceType instanceof Schema.IfaceType.Struct)
                .space().suffix_s(ifaceType.iface.getSimpleName()).braceNlIndented(_ -> {
                    //System.out.println(ifaceTypeNode);
                    int fieldCount = ifaceType.fields.size();
                    StreamCounter.of(ifaceType.fields, (c, field) -> {
                        nlIf(c.isNotFirst());
                        boolean isLast = c.value() == fieldCount - 1;
                        if (field instanceof Schema.FieldNode.AbstractPrimitiveField primitiveField) {
                            typeName(primitiveField.type.getSimpleName());
                            space().typeName(primitiveField.name);
                            if (primitiveField instanceof Schema.FieldNode.PrimitiveArray array) {
                                if (array instanceof Schema.FieldNode.PrimitiveFieldControlledArray fieldControlledArray) {
                                    if (isLast && ifaceType.parent == null) {
                                        sbrace(_ -> literal(1));
                                    } else {
                                        boolean[] done = new boolean[]{false};
                                        if (boundSchema != null) {
                                            boundSchema.boundArrayFields().forEach(a -> {
                                                if (a.field.equals(array)) {
                                                    sbrace(_ -> literal(a.len));
                                                    done[0] = true;
                                                }
                                            });
                                            if (!done[0]) {
                                                throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                            }
                                        }else {
                                            throw new IllegalStateException("bound schema is null  !");
                                        }
                                    }
                                } else if (array instanceof Schema.FieldNode.PrimitiveFixedArray fixed) {
                                    sbrace(_ -> literal(Math.max(1, fixed.len)));
                                } else {
                                    throw new IllegalStateException("what kind of array ");
                                }
                            }
                        } else if (field instanceof Schema.FieldNode.AbstractIfaceField ifaceField) {
                            suffix_t(ifaceField.ifaceType.iface.getSimpleName());
                            space().typeName(ifaceField.name);
                            if (ifaceField instanceof Schema.FieldNode.IfaceArray array) {
                                if (array instanceof Schema.FieldNode.IfaceFieldControlledArray fieldControlledArray) {
                                    if (isLast && ifaceType.parent == null) {
                                        sbrace(_ -> literal(1));
                                    } else {
                                        if (boundSchema != null) {
                                            boolean[] done = new boolean[]{false};
                                            boundSchema.boundArrayFields().forEach(a -> {
                                                if (a.field.equals(ifaceField)) {
                                                    sbrace(_ -> literal(a.len));
                                                    done[0] = true;
                                                }
                                            });
                                            if (!done[0]) {
                                                throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                            }
                                        }else {
                                        throw new IllegalStateException("bound schema is null  !");
                                        }
                                    }
                                } else if (array instanceof Schema.FieldNode.IfaceFixedArray fixed) {
                                    sbrace(_ -> literal(Math.max(1, fixed.len)));
                                } else {
                                    throw new IllegalStateException("what kind of array ");
                                }
                            }
                        } else if (field instanceof Schema.SchemaNode.Padding) {
                            // SKIP
                        } else {
                            throw new IllegalStateException("hmm");
                        }


                        semicolon();
                    });
                }).suffix_t(ifaceType.iface.getSimpleName()).semicolon().nl().nl();
        return self();
    }

    public T atomicInc(CodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        throw new IllegalStateException("atomicInc not implemented");
    }

    @Override
    public T methodCall(CodeBuilderContext buildContext, InvokeOpWrapper invokeOpWrapper) {
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
                    recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),instanceResult.op()));
                    rarrow().identifier(name);
                    //if (invokeOpWrapper.name().equals("value") || invokeOpWrapper.name().equals("anon")){
                    //System.out.println("value|anon");
                    // }
                    if (returnType instanceof PrimitiveType primitiveType && primitiveType.isVoid()) {
                        //   setter
                        switch (operandCount) {
                            case 2: {
                                if (invokeOpWrapper.operandNAsResult(1) instanceof Op.Result result1) {
                                    equals().recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),result1.op()));
                                } else {
                                    throw new IllegalStateException("How ");
                                }
                                break;
                            }
                            case 3: {
                                if (invokeOpWrapper.operandNAsResult(1) instanceof Op.Result result1
                                        && invokeOpWrapper.operandNAsResult(2) instanceof Op.Result result2) {
                                    sbrace(_ -> recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),result1.op())));
                                    equals().recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),result2.op()));
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
                            var rhs = OpWrapper.wrap(buildContext.lookup(),result1.op());
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
                            recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),result.op()));
                        } else {
                            throw new IllegalStateException("wtf?");
                        }
                    })
            );
        }
        return self();
    }

    @Override
    public T ternary(CodeBuilderContext buildContext, TernaryOpWrapper ternaryOpWrapper) {
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
    public T parencedence(CodeBuilderContext buildContext, Op parent, OpWrapper<?> child) {
        return parenWhen(precedenceOf(parent) < precedenceOf(child.op()), _ -> recurse(buildContext, child));
    }

    public T parencedence(CodeBuilderContext buildContext, OpWrapper<?> parent, OpWrapper<?> child) {
        return parenWhen(precedenceOf(parent.op()) < precedenceOf(child.op()), _ -> recurse(buildContext, child));
    }
@Override
    public T parencedence(CodeBuilderContext buildContext,  Op parent, Op child) {
        return parenWhen(precedenceOf(parent) < precedenceOf(child), _ -> recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),child)));
    }

    public T parencedence(CodeBuilderContext buildContext, OpWrapper<?> parent, Op child) {
        return parenWhen(precedenceOf(parent.op()) < precedenceOf(child), _ -> recurse(buildContext, OpWrapper.wrap(buildContext.lookup(),child)));
    }


    @Override
    public T ret(CodeBuilderContext buildContext, ReturnOpWrapper returnOpWrapper) {
        returnKeyword();
        if (returnOpWrapper.hasOperands()) {
            space().parencedence(buildContext, returnOpWrapper, returnOpWrapper.operandNAsResult(0).op());
        }
        return self();
    }
}
