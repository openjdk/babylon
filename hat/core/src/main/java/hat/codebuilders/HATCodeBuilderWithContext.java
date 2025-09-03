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
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import hat.optools.OpTk;
import hat.util.StreamMutable;
import hat.util.StreamCounter;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;

import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
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


    public T type(HATCodeBuilderContext buildContext, JavaType javaType) {
        if (OpTk.isAssignable(buildContext.lookup,javaType, MappableIface.class)
                //isIfaceUsingLookup(buildContext.lookup,javaType)
                        && javaType instanceof ClassType classType) {
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
    public T varLoad(HATCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varLoadOp.operands().getFirst());
        varName(varOp);
        return self();
    }

    @Override
    public T varStore(HATCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varStoreOp.operands().getFirst());
        varName(varOp).equals();
        parencedence(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }

    private String extractClassType(HATCodeBuilderContext buildContext, JavaType javaType, ClassType classType) {
        String name = classType.toClassName();
        int dotIdx = name.lastIndexOf('.');
        int dollarIdx = name.lastIndexOf('$');
        int idx = Math.max(dotIdx, dollarIdx);
        if (idx > 0) {
            name = name.substring(idx + 1);
        }
        return name;
    }

    record LocalArrayDeclaration(String typeStructName, String varName) {}
    private Stack<LocalArrayDeclaration> localArrayDeclarations = new Stack<>();
    private Set<String> privateAndLocalTypes = new HashSet<>();

    @Override
    public T varDeclaration(HATCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        if (varOp.isUninitialized()) {
            // Variable is uninitialized
            type(buildContext, OpTk.javaType(varOp)).space().identifier(OpTk.varName(varOp));
        } else {
            // if type is Buffer (iMappable), then we ignore it and pass it along to the methodCall
            JavaType javaType = OpTk.javaType(varOp);
            if (OpTk.isAssignable(buildContext.lookup,javaType, MappableIface.class) && javaType instanceof ClassType classType) {
                String typeName = extractClassType(buildContext, javaType, classType);
                String variableName = OpTk.varName(varOp);
                privateAndLocalTypes.add(variableName);
                localArrayDeclarations.push(new LocalArrayDeclaration(typeName, variableName));
                parencedence(buildContext, varOp, ((Op.Result)varOp.operands().getFirst()).op());
            } else {
                type(buildContext, OpTk.javaType(varOp)).space().identifier(OpTk.varName(varOp)).space().equals().space();
                parencedence(buildContext, varOp, ((Op.Result)varOp.operands().getFirst()).op());
            }
        }
        return self();
    }

    @Override
    public T varFuncDeclaration(HATCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        // append("/* skipping ").type(varFuncDeclarationOpWrapper.javaType()).append(" param declaration  */");
        return self();
    }

    @Override
    public T fieldLoad(HATCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (OpTk.isKernelContextAccess(fieldLoadOp)) {
            identifier("kc").rarrow().identifier(OpTk.fieldName(fieldLoadOp));
        } else if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) {
            Object value = OpTk.getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp);
            literal(value.toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp);
        }
        return self();
    }

    @Override
    public T fieldStore(HATCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp) {
      //  throw new IllegalStateException("What is this field store ?" + fieldStoreOp);
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
    public T unaryOperation(HATCodeBuilderContext buildContext, JavaOp.UnaryOp unaryOp) {
        symbol(unaryOp);
        parencedence(buildContext, unaryOp, ((Op.Result)unaryOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public T binaryOperation(HATCodeBuilderContext buildContext, Op binaryOp) {
        parencedence(buildContext, binaryOp, OpTk.lhsAsOp(binaryOp));
        symbol(binaryOp);
        parencedence(buildContext, binaryOp, OpTk.rhsAsOp(binaryOp));
        return self();
    }

    @Override
    public T logical(HATCodeBuilderContext buildContext, JavaOp.JavaConditionalOp logicalOp) {
        OpTk.lhsYieldOpStream(logicalOp).forEach(o ->  recurse(buildContext, o));
        space().symbol(logicalOp).space();
        OpTk.rhsYieldOpStream(logicalOp).forEach(o-> recurse(buildContext, o));
        return self();
    }

    @Override
    public T binaryTest(HATCodeBuilderContext buildContext, Op binaryTestOp) {
        parencedence(buildContext, binaryTestOp, OpTk.lhsAsOp(binaryTestOp));
        symbol(binaryTestOp);
        parencedence(buildContext, binaryTestOp, OpTk.rhsAsOp(binaryTestOp));
        return self();
    }

    @Override
    public T conv(HATCodeBuilderContext buildContext, JavaOp.ConvOp convOp) {
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(buildContext,JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type(buildContext,(JavaType)convOp.resultType()));
        }
        parencedence(buildContext, convOp, ((Op.Result)convOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public T constant(HATCodeBuilderContext buildContext, CoreOp.ConstantOp constantOp) {
        Object object = constantOp.value();
        if (object == null) {
            nullKeyword();
        } else {
            literal(constantOp.value().toString());
        }
        return self();
    }

    @Override
    public T javaYield(HATCodeBuilderContext buildContext, CoreOp.YieldOp yieldOp) {
        if (yieldOp.operands().getFirst() instanceof Op.Result result) {
            recurse(buildContext, result.op());
        }
        return self();
    }

    @Override
    public T lambda(HATCodeBuilderContext buildContext, JavaOp.LambdaOp lambdaOp) {
        return commented("/*LAMBDA*/");
    }

    @Override
    public T tuple(HATCodeBuilderContext buildContext, CoreOp.TupleOp tupleOp) {
        StreamCounter.of(tupleOp.operands(), (c, operand) -> {
            if (c.isNotFirst()) {
                comma().space();
            }
            if (operand instanceof Op.Result result) {
                recurse(buildContext, result.op());
            } else {
                commented("/*nothing to tuple*/");
            }
        });
        return self();
    }

    @Override
    public T funcCall(HATCodeBuilderContext buildContext, CoreOp.FuncCallOp funcCallOp) {
          identifier(funcCallOp.funcName());
        paren(_ -> {
            commaSeparated(funcCallOp.operands(), (e) -> {
                if (e instanceof Op.Result result) {
                    parencedence(buildContext, funcCallOp, result.op());
                } else {
                    throw new IllegalStateException("Value?");
                }
            });
        });
        return self();
    }

    @Override
    public T javaLabeled(HATCodeBuilderContext buildContext, JavaOp.LabeledOp labeledOp) {
        var labelNameOp = labeledOp.bodies().getFirst().entryBlock().ops().getFirst();
        CoreOp.ConstantOp constantOp = (CoreOp.ConstantOp) labelNameOp;
        literal(constantOp.value().toString()).colon().nl();
        var forLoopOp = labeledOp.bodies().getFirst().entryBlock().ops().get(1);
        recurse(buildContext,forLoopOp);
        return self();
    }

    public T javaBreak(HATCodeBuilderContext buildContext, JavaOp.BreakOp breakOp) {
        breakKeyword();
        if (!breakOp.operands().isEmpty() && breakOp.operands().getFirst() instanceof Op.Result result) {
            space();
            if (result.op() instanceof CoreOp.ConstantOp c) {
                literal(c.value().toString());
            }
        }
        return self();
    }

    public T javaContinue(HATCodeBuilderContext buildContext, JavaOp.ContinueOp continueOp) {
        if (!continueOp.operands().isEmpty()
                && continueOp.operands().getFirst() instanceof Op.Result result
                && result.op() instanceof CoreOp.ConstantOp c
        ) {
            continueKeyword().space().literal(c.value().toString());
        } else if (buildContext.scope.parent instanceof HATCodeBuilderContext.LoopScope<?>) {
            // nope
        } else {
            continueKeyword();
        }

        return self();
    }

    @Override
    public T javaIf(HATCodeBuilderContext buildContext, JavaOp.IfOp ifOp) {
        buildContext.scope(ifOp, () -> {
            var lastWasBody = StreamMutable.of(false);
            StreamCounter.of(ifOp.bodies(), (c, b) -> {
                if (b.yieldType() instanceof JavaType javaType && javaType == JavaType.VOID) {
                    if (OpTk.hasElse(ifOp,c.value())) { // we might have more than one else
                        if (lastWasBody.get()) {
                            elseKeyword();
                        }
                        braceNlIndented(_ ->
                                StreamCounter.of(OpTk.rootsExcludingVarFuncDeclarationsAndYields(ifOp.bodies().get(c.value()).entryBlock()), (innerc, root) ->
                                        nlIf(innerc.isNotFirst()).recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
                                )
                        );
                    }
                    lastWasBody.set(true);
                } else {
                    if (c.isNotFirst()) {
                        elseKeyword().space();
                    }

                    ifKeyword().paren(_ ->
                            ifOp.bodies().get(c.value()).entryBlock()            // get the entryblock if bodies[c.value]
                                    .ops().stream().filter(o->o instanceof CoreOp.YieldOp) // we want all the yields
                                    .forEach((yield) -> recurse(buildContext, yield))
                    );
                    lastWasBody.set(false);
                }
            });
        });
        return self();
    }

    @Override
    public T javaWhile(HATCodeBuilderContext buildContext, JavaOp.WhileOp whileOp) {
        whileKeyword().paren(_ ->
                OpTk.conditionYieldOpStream(whileOp).forEach(o -> recurse(buildContext, o))
        ).braceNlIndented(_ ->
                StreamCounter.of(OpTk.loopRootOpStream(buildContext.lookup,whileOp), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
                )
        );
        return self();
    }

    @Override
    public T javaFor(HATCodeBuilderContext buildContext, JavaOp.ForOp forOp) {
        buildContext.scope(forOp, () ->
                forKeyword().paren(_ -> {
                    OpTk.initYieldOpStream(forOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    OpTk.conditionYieldOpStream(forOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    StreamCounter.of(OpTk.mutateRootOpStream(buildContext.lookup,forOp), (c, wrapped) ->
                            commaSpaceIf(c.isNotFirst()).recurse(buildContext, wrapped)
                    );
                }).braceNlIndented(_ ->
                        StreamCounter.of(OpTk.loopRootOpStream(buildContext.lookup,forOp), (c, root) ->
                                nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
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
                        } else if (field instanceof Schema.SchemaNode.Padding padding) {
                            emitText(padding.toC99()).semicolon().nl();
                        } else {
                            throw new IllegalStateException("hmm");
                        }


                        semicolon();
                    });
                }).suffix_t(ifaceType.iface.getSimpleName()).semicolon().nl().nl();
        return self();
    }

    public T atomicInc(HATCodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        throw new IllegalStateException("atomicInc not implemented");
    }

    @Override
    public T methodCall(HATCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        var name = invokeOp.invokeDescriptor().name();//OpTk.name(invokeOp);

        if (OpTk.isIfaceBufferMethod(buildContext.lookup,invokeOp)) {
            var operandCount = invokeOp.operands().size();
            var returnType = OpTk.javaReturnType(invokeOp);

            if (operandCount == 1 && name.startsWith("atomic") && name.endsWith("Inc")
                    && returnType instanceof PrimitiveType primitiveType && primitiveType.equals(JavaType.INT)) {
                // this is a bit of a hack for atomics.
                if (invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
                    atomicInc(buildContext, instanceResult, name.substring(0, name.length() - 3));
                    //identifier("atomic_inc").paren(_ -> {
                    //    ampersand().recurse(buildContext, OpWrapper.wrap(instanceResult.op()));
                    //    rarrow().identifier(name.substring(0, name.length() - 3));
                    //});
                } else {
                    throw new IllegalStateException("bad atomic");
                }
            } else {
                if (invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
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
                    if (name.startsWith("createPrivate")) {
                        LocalArrayDeclaration declaration = localArrayDeclarations.pop();
                        String typeStruct = declaration.typeStructName;
                        suffix_t(typeStruct)
                                .space()
                                .emitText(declaration.varName).nl();
                    } else if (name.startsWith("createLocal")) {
                        LocalArrayDeclaration declaration = localArrayDeclarations.pop();
                        String varName = declaration.varName + "$shared";
                        emitCastToLocal(declaration.typeStructName, declaration.varName, varName);

                    } else {
                        if (returnType instanceof ClassType classType) {
                            ampersand();
                            /* This is way more complicated I think we need to determine the expression type.
                             * sumOfThisStage=sumOfThisStage+&left->anon->value; from    sumOfThisStage += left.anon().value();
                             */
                        }

                        // Check if the varOpLoad that could follow corresponds to a local/private type
                        boolean isLocal = false;
                        Op op = instanceResult.op();
                        if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                            CoreOp.VarOp resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
                            if (privateAndLocalTypes.contains(resolve.varName())) {
                                isLocal = true;
                            }
                        }

                        recurse(buildContext, instanceResult.op());

                        if (!isLocal) {
                            // If it is not local or private, it generates an arrow
                            rarrow();
                        } else {
                            // Otherwise, it generates a do (access members without pointers)
                            dot();
                        }
                        identifier(name);


                        //if (invokeOpWrapper.name().equals("value") || invokeOpWrapper.name().equals("anon")){
                        //System.out.println("value|anon");
                        // }
                        if (returnType instanceof PrimitiveType primitiveType && primitiveType.isVoid()) {
                            //   setter
                            switch (operandCount) {
                                case 2: {
                                    if (invokeOp.operands().get(1) instanceof Op.Result result1) {
                                        equals().recurse(buildContext, result1.op());
                                    } else {
                                        throw new IllegalStateException("How ");
                                    }
                                    break;
                                }
                                case 3: {
                                    if (invokeOp.operands().get(1) instanceof Op.Result result1
                                            && invokeOp.operands().get(2) instanceof Op.Result result2) {
                                        sbrace(_ -> recurse(buildContext, result1.op()));
                                        equals().recurse(buildContext, result2.op());
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
                            if (invokeOp.operands().size() > 1 && invokeOp.operands().get(1) instanceof Op.Result result1) {
                                sbrace(_ -> recurse(buildContext, result1.op()));
                            } else {
                                // This is a simple usage.   So scaleTable->multiScaleAccumRange
                            }
                        }
                    }
                } else {
                    throw new IllegalStateException("[Illegal] Expected a parameter for the InvokOpWrapper Node");
                }
            }
        } else {
            // Detect well-known constructs
            if (name.equals("barrier")) {
                List<Value> operands = invokeOp.operands();
                for (Value value : operands) {
                    if (value instanceof Op.Result instanceResult) {
                        FunctionType functionType = instanceResult.op().opType();
                        // if it is a barrier from the kernel context, then we generate
                        // a local barrier.
                        if (functionType.returnType().toString().equals("hat.KernelContext")) {
                            syncBlockThreads();
                        }
                    }
                }
            } else {
                // General case
                identifier(name).paren(_ ->
                        commaSeparated(invokeOp.operands(), (op) -> {
                            if (op instanceof Op.Result result) {
                                recurse(buildContext, result.op());
                            } else {
                                throw new IllegalStateException("wtf?");
                            }
                        })
                );
            }
        }
        return self();
    }

    private Integer obtainSize(Value parameter) {
        if (parameter instanceof Op.Result opResult) {
            if (opResult.op() instanceof CoreOp.ConstantOp constantOp) {
                if (constantOp.value() instanceof Integer intValue) {
                    return intValue;
                }
            }
        }
        return null;
    }

    public abstract T emitCastToLocal(String typeName, String varName, String localVarS);

    public abstract T syncBlockThreads();

    @Override
    public T ternary(HATCodeBuilderContext buildContext, JavaOp.ConditionalExpressionOp ternaryOp) {
        OpTk.conditionYieldOpStream(ternaryOp).forEach(o -> recurse(buildContext, o));
        questionMark();
        OpTk.thenYieldOpStream(ternaryOp).forEach(o -> recurse(buildContext, o));
        colon();
        OpTk.elseYieldOpStream(ternaryOp).forEach(o -> recurse(buildContext, o));
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
  /*  @Override
    public T parencedence(HATCodeBuilderContext buildContext, Op parent, OpWrapper<?> child) {
        return parenWhen(precedenceOf(parent) < precedenceOf(child.op), _ -> recurse(buildContext, child));
    }

    @Override
    public T parencedence(HATCodeBuilderContext buildContext, OpWrapper<?> parent, OpWrapper<?> child) {
        return parenWhen(precedenceOf(parent.op) < precedenceOf(child.op), _ -> recurse(buildContext, child));
    } */

    @Override
    public T parencedence(HATCodeBuilderContext buildContext, Op parent, Op child) {
        return parenWhen(precedenceOf(parent) < precedenceOf(child), _ -> recurse(buildContext, child));
    }

  /*  @Override
    public T parencedence(HATCodeBuilderContext buildContext, OpWrapper<?> parent, Op child) {
        return parenWhen(precedenceOf(parent.op) < precedenceOf(child), _ -> recurse(buildContext, child));
    } */


    @Override
    public T ret(HATCodeBuilderContext buildContext, CoreOp.ReturnOp returnOp) {
        returnKeyword();
        if (!returnOp.operands().isEmpty()) {
            space().parencedence(buildContext, returnOp, ((Op.Result)returnOp.operands().getFirst()).op());
        }
        return self();
    }
}
