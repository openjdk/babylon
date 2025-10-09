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

import hat.dialect.HatBarrierOp;
import hat.dialect.HatVSelectLoadOp;
import hat.dialect.HatVSelectStoreOp;
import hat.dialect.HatVectorBinaryOp;
import hat.dialect.HatVectorLoadOp;
import hat.dialect.HatVectorStoreView;
import hat.dialect.HatLocalVarOp;
import hat.dialect.HatMemoryOp;
import hat.dialect.HatPrivateVarOp;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import hat.util.StreamMutable;

import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

public abstract class HATCodeBuilderWithContext<T extends HATCodeBuilderWithContext<T>> extends HATCodeBuilder<T> implements BabylonOpBuilder<T> {

    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        if (OpTk.isAssignable(buildContext.lookup, javaType, MappableIface.class)
                        && javaType instanceof ClassType classType) {
            suffix_t(classType).asterisk();
        } else {
            typeName(javaType.toBasicType().toString());
        }
        return self();
    }

    @Override
    public T varLoadOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
        switch (resolve) {
            case CoreOp.VarOp $ -> varName($);
            case HatMemoryOp $ -> varName($);
            case HatVectorLoadOp $ -> varName($);
            case HatVectorBinaryOp $ -> varName($);
            case null, default -> {
            }
        }
        return self();
    }

    @Override
    public T varStoreOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        CoreOp.VarOp varOp = (CoreOp.VarOp) buildContext.scope.resolve(varStoreOp.operands().getFirst());
        varName(varOp).equals();
        parenthesisIfNeeded(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }

    public record LocalArrayDeclaration(ClassType classType, HatMemoryOp varOp) {}

    private void varDeclarationWithInitialization(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        if (buildContext.isVarOpFinal(varOp)) {
            constKeyword().space();
        }
        type(buildContext, (JavaType) varOp.varValueType()).space().varName(varOp).space().equals().space();
        parenthesisIfNeeded(buildContext, varOp, ((Op.Result)varOp.operands().getFirst()).op());
    }

    @Override
    public T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        if (varOp.isUninitialized()) {
            type(buildContext, (JavaType) varOp.varValueType()).space().varName(varOp);
        } else {
            varDeclarationWithInitialization(buildContext, varOp);
        }
        return self();
    }

    @Override
    public T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HatLocalVarOp hatLocalVarOp) {
        LocalArrayDeclaration localArrayDeclaration = new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp);
        localDeclaration(localArrayDeclaration);
        return self();
    }

    @Override
    public T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HatPrivateVarOp hatLocalVarOp) {
        LocalArrayDeclaration localArrayDeclaration = new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp);
        privateDeclaration(localArrayDeclaration);
        return self();
    }

    @Override
    public T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp, OpTk.ParamVar paramVar) {
        return self();
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) {
            Object value = OpTk.getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp);
            literal(value.toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp);
        }
        return self();
    }

    @Override
    public T fieldStoreOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp) {
        return self();
    }


    @Override
    public T unaryOp(ScopedCodeBuilderContext buildContext, JavaOp.UnaryOp unaryOp) {
        symbol(unaryOp).parenthesisIfNeeded(buildContext, unaryOp, ((Op.Result)unaryOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public T binaryOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryOp binaryOp) {
        parenthesisIfNeeded(buildContext, binaryOp, OpTk.lhsResult(binaryOp).op());
        symbol(binaryOp);
        parenthesisIfNeeded(buildContext, binaryOp, OpTk.rhsResult(binaryOp).op());
        return self();
    }


    @Override
    public T conditionalOp(ScopedCodeBuilderContext buildContext, JavaOp.JavaConditionalOp logicalOp) {
        OpTk.lhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o ->  recurse(buildContext, o));
        space().symbol(logicalOp).space();
        OpTk.rhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o-> recurse(buildContext, o));
        return self();
    }

    @Override
    public T binaryTestOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryTestOp binaryTestOp) {
        parenthesisIfNeeded(buildContext, binaryTestOp, OpTk.lhsResult(binaryTestOp).op());
        symbol(binaryTestOp);
        parenthesisIfNeeded(buildContext, binaryTestOp, OpTk.rhsResult(binaryTestOp).op());
        return self();
    }

    @Override
    public T convOp(ScopedCodeBuilderContext buildContext, JavaOp.ConvOp convOp) {
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(buildContext,JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type(buildContext,(JavaType)convOp.resultType()));
        }
        parenthesisIfNeeded(buildContext, convOp, OpTk.result(convOp).op());
        return self();
    }

    @Override
    public T constantOp(ScopedCodeBuilderContext buildContext, CoreOp.ConstantOp constantOp) {
        if (constantOp.value() == null) {
            nullConst();
        } else {
            literal(constantOp.value().toString());
        }
        return self();
    }

    @Override
    public T yieldOp(ScopedCodeBuilderContext buildContext, CoreOp.YieldOp yieldOp) {
        if (yieldOp.operands().getFirst() instanceof Op.Result result) {
            recurse(buildContext, result.op());
        }
        return self();
    }

    @Override
    public T lambdaOp(ScopedCodeBuilderContext buildContext, JavaOp.LambdaOp lambdaOp) {
        return comment("/*LAMBDA*/");
    }

    @Override
    public T tupleOp(ScopedCodeBuilderContext buildContext, CoreOp.TupleOp tupleOp) {
        separated(tupleOp.operands(),(_)->commaSpace(),operand->{
            if (operand instanceof Op.Result result) {
                recurse(buildContext, result.op());
            } else {
                comment("/*nothing to tuple*/");
            }
        });
        return self();
    }

    @Override
    public T funcCallOp(ScopedCodeBuilderContext buildContext, CoreOp.FuncCallOp funcCallOp) {
        funcName(funcCallOp);
        paren(_ ->
            separated(funcCallOp.operands().stream()
                    .filter(e->e instanceof Op.Result ).map(e->(Op.Result)e),(_)->commaSpace(), result ->
                     recurse(buildContext,result.op())
            )
        );
        return self();
    }

    @Override
    public T labeledOp(ScopedCodeBuilderContext buildContext, JavaOp.LabeledOp labeledOp) {
        var labelNameOp = labeledOp.bodies().getFirst().entryBlock().ops().getFirst();
        CoreOp.ConstantOp constantOp = (CoreOp.ConstantOp) labelNameOp;
        literal(constantOp.value().toString()).colon().nl();
        var forLoopOp = labeledOp.bodies().getFirst().entryBlock().ops().get(1);
        recurse(buildContext,forLoopOp);
        return self();
    }

    public T breakOp(ScopedCodeBuilderContext buildContext, JavaOp.BreakOp breakOp) {
        breakKeyword();
        if (!breakOp.operands().isEmpty() && breakOp.operands().getFirst() instanceof Op.Result result) {
            space();
            if (result.op() instanceof CoreOp.ConstantOp c) {
                literal(c.value().toString());
            }
        }
        return self();
    }

    public T continueOp(ScopedCodeBuilderContext buildContext, JavaOp.ContinueOp continueOp) {
        if (!continueOp.operands().isEmpty()
                && continueOp.operands().getFirst() instanceof Op.Result result
                && result.op() instanceof CoreOp.ConstantOp c
        ) {
            continueKeyword().space().literal(c.value().toString());
        } else if (buildContext.scope.parent instanceof ScopedCodeBuilderContext.ForScope) {
            // nope
        } else {
            continueKeyword();
        }

        return self();
    }

    @Override
    public T ifOp(ScopedCodeBuilderContext buildContext, JavaOp.IfOp ifOp) {
        buildContext.ifScope(ifOp, () -> {
            var lastWasBody = StreamMutable.of(false);
            var i = StreamMutable.of(0);
            // We probably should just use a regular for loop here ;)
            ifOp.bodies().forEach(b->{
                int idx = i.get();
                if (b.yieldType() instanceof JavaType javaType && javaType == JavaType.VOID) {
                    if (ifOp.bodies().size() > idx && ifOp.bodies().get(idx).entryBlock().ops().size() > 1){
                        if (lastWasBody.get()) {
                            elseKeyword();
                        }
                        braceNlIndented(_ ->
                                separated(OpTk.statements(ifOp.bodies().get(idx).entryBlock()),(_)->nl(), root->
                                        statement(buildContext,root)
                                )
                        );
                    }
                    lastWasBody.set(true);
                } else {
                    if (idx>0) {
                        elseKeyword().space();
                    }
                    ifKeyword().paren(_ ->
                            ifOp.bodies().get(idx).entryBlock()            // get the entryblock if bodies[c.value]
                                    .ops().stream().filter(o->o instanceof CoreOp.YieldOp) // we want all the yields
                                    .forEach((yield) -> recurse(buildContext, yield))
                    );
                    lastWasBody.set(false);
                }
                i.set(i.get()+1);
            });
        });
        return self();
    }

    @Override
    public T whileOp(ScopedCodeBuilderContext buildContext, JavaOp.WhileOp whileOp) {
        whileKeyword().paren(_ ->
                OpTk.condBlock(whileOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp)
                        .forEach(o -> recurse(buildContext, o))
        );
        braceNlIndented(_ ->
                separated(OpTk.statements(whileOp),(_)->nl(), statement->statement(buildContext,statement)
                       // recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
                )
        );
        return self();
    }

    @Override
    public T forOp(ScopedCodeBuilderContext buildContext, JavaOp.ForOp forOp) {
        buildContext.forScope(forOp, () ->
                forKeyword().paren(_ -> {
                    OpTk.initBlock(forOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    OpTk.condBlock(forOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    separated(OpTk.statements(OpTk.mutateBlock(forOp)), (_)->commaSpace(),
                            op -> recurse(buildContext, op)
                    );
                }).braceNlIndented(_ ->
                        separated(OpTk.statements(forOp), (_)->nl(),statement ->statement(buildContext,statement)
                              //  root-> recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
                        )
                )
        );
        return self();
    }


    public T typedef(BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        typedefKeyword().space().structOrUnion(ifaceType instanceof Schema.IfaceType.Struct)
                .space().suffix_s(ifaceType.iface.getSimpleName()).braceNlIndented(_ -> {
                    int fieldCount = ifaceType.fields.size();
                    var fieldIdx = StreamMutable.of(0);
                    separated(ifaceType.fields,(_)->nl(), field->{
                        boolean isLast =fieldIdx.get() == fieldCount - 1;
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
                        fieldIdx.set(fieldIdx.get()+1);
                    });
                }).suffix_t(ifaceType.iface.getSimpleName()).semicolon().nl().nl();
        return self();
    }

    public T atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        throw new IllegalStateException("atomicInc not implemented");
    }

    @Override
    public T barrier(ScopedCodeBuilderContext buildContext, HatBarrierOp barrierOp) {
        return syncBlockThreads();
    }

    @Override
    public T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        if (OpTk.isIfaceBufferMethod(buildContext.lookup, invokeOp)) {
            if (invokeOp.operands().size() == 1
                    && OpTk.funcName(invokeOp) instanceof String funcName
                    && funcName.startsWith("atomic")
                    && funcName.endsWith("Inc")
                    && OpTk.javaReturnType(invokeOp).equals(JavaType.INT)) {
                // this is a bit of a hack for atomics.
                if (invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
                    atomicInc(buildContext, instanceResult, funcName.substring(0, funcName.length() - "Inc".length()));
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
                    if (OpTk.javaReturnType(invokeOp) instanceof ClassType) { // isAssignable?
                        ampersand();
                        /* This is way more complicated I think we need to determine the expression type.
                         * sumOfThisStage=sumOfThisStage+&left->anon->value; from    sumOfThisStage += left.anon().value();
                         */
                    }

                    recurse(buildContext, instanceResult.op());

                    // Check if the varOpLoad that could follow corresponds to a local/private type
                    boolean isLocalOrPrivateDS = false;
                    if (instanceResult.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        Op resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
                        //if (localDataStructures.contains(resolve)) {
                        if (resolve instanceof HatMemoryOp) {
                            isLocalOrPrivateDS = true;
                        }
                    }

                    either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

                    funcName(invokeOp);

                    if (OpTk.javaReturnTypeIsVoid(invokeOp)) {
                        //   setter
                        switch (invokeOp.operands().size()) {
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
                        if (OpTk.resultOrNull(invokeOp,1) instanceof Op.Result result1) {
                            sbrace(_ -> recurse(buildContext, result1.op()));
                        } else {
                            // This is a simple usage.   So scaleTable->multiScaleAccumRange
                        }
                    }
                } else {
                    throw new IllegalStateException("[Illegal] Expected a parameter for the InvokOpWrapper Node");
                }
            }
        } else {
            // General case
            funcName(invokeOp).paren(_ ->
                    separated(invokeOp.operands(), ($) -> $.comma().space(), (op) -> {
                        if (op instanceof Op.Result result) {
                            recurse(buildContext, result.op());
                        }
                    })
            );
        }
        return self();
    }



    public abstract T privateDeclaration(LocalArrayDeclaration localArrayDeclaration);

    public abstract T localDeclaration(LocalArrayDeclaration localArrayDeclaration);

    public abstract T syncBlockThreads();

    @Override
    public T conditionalExpressionOp(ScopedCodeBuilderContext buildContext, JavaOp.ConditionalExpressionOp ternaryOp) {
        OpTk.condBlock(ternaryOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        questionMark();
        OpTk.thenBlock(ternaryOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        colon();
        OpTk.elseBlock(ternaryOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        return self();
    }

    /**
     * Wrap paren() of precedence of op is higher than parent.
     *
     * @param buildContext
     * @param parent
     * @param child
     */
    @Override
    public T parenthesisIfNeeded(ScopedCodeBuilderContext buildContext, Op parent, Op child) {
        return parenWhen(OpTk.needsParenthesis(parent,child), _ -> recurse(buildContext, child));
    }

    @Override
    public T returnOp(ScopedCodeBuilderContext buildContext, CoreOp.ReturnOp returnOp) {
        returnKeyword().when(!returnOp.operands().isEmpty(),
                        $-> $.space().parenthesisIfNeeded(buildContext, returnOp, OpTk.result(returnOp).op())
                );
        return self();
    }

    public T statement(ScopedCodeBuilderContext buildContext,Op op) {
        recurse(buildContext, op);
        if (switch (op){
                case JavaOp.ForOp _ -> false;
                case JavaOp.WhileOp _ -> false;
                case JavaOp.IfOp _ -> false;
                case JavaOp.LabeledOp _ -> false;
                case JavaOp.YieldOp _ -> false;
                case CoreOp.TupleOp _ ->false;
                default -> true;
            }
        ){
            semicolon();
        }
        return self();
    }

    public T suffix_t(ClassType type){
        String name = type.toClassName();
        int dotIdx = name.lastIndexOf('.');
        int dollarIdx = name.lastIndexOf('$');
        int idx = Math.max(dotIdx, dollarIdx);
        if (idx > 0) {
            name = name.substring(idx + 1);
        }
        return suffix_t(name);
    }

    public T declareParam(ScopedCodeBuilderContext buildContext, FuncOpParams.Info param){
        return  type(buildContext,(JavaType) param.parameter.type()).space().varName(param.varOp);
    }

    @Override
    public T hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {

        Value dest = hatVectorStoreView.operands().get(0);
        Value vector = hatVectorStoreView.operands().get(1);
        Value index = hatVectorStoreView.operands().get(2);

        // emitText("vstore4(vC, 0, &c->array[index * 4])");

        emitText("vstore" + hatVectorStoreView.storeN())
                .oparen()
                .varName(hatVectorStoreView)
                .comma()
                .space()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (dest instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        rarrow().emitText("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen();

        return self();
    }

    @Override
    public T hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {
        //emitText("float4 vC = vA + vB").semicolon().nl();

        typeName("float4")
                .space()
                .varName(hatVectorBinaryOp)
                .space().equals().space();

        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        if (op1 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        emitText(hatVectorBinaryOp.operationType().symbol()).space();

        if (op2 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        return self();
    }

    @Override
    public T hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {

        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        typeName(hatVectorLoadOp.buildType())
                .space()
                .varName(hatVectorLoadOp)
                .space().equals().space()
                .emitText("vload" + hatVectorLoadOp.loadN())
                .oparen()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (source instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        rarrow().emitText("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen();

        return self();
    }

    @Override
    public T hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        identifier(hatVSelectLoadOp.varName())
                .dot()
                .emitText(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public T hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        identifier(hatVSelectStoreOp.varName())
                .dot()
                .emitText(hatVSelectStoreOp.mapLane())
                .space().equals().space();
        if (hatVSelectStoreOp.resultValue() != null) {
            // We have detected a direct resolved result (resolved name)
            varName(hatVSelectStoreOp.resultValue());
        } else {
            // otherwise, we traverse to resolve the expression
            Value storeValue = hatVSelectStoreOp.operands().get(1);
            if (storeValue instanceof Op.Result r) {
                recurse(buildContext, r.op());
            }
        }
        return self();
    }
}
