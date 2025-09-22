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


import hat.Space;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import hat.util.StreamMutable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;

import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
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
        CoreOp.VarOp varOp = buildContext.scope.resolve(varLoadOp.operands().getFirst());
        varName(varOp);
        return self();
    }

    @Override
    public T varStoreOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        CoreOp.VarOp varOp = buildContext.scope.resolve(varStoreOp.operands().getFirst());
        varName(varOp).equals();
        parenthesisIfNeeded(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }



    public record LocalArrayDeclaration(ClassType classType, CoreOp.VarOp varOp) {}
    private final Stack<LocalArrayDeclaration> localArrayDeclarations = new Stack<>();
    private final Set<CoreOp.VarOp> localDataStructures = new HashSet<>();

    private boolean isMappableIFace(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        return (OpTk.isAssignable(buildContext.lookup,javaType, MappableIface.class));
    }

    private void annotateTypeAndName( ClassType classType, CoreOp.VarOp varOp) {
        localArrayDeclarations.push(new LocalArrayDeclaration(classType, varOp));
    }

    private void varDeclarationWithInitialization(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        type(buildContext, (JavaType) varOp.varValueType()).space().varName(varOp).space().equals().space();
        if (isMappableIFace(buildContext, (JavaType) varOp.varValueType()) && (JavaType) varOp.varValueType() instanceof ClassType classType) {
            annotateTypeAndName( classType, varOp);
        }
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
    public T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp, OpTk.ParamVar paramVar) {
        return self();
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (OpTk.isKernelContextAccess(fieldLoadOp)) {
            identifier("kc").rarrow().fieldName(fieldLoadOp);
        } else if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) {
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
        oparen();
        parenthesisIfNeeded(buildContext, binaryOp, OpTk.lhsResult(binaryOp).op());
        symbol(binaryOp);
        parenthesisIfNeeded(buildContext, binaryOp, OpTk.rhsResult(binaryOp).op());
        cparen();
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
        oparen();
        parenthesisIfNeeded(buildContext, binaryTestOp, OpTk.lhsResult(binaryTestOp).op());
        symbol(binaryTestOp);
        parenthesisIfNeeded(buildContext, binaryTestOp, OpTk.rhsResult(binaryTestOp).op());
        cparen();
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
                if (OpTk.funcName(invokeOp).equals("create")) {
                    // If we decide to keep the version in which we pass the enum with the memory space
                    // to allocate a particular data structure (E.g., shared, or private)

                    // Obtain the space in the first parameter
                    List<Value> operands = invokeOp.operands();
                    if (operands.size() != 1) {
                        throw new RuntimeException("[Fail] `create` method expects one parameter for the space");
                    }
                    Value spaceValue = operands.getFirst();
                    if (spaceValue instanceof Op.Result instanceResult) {
                        if (instanceResult.op() instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp ) {
                            // check type of field load
                            TypeElement typeElement = fieldLoadOp.fieldDescriptor().refType();
                            if (typeElement instanceof ClassType classType) {
                                if (!classType.toClassName().equals(Space.class.getCanonicalName())) {
                                    throw new RuntimeException("[Fail] Expected an instance from Space");
                                }
                            }

                            // If the type is correct, then we obtain the enum value and invoke the
                            // corresponding declaration
                            String spaceName = fieldLoadOp.fieldDescriptor().name();
                            LocalArrayDeclaration declaration = localArrayDeclarations.pop();
                            if (spaceName.equals(Space.PRIVATE.name())) {
                                privateDeclaration(declaration);
                            } else if (spaceName.equals(Space.SHARED.name())) {
                                localDeclaration(declaration);
                            }
                        }
                    }
                } else if (OpTk.funcName(invokeOp).equals("createLocal")) {
                    LocalArrayDeclaration declaration = localArrayDeclarations.pop();
                    localDeclaration(declaration);
                    localDataStructures.add(declaration.varOp);
                } else if (invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
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
                    boolean isLocal = false;
                    if (instanceResult.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        CoreOp.VarOp resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
                        if (localDataStructures.contains(resolve)) {
                            isLocal = true;
                        }
                    }

                    either(isLocal, CodeBuilder::dot, CodeBuilder::rarrow);

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
            // Detect well-known constructs

            if (OpTk.funcName(invokeOp).equals("barrier")) { // TODO:  only on kernel context?
                List<Value> operands = invokeOp.operands(); // map to Result and use stream filter and  find
                for (Value value : operands) {
                    if (value instanceof Op.Result instanceResult) {
                        FunctionType functionType = instanceResult.op().opType();
                        // if it is a barrier from the kernel context, then we generate
                        // a local barrier.
                        if (functionType.returnType().toString().equals("hat.KernelContext")) {  // OpTk.isAssignable?
                            syncBlockThreads();
                        }
                    }
                }
            } else {
                // General case
                funcName(invokeOp).paren(_ ->
                        separated(invokeOp.operands(), ($)->$.comma().space(), (op) -> {
                            if (op instanceof Op.Result result) {
                                recurse(buildContext, result.op());
                            }
                        })
                );
            }
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
                case JavaOp.LabeledOp _ ->false;
                case JavaOp.YieldOp _ ->false;
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
        return   type(buildContext,(JavaType) param.parameter.type()).space().varName(param.varOp);
    }
}
