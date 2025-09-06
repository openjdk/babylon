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
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

public abstract class HATCodeBuilderWithContext<T extends HATCodeBuilderWithContext<T>> extends HATCodeBuilder<T> implements HATCodeBuilder.CodeBuilderInterface<T> {

    public T typedefStructOrUnion(MemoryLayout memoryLayout, String name) {
        return typedefKeyword().space().structOrUnion(memoryLayout).space().suffix_s(name);
    }

    T structOrUnion(MemoryLayout memoryLayout) {
        return structOrUnion(memoryLayout instanceof StructLayout);
    }


    public T type(HATCodeBuilderContext buildContext, JavaType javaType) {
        if (OpTk.isAssignable(buildContext.lookup, javaType, MappableIface.class)
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
        parenthesisIfNeeded(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }


    record IfaceStruct(ClassType classType){
        String name(){
            String name = classType.toClassName();
            int dotIdx = name.lastIndexOf('.');
            int dollarIdx = name.lastIndexOf('$');
            int idx = Math.max(dotIdx, dollarIdx);
            if (idx > 0) {
                name = name.substring(idx + 1);
            }
            return name;
        }
    }
    record LocalArrayDeclaration(IfaceStruct ifaceStruct, CoreOp.VarOp varOp) {}
    private final Stack<LocalArrayDeclaration> localArrayDeclarations = new Stack<>();
    private final Set<CoreOp.VarOp> localDataStructures = new HashSet<>();

    private boolean isMappableIFace(HATCodeBuilderContext buildContext, JavaType javaType) {
        return (OpTk.isAssignable(buildContext.lookup,javaType, MappableIface.class));
    }

    private void annotateTypeAndName( ClassType classType, CoreOp.VarOp varOp) {

        localArrayDeclarations.push(new LocalArrayDeclaration(new IfaceStruct(classType), varOp));
    }

    private void varDeclarationWithInitialization(HATCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        type(buildContext, (JavaType) varOp.varValueType()).space().identifier(varOp.varName()).space().equals().space();
        if (isMappableIFace(buildContext, (JavaType) varOp.varValueType()) && (JavaType) varOp.varValueType() instanceof ClassType classType) {
            annotateTypeAndName( classType, varOp);
        }
        parenthesisIfNeeded(buildContext, varOp, ((Op.Result)varOp.operands().getFirst()).op());
    }

    @Override
    public T varDeclaration(HATCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        if (varOp.isUninitialized()) {
            type(buildContext, (JavaType) varOp.varValueType()).space().identifier(varOp.varName());
        } else {
            varDeclarationWithInitialization(buildContext, varOp);
        }
        return self();
    }

    @Override
    public T varFuncDeclaration(HATCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
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
        return self();
    }



    @Override
    public T unaryOperation(HATCodeBuilderContext buildContext, JavaOp.UnaryOp unaryOp) {
        symbol(unaryOp).parenthesisIfNeeded(buildContext, unaryOp, ((Op.Result)unaryOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public T binaryOperation(HATCodeBuilderContext buildContext, JavaOp.BinaryOp binaryOp) {
        parenthesisIfNeeded(buildContext, binaryOp, OpTk.lhsResult(binaryOp).op());
        symbol(binaryOp);
        parenthesisIfNeeded(buildContext, binaryOp, OpTk.rhsResult(binaryOp).op());
        return self();
    }


    public static List<Op> ops(JavaOp.JavaConditionalOp javaConditionalOp, int idx){
        return javaConditionalOp.bodies().get(idx).entryBlock().ops();
    }
    public static List<Op> lhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
        return ops(javaConditionalOp,0);
    }
    public static List<Op> rhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
        return ops(javaConditionalOp,1);
    }
    @Override
    public T logical(HATCodeBuilderContext buildContext, JavaOp.JavaConditionalOp logicalOp) {
        lhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o ->  recurse(buildContext, o));
        space().symbol(logicalOp).space();
        rhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o-> recurse(buildContext, o));
        return self();
    }
    public static Op.Result result(JavaOp.BinaryTestOp binaryTestOp, int idx){
        return (Op.Result)binaryTestOp.operands().get(idx);
    }
    public static Op.Result lhsResult(JavaOp.BinaryTestOp binaryTestOp){
        return result(binaryTestOp,0);
    }
    public static Op.Result rhsResult(JavaOp.BinaryTestOp binaryTestOp){
        return result(binaryTestOp,1);
    }
    @Override
    public T binaryTest(HATCodeBuilderContext buildContext, JavaOp.BinaryTestOp binaryTestOp) {
        parenthesisIfNeeded(buildContext, binaryTestOp, lhsResult(binaryTestOp).op());
        symbol(binaryTestOp);
        parenthesisIfNeeded(buildContext, binaryTestOp, rhsResult(binaryTestOp).op());
        return self();
    }
    public static Op.Result result(JavaOp.ConvOp convOp){
        return (Op.Result)convOp.operands().getFirst();
    }
    @Override
    public T conv(HATCodeBuilderContext buildContext, JavaOp.ConvOp convOp) {
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(buildContext,JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type(buildContext,(JavaType)convOp.resultType()));
        }
        parenthesisIfNeeded(buildContext, convOp, result(convOp).op());
        return self();
    }

    @Override
    public T constant(HATCodeBuilderContext buildContext, CoreOp.ConstantOp constantOp) {
        if (constantOp.value() == null) {
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
        paren(_ ->
            commaSeparated(funcCallOp.operands(), (e) -> {
                if (e instanceof Op.Result result) {
                    parenthesisIfNeeded(buildContext, funcCallOp, result.op());
                } else {
                    throw new IllegalStateException("Value?");
                }
            })
        );
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
                    int idx = c.value();
                    if (ifOp.bodies().size() > idx && ifOp.bodies().get(idx).entryBlock().ops().size() > 1){
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
                whileOp.bodies().getFirst().entryBlock().ops().stream() // cond
                        .filter(o -> o instanceof CoreOp.YieldOp)
                        .forEach(o -> recurse(buildContext, o))
        ).braceNlIndented(_ ->
                StreamCounter.of(OpTk.loopRootOpStream(whileOp), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
                )
        );
        return self();
    }

    @Override
    public T javaFor(HATCodeBuilderContext buildContext, JavaOp.ForOp forOp) {
        buildContext.scope(forOp, () ->
                forKeyword().paren(_ -> {
                    forOp.init().entryBlock().ops().stream()
                            .filter(o -> o instanceof CoreOp.YieldOp)
                            .forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    forOp.cond().entryBlock().ops().stream()
                            .filter(o -> o instanceof CoreOp.YieldOp)
                            .forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    StreamCounter.of(
                            OpTk.rootsExcludingVarFuncDeclarationsAndYields( forOp.bodies().get(2).entryBlock()) //mutate block
                            , (c, op) -> commaSpaceIf(c.isNotFirst()).recurse(buildContext, op)
                    );
                }).braceNlIndented(_ ->
                        StreamCounter.of(OpTk.loopRootOpStream(forOp), (c, root) ->
                                nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root))
                        )
                )
        );
        return self();
    }


    public T typedef(BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        typedefKeyword().space().structOrUnion(ifaceType instanceof Schema.IfaceType.Struct)
                .space().suffix_s(ifaceType.iface.getSimpleName()).braceNlIndented(_ -> {
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
        if (OpTk.isIfaceBufferMethod(buildContext.lookup, invokeOp)) {
          //  var returnType = OpTk.javaReturnType(invokeOp);

            if (invokeOp.operands().size() == 1 && invokeOp.invokeDescriptor().name().startsWith("atomic") && invokeOp.invokeDescriptor().name().endsWith("Inc")
                    && OpTk.javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.equals(JavaType.INT)) {
                // this is a bit of a hack for atomics.
                if (invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
                    atomicInc(buildContext, instanceResult, invokeOp.invokeDescriptor().name().substring(0, invokeOp.invokeDescriptor().name().length() - 3));
                } else {
                    throw new IllegalStateException("bad atomic");
                }
            } else {
                if (invokeOp.invokeDescriptor().name().equals("create")) { // TODO:  only on iface buffers
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
                                privateDeclaration(declaration.ifaceStruct.name(), declaration.varOp);
                            } else if (spaceName.equals(Space.SHARED.name())) {
                                localDeclaration(declaration.ifaceStruct.name(), declaration.varOp);
                            }
                        }
                    }
                } else if (invokeOp.invokeDescriptor().name().equals("createLocal")) { // TODO:  only on kernel iface buffers
                    LocalArrayDeclaration declaration = localArrayDeclarations.pop();
                    localDeclaration(declaration.ifaceStruct.name(), declaration.varOp);
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
                    if (OpTk.javaReturnType(invokeOp) instanceof ClassType) {
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

                    identifier(invokeOp.invokeDescriptor().name());

                    if (OpTk.javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid()) {
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
                        if (invokeOp.operands().size() > 1 && invokeOp.operands().get(1) instanceof Op.Result result1) {
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
            if (invokeOp.invokeDescriptor().name().equals("barrier")) { // TODO:  only on kernel context?
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
                identifier(invokeOp.invokeDescriptor().name()).paren(_ ->
                        commaSeparated(invokeOp.operands(), (op) -> {
                            if (op instanceof Op.Result result) {
                                recurse(buildContext, result.op());
                            }
                        })
                );
            }
        }
        return self();
    }

    public abstract T privateDeclaration(String typeName, CoreOp.VarOp varOp);

    public abstract T localDeclaration(String typeName, CoreOp.VarOp varOp);

    public abstract T syncBlockThreads();



    @Override
    public T ternary(HATCodeBuilderContext buildContext, JavaOp.ConditionalExpressionOp ternaryOp) {
        ternaryOp.bodies().get(0).entryBlock().ops().stream()
                .filter(o -> o instanceof CoreOp.YieldOp) // cond
                .forEach(o -> recurse(buildContext, o));
        questionMark();
        ternaryOp.bodies().get(1).entryBlock().ops().stream()
                .filter(o -> o instanceof CoreOp.YieldOp) // iff yield
                .forEach(o -> recurse(buildContext, o));
        colon();
        ternaryOp.bodies().get(2).entryBlock().ops().stream()
                 .filter(o -> o instanceof CoreOp.YieldOp) // else yield
                 .forEach(o -> recurse(buildContext, o));
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
    public T parenthesisIfNeeded(HATCodeBuilderContext buildContext, Op parent, Op child) {
        return parenWhen(OpTk.needsParenthesis(parent,child), _ -> recurse(buildContext, child));
    }

    public static Op.Result result( CoreOp.ReturnOp returnOp){
       return (Op.Result)returnOp.operands().getFirst();
    }

    @Override
    public T ret(HATCodeBuilderContext buildContext, CoreOp.ReturnOp returnOp) {
        returnKeyword().when(!returnOp.operands().isEmpty(),
                        $-> $.space().parenthesisIfNeeded(buildContext, returnOp, result(returnOp).op())
                );
        return self();
    }
}
