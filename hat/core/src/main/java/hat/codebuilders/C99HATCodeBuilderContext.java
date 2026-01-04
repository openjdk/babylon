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

import hat.dialect.HATF16Op;
import hat.dialect.HATVectorOp;
import hat.types.HAType;
import hat.device.DeviceType;
import hat.dialect.HATMemoryVarOp;
import optkl.FieldAccess;
import optkl.FuncOpParams;
import optkl.Invoke;
import optkl.OpTkl;
import optkl.ParamVar;
import optkl.ifacemapper.MappableIface;
import optkl.util.ops.Precedence;
import optkl.util.Regex;
import optkl.util.StreamMutable;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.codebuilders.BabylonCoreOpBuilder;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

import static optkl.FieldAccess.fieldAccessOpHelper;
import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.condBlock;
import static optkl.OpTkl.elseBlock;
import static optkl.OpTkl.initBlock;
import static optkl.OpTkl.lhsOps;
import static optkl.OpTkl.lhsResult;
import static optkl.OpTkl.updateBlock;
import static optkl.OpTkl.result;
import static optkl.OpTkl.rhsOps;
import static optkl.OpTkl.rhsResult;
import static optkl.OpTkl.thenBlock;

public abstract class C99HATCodeBuilderContext<T extends C99HATCodeBuilderContext<T>> extends C99HATCodeBuilder<T>
        implements BabylonCoreOpBuilder<T, ScopedCodeBuilderContext> {


    @Override
    public final T varLoadOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
        switch (resolve) {
            case CoreOp.VarOp $ -> varName($);
            case HATMemoryVarOp $ -> varName($);
            case HATVectorOp.HATVectorVarOp $ -> varName($);
            case HATVectorOp.HATVectorLoadOp $ -> varName($);
            case HATVectorOp.HATVectorBinaryOp $ -> varName($);
            case HATF16Op.HATF16VarOp $ -> varName($);
            case null, default -> {
            }
        }
        return self();
    }

    @Override
    public final T varStoreOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        Op op = buildContext.scope.resolve(varStoreOp.operands().getFirst());

        //TODO see if VarLikeOp marker interface fixes this

        // TODO: each of these is delegating to varName().... maybe varName should be handling these types.

        // When the op is intended to operate as VarOp, then we need to include it in the following switch.
        // This is because HAT has its own dialect, and some of the Ops operate on HAT Types (not included in the Java
        // dialect). For instance, private data structures, local data structures, vector types, etc.
        switch (op) {
            case CoreOp.VarOp varOp -> varName(varOp);
            case HATF16Op.HATF16VarOp hatf16VarOp -> varName(hatf16VarOp);
            case HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp -> varName(hatPrivateInitVarOp);
            case HATMemoryVarOp.HATPrivateVarOp hatPrivateVarOp -> varName(hatPrivateVarOp);
            case HATMemoryVarOp.HATLocalVarOp hatLocalVarOp -> varName(hatLocalVarOp);
            case HATVectorOp.HATVectorVarOp hatVectorVarOp -> varName(hatVectorVarOp);
            case null, default -> throw new IllegalStateException("What type of varStoreOp is this?");
        }
        equals().parenthesisIfNeeded(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }

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
    public T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp, ParamVar paramVar) {
        varName(varOp);
        return self();
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp1) {
        if (fieldAccessOpHelper(buildContext.lookup,fieldLoadOp1) instanceof FieldAccess fieldAccess
              &&  fieldAccess.operandCount()==0 && fieldAccess.isPrimitive() ) {
            Object value = fieldAccess.getStaticFinalPrimitiveValue();
            literal(value.toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp1);
        }
        return self();
    }

    @Override
    public T fieldStoreOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp) {
        throw new IllegalStateException("What is this field store ?" + fieldStoreOp);
       // return self();
    }


    @Override
    public T unaryOp(ScopedCodeBuilderContext buildContext, JavaOp.UnaryOp unaryOp) {
        symbol(unaryOp).parenthesisIfNeeded(buildContext, unaryOp, ((Op.Result)unaryOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public T binaryOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryOp binaryOp) {
        parenthesisIfNeeded(buildContext, binaryOp, lhsResult(binaryOp).op());
        symbol(binaryOp);
        parenthesisIfNeeded(buildContext, binaryOp, rhsResult(binaryOp).op());
        return self();
    }


    @Override
    public T conditionalOp(ScopedCodeBuilderContext buildContext, JavaOp.JavaConditionalOp logicalOp) {
        lhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o ->  recurse(buildContext, o));
        space().symbol(logicalOp).space();
        rhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o-> recurse(buildContext, o));
        return self();
    }

    @Override
    public T binaryTestOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryTestOp binaryTestOp) {
        parenthesisIfNeeded(buildContext, binaryTestOp, lhsResult(binaryTestOp).op());
        symbol(binaryTestOp);
        parenthesisIfNeeded(buildContext, binaryTestOp, rhsResult(binaryTestOp).op());
        return self();
    }

    @Override
    public T convOp(ScopedCodeBuilderContext buildContext, JavaOp.ConvOp convOp) {
        // TODO: I think we need to work out how to handle doubles. If I remove this OpenCL on MAC complains (no FP64)
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(buildContext,JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type(buildContext,(JavaType)convOp.resultType()));
        }
        parenthesisIfNeeded(buildContext, convOp, result(convOp).op());
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
        commaSpaceSeparated(tupleOp.operands(),operand->{
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
            commaSpaceSeparated(
                    funcCallOp.operands().stream().filter(e->e instanceof Op.Result ).map(e->(Op.Result)e),
                    result -> recurse(buildContext,result.op())
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

    @Override
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

    @Override
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
                                        nlSeparated(OpTkl.statements(ifOp.bodies().get(idx).entryBlock()),
                                        root-> statement(buildContext,root)
                                        ));
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
                condBlock(whileOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp)
                        .forEach(o -> recurse(buildContext, o))
        );
        braceNlIndented(_ ->
                        nlSeparated(OpTkl.loopBodyStatements(whileOp),
                        statement->statement(buildContext,statement)
                )
        );
        return self();
    }

    @Override
    public T forOp(ScopedCodeBuilderContext buildContext, JavaOp.ForOp forOp) {
        buildContext.forScope(forOp, () ->
                forKeyword().paren(_ -> {
                    initBlock(forOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    condBlock(forOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    commaSpaceSeparated(
                            OpTkl.statements(updateBlock(forOp)),
                            op -> recurse(buildContext, op)
                    );
                }).braceNlIndented(_ ->
                            nlSeparated(OpTkl.loopBodyStatements(forOp),
                                    statement ->statement(buildContext,statement)
                        )
                )
        );
        return self();
    }

    public abstract  T atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name);

    static Regex atomicIncRegex = Regex.of("(atomic.*)Inc");

    @Override
    public T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        var invoke = invokeOpHelper(buildContext.lookup,invokeOp);
        if ( invoke.refIs(MappableIface.class,HAType.class,DeviceType.class)) {
            if (invoke.isInstance() && invoke.operandCount() == 1 && invoke.returnsInt() && invoke.named(atomicIncRegex)) {
                if (invoke.operandNAsResultOrThrow(0) instanceof Op.Result instanceResult) {
                    atomicInc(buildContext, instanceResult,
                            ((Regex.Match)atomicIncRegex.is(invoke.name())).stringOf(1) // atomicXXInc -> atomicXX
                    );
                }
            } else if (invoke.isInstance() && invoke.operandNAsResultOrThrow(0) instanceof Op.Result instance) {
                parenWhen(
                   // When we have patterns like:
                   //
                   // myiFaceArray.array().value(storeAValue);
                   //
                   // We need to generate extra parenthesis to make the struct pointer accessor "->" correct.
                   // This is a common pattern when we have a IFace type that contains a subtype based on
                   // struct or union.
                   // An example of this is for the type F16Array.
                   // The following expression checks that the current invokeOp has at least 2 operands:
                    // Why 2?
                    // - The first one is another invokeOp to load the inner struct from an IFace data structure.
                    //   The first operand is also assignable.
                    // - The second one is the store value, but this depends on the semantics and definition
                    //   of the user code.
                    invoke.operandCount() > 1
                                && invokeOpHelper(buildContext.lookup,instance.op()) instanceof Invoke invoke0
                                && invoke0.returnsClassType()
                        , _->{
                    when(invoke.returnsClassType(), _ -> ampersand());
                    recurse(buildContext, instance.op());
                });

                // Check if the varOpLoad that could follow corresponds to a local/private type
                boolean isLocalOrPrivateDS = (instance.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                        && buildContext.scope.resolve(varLoadOp.operands().getFirst()) instanceof HATMemoryVarOp);

                either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

                funcName(invoke.op());

                if (invoke.returnsVoid()) {//   setter
                    switch (invoke.operandCount()) {
                        case 2-> {
                            if (invoke.opFromOperandNAsResultOrNull(1) instanceof Op op) {
                                equals().recurse(buildContext, op);
                            }
                        }
                        case 3-> {
                            if ( invoke.opFromOperandNAsResultOrThrow(1) instanceof Op op1
                                 && invoke.opFromOperandNAsResultOrThrow(2) instanceof Op op2) {
                                 sbrace(_ -> recurse(buildContext, op1)).equals().recurse(buildContext, op2);
                            }
                        }
                        default -> throw new IllegalStateException("How ");
                    }
                } else {
                    if (invoke.opFromOperandNAsResultOrNull(1) instanceof Op op) {
                        sbrace(_ -> recurse(buildContext, op));
                    }else{
                            // this is just call.
                    }
                }
            }
        } else {// General case
            funcName(invoke.op()).paren(_ ->
                    commaSpaceSeparated(invoke.op().operands(),
                            op -> {if (op instanceof Op.Result result) {recurse(buildContext, result.op());}
                    })
            );
        }
        return self();
    }

    @Override
    public T conditionalExpressionOp(ScopedCodeBuilderContext buildContext, JavaOp.ConditionalExpressionOp ternaryOp) {
        condBlock(ternaryOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        questionMark();
        thenBlock(ternaryOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        colon();
        elseBlock(ternaryOp).ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
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
        return parenWhen(Precedence.needsParenthesis(parent,child), _ -> recurse(buildContext, child));
    }

    @Override
    public T returnOp(ScopedCodeBuilderContext buildContext, CoreOp.ReturnOp returnOp) {
        returnKeyword().when(!returnOp.operands().isEmpty(),
                        $-> $.space().parenthesisIfNeeded(buildContext, returnOp, OpTkl.result(returnOp).op())
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

    public T declareParam(ScopedCodeBuilderContext buildContext, FuncOpParams.Info param){
        return  type(buildContext,(JavaType) param.parameter.type()).space().varName(param.varOp);
    }
}
