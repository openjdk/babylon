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
package optkl.codebuilders;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.FuncOpParams;
import optkl.OpHelper;
import optkl.ParamVar;
import optkl.util.StreamMutable;
import optkl.util.ops.Precedence;

import java.util.function.Consumer;

import static optkl.OpHelper.Named.NamedStaticOrInstance.FieldAccess.fieldAccess;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke.invoke;
import static optkl.OpHelper.Ternary.ternary;

public class JavaOrC99StyleCodeBuilder<T extends JavaOrC99StyleCodeBuilder<T>> extends CodeBuilder<T>  implements BabylonOpDispatcher<T,ScopedCodeBuilderContext>{

    public final T assign(Consumer<T> lhs, Consumer<T> rhs){
        lhs.accept(self());
        space().equals().space();
        rhs.accept(self());
        return self();
    }

    public final T cast(Consumer<T> type){
        return paren(_-> type.accept(self()));
    }

    public final T returnKeyword(Consumer<T> exp){
        return returnKeyword().space().paren(_-> exp.accept(self())).semicolon();
    }
    public final T forLoop(Consumer<T> init, Consumer<T> test, Consumer<T>mutate, Consumer<T>body) {
        return  forKeyword()
                .paren(_->{
                    init.accept(self());
                    semicolon().space();
                    test.accept(self());
                    semicolon().space();mutate.accept(self());
                })
                .braceNlIndented(body::accept);
    }

    public final T literal(TypeElement typeElement, String string){
        if (typeElement.toString().equals("java.lang.String")){
            dquote().escaped(string).dquote();
        }else{
            literal(string);
        }
        return self();
    }


    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        return typeName(javaType.toString());
    }


    @Override
    public T varLoadOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = buildContext.scope.resolve(varLoadOp.operands().getFirst());
        if (resolve instanceof CoreOp.VarOp varOp) {
            varName(varOp);
        }else if (resolve instanceof CoreOp.VarAccessOp.VarLoadOp){
            varName(varLoadOp);
        }
        return self();
    }

    @Override
    public T varStoreOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        Op op = buildContext.scope.resolve(varStoreOp.operands().getFirst());
        varName((CoreOp.VarOp) op);
        equals().parenthesisIfNeeded(buildContext, varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }


    @Override
    public final T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp) {
        if (varOp.isUninitialized()) {
            type(buildContext, (JavaType) varOp.varValueType()).space().varName(varOp);
        } else {
            if (buildContext.isVarOpFinal(varOp)) {
                constKeyword().space();
            }
            type(buildContext, (JavaType) varOp.varValueType()).space().varName(varOp).space().equals().space();
            var first = varOp.operands().getFirst();
            if (first instanceof Op.Result result) {
                parenthesisIfNeeded(buildContext, varOp, result.op());
            }else if (first instanceof Block.Parameter parameter) {
               var p1 =  parameter.declaringBlock().parameters().getFirst();

                var r = parameter.uses().iterator().next();
                //parenthesisIfNeeded(buildContext, varOp, r.op());
               // if (r.op() instanceof CoreOp.VarOp varOp1){
                 //   identifier(varOp1.varName());
               // }
              blockInlineComment("param "+r);
            }else{
                blockInlineComment("look at varOp "+first);
            }
        }
        return self();
    }

    @Override
    public final T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp, ParamVar paramVar) {
        varName(varOp);
        return self();
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        var fieldAccess = fieldAccess(buildContext.lookup,fieldLoadOp);
        if (fieldAccess.operandCount()==0 && fieldAccess.isPrimitive() ) {
            literal(fieldAccess.getStaticFinalPrimitiveValue().toString());
        } else {
          identifier(fieldAccess.name());
        }
        return self();
    }

    @Override
    public final T fieldStoreOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp) {
        var fieldAccess = fieldAccess(buildContext.lookup,fieldStoreOp);
        identifier(fieldAccess.name()).space().equals().space();
        recurse(buildContext,((Op.Result)fieldAccess.op().operands().get(0)).op());
        dot();
        recurse(buildContext,((Op.Result)fieldAccess.op().operands().get(1)).op());
        return self();
    }


    @Override
    public final  T unaryOp(ScopedCodeBuilderContext buildContext, JavaOp.UnaryOp unaryOp) {
        symbol(unaryOp).parenthesisIfNeeded(buildContext, unaryOp, ((Op.Result)unaryOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public final  T binaryOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryOp binaryOp) {
        parenthesisIfNeeded(buildContext, binaryOp, OpHelper.lhsResult(binaryOp).op());
        symbol(binaryOp);
        parenthesisIfNeeded(buildContext, binaryOp, OpHelper.rhsResult(binaryOp).op());
        return self();
    }


    @Override
    public final T conditionalOp(ScopedCodeBuilderContext buildContext, JavaOp.JavaConditionalOp logicalOp) {
        OpHelper.lhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o ->  recurse(buildContext, o));
        space().symbol(logicalOp).space();
        OpHelper.rhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o-> recurse(buildContext, o));
        return self();
    }

    @Override
    public final T binaryTestOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryTestOp binaryTestOp) {
        parenthesisIfNeeded(buildContext, binaryTestOp, OpHelper.lhsResult(binaryTestOp).op());
        symbol(binaryTestOp);
        parenthesisIfNeeded(buildContext, binaryTestOp, OpHelper.rhsResult(binaryTestOp).op());
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
        parenthesisIfNeeded(buildContext, convOp, ((Op.Result) convOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public final T constantOp(ScopedCodeBuilderContext buildContext, CoreOp.ConstantOp constantOp) {
        if (constantOp.value() == null) {
            nullConst();
        } else {
            literal(constantOp.resultType(),constantOp.value().toString());
        }
        return self();
    }

    @Override
    public final  T yieldOp(ScopedCodeBuilderContext buildContext, CoreOp.YieldOp yieldOp) {
        if (yieldOp.operands().getFirst() instanceof Op.Result result) {
            recurse(buildContext, result.op());
        }
        return self();
    }



    @Override
    public final  T tupleOp(ScopedCodeBuilderContext buildContext, CoreOp.TupleOp tupleOp) {
        commaSpaceSeparated(tupleOp.operands(),operand->{
            if (operand instanceof Op.Result result) {
                recurse(buildContext, result.op());
            } else {
                throw new IllegalStateException("handle tuple");
                //comment("/* nothing to tuple */");
            }
        });
        return self();
    }

    @Override
    public final T funcCallOp(ScopedCodeBuilderContext buildContext, CoreOp.FuncCallOp funcCallOp) {
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
    public final T labeledOp(ScopedCodeBuilderContext buildContext, JavaOp.LabeledOp labeledOp) {
        var labelNameOp = labeledOp.bodies().getFirst().entryBlock().ops().getFirst();
        CoreOp.ConstantOp constantOp = (CoreOp.ConstantOp) labelNameOp;
        literal(constantOp.value().toString()).colon().nl();
        var forLoopOp = labeledOp.bodies().getFirst().entryBlock().ops().get(1);
        recurse(buildContext,forLoopOp);
        return self();
    }

    @Override
    public final T breakOp(ScopedCodeBuilderContext buildContext, JavaOp.BreakOp breakOp) {
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
    public final T continueOp(ScopedCodeBuilderContext buildContext, JavaOp.ContinueOp continueOp) {
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
    public final T ifOp(ScopedCodeBuilderContext buildContext, JavaOp.IfOp ifOp) {
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
                                nlSeparated(OpHelper.Statement.statements(ifOp.bodies().get(idx).entryBlock()),
                                        root-> statement(buildContext,root)
                                ));
                    }
                    lastWasBody.set(true);
                } else {
                    when(idx>0,_-> elseKeyword().space());
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
    public final T whileOp(ScopedCodeBuilderContext buildContext, JavaOp.WhileOp whileOp) {
        whileKeyword().paren(_ ->
                OpHelper.entryBlockOfBodyN(whileOp, 0)
                        .ops().stream().filter(o -> o instanceof CoreOp.YieldOp)
                        .forEach(o -> recurse(buildContext, o))
        );
        braceNlIndented(_ ->
                nlSeparated(OpHelper.Statement.bodyStatements(whileOp.loopBody()),
                        statement->statement(buildContext,statement)
                )
        );
        return self();
    }

    @Override
    public final T forOp(ScopedCodeBuilderContext buildContext, JavaOp.ForOp forOp) {
        buildContext.forScope(forOp, () ->
                forKeyword().paren(_ -> {
                    forOp.init().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    forOp.cond().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
                    semicolon().space();
                    commaSpaceSeparated(
                            OpHelper.Statement.statements(forOp.update().entryBlock()),
                            op -> recurse(buildContext, op)
                    );
                }).braceNlIndented(_ ->
                        nlSeparated(OpHelper.Statement.bodyStatements(forOp.loopBody()),
                                statement ->statement(buildContext,statement)
                        )
                )
        );
        return self();
    }

    @Override
    public T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        var invoke = invoke(buildContext.lookup,invokeOp);

            funcName(invoke.op()).paren(_ ->
                    commaSpaceSeparated(invoke.op().operands(),
                            op -> {if (op instanceof Op.Result result) {recurse(buildContext, result.op());}
                            })
            );

        return self();
    }

    @Override
    public final T conditionalExpressionOp(ScopedCodeBuilderContext buildContext, JavaOp.ConditionalExpressionOp ternaryOp) {
        OpHelper.Ternary ternary = ternary(buildContext.lookup,ternaryOp);
        ternary.condBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        questionMark();
        ternary.thenBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
        colon();
        ternary.elseBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(buildContext, o));
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
    public final  T parenthesisIfNeeded(ScopedCodeBuilderContext buildContext, Op parent, Op child) {
        return parenWhen(Precedence.needsParenthesis(parent,child), _ -> recurse(buildContext, child));
    }

    @Override
    public final  T returnOp(ScopedCodeBuilderContext buildContext, CoreOp.ReturnOp returnOp) {
        returnKeyword().when(!returnOp.operands().isEmpty(),
                $-> $.space().parenthesisIfNeeded(buildContext, returnOp, ((Op.Result) returnOp.operands().getFirst()).op())
        );
        return self();
    }

    public final  T statement(ScopedCodeBuilderContext buildContext,Op op) {
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

    public final  T declareParam(ScopedCodeBuilderContext buildContext, FuncOpParams.Info param){
        return  type(buildContext,(JavaType) param.parameter.type()).space().varName(param.varOp);
    }

    @Override
    public T newOp(ScopedCodeBuilderContext buildContext, JavaOp.NewOp newOp) {
         newKeyword().space().type(buildContext,(JavaType) newOp.type());
       if (newOp.operands().isEmpty()){
           ocparen();
       }else {
           if (newOp.type() instanceof ArrayType){
               brace(_ -> {
                   commaSpaceSeparated(newOp.operands(),
                           op -> {
                               if (op instanceof Op.Result result) {
                                   recurse(buildContext, result.op());
                               }
                           });
               });
           }else {
               paren(_ -> {
                   commaSpaceSeparated(newOp.operands(),
                           op -> {
                               if (op instanceof Op.Result result) {
                                   recurse(buildContext, result.op());
                               }
                           });
               });
           }
       }
       return self();
    }
    @Override
    public T arrayLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp){
        recurse(buildContext,((Op.Result)arrayLoadOp.operands().get(0)).op());
        sbrace(_-> recurse(buildContext,((Op.Result)arrayLoadOp.operands().get(1)).op()));
        return self();
    }

    @Override
    public T arrayStoreOp(ScopedCodeBuilderContext buildContext, JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp){
        recurse(buildContext,((Op.Result)arrayStoreOp.operands().get(0)).op());
        sbrace(_-> recurse(buildContext,((Op.Result)arrayStoreOp.operands().get(1)).op()));
        space().equals().space();
        recurse(buildContext,((Op.Result)arrayStoreOp.operands().get(2)).op());
        return self();
    }

    @Override
    public T enhancedForOp(ScopedCodeBuilderContext builderContext, JavaOp.EnhancedForOp enhancedForOp){
        forKeyword().paren(_-> {
            enhancedForOp.initialization().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(builderContext, o));
            space().colon().space().blockInlineComment("Get rid of = before this");
            enhancedForOp.expression().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse(builderContext, o));
        }).braceNlIndented(_->
            nlSeparated(OpHelper.Statement.bodyStatements(enhancedForOp.loopBody()),
                    statement ->statement(builderContext,statement)
            )

        );
        return self();
    }

    @Override
    public T blockOp(ScopedCodeBuilderContext buildContext, JavaOp.BlockOp blockOp) {
      return braceNlIndented(_-> nlSeparated(OpHelper.Statement.statements(blockOp.body().entryBlock()), statement ->statement(buildContext,statement)));
    }

    @Override
    public T concatOp(ScopedCodeBuilderContext buildContext, JavaOp.ConcatOp concatOp) {
        return
                recurse(buildContext, ((Op.Result)concatOp.operands().get(0)).op()).
        add().recurse(buildContext, ((Op.Result)concatOp.operands().get(1)).op());
      //  blockInlineComment("concat");
    }

    @Override
    public final T lambdaOp(ScopedCodeBuilderContext buildContext, JavaOp.LambdaOp lambdaOp) {
        braceNlIndented(_-> {
            blockInlineComment("LAMBDA");
            nlSeparated(OpHelper.Statement.bodyStatements(lambdaOp.body()),
                    statement -> statement(buildContext, statement)
            );
            blockInlineComment("ADBMAL");
        });
        return self();
    }
}
