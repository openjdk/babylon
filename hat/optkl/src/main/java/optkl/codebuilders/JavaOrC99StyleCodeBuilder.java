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
import optkl.util.Mutable;
import optkl.util.ops.Precedence;

import java.util.function.Consumer;

import static optkl.OpHelper.FieldAccess.fieldAccess;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.Ternary.ternary;

public abstract class JavaOrC99StyleCodeBuilder<T extends JavaOrC99StyleCodeBuilder<T,SCBC>,SCBC extends ScopedCodeBuilderContext>
        extends ScopedCodeBuilder<T,SCBC>
        implements BabylonOpDispatcher<T,SCBC>{

     ScopedCodeBuilderContext scopedCodeBuilderContext(){
         throw new RuntimeException("we are fake. This should be removed ");
     }

    public T body(Consumer<T> consumer) {
        return braceNlIndented(consumer);
    }

    public T tern(Consumer<T> test, Consumer<T> ifTrue, Consumer<T> ifFalse){
        return paren(test).questionMark().paren(ifTrue).colon().paren(ifFalse);
    }

   public T stmnt(Consumer<T> consumer){
        consumer.accept(self());
        return semicolon();
    }

    public T func(
            Consumer<T> type,
            String funcName,
            Consumer<T> args,
            Consumer<T> body) {
        type.accept(self());
        return space().identifier(funcName).paren(args).body(body).nl().nl();
    }

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
      //  return returnKeyword().space().paren(_-> exp.accept(self())).semicolon(); // This looks wrong.  it is very rare for us to have to add trailing semicolons
        returnKeyword().space();
        exp.accept(self());
        return semicolon();
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
    public T type( JavaType javaType) {
        return typeName(javaType.toString());
    }


    @Override
    public T varLoadOp( CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
        if (resolve instanceof CoreOp.VarOp varOp) {
            varName(varOp);
        }else if (resolve instanceof CoreOp.VarAccessOp.VarLoadOp){
            varName(varLoadOp);
        }
        return self();
    }

    @Override
    public T varStoreOp( CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        Op op = scopedCodeBuilderContext().resolve(varStoreOp.operands().getFirst());
        varName((CoreOp.VarOp) op);
        equals().parenthesisIfNeeded( varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }


    @Override
    public final T varOp( CoreOp.VarOp varOp) {
        if (varOp.isUninitialized()) {
            type( (JavaType) varOp.varValueType()).space().varName(varOp);
        } else {
            if (scopedCodeBuilderContext().isVarOpFinal(varOp)) {
                constKeyword().space();
            }
            type( (JavaType) varOp.varValueType()).space().varName(varOp).space().equals().space();
            var first = varOp.operands().getFirst();
            if (first instanceof Op.Result result) {
                parenthesisIfNeeded( varOp, result.op());
            }else if (first instanceof Block.Parameter parameter) {
               var p1 =  parameter.declaringBlock().parameters().getFirst();

                var r = parameter.uses().iterator().next();
                //parenthesisIfNeeded( varOp, r.op());
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
    public final T varOp( CoreOp.VarOp varOp, ParamVar paramVar) {
        varName(varOp);
        return self();
    }

    @Override
    public T fieldLoadOp( JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        var fieldAccess = fieldAccess(scopedCodeBuilderContext().lookup(),fieldLoadOp);
        if (fieldAccess.operandCount()==0 && fieldAccess.isPrimitive() ) {
            literal(fieldAccess.getStaticFinalPrimitiveValue().toString());
        } else {
          identifier(fieldAccess.name());
        }
        return self();
    }

    @Override
    public final T fieldStoreOp( JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp) {
        var fieldAccess = fieldAccess(scopedCodeBuilderContext().lookup(),fieldStoreOp);
        identifier(fieldAccess.name()).space().equals().space();
        recurse(((Op.Result)fieldAccess.op().operands().get(0)).op());
        dot();
        recurse(((Op.Result)fieldAccess.op().operands().get(1)).op());
        return self();
    }


    @Override
    public final  T unaryOp( JavaOp.UnaryOp unaryOp) {
        symbol(unaryOp).parenthesisIfNeeded( unaryOp, ((Op.Result)unaryOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public final  T binaryOp( JavaOp.BinaryOp binaryOp) {
        parenthesisIfNeeded( binaryOp, OpHelper.lhsResult(binaryOp).op());
        symbol(binaryOp);
        parenthesisIfNeeded( binaryOp, OpHelper.rhsResult(binaryOp).op());
        return self();
    }


    @Override
    public final T conditionalOp( JavaOp.JavaConditionalOp logicalOp) {
        OpHelper.lhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o ->  recurse( o));
        space().symbol(logicalOp).space();
        OpHelper.rhsOps(logicalOp).stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o-> recurse( o));
        return self();
    }

    @Override
    public final T compareOp( JavaOp.CompareOp compareOp) {
        parenthesisIfNeeded( compareOp, OpHelper.lhsResult(compareOp).op());
        symbol(compareOp);
        parenthesisIfNeeded( compareOp, OpHelper.rhsResult(compareOp).op());
        return self();
    }

    @Override
    public T convOp( JavaOp.ConvOp convOp) {
        // TODO: I think we need to work out how to handle doubles. If I remove this OpenCL on MAC complains (no FP64)
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type((JavaType)convOp.resultType()));
        }
        parenthesisIfNeeded( convOp, ((Op.Result) convOp.operands().getFirst()).op());
        return self();
    }

    @Override
    public final T constantOp( CoreOp.ConstantOp constantOp) {
        if (constantOp.value() == null) {
            nullConst();
        } else {
            literal(constantOp.resultType(),constantOp.value().toString());
        }
        return self();
    }

    @Override
    public final  T yieldOp( CoreOp.YieldOp yieldOp) {
        if (yieldOp.operands().getFirst() instanceof Op.Result result) {
            recurse( result.op());
        }
        return self();
    }



    @Override
    public final  T tupleOp( CoreOp.TupleOp tupleOp) {
        commaSpaceSeparated(tupleOp.operands(),operand->{
            if (operand instanceof Op.Result result) {
                recurse( result.op());
            } else {
                throw new IllegalStateException("handle tuple");
                //comment("/* nothing to tuple */");
            }
        });
        return self();
    }

    @Override
    public final T funcCallOp( CoreOp.FuncCallOp funcCallOp) {
        funcName(funcCallOp);
        paren(_ ->
                commaSpaceSeparated(
                        funcCallOp.operands().stream().filter(e->e instanceof Op.Result ).map(e->(Op.Result)e),
                        result -> recurse(result.op())
                )
        );
        return self();
    }

    @Override
    public final T labeledOp( JavaOp.LabeledOp labeledOp) {
        var labelNameOp = labeledOp.bodies().getFirst().entryBlock().ops().getFirst();
        CoreOp.ConstantOp constantOp = (CoreOp.ConstantOp) labelNameOp;
        literal(constantOp.value().toString()).colon().nl();
        var forLoopOp = labeledOp.bodies().getFirst().entryBlock().ops().get(1);
        recurse(forLoopOp);
        return self();
    }

    @Override
    public final T breakOp( JavaOp.BreakOp breakOp) {
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
    public final T continueOp( JavaOp.ContinueOp continueOp) {
        if (!continueOp.operands().isEmpty()
                && continueOp.operands().getFirst() instanceof Op.Result result
                && result.op() instanceof CoreOp.ConstantOp c
        ) {
            continueKeyword().space().literal(c.value().toString());
        } else if (scopedCodeBuilderContext().isInFor()) {
            // nope
        } else {
            continueKeyword();
        }

        return self();
    }

    @Override
    public final T ifOp( JavaOp.IfOp ifOp) {
        scopedCodeBuilderContext().ifScope(ifOp, () -> {
            var lastWasBody = Mutable.of(false);
            var i = Mutable.of(0);
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
                                        root-> statement(root)
                                ));
                    }
                    lastWasBody.set(true);
                } else {
                    when(idx>0,_-> elseKeyword().space());
                    ifKeyword().paren(_ ->
                            ifOp.bodies().get(idx).entryBlock()            // get the entryblock if bodies[c.value]
                                    .ops().stream().filter(o->o instanceof CoreOp.YieldOp) // we want all the yields
                                    .forEach((yield) -> recurse( yield))
                    );
                    lastWasBody.set(false);
                }
                i.set(i.get()+1);
            });
        });
        return self();
    }

    @Override
    public final T whileOp( JavaOp.WhileOp whileOp) {
        whileKeyword().paren(_ ->
                OpHelper.entryBlockOfBodyN(whileOp, 0)
                        .ops().stream().filter(o -> o instanceof CoreOp.YieldOp)
                        .forEach(o -> recurse( o))
        );
        braceNlIndented(_ ->
                nlSeparated(OpHelper.Statement.bodyStatements(whileOp.loopBody()),
                        statement->statement(statement)
                )
        );
        return self();
    }

    @Override
    public final T forOp( JavaOp.ForOp forOp) {
        scopedCodeBuilderContext().forScope(forOp, () ->
                forKeyword().paren(_ -> {
                    forOp.init().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
                    semicolon().space();
                    forOp.cond().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
                    semicolon().space();
                    commaSpaceSeparated(
                            OpHelper.Statement.statements(forOp.update().entryBlock()),
                            op -> recurse( op)
                    );
                }).braceNlIndented(_ ->
                        nlSeparated(OpHelper.Statement.bodyStatements(forOp.loopBody()),
                                statement ->statement(statement)
                        )
                )
        );
        return self();
    }

    @Override
    public T invokeOp( JavaOp.InvokeOp invokeOp) {
        var invoke = invoke(scopedCodeBuilderContext().lookup(),invokeOp);

            funcName(invoke.op()).paren(_ ->
                    commaSpaceSeparated(invoke.op().operands(),
                            op -> {if (op instanceof Op.Result result) {recurse( result.op());}
                            })
            );

        return self();
    }

    @Override
    public final T conditionalExpressionOp( JavaOp.ConditionalExpressionOp ternaryOp) {
        OpHelper.Ternary ternary = ternary(scopedCodeBuilderContext().lookup(),ternaryOp);
        ternary.condBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
        questionMark();
        ternary.thenBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
        colon();
        ternary.elseBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
        return self();
    }

    /**
     * Wrap paren() of precedence of op is higher than parent.
     *

     * @param parent
     * @param child
     */
    @Override
    public final  T parenthesisIfNeeded( Op parent, Op child) {
        return parenWhen(Precedence.needsParenthesis(parent,child), _ -> recurse( child));
    }

    @Override
    public final  T returnOp( CoreOp.ReturnOp returnOp) {
        returnKeyword().when(!returnOp.operands().isEmpty(),
                $-> $.space().parenthesisIfNeeded( returnOp, ((Op.Result) returnOp.operands().getFirst()).op())
        );
        return self();
    }

    public final  T statement(Op op) {
        recurse( op);
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

    public final  T declareParam( FuncOpParams.Info param){
        return  type((JavaType) param.parameter.type()).space().varName(param.varOp);
    }

    @Override
    public T newOp( JavaOp.NewOp newOp) {
         newKeyword().space().type((JavaType) newOp.type());
       if (newOp.operands().isEmpty()){
           ocparen();
       }else {
           if (newOp.type() instanceof ArrayType){
               brace(_ -> {
                   commaSpaceSeparated(newOp.operands(),
                           op -> {
                               if (op instanceof Op.Result result) {
                                   recurse( result.op());
                               }
                           });
               });
           }else {
               paren(_ -> {
                   commaSpaceSeparated(newOp.operands(),
                           op -> {
                               if (op instanceof Op.Result result) {
                                   recurse( result.op());
                               }
                           });
               });
           }
       }
       return self();
    }
    @Override
    public T arrayLoadOp( JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp){
        recurse(((Op.Result)arrayLoadOp.operands().get(0)).op());
        sbrace(_-> recurse(((Op.Result)arrayLoadOp.operands().get(1)).op()));
        return self();
    }

    @Override
    public T arrayStoreOp( JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp){
        recurse(((Op.Result)arrayStoreOp.operands().get(0)).op());
        sbrace(_-> recurse(((Op.Result)arrayStoreOp.operands().get(1)).op()));
        space().equals().space();
        recurse(((Op.Result)arrayStoreOp.operands().get(2)).op());
        return self();
    }

    @Override
    public T enhancedForOp(JavaOp.EnhancedForOp enhancedForOp){
        forKeyword().paren(_-> {
            enhancedForOp.initialization().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
            space().colon().space().blockInlineComment("Get rid of = before this");
            enhancedForOp.expression().entryBlock().ops().stream().filter(o -> o instanceof CoreOp.YieldOp).forEach(o -> recurse( o));
        }).braceNlIndented(_->
            nlSeparated(OpHelper.Statement.bodyStatements(enhancedForOp.loopBody()),
                    this::statement
            )

        );
        return self();
    }

    @Override
    public T blockOp( JavaOp.BlockOp blockOp) {
      return braceNlIndented(_-> nlSeparated(OpHelper.Statement.statements(blockOp.body().entryBlock()), this::statement));
    }

    @Override
    public T concatOp( JavaOp.ConcatOp concatOp) {
        return
                recurse( ((Op.Result)concatOp.operands().get(0)).op()).
        add().recurse( ((Op.Result)concatOp.operands().get(1)).op());
    }

    @Override
    public final T lambdaOp( JavaOp.LambdaOp lambdaOp) {
        scopedCodeBuilderContext().lambdaScope(lambdaOp,()-> {
                    var parameters = lambdaOp.body().entryBlock().parameters();
                  //  if (parameters.isEmpty()) {
                    //    ocparen();
                    //}else{
                        parenWhen(parameters.size()!=1,_->{
                           commaSpaceSeparated(parameters,$->
                               varName((CoreOp.VarOp)$.uses().stream().findFirst().get().op())
                           );
                        });
                    //}
                    space().rarrow().space();
                    braceNlIndented(_ -> {
                        nlSeparated(OpHelper.Statement.bodyStatements(lambdaOp.body()),
                                op->{
                                    if (op instanceof CoreOp.VarOp varOp
                                            && varOp.operands().getFirst() instanceof Block.Parameter parameter
                                           ){
                                        varName(varOp);
                                    }else {
                                        statement(op);
                                    }
                                }
                        );
                    });
                }
        );
        return self();
    }
}
