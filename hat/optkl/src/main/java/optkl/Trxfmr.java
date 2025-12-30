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
package optkl;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.util.CallSite;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Predicate;

import static optkl.OpTkl.copyLocation;
import static optkl.OpTkl.operandOrNull;

public class Trxfmr {
    public static Trxfmr of(CoreOp.FuncOp funcOp) {
        return new Trxfmr(funcOp);
    }
    public static Trxfmr of(CallSite callSite,CoreOp.FuncOp funcOp) {
        return new Trxfmr(callSite,funcOp);
    }

    interface TransformerCarrier {
        Trxfmr trxfmr();
    }
    interface CursorCarrier<T extends Cursor>{
        T cursor();
    }

    public interface  Walker extends TransformerCarrier {
        void op(Op op);
        Op op();
        void funcOp(CoreOp.FuncOp funcOp);
        CoreOp.FuncOp funcOp();
              class Impl implements TransformerCarrier,Walker {
                private final Trxfmr trxfmr;
                public Trxfmr trxfmr() {
                    return trxfmr;
                }
                private Op op;
                private CoreOp.FuncOp funcOp;
                @Override
                public void op(Op op) {
                    this.op = op;
                }

                @Override
                public Op op() {
                    return this.op;
                }

                @Override
                public void funcOp(CoreOp.FuncOp funcOp) {
                    this.funcOp = funcOp;
                }

                @Override
                public CoreOp.FuncOp funcOp() {
                    return this.funcOp;
                }

                Impl(Trxfmr trxfmr, CoreOp.FuncOp funcOp) {
                    this.trxfmr = trxfmr;
                    this.funcOp = funcOp;
                }
            }
        static Walker of(Trxfmr trxfmr, CoreOp.FuncOp funcOp){
            return new Impl(trxfmr,funcOp);
        }
    }

    public interface  Cursor extends TransformerCarrier, Walker {
        enum Action{NONE,REMOVED,REPLACE,ADDED };
        void action(Action action);
        Action action();
        void builder(Block.Builder builder);
        Block.Builder builder();
        void handled(boolean handled);
        boolean handled();
        Op.Result replace(Op op, Consumer<Mapper<?>> mapperConsumer);
        Op.Result add(Op op, Consumer<Mapper<?>> mapperConsumer);
        default Op.Result replace(Op op){
            return replace(op, _->{});
        }

        default Op.Result add(Op op){
            return add(op, _->{});
        }
         default void remove(Consumer<Mapper<?>> mapperConsumer) {
            mapperConsumer.accept(Mapper.of(this));
        }
        default void remove() {
            remove(_->{});
        }
        default Op.Result remove(Op op){
            return replace(op, (m)->{});
        }
        static Cursor of(Trxfmr trxfmr, CoreOp.FuncOp funcOp, Block.Builder builder, Op op){
            class Impl extends Walker.Impl implements Cursor {
                private Action action;
                private Block.Builder builder;
                private boolean handled;
                @Override
                public void handled(boolean handled) {
                    this.handled = handled;
                }

                @Override
                public boolean  handled() {
                    return this.handled;
                }

                @Override
                public void action(Action action) {
                    this.action=action;
                }

                @Override
                public Action action() {
                    return action;
                }

                @Override
                public void builder(Block.Builder builder) {
                    this.builder = builder;
                }

                @Override
                public Block.Builder builder() {
                    return this.builder;
                }
                @Override
                public Op.Result replace(Op replacement, Consumer<Mapper<?>> mapperConsumer) {
                    handled(true);
                    action(Action.REPLACE);
                    var result = trxfmr.opToResultOp(op(),builder().op(copyLocation(op(), replacement)));
                    if (result.type() instanceof PrimitiveType primitiveType && primitiveType.isVoid()) {
                    }else {
                        mapperConsumer.accept(Mapper.of(this).map(op().result(), result));
                    }
                    return result;
                }
                public Op.Result add(Op newOne, Consumer<Mapper<?>> mapperConsumer) {
                    handled(true);
                    action(Action.ADDED);
                    var result = trxfmr.opToResultOp(op(),builder().op(copyLocation(op(), newOne)));
                    if (result.type() instanceof PrimitiveType primitiveType && primitiveType.isVoid()) {
                    }else{
                        mapperConsumer.accept(Mapper.of(this).map(op().result(), result));
                    }
                    return result;
                }
                @Override
                public void remove( Consumer<Mapper<?>> mapperConsumer) {
                    handled(true);
                    action(Action.REMOVED);
                }
                Impl(Trxfmr hatTransformer, CoreOp.FuncOp funcOp, Block.Builder builder, Op op) {
                    super(hatTransformer,funcOp);
                    builder(builder);
                    op(op);
                }
            }
            return new Impl(trxfmr,funcOp, builder,op);
        }

        default Value getValue(Value value){
            return builder().context().getValue(value);
        }

        default Value mappedOperand(int idx){
            return getValue(operandOrNull(op(),idx));
        }
    }

    public interface Selector<T extends Selector<T>> extends TransformerCarrier {
        default T  select(Op...ops){
            trxfmr().selected.addAll(List.of(ops));
            return (T)this;
        }
       static Selector<?> of(Trxfmr trxfmr){
            record SelectorImpl(Trxfmr trxfmr) implements Selector<SelectorImpl>{}
            return  new SelectorImpl(trxfmr);
        }
    }


    public interface Mapper<T extends Mapper<T>> extends CursorCarrier{
        default  T map(Value from, Value to) {
            cursor().builder().context().mapValue(from, to);
            return (T)this;
        }
        default  T map(Op fromOp, Value to) {
            map(fromOp.result(), to);
            return (T)this;
        }
        default  T mapOperand(Op fromOp, List<Value> operands, int n) {
           return map(fromOp,operands.get(n));
        }
        default  T mapOperands(Op fromOp, List<Value> operands) {
            operands.forEach(v -> {
                map(fromOp,v);
            });
            return (T)this;
        }
        default  T mapOperands(Op fromOp, Op to) {
            return mapOperands(fromOp,to.operands());
        }
        default  T mapOperand(Op fromOp, Op to, int index) {
            return map(fromOp,to.operands().get(index));
        }
        static Mapper<?> of(Cursor cursor){
            record MapperImpl(Cursor cursor) implements Mapper<MapperImpl> { }
            return new MapperImpl(cursor);
        }
    }

    public final Set<Op> selected = new LinkedHashSet<>();
    public final Map<Op, Op> opmap = new HashMap<>();
    public final CallSite callSite;
    public CoreOp.FuncOp funcOp;

    public CoreOp.FuncOp funcOp(){
        return funcOp;
    }
    public CoreOp.FuncOp funcOp(CoreOp.FuncOp funcOp){
        return this.funcOp=funcOp;
    }

    public Trxfmr(CallSite callSite, CoreOp.FuncOp funcOp) {
        this.callSite = callSite;
        this.funcOp =  funcOp;
        if (callSite!=null && callSite.tracing()) {
            IO.println("[INFO] Code model after [" + callSite.clazz().getSimpleName() + "#" + callSite.methodName() + "]: " + System.lineSeparator() + funcOp.toText());
        }
    }

    public Trxfmr(CoreOp.FuncOp funcOp) {
        this (null,funcOp);

    }
    public Trxfmr select(Predicate<Op> codeElementPredicate, BiConsumer<Selector<?>,Op> selectorConsumer) {
        Selector<?> selector = Selector.of(this);
        funcOp().elements().filter(ce->ce instanceof Op).map(ce->(Op)ce).filter(codeElementPredicate).forEach(op->
                selectorConsumer.accept(selector,op)
        );
        return this;
    }


    public Trxfmr done() {
        if (callSite!=null && callSite.tracing()) {
            IO.println("[INFO] Code model after [" + callSite.clazz().getSimpleName() + "#" + callSite.methodName() + "]: " + System.lineSeparator()
                    +funcOp().toText());
        }
        return this;
    }



    private Op opToOp(Op from, Op to){
        opmap.put(from,to);
        return to;
    }
    private Op.Result opToResultOp(Op from, Op.Result result){
        opToOp(from, result.op());
        return result;
    }

    public Trxfmr transform(Predicate<CodeElement<?,?>> predicate, Consumer<Cursor> cursorConsumer) {
        if (callSite != null && callSite.tracing()) {
            System.out.println(callSite);
        }
        var newFuncOp = funcOp().transform((blockBuilder, op) -> {
            Cursor cursor = Cursor.of(this, funcOp, blockBuilder,op);
            cursor.builder(blockBuilder);
            cursor.op(op);
            cursor.handled(false);
            cursor.action(Cursor.Action.NONE);
            boolean isEmpty = selected.isEmpty();
            boolean isInSelected = selected.contains(op);
            boolean isSelected = isEmpty|isInSelected;
            boolean passesPredicate = predicate.test(op);
            if (isSelected && passesPredicate) {
                cursorConsumer.accept(cursor);
                if (!cursor.handled()){
                    opToOp(op,cursor.builder().op(op).op());
                }
            } else {
                opToOp(op,cursor.builder().op(op).op());
            }
            return blockBuilder;
        });
        funcOp(newFuncOp);
        return this;
    }
    public Trxfmr transform(Consumer<Cursor> transformer) {
        return transform(_->true,transformer);
    }

    public Trxfmr transform(Predicate<Op> predicate, CodeTransformer codeTransformer) {
        if (callSite != null && callSite.tracing()) {
            System.out.println(callSite);
        }
        var currentFuncOp = funcOp();
        var newFuncOp = currentFuncOp.transform((blockBuilder, op) -> {
            Cursor cursor = Cursor.of(this,funcOp,blockBuilder,op);
            if ((selected.isEmpty() || selected.contains(op)) &&  predicate.test(op)) {
                codeTransformer.acceptOp(cursor.builder(),op);
            } else {
                opToOp(op,cursor.builder().op(op).op());
            }
            return cursor.builder();
        });
        opmap.put(currentFuncOp, newFuncOp);
        funcOp(newFuncOp);
        opmap.forEach((from, to) -> { selected.remove(from);selected.add(to);});
        return this;
    }

    public Trxfmr transform(CodeTransformer codeTransformer ) {
        if (callSite != null && callSite.tracing()) {
            System.out.println(callSite);
        }
        funcOp(funcOp().transform(codeTransformer));
        return this;
    }


    public interface Edge<F extends CodeElement<?,?>, T extends CodeElement<?,?>> {
        F f();
        T t();
        Set<Op> ops();
         class Selector<F extends CodeElement<?,?>, T extends CodeElement<?,?>> {
            Map<F, Edge<F, T>> fromMap = new HashMap<>();
            Map<T, Edge<F, T>> toMap = new HashMap<>();

            public Selector<F, T> add(Edge<F, T> edge) {
                fromMap.put(edge.f(), edge);
                toMap.put(edge.t(), edge);
                return this;
            }

            Edge<F, T> from(F f) {
                return fromMap.get(f);
            }

            Edge<F, T> to(T t) {
                return toMap.get(t);
            }

            Predicate<CodeElement<?,?>> predicate =ce->fromMap.containsKey((F) ce) || toMap.containsKey((T) ce);

            public boolean contains(Op op) {
                return predicate.test(op);
            }

             public CoreOp.FuncOp transform(CoreOp.FuncOp funcOp, Consumer<Cursor> c) {
                 return new Trxfmr(CallSite.of(this.getClass()), funcOp)
                         .transform(this.predicate,c).done().funcOp();
             }
         }
    }


}
