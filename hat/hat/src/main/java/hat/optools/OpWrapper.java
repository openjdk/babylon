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
package hat.optools;

import hat.util.Result;
import hat.util.StreamCounter;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExtendedOp;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class OpWrapper<T extends Op> {
    public static <O extends Op, OW extends OpWrapper<O>> OW wrap(O op) {
        // We have one special case
        if (op instanceof CoreOp.VarOp varOp) {
            // this gets called a lot and we can't wrap yet or we recurse so we
            // use the raw model. Basically we want a different wrapper for VarDeclations
            // which  relate to func parameters.
            // This saves us asking each time if a var is indeed a func param.

            if (varOp.parentBlock().parentBody().parentOp() instanceof CoreOp.FuncOp funcOp) {
                Result<OW> result = new Result<>();
                StreamCounter.of(funcOp.bodies().getFirst().blocks().getFirst().parameters(), (c, parameter) -> {
                            if (parameter.uses().stream().findFirst().get().op().equals(varOp)) {
                                result.of((OW) new VarFuncDeclarationOpWrapper(varOp, funcOp, parameter, c.value()));
                            }
                        }
                );
                if (result.isPresent()) {
                    return result.get();
                }
            }
        }
        return switch (op) {
            case ExtendedOp.JavaForOp $ -> (OW) new ForOpWrapper($);
            case ExtendedOp.JavaWhileOp $ -> (OW) new WhileOpWrapper($);
            case ExtendedOp.JavaIfOp $ -> (OW) new IfOpWrapper($);
            case CoreOp.BinaryOp $ -> (OW) new BinaryArithmeticOrLogicOperation($);
            case CoreOp.BinaryTestOp $ -> (OW) new BinaryTestOpWrapper($);
            case CoreOp.FuncOp $ -> (OW) new FuncOpWrapper($);
            case CoreOp.VarOp $ -> (OW) new VarDeclarationOpWrapper($);
            case CoreOp.YieldOp $ -> (OW) new YieldOpWrapper($);
            case CoreOp.FuncCallOp $ -> (OW) new FuncCallOpWrapper($);
            case CoreOp.ConvOp $ -> (OW) new ConvOpWrapper($);
            case CoreOp.ConstantOp $ -> (OW) new ConstantOpWrapper($);
            case CoreOp.ReturnOp $ -> (OW) new ReturnOpWrapper($);
            case CoreOp.VarAccessOp.VarStoreOp $ -> (OW) new VarStoreOpWrapper($);
            case CoreOp.VarAccessOp.VarLoadOp $ -> (OW) new VarLoadOpWrapper($);
            case CoreOp.FieldAccessOp.FieldStoreOp $ -> (OW) new FieldStoreOpWrapper($);
            case CoreOp.FieldAccessOp.FieldLoadOp $ -> (OW) new FieldLoadOpWrapper($);
            case CoreOp.InvokeOp $ -> (OW) new InvokeOpWrapper($);
            case CoreOp.TupleOp $ -> (OW) new TupleOpWrapper($);
            case CoreOp.LambdaOp $ -> (OW) new LambdaOpWrapper($);
            case ExtendedOp.JavaConditionalOp $ -> (OW) new LogicalOpWrapper($);
            case ExtendedOp.JavaConditionalExpressionOp $ -> (OW) new TernaryOpWrapper($);
            case ExtendedOp.JavaLabeledOp $ -> (OW) new JavaLabeledOpWrapper($);
            case ExtendedOp.JavaBreakOp $ -> (OW) new JavaBreakOpWrapper($);
            case ExtendedOp.JavaContinueOp $ -> (OW) new JavaContinueOpWrapper($);
            default -> (OW) new OpWrapper<>(op);
        };
    }

    private final T op;

    OpWrapper(T op) {
        this.op = op;
    }

    public T op() {
        return (T) op;
    }

    public Body firstBody() {
        if (op.bodies().isEmpty()) {
            throw new IllegalStateException("no body!");
        }
        return op.bodies().getFirst();
    }

    public Body onlyBody() {
        if (op.bodies().size() != 1) {
            throw new IllegalStateException("not the only body!");
        }
        return firstBody();
    }

    public void onlyBody(Consumer<BodyWrapper> bodyWrapperConsumer) {
        bodyWrapperConsumer.accept(new BodyWrapper(onlyBody()));
    }

    public final Stream<Body> bodies() {
        return op.bodies().stream();
    }

    public void selectOnlyBlockOfOnlyBody(Consumer<BlockWrapper> blockWrapperConsumer) {
        onlyBody(w -> {
            w.onlyBlock(blockWrapperConsumer);
        });
    }

    public void selectCalls(Consumer<InvokeOpWrapper> consumer) {
        this.op.traverse(null, (map, op) -> {
            if (op instanceof CoreOp.InvokeOp invokeOp) {
                consumer.accept(wrap(invokeOp));
            }
            return map;
        });
    }

    public BlockWrapper parentBlock() {
        return new BlockWrapper(op.parentBlock());
    }

    public BodyWrapper parentBodyOfParentBlock() {
        return parentBlock().parentBody();
    }


    public Op.Result operandNAsResult(int i) {
        if (operandNAsValue(i) instanceof Op.Result result) {
            return result;
        } else {
            return null;
        }
    }

    public Value operandNAsValue(int i) {
        return hasOperandN(i) ? op().operands().get(i) : null;
    }

    public boolean hasOperandN(int i) {
        return operandCount() > i;
    }

    public int operandCount() {
        return op().operands().size();
    }

    public boolean hasOperands() {
        return !hasNoOperands();
    }

    public boolean hasNoOperands() {
        return operands().isEmpty();
    }

    public List<Value> operands() {
        return op.operands();
    }

    public Body bodyN(int i) {
        return op().bodies().get(i);
    }

    public boolean hasBodyN(int i) {
        return op().bodies().size() > i;
    }

    public Block firstBlockOfBodyN(int i) {
        return bodyN(i).blocks().getFirst();
    }

    public Block firstBlockOfFirstBody() {
        return op().bodies().getFirst().blocks().getFirst();
    }

    public String toText() {
        return op().toText();
    }

    public Stream<OpWrapper<?>> wrappedOpStream(Block block) {
        return block.ops().stream().map(OpWrapper::wrap);
    }

    public Stream<OpWrapper<?>> wrappedYieldOpStream(Block block) {
        return wrappedOpStream(block).filter(wrapped -> wrapped instanceof YieldOpWrapper);
    }

    private Stream<OpWrapper<?>> roots(Block block) {
        var rootSet = RootSet.getRootSet(block.ops().stream());
        return block.ops().stream().filter(rootSet::contains).map(OpWrapper::wrap);
    }

    private Stream<OpWrapper<?>> rootsWithoutVarFuncDeclarations(Block block) {
        return roots(block).filter(w -> !(w instanceof VarFuncDeclarationOpWrapper));
    }

    private Stream<OpWrapper<?>> rootsWithoutVarFuncDeclarationsOrYields(Block block) {
        return rootsWithoutVarFuncDeclarations(block).filter(w -> !(w instanceof YieldOpWrapper));
    }

    public Stream<OpWrapper<?>> wrappedRootOpStream(Block block) {
        return rootsWithoutVarFuncDeclarationsOrYields(block);
    }

    public Stream<OpWrapper<?>> wrappedRootOpStreamSansFinalContinue(Block block) {
        var list = new ArrayList<>(rootsWithoutVarFuncDeclarationsOrYields(block).toList());
        if (list.getLast() instanceof JavaContinueOpWrapper javaContinueOpWrapper) {
            list.removeLast();
        }
        return list.stream();
    }

    public Op.Result result() {
        return op.result();
    }

    public TypeElement resultType() {
        return op.resultType();
    }
}
