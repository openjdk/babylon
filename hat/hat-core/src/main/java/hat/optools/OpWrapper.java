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

import hat.buffer.Buffer;

import hat.ifacemapper.MappableIface;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.type.ClassType;
import jdk.incubator.code.type.JavaType;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class OpWrapper<T extends Op> {
    @SuppressWarnings("unchecked")
    public static <O extends Op, OW extends OpWrapper<O>> OW wrap(MethodHandles.Lookup lookup,O op) {
        // We have one special case
        // This is possibly a premature optimization. But it allows us to treat vardeclarations differently from params.
        if (op instanceof CoreOp.VarOp varOp && !varOp.isUninitialized()) {
            // this gets called a lot and we can't wrap yet or we recurse so we
            // use the raw model. Basically we want a different wrapper for VarDeclarations
            // which  relate to func parameters.
            // This saves us asking each time if a var is indeed a func param.

            if (varOp.operands().getFirst() instanceof Block.Parameter parameter &&
                    parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp) {
                return (OW) new VarFuncDeclarationOpWrapper(lookup,varOp, funcOp, parameter);
            }
        }
        return switch (op) {
            case CoreOp.ModuleOp $ -> (OW) new ModuleOpWrapper(lookup, $);
            case ExtendedOp.JavaForOp $ -> (OW) new ForOpWrapper(lookup, $);
            case ExtendedOp.JavaWhileOp $ -> (OW) new WhileOpWrapper(lookup, $);
            case ExtendedOp.JavaIfOp $ -> (OW) new IfOpWrapper(lookup, $);
            case CoreOp.NotOp $ -> (OW) new UnaryArithmeticOrLogicOpWrapper(lookup, $);
            case CoreOp.NegOp $ -> (OW) new UnaryArithmeticOrLogicOpWrapper(lookup, $);
            case CoreOp.BinaryOp $ -> (OW) new BinaryArithmeticOrLogicOperation(lookup, $);
            case CoreOp.BinaryTestOp $ -> (OW) new BinaryTestOpWrapper(lookup, $);
            case CoreOp.FuncOp $ -> (OW) new FuncOpWrapper(lookup, $);
            case CoreOp.VarOp $ -> (OW) new VarDeclarationOpWrapper(lookup, $);
            case CoreOp.YieldOp $ -> (OW) new YieldOpWrapper(lookup, $);
            case CoreOp.FuncCallOp $ -> (OW) new FuncCallOpWrapper(lookup, $);
            case CoreOp.ConvOp $ -> (OW) new ConvOpWrapper(lookup, $);
            case CoreOp.ConstantOp $ -> (OW) new ConstantOpWrapper(lookup, $);
            case CoreOp.ReturnOp $ -> (OW) new ReturnOpWrapper(lookup, $);
            case CoreOp.VarAccessOp.VarStoreOp $ -> (OW) new VarStoreOpWrapper(lookup, $);
            case CoreOp.VarAccessOp.VarLoadOp $ -> (OW) new VarLoadOpWrapper(lookup, $);
            case CoreOp.FieldAccessOp.FieldStoreOp $ -> (OW) new FieldStoreOpWrapper(lookup, $);
            case CoreOp.FieldAccessOp.FieldLoadOp $ -> (OW) new FieldLoadOpWrapper(lookup, $);
            case CoreOp.InvokeOp $ -> (OW) new InvokeOpWrapper(lookup, $);
            case CoreOp.TupleOp $ -> (OW) new TupleOpWrapper(lookup, $);
            case CoreOp.LambdaOp $ -> (OW) new LambdaOpWrapper(lookup, $);
            case ExtendedOp.JavaConditionalOp $ -> (OW) new LogicalOpWrapper(lookup, $);
            case ExtendedOp.JavaConditionalExpressionOp $ -> (OW) new TernaryOpWrapper(lookup, $);
            case ExtendedOp.JavaLabeledOp $ -> (OW) new JavaLabeledOpWrapper(lookup, $);
            case ExtendedOp.JavaBreakOp $ -> (OW) new JavaBreakOpWrapper(lookup, $);
            case ExtendedOp.JavaContinueOp $ -> (OW) new JavaContinueOpWrapper(lookup, $);
            default -> (OW) new OpWrapper<>(lookup,op);
        };
    }

    private final T op;
public MethodHandles.Lookup lookup;
    OpWrapper( MethodHandles.Lookup lookup,T op) {
        this.lookup= lookup;
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
                consumer.accept(wrap(lookup,invokeOp));
            }
            return map;
        });
    }
    public void selectAssignments(Consumer<VarOpWrapper> consumer) {
        this.op.traverse(null, (map, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                consumer.accept(wrap(lookup,varOp));
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
        return bodyN(i).entryBlock();
    }

    public Block firstBlockOfFirstBody() {
        return op().bodies().getFirst().entryBlock();
    }

    public String toText() {
        return op().toText();
    }

    public Stream<OpWrapper<?>> wrappedOpStream(Block block) {
        return block.ops().stream().map(o->wrap(lookup,o));
    }

    public Stream<OpWrapper<?>> wrappedYieldOpStream(Block block) {
        return wrappedOpStream(block).filter(wrapped -> wrapped instanceof YieldOpWrapper);
    }

    private Stream<OpWrapper<?>> roots(Block block) {
        var rootSet = RootSet.getRootSet(block.ops().stream());
        return block.ops().stream().filter(rootSet::contains).map(o->wrap(lookup,o));
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
    public  static boolean isIfaceUsingLookup(MethodHandles.Lookup lookup,JavaType javaType) {
        return  (isAssignableUsingLookup(lookup,javaType, MappableIface.class));
    }
    public  boolean isIface(JavaType javaType) {
        return  (isAssignable(javaType, MappableIface.class));
    }

    public  static Type classTypeToTypeUsingLookup(MethodHandles.Lookup lookup,ClassType classType) {
        Type javaTypeClass = null;
        try {
            javaTypeClass = classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        return javaTypeClass;

    }
    public  Type classTypeToType(ClassType classType) {
       return classTypeToTypeUsingLookup(lookup,classType);
    }
    public  static boolean isAssignableUsingLookup(MethodHandles.Lookup lookup,JavaType javaType, Class<?> ... classes) {
        if (javaType instanceof ClassType classType) {
            Type type = classTypeToTypeUsingLookup(lookup,classType);
            for (Class<?> clazz : classes) {
                if (clazz.isAssignableFrom((Class<?>) type)) {
                    return true;
                }
            }
        }
        return false;

    }
    public  boolean isAssignable(JavaType javaType, Class<?> ... classes) {
       return isAssignableUsingLookup(lookup,javaType,classes);

    }
}
