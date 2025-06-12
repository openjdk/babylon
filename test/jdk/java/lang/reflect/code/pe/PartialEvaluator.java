/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Array;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

final class PartialEvaluator {
    final Set<Value> constants;
    final Predicate<Op> opConstant;

    PartialEvaluator(Set<Value> constants, Predicate<Op> opConstant) {
        this.constants = new LinkedHashSet<>(constants);
        this.opConstant = opConstant;
    }

    public static
    CoreOp.FuncOp evaluate(MethodHandles.Lookup l,
                           Predicate<Op> opConstant, Set<Value> constants,
                           CoreOp.FuncOp op) {
        PartialEvaluator pe = new PartialEvaluator(constants, opConstant);
        Body.Builder outBody = pe.evaluateBody(l, op.body());
        return CoreOp.func(op.funcName(), outBody);
    }


    @SuppressWarnings("serial")
    public static final class EvaluationException extends RuntimeException {
        private EvaluationException(Throwable cause) {
            super(cause);
        }
    }

    static EvaluationException evaluationException(Throwable cause) {
        return new EvaluationException(cause);
    }

    static final class BodyContext {
        final BodyContext parent;

        final Map<Block, List<Block>> evaluatedPredecessors;
        final Map<Value, Object> evaluatedValues;

        final Queue<Block> blockStack;
        final BitSet visited;

        BodyContext(Block entryBlock) {
            this.parent = null;

            this.evaluatedPredecessors = new HashMap<>();
            this.evaluatedValues = new HashMap<>();
            this.blockStack = new PriorityQueue<>(Comparator.comparingInt(Block::index));

            this.visited = new BitSet();
        }

        Object getValue(Value v) {
            Object rv = evaluatedValues.get(v);
            if (rv != null) {
                return rv;
            }

            throw evaluationException(new IllegalArgumentException("Undefined value: " + v));
        }

        void setValue(Value v, Object o) {
            evaluatedValues.put(v, o);
        }
    }

    Body.Builder evaluateBody(MethodHandles.Lookup l,
                              Body inBody) {
        Block inEntryBlock = inBody.entryBlock();

        Body.Builder outBody = Body.Builder.of(null, inBody.bodyType());
        Block.Builder outEntryBlock = outBody.entryBlock();

        CopyContext cc = outEntryBlock.context();
        cc.mapBlock(inEntryBlock, outEntryBlock);
        cc.mapValues(inEntryBlock.parameters(), outEntryBlock.parameters());

        evaluateEntryBlock(l, inEntryBlock, outEntryBlock, new BodyContext(inEntryBlock));

        return outBody;
    }

    void evaluateEntryBlock(MethodHandles.Lookup l,
                            Block inEntryBlock,
                            Block.Builder outEntryBlock,
                            BodyContext bc) {
        assert inEntryBlock.isEntryBlock();

        Map<Block, LoopAnalyzer.Loop> loops = new HashMap<>();
        Set<Block> loopNoPeeling = new HashSet<>();

        // The first block cannot have any successors so the queue will have at least one entry
        bc.blockStack.add(inEntryBlock);
        while (!bc.blockStack.isEmpty()) {
            final Block inBlock = bc.blockStack.poll();
            if (bc.visited.get(inBlock.index())) {
                continue;
            }
            bc.visited.set(inBlock.index());

            final Block.Builder outBlock = outEntryBlock.context().getBlock(inBlock);

            nopeel: if (inBlock.predecessors().size() > 1 && bc.evaluatedPredecessors.get(inBlock).size() == 1) {
                // If we reached to this block through just one evaluated predecessor
                Block inBlockPred = bc.evaluatedPredecessors.get(inBlock).getFirst();
                Block.Reference inBlockRef = inBlockPred.terminatingOp().successors().stream()
                        .filter(r -> r.targetBlock() == inBlock)
                        .findFirst().get();
                List<Value> args = inBlockRef.arguments();
                List<Boolean> argConstant = args.stream().map(constants::contains).toList();

                LoopAnalyzer.Loop loop = loops.computeIfAbsent(inBlock, b -> LoopAnalyzer.isLoop(inBlock).orElse(null));
                if (loop != null && inBlockPred.isDominatedBy(loop.header())) {
                    // Entering loop header from latch
                    assert loop.latches().contains(inBlockPred);

                    // Linear constant path from each exiting block (or nearest evaluated present dominator) to loop header
                    boolean constantExits = true;
                    for (LoopAnalyzer.LoopExit loopExitPair : loop.exits()) {
                        Block loopExit = loopExitPair.exit();

                        // Find nearest evaluated dominator
                        List<Block> ePreds = bc.evaluatedPredecessors.get(loopExit);
                        while (ePreds == null) {
                            loopExit = loopExit.immediateDominator();
                            ePreds = bc.evaluatedPredecessors.get(loopExit);
                        }
                        assert loop.body().contains(loopExit);

                        if (ePreds.size() != 1 ||
                                !(loopExit.terminatingOp() instanceof CoreOp.ConditionalBranchOp cbr) ||
                                !constants.contains(cbr.result())) {
                            // If there are multiple encounters, or terminal op is not a constant conditional branch
                            constantExits = false;
                            break;
                        }
                    }

                    // Determine if constant args, before reset
                    boolean constantArgs = constants.containsAll(args);

                    // Reset state within loop body
                    for (Block block : loop.body()) {
                        // Reset visits, but not for loop header
                        if (block != loop.header()) {
                            bc.evaluatedPredecessors.remove(block);
                            bc.visited.set(block.index(), false);
                        }

                        // Reset constants
                        for (Op op : block.ops()) {
                            constants.remove(op.result());
                        }
                        constants.removeAll(block.parameters());

                        // Reset no peeling for any nested loops
                        loopNoPeeling.remove(block);
                    }

                    if (!constantExits || !constantArgs) {
                        // Finish peeling
                        // No constant exit and no constant args
                        loopNoPeeling.addAll(loop.latches());
                        break nopeel;
                    }
                    // Peel next iteration
                }

                // Propagate constant arguments
                for (int i = 0; i < args.size(); i++) {
                    Value inArgument = args.get(i);
                    if (argConstant.get(i)) {
                        Block.Parameter inParameter = inBlock.parameters().get(i);

                        // Map input parameter to output argument
                        outBlock.context().mapValue(inParameter, outBlock.context().getValue(inArgument));
                        // Set parameter constant
                        constants.add(inParameter);
                        bc.setValue(inParameter, bc.getValue(inArgument));
                    }
                }
            }

            // Process all but the terminating operation
            int nops = inBlock.ops().size();
            for (int i = 0; i < nops - 1; i++) {
                Op op = inBlock.ops().get(i);

                if (isConstant(op)) {
                    // Evaluate operation
                    // @@@ Handle exceptions
                    Object result = interpretOp(l, bc, op);
                    bc.setValue(op.result(), result);

                    if (op instanceof CoreOp.VarOp) {
                        // @@@ Do not turn into constant to avoid conflicts with the interpreter
                        // and its runtime representation of vars
                        outBlock.op(op);
                    } else {
                        // Result was evaluated, replace with constant operation
                        Op.Result constantResult = outBlock.op(CoreOp.constant(op.resultType(), result));
                        outBlock.context().mapValue(op.result(), constantResult);
                    }
                } else {
                    // Copy unevaluated operation
                    Op.Result r = outBlock.op(op);
                    // Explicitly remap result, since the op can be copied more than once in pealed loops
                    // @@@ See comment Block.op code which implicitly limits this
                    outBlock.context().mapValue(op.result(), r);
                }
            }

            // Process the terminating operation
            Op to = inBlock.terminatingOp();
            switch (to) {
                case CoreOp.ConditionalBranchOp cb -> {
                    if (isConstant(to)) {
                        boolean p = switch (bc.getValue(cb.predicate())) {
                            case Boolean bp -> bp;
                            case Integer ip ->
                                // @@@ This is required when lifting up from bytecode, since boolean values
                                // are erased to int values, abd the bytecode lifting implementation is not currently
                                // sophisticated enough to recover the type information
                                    ip != 0;
                            default -> throw evaluationException(
                                    new UnsupportedOperationException("Unsupported type input to operation: " + cb));
                        };

                        Block.Reference nextInBlockRef = p ? cb.trueBranch() : cb.falseBranch();
                        Block nextInBlock = nextInBlockRef.targetBlock();

                        // @@@ might be latch to loop
                        assert !inBlock.isDominatedBy(nextInBlock);

                        processBlock(bc, inBlock, nextInBlock, outBlock);

                        outBlock.op(CoreOp.branch(outBlock.context().getSuccessorOrCreate(nextInBlockRef)));
                    } else {
                        // @@@ might be non-constant latch to loop
                        processBlock(bc, inBlock, cb.falseBranch().targetBlock(), outBlock);
                        processBlock(bc, inBlock, cb.trueBranch().targetBlock(), outBlock);

                        outBlock.op(to);
                    }
                }
                case CoreOp.BranchOp b -> {
                    Block.Reference nextInBlockRef = b.branch();
                    Block nextInBlock = nextInBlockRef.targetBlock();

                    if (inBlock.isDominatedBy(nextInBlock)) {
                        // latch to loop header
                        assert bc.visited.get(nextInBlock.index());
                        if (!loopNoPeeling.contains(inBlock) && constants.containsAll(nextInBlock.parameters())) {
                            // Reset loop body to peel off another iteration
                            bc.visited.set(nextInBlock.index(), false);
                            bc.evaluatedPredecessors.remove(nextInBlock);
                        }
                    }

                    processBlock(bc, inBlock, nextInBlock, outBlock);

                    outBlock.op(b);
                }
                case CoreOp.ReturnOp _ -> outBlock.op(to);
                default -> throw evaluationException(
                        new UnsupportedOperationException("Unsupported terminating operation: " + to.opName()));
            }
        }
    }

    boolean isConstant(Op op) {
        if (constants.contains(op.result())) {
            return true;
        } else if (constants.containsAll(op.operands()) && opConstant.test(op)) {
            constants.add(op.result());
            return true;
        } else {
            return false;
        }
    }

    void processBlock(BodyContext bc, Block inBlock, Block nextInBlock, Block.Builder outBlock) {
        bc.blockStack.add(nextInBlock);
        if (!bc.evaluatedPredecessors.containsKey(nextInBlock)) {
            // Copy block
            Block.Builder nextOutBlock = outBlock.block(nextInBlock.parameterTypes());
            outBlock.context().mapBlock(nextInBlock, nextOutBlock);
            outBlock.context().mapValues(nextInBlock.parameters(), nextOutBlock.parameters());
        }
        bc.evaluatedPredecessors.computeIfAbsent(nextInBlock, _ -> new ArrayList<>()).add(inBlock);
    }

    @SuppressWarnings("unchecked")
    public static <E extends Throwable> void eraseAndThrow(Throwable e) throws E {
        throw (E) e;
    }

    // @@@ This could be shared with the interpreter if it was more extensible
    Object interpretOp(MethodHandles.Lookup l, BodyContext bc, Op o) {
        switch (o) {
            case CoreOp.ConstantOp co -> {
                if (co.resultType().equals(JavaType.J_L_CLASS)) {
                    return resolveToClass(l, (JavaType) co.value());
                } else {
                    return co.value();
                }
            }
            case JavaOp.InvokeOp co -> {
                MethodType target = resolveToMethodType(l, o.opType());
                MethodHandles.Lookup il = switch (co.invokeKind()) {
                    case STATIC, INSTANCE -> l;
                    case SUPER -> l.in(target.parameterType(0));
                };
                MethodHandle mh = resolveToMethodHandle(il, co.invokeDescriptor(), co.invokeKind());

                mh = mh.asType(target).asFixedArity();
                Object[] values = o.operands().stream().map(bc::getValue).toArray();
                return invoke(mh, values);
            }
            case JavaOp.NewOp no -> {
                Object[] values = o.operands().stream().map(bc::getValue).toArray();
                JavaType nType = (JavaType) no.resultType();
                if (nType instanceof ArrayType at) {
                    if (values.length > at.dimensions()) {
                        throw evaluationException(new IllegalArgumentException("Bad constructor NewOp: " + no));
                    }
                    int[] lengths = Stream.of(values).mapToInt(v -> (int) v).toArray();
                    for (int length : lengths) {
                        nType = ((ArrayType) nType).componentType();
                    }
                    return Array.newInstance(resolveToClass(l, nType), lengths);
                } else {
                    MethodHandle mh = constructorHandle(l, no.constructorDescriptor().type());
                    return invoke(mh, values);
                }
            }
            case CoreOp.VarOp vo -> {
                Object[] vbox = vo.isUninitialized()
                        ? new Object[] { null, false }
                        : new Object[] { bc.getValue(o.operands().get(0)) };
                return vbox;
            }
            case CoreOp.VarAccessOp.VarLoadOp vlo -> {
                // Cast to CoreOp.Var, since the instance may have originated as an external instance
                // via a captured value map
                Object[] vbox = (Object[]) bc.getValue(o.operands().get(0));
                if (vbox.length == 2 && !((Boolean) vbox[1])) {
                    throw evaluationException(new IllegalStateException("Loading from uninitialized variable"));
                }
                return vbox[0];
            }
            case CoreOp.VarAccessOp.VarStoreOp vso -> {
                Object[] vbox = (Object[]) bc.getValue(o.operands().get(0));
                if (vbox.length == 2) {
                    vbox[1] = true;
                }
                vbox[0] = bc.getValue(o.operands().get(1));
                return null;
            }
            case CoreOp.TupleOp to -> {
                return o.operands().stream().map(bc::getValue).toList();
            }
            case CoreOp.TupleLoadOp tlo -> {
                @SuppressWarnings("unchecked")
                List<Object> tb = (List<Object>) bc.getValue(o.operands().get(0));
                return tb.get(tlo.index());
            }
            case CoreOp.TupleWithOp two -> {
                @SuppressWarnings("unchecked")
                List<Object> tb = (List<Object>) bc.getValue(o.operands().get(0));
                List<Object> copy = new ArrayList<>(tb);
                copy.set(two.index(), bc.getValue(o.operands().get(1)));
                return Collections.unmodifiableList(copy);
            }
            case JavaOp.FieldAccessOp.FieldLoadOp fo -> {
                if (fo.operands().isEmpty()) {
                    VarHandle vh = fieldStaticHandle(l, fo.fieldDescriptor());
                    return vh.get();
                } else {
                    Object v = bc.getValue(o.operands().get(0));
                    VarHandle vh = fieldHandle(l, fo.fieldDescriptor());
                    return vh.get(v);
                }
            }
            case JavaOp.FieldAccessOp.FieldStoreOp fo -> {
                if (fo.operands().size() == 1) {
                    Object v = bc.getValue(o.operands().get(0));
                    VarHandle vh = fieldStaticHandle(l, fo.fieldDescriptor());
                    vh.set(v);
                } else {
                    Object r = bc.getValue(o.operands().get(0));
                    Object v = bc.getValue(o.operands().get(1));
                    VarHandle vh = fieldHandle(l, fo.fieldDescriptor());
                    vh.set(r, v);
                }
                return null;
            }
            case JavaOp.InstanceOfOp io -> {
                Object v = bc.getValue(o.operands().get(0));
                return isInstance(l, io.type(), v);
            }
            case JavaOp.CastOp co -> {
                Object v = bc.getValue(o.operands().get(0));
                return cast(l, co.type(), v);
            }
            case JavaOp.ArrayLengthOp arrayLengthOp -> {
                Object a = bc.getValue(o.operands().get(0));
                return Array.getLength(a);
            }
            case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                Object a = bc.getValue(o.operands().get(0));
                Object index = bc.getValue(o.operands().get(1));
                return Array.get(a, (int) index);
            }
            case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                Object a = bc.getValue(o.operands().get(0));
                Object index = bc.getValue(o.operands().get(1));
                Object v = bc.getValue(o.operands().get(2));
                Array.set(a, (int) index, v);
                return null;
            }
            case JavaOp.ArithmeticOperation arithmeticOperation -> {
                MethodHandle mh = opHandle(l, o.opName(), o.opType());
                Object[] values = o.operands().stream().map(bc::getValue).toArray();
                return invoke(mh, values);
            }
            case JavaOp.TestOperation testOperation -> {
                MethodHandle mh = opHandle(l, o.opName(), o.opType());
                Object[] values = o.operands().stream().map(bc::getValue).toArray();
                return invoke(mh, values);
            }
            case JavaOp.ConvOp convOp -> {
                MethodHandle mh = opHandle(l, o.opName() + "_" + o.opType().returnType(), o.opType());
                Object[] values = o.operands().stream().map(bc::getValue).toArray();
                return invoke(mh, values);
            }
            case JavaOp.ConcatOp concatOp -> {
                return o.operands().stream()
                        .map(bc::getValue)
                        .map(String::valueOf)
                        .collect(Collectors.joining());
            }
            // @@@
//            case CoreOp.LambdaOp lambdaOp -> {
//                interpretEntryBlock(l, lambdaOp.body().entryBlock(), oc, new HashMap<>());
//                unevaluatedOperations.add(o);
//                return null;
//            }
//            case CoreOp.FuncOp funcOp -> {
//                interpretEntryBlock(l, funcOp.body().entryBlock(), oc, new HashMap<>());
//                unevaluatedOperations.add(o);
//                return null;
//            }
            case null, default -> throw evaluationException(
                    new UnsupportedOperationException("Unsupported operation: " + o.opName()));
        }
    }


    static MethodHandle opHandle(MethodHandles.Lookup l, String opName, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft).erase();
        try {
            return MethodHandles.lookup().findStatic(InvokableLeafOps.class, opName, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw evaluationException(e);
        }
    }

    static MethodHandle constructorHandle(MethodHandles.Lookup l, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft);

        if (mt.returnType().isArray()) {
            if (mt.parameterCount() != 1 || mt.parameterType(0) != int.class) {
                throw evaluationException(new IllegalArgumentException("Bad constructor descriptor: " + ft));
            }
            return MethodHandles.arrayConstructor(mt.returnType());
        } else {
            try {
                return l.findConstructor(mt.returnType(), mt.changeReturnType(void.class));
            } catch (NoSuchMethodException | IllegalAccessException e) {
                throw evaluationException(e);
            }
        }
    }

    static VarHandle fieldStaticHandle(MethodHandles.Lookup l, FieldRef d) {
        return resolveToVarHandle(l, d);
    }

    static VarHandle fieldHandle(MethodHandles.Lookup l, FieldRef d) {
        return resolveToVarHandle(l, d);
    }

    static Object isInstance(MethodHandles.Lookup l, TypeElement d, Object v) {
        Class<?> c = resolveToClass(l, d);
        return c.isInstance(v);
    }

    static Object cast(MethodHandles.Lookup l, TypeElement d, Object v) {
        Class<?> c = resolveToClass(l, d);
        return c.cast(v);
    }

    static MethodHandle resolveToMethodHandle(MethodHandles.Lookup l, MethodRef d, JavaOp.InvokeOp.InvokeKind kind) {
        try {
            return d.resolveToHandle(l, kind);
        } catch (ReflectiveOperationException e) {
            throw evaluationException(e);
        }
    }

    static VarHandle resolveToVarHandle(MethodHandles.Lookup l, FieldRef d) {
        try {
            return d.resolveToHandle(l);
        } catch (ReflectiveOperationException e) {
            throw evaluationException(e);
        }
    }

    public static MethodType resolveToMethodType(MethodHandles.Lookup l, FunctionType ft) {
        try {
            return MethodRef.toNominalDescriptor(ft).resolveConstantDesc(l);
        } catch (ReflectiveOperationException e) {
            throw evaluationException(e);
        }
    }

    public static Class<?> resolveToClass(MethodHandles.Lookup l, TypeElement d) {
        try {
            if (d instanceof JavaType jt) {
                return (Class<?>) jt.erasure().resolve(l);
            } else {
                throw new ReflectiveOperationException();
            }
        } catch (ReflectiveOperationException e) {
            throw evaluationException(e);
        }
    }

    static Object invoke(MethodHandle m, Object... args) {
        try {
            return m.invokeWithArguments(args);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable e) {
            eraseAndThrow(e);
            throw new InternalError("should not reach here");
        }
    }
}
