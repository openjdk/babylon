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

package java.lang.reflect.code.interpreter;

import java.lang.invoke.*;
import java.lang.reflect.Array;
import java.lang.reflect.Proxy;
import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.ArrayType;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.VarType;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toMap;

public final class Interpreter {
    private Interpreter() {
    }

    @SuppressWarnings("serial")
    public static class InterpreterException extends RuntimeException {
        public InterpreterException(Throwable cause) {
            super(cause);
        }
    }

    static InterpreterException interpreterException(Throwable cause) {
        return new InterpreterException(cause);
    }

    record BlockContext(Block b, Map<Value, Object> values) {
    }

    static final ConcurrentHashMap<Object, ReentrantLock> locks = new ConcurrentHashMap<>();

    static final class OpContext {
        final Map<Object, ReentrantLock> locks = new HashMap<>();
        final Deque<BlockContext> stack = new ArrayDeque<>();
        final Deque<ExceptionRegionRecord> erStack = new ArrayDeque<>();

        Object getValue(Value v) {
            // @@@ Only dominating values are accessible
            BlockContext bc = findContext(v);
            if (bc != null) {
                return bc.values.get(v);
            } else {
                throw interpreterException(new IllegalArgumentException("Undefined value: " + v));
            }
        }

        Object setValue(Value v, Object o) {
            BlockContext bc = findContext(v);
            if (bc != null) {
                throw interpreterException(new IllegalArgumentException("Value already defined: " + v));
            }
            stack.peek().values.put(v, o);
            return o;
        }

        BlockContext findContext(Value v) {
            Optional<BlockContext> ob = stack.stream().filter(b -> b.values.containsKey(v)).findFirst();
            return ob.orElse(null);
        }

        boolean contains(Block.Reference s) {
            Block sb = s.targetBlock();
            return stack.stream().anyMatch(bc -> bc.b.equals(sb));
        }

        void successor(Block.Reference sb) {
            List<Object> sbValues = sb.arguments().stream().map(this::getValue).toList();

            Block b = sb.targetBlock();
            Map<Value, Object> bValues = new HashMap<>();
            for (int i = 0; i < sbValues.size(); i++) {
                bValues.put(b.parameters().get(i), sbValues.get(i));
            }

            if (contains(sb)) {
                // if block is already dominating pop back up from the back branch to the block
                // before the successor block
                while (!stack.peek().b.equals(sb.targetBlock())) {
                    stack.pop();
                }
                stack.pop();
            }
            stack.push(new BlockContext(b, bValues));
        }

        void successor(Block b, Map<Value, Object> bValues) {
            stack.push(new BlockContext(b, bValues));
        }

        void popTo(BlockContext bc) {
            while (!stack.peek().equals(bc)) {
                stack.pop();
            }
        }

        void pushExceptionRegion(ExceptionRegionRecord erb) {
            erStack.push(erb);
        }

        void popExceptionRegion(CoreOp.ExceptionRegionEnter ers) {
            if (erStack.peek().ers != ers) {
                // @@@ Use internal exception type
                throw interpreterException(new IllegalStateException("Mismatched exception regions"));
            }
            erStack.pop();
        }

        Block exception(MethodHandles.Lookup l, Throwable e) {
            // Find the first matching exception region
            // with a catch block whose argument type is assignable-compatible to the throwable
            ExceptionRegionRecord er;
            Block cb = null;
            while ((er = erStack.poll()) != null &&
                    (cb = er.match(l, e)) == null) {
            }

            if (er == null) {
                return null;
            }

            // Pop the block context to the block defining the start of the exception region
            popTo(er.mark);
            return cb;
        }
    }

    static final class VarBox
            implements CoreOp.Var<Object> {
        Object value;

        public Object value() {
            return value;
        }

        VarBox(Object value) {
            this.value = value;
        }
    }

    record ClosureRecord(CoreOp.ClosureOp op,
                         Map<Value, Object> capturedValues) {
    }

    record TupleRecord(List<Object> components) {
        Object getComponent(int index) {
            return components.get(index);
        }

        TupleRecord with(int index, Object value) {
            List<Object> copy = new ArrayList<>(components);
            copy.set(index, value);
            return new TupleRecord(copy);
        }
    }

    record ExceptionRegionRecord(BlockContext mark, CoreOp.ExceptionRegionEnter ers)
            implements CoreOp.ExceptionRegion {
        Block match(MethodHandles.Lookup l, Throwable e) {
            for (Block.Reference catchBlock : ers.catchBlocks()) {
                Block target = catchBlock.targetBlock();
                List<Block.Parameter> args = target.parameters();
                if (args.size() != 1) {
                    throw interpreterException(new IllegalStateException("Catch block must have one argument"));
                }
                TypeElement et = args.get(0).type();
                if (et instanceof VarType vt) {
                    et = vt.valueType();
                }
                if (resolveToClass(l, et).isInstance(e)) {
                    return target;
                }
            }
            return null;
        }
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(T op,
                  Object... args) {
        return invoke(MethodHandles.publicLookup(), op, args);
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(MethodHandles.Lookup l, T op,
                  Object... args) {
        return invoke(l, op, new ArrayList<>(Arrays.asList(args)));
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(MethodHandles.Lookup l, T op,
                  Map<Value, Object> capturedValues,
                  Object... args) {
        return invoke(l, op, capturedValues, new ArrayList<>(Arrays.asList(args)));
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(T op,
                  List<Object> args) {
        return invoke(MethodHandles.publicLookup(), op, args);
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(T op,
                  Map<Value, Object> capturedValues,
                  List<Object> args) {
        return invoke(MethodHandles.publicLookup(), op, capturedValues, args);
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(MethodHandles.Lookup l, T op,
                  List<Object> args) {
        return invoke(l, op, Map.of(), args);
    }

    public static <T extends Op & Op.Invokable>
    Object invoke(MethodHandles.Lookup l, T invokableOp,
                  Map<Value, Object> capturedValues,
                  List<Object> args) {
        Body r = invokableOp.bodies().get(0);
        return invoke(l, r, capturedValues, new OpContext(), args);

    }
    static Object invoke(MethodHandles.Lookup l, Body r,
                                 Map<Value, Object> capturedValues, OpContext oc,
                                 List<Object> args) {
        return invoke(l,r.entryBlock(), capturedValues, oc, args);
    }

    private static Object invoke(MethodHandles.Lookup l, Block first,
                  Map<Value, Object> capturedValues, OpContext oc,
                  List<Object> args) {

        if (args.size() != first.parameters().size()) {
            throw interpreterException(new IllegalArgumentException("Incorrect number of arguments"));
        }
        Map<Value, Object> values = new HashMap<>();
        for (int i = 0; i < first.parameters().size(); i++) {
            values.put(first.parameters().get(i), args.get(i));
        }

        // Note that first block cannot have any successors so the queue will have at least one entry
        oc.stack.push(new BlockContext(first, values));
        capturedValues.forEach(oc::setValue);
        while (true) {
            BlockContext bc = oc.stack.peek();

            // Execute all but the terminating operation
            int nops = bc.b.ops().size();
            try {
                for (int i = 0; i < nops - 1; i++) {
                    Op op = bc.b.ops().get(i);
                    assert !(op instanceof Op.Terminating) : op.opName();

                    Object result = exec(l, oc, op);
                    oc.setValue(op.result(), result);
                }
            } catch (InterpreterException e) {
                throw e;
            } catch (Throwable t) {
                processThrowable(oc, l, t);
                continue;
            }

            // Execute the terminating operation
            Op to = bc.b.terminatingOp();
            if (to instanceof CoreOp.ConditionalBranchOp cb) {
                boolean p;
                Object bop = oc.getValue(cb.predicate());
                if (bop instanceof Boolean bp) {
                    p = bp;
                } else if (bop instanceof Integer ip) {
                    // @@@ This is required when lifting up from bytecode, since boolean values
                    // are erased to int values, abd the bytecode lifting implementation is not currently
                    // sophisticated enough to recover the type information
                    p = ip != 0;
                } else {
                    throw interpreterException(
                            new UnsupportedOperationException("Unsupported type input to operation: " + cb));
                }
                Block.Reference sb = p ? cb.trueBranch() : cb.falseBranch();
                oc.successor(sb);
            } else if (to instanceof CoreOp.BranchOp b) {
                Block.Reference sb = b.branch();

                oc.successor(sb);
            } else if (to instanceof CoreOp.ThrowOp _throw) {
                Throwable t = (Throwable) oc.getValue(_throw.argument());
                processThrowable(oc, l, t);
            } else if (to instanceof CoreOp.ReturnOp ret) {
                Value rv = ret.returnValue();
                return rv == null ? null : oc.getValue(rv);
            } else if (to instanceof CoreOp.YieldOp yop) {
                Value yv = yop.yieldValue();
                return yv == null ? null : oc.getValue(yv);
            } else if (to instanceof CoreOp.ExceptionRegionEnter ers) {
                var er = new ExceptionRegionRecord(oc.stack.peek(), ers);
                oc.setValue(ers.result(), er);

                oc.pushExceptionRegion(er);

                oc.successor(ers.start());
            } else if (to instanceof CoreOp.ExceptionRegionExit ere) {
                oc.popExceptionRegion(ere.regionStart());

                oc.successor(ere.end());
            } else {
                throw interpreterException(
                        new UnsupportedOperationException("Unsupported terminating operation: " + to.opName()));
            }
        }
    }

    static <T extends Op & Op.Invokable>
    Object interpretBody(MethodHandles.Lookup l, Body r,
                         OpContext oc) {
        return invoke(l, r, Map.of(), oc, List.of());
    }

    static void processThrowable(OpContext oc, MethodHandles.Lookup l, Throwable t) {
        // Find a matching catch block
        Block cb = oc.exception(l, t);
        if (cb == null) {
            // If there is no matching catch bock then rethrow back to the caller
            eraseAndThrow(t);
            throw new InternalError("should not reach here");
        }

        // Add a new block context to the catch block with the exception as the argument
        Map<Value, Object> bValues = new HashMap<>();
        Block.Parameter eArg = cb.parameters().get(0);
        if (eArg.type() instanceof VarType) {
            bValues.put(eArg, new VarBox(t));
        } else {
            bValues.put(eArg, t);
        }
        oc.successor(cb, bValues);
    }



    @SuppressWarnings("unchecked")
    public static <E extends Throwable> void eraseAndThrow(Throwable e) throws E {
        throw (E) e;
    }

    static Object exec(MethodHandles.Lookup l, OpContext oc, Op o) {
        if (o instanceof CoreOp.ConstantOp co) {
            if (co.resultType().equals(JavaType.J_L_CLASS)) {
                return resolveToClass(l, (JavaType) co.value());
            } else {
                return co.value();
            }
        } else if (o instanceof CoreOp.FuncCallOp fco) {
            String name = fco.funcName();

            // Find top-level op
            Op top = fco;
            while (top.ancestorBody() != null) {
                top = top.ancestorBody().parentOp();
            }

            // Ensure top-level op is a module and function name
            // is in the module's function table
            if (top instanceof CoreOp.ModuleOp mop) {
                CoreOp.FuncOp funcOp = mop.functionTable().get(name);
                if (funcOp == null) {
                    throw interpreterException(
                            new IllegalStateException
                                    ("Function " + name + " cannot be resolved: not in module's function table"));
                }

                Object[] values = o.operands().stream().map(oc::getValue).toArray();
                return Interpreter.invoke(funcOp, values);
            } else {
                throw interpreterException(
                        new IllegalStateException(
                                "Function " + name + " cannot be resolved: top level op is not a module"));
            }
        } else if (o instanceof CoreOp.InvokeOp co) {
            MethodType target = resolveToMethodType(l, o.opType());
            MethodHandles.Lookup il = switch (co.invokeKind()) {
                case STATIC, INSTANCE -> l;
                case SUPER -> l.in(target.parameterType(0));
            };
            MethodHandle mh = resolveToMethodHandle(il, co.invokeDescriptor(), co.invokeKind());

            mh = mh.asType(target).asFixedArity();
            Object[] values = o.operands().stream().map(oc::getValue).toArray();
            return invoke(mh, values);
        } else if (o instanceof CoreOp.NewOp no) {
            Object[] values = o.operands().stream().map(oc::getValue).toArray();
            JavaType nType = (JavaType) no.constructorType().returnType();
            if (nType instanceof ArrayType at) {
                if (values.length > at.dimensions()) {
                    throw interpreterException(new IllegalArgumentException("Bad constructor NewOp: " + no));
                }
                int[] lengths = Stream.of(values).mapToInt(v -> (int) v).toArray();
                for (int length : lengths) {
                    nType = ((ArrayType)nType).componentType();
                }
                return Array.newInstance(resolveToClass(l, nType), lengths);
            } else {
                MethodHandle mh = constructorHandle(l, no.constructorType());
                return invoke(mh, values);
            }
        } else if (o instanceof CoreOp.QuotedOp qo) {
            Map<Value, Object> capturedValues = qo.capturedValues().stream()
                    .collect(toMap(v -> v, oc::getValue, (v, _) -> v, LinkedHashMap::new));
            return new Quoted(qo.quotedOp(), capturedValues);
        } else if (o instanceof CoreOp.LambdaOp lo) {
            Map<Value, Object> capturedValues = lo.capturedValues().stream()
                    .collect(toMap(v -> v, oc::getValue, (v, _) -> v, LinkedHashMap::new));
            Class<?> fi = resolveToClass(l, lo.functionalInterface());

            MethodHandle fProxy = INVOKE_LAMBDA_MH.bindTo(l).bindTo(lo).bindTo(capturedValues)
                    .asCollector(Object[].class, lo.body().entryBlock().parameters().size());
            Object fiInstance = MethodHandleProxies.asInterfaceInstance(fi, fProxy);

            // If a quotable lambda proxy again to implement Quotable
            if (Quotable.class.isAssignableFrom(fi)) {
                return Proxy.newProxyInstance(l.lookupClass().getClassLoader(), new Class<?>[]{fi},
                        (_, method, args) -> {
                            if (method.getDeclaringClass() == Quotable.class) {
                                // Implement Quotable::quoted
                                return new Quoted(lo, capturedValues);
                            } else {
                                // Delegate to FI instance
                                return method.invoke(fiInstance, args);
                            }
                        });
            } else {
                return fiInstance;
            }
        } else if (o instanceof CoreOp.ClosureOp co) {
            Map<Value, Object> capturedValues = co.capturedValues().stream()
                    .collect(toMap(v -> v, oc::getValue));
            return new ClosureRecord(co, capturedValues);
        } else if (o instanceof CoreOp.ClosureCallOp cco) {
            List<Object> values = o.operands().stream().map(oc::getValue).toList();
            ClosureRecord cr = (ClosureRecord) values.get(0);

            return Interpreter.invoke(l, cr.op(), cr.capturedValues, values.subList(1, values.size()));
        } else if (o instanceof CoreOp.VarOp vo) {
            return new VarBox(oc.getValue(o.operands().get(0)));
        } else if (o instanceof CoreOp.VarAccessOp.VarLoadOp vlo) {
            // Cast to CoreOp.Var, since the instance may have originated as an external instance
            // via a captured value map
            CoreOp.Var<?> vb = (CoreOp.Var<?>) oc.getValue(o.operands().get(0));
            return vb.value();
        } else if (o instanceof CoreOp.VarAccessOp.VarStoreOp vso) {
            VarBox vb = (VarBox) oc.getValue(o.operands().get(0));
            vb.value = oc.getValue(o.operands().get(1));
            return null;
        } else if (o instanceof CoreOp.TupleOp to) {
            List<Object> values = o.operands().stream().map(oc::getValue).toList();
            return new TupleRecord(values);
        } else if (o instanceof CoreOp.TupleLoadOp tlo) {
            TupleRecord tb = (TupleRecord) oc.getValue(o.operands().get(0));
            return tb.getComponent(tlo.index());
        } else if (o instanceof CoreOp.TupleWithOp two) {
            TupleRecord tb = (TupleRecord) oc.getValue(o.operands().get(0));
            return tb.with(two.index(), oc.getValue(o.operands().get(1)));
        } else if (o instanceof CoreOp.FieldAccessOp.FieldLoadOp fo) {
            if (fo.operands().isEmpty()) {
                VarHandle vh = fieldStaticHandle(l, fo.fieldDescriptor());
                return vh.get();
            } else {
                Object v = oc.getValue(o.operands().get(0));
                VarHandle vh = fieldHandle(l, fo.fieldDescriptor());
                return vh.get(v);
            }
        } else if (o instanceof CoreOp.FieldAccessOp.FieldStoreOp fo) {
            if (fo.operands().size() == 1) {
                Object v = oc.getValue(o.operands().get(0));
                VarHandle vh = fieldStaticHandle(l, fo.fieldDescriptor());
                vh.set(v);
            } else {
                Object r = oc.getValue(o.operands().get(0));
                Object v = oc.getValue(o.operands().get(1));
                VarHandle vh = fieldHandle(l, fo.fieldDescriptor());
                vh.set(r, v);
            }
            return null;
        } else if (o instanceof CoreOp.InstanceOfOp io) {
            Object v = oc.getValue(o.operands().get(0));
            return isInstance(l, io.type(), v);
        } else if (o instanceof CoreOp.CastOp co) {
            Object v = oc.getValue(o.operands().get(0));
            return cast(l, co.type(), v);
        } else if (o instanceof CoreOp.ArrayLengthOp) {
            Object a = oc.getValue(o.operands().get(0));
            return Array.getLength(a);
        } else if (o instanceof CoreOp.ArrayAccessOp.ArrayLoadOp) {
            Object a = oc.getValue(o.operands().get(0));
            Object index = oc.getValue(o.operands().get(1));
            return Array.get(a, (int) index);
        } else if (o instanceof CoreOp.ArrayAccessOp.ArrayStoreOp) {
            Object a = oc.getValue(o.operands().get(0));
            Object index = oc.getValue(o.operands().get(1));
            Object v = oc.getValue(o.operands().get(2));
            Array.set(a, (int) index, v);
            return null;
        } else if (o instanceof CoreOp.ArithmeticOperation || o instanceof CoreOp.TestOperation) {
            MethodHandle mh = opHandle(o.opName(), o.opType());
            Object[] values = o.operands().stream().map(oc::getValue).toArray();
            return invoke(mh, values);
        } else if (o instanceof CoreOp.ConvOp) {
            MethodHandle mh = opHandle(o.opName() + "_" + o.opType().returnType(), o.opType());
            Object[] values = o.operands().stream().map(oc::getValue).toArray();
            return invoke(mh, values);
        } else if (o instanceof CoreOp.AssertOp _assert) {
            //Note: The nature of asserts and munged bodies may require a re-visiting.
            //This code seems to work without poisoning contexts. See TestAssert.java in tests for relevant test coverage.
            Body testBody = _assert.bodies.get(0);
            boolean testResult = (boolean) interpretBody(l, testBody, oc);
            if (!testResult) {
                if (_assert.bodies.size() > 1) {
                    Body messageBody = _assert.bodies.get(1);
                    String message = String.valueOf(interpretBody(l, messageBody, oc));
                    throw new AssertionError(message);
                } else {
                    throw new AssertionError();
                }
            }
            return null;
        } else if (o instanceof CoreOp.ConcatOp) {
            return o.operands().stream()
                    .map(oc::getValue)
                    .map(String::valueOf)
                    .collect(Collectors.joining());
        } else if (o instanceof CoreOp.MonitorOp.MonitorEnterOp) {
            Object monitorTarget = oc.getValue(o.operands().get(0));
            if (monitorTarget == null) {
                throw new NullPointerException();
            }
            ReentrantLock lock = oc.locks.computeIfAbsent(monitorTarget, _ -> new ReentrantLock());
            lock.lock();
            return null;
        } else if (o instanceof CoreOp.MonitorOp.MonitorExitOp) {
            Object monitorTarget = oc.getValue(o.operands().get(0));
            if (monitorTarget == null) {
                throw new NullPointerException();
            }
            ReentrantLock lock = oc.locks.get(monitorTarget);
            if (lock == null) {
                throw new IllegalMonitorStateException();
            }
            lock.unlock();
            return null;
        } else {
            throw interpreterException(
                    new UnsupportedOperationException("Unsupported operation: " + o.opName()));
        }
    }

    static final MethodHandle INVOKE_LAMBDA_MH;
    static {
        try {
            INVOKE_LAMBDA_MH = MethodHandles.lookup().findStatic(Interpreter.class, "invokeLambda",
                    MethodType.methodType(Object.class, MethodHandles.Lookup.class,
                            CoreOp.LambdaOp.class, Map.class, Object[].class));
        } catch (Throwable t) {
            throw new InternalError(t);
        }
    }

    static Object invokeLambda(MethodHandles.Lookup l, CoreOp.LambdaOp op, Map<Value, Object> capturedValues, Object[] args) {
        return invoke(l, op, capturedValues, args);
    }

    static MethodHandle opHandle(String opName, FunctionType ft) {
        MethodType mt = resolveToMethodType(MethodHandles.lookup(), ft).erase();
        try {
            return MethodHandles.lookup().findStatic(InvokableLeafOps.class, opName, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw interpreterException(e);
        }
    }

    static MethodHandle constructorHandle(MethodHandles.Lookup l, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft);

        if (mt.returnType().isArray()) {
            if (mt.parameterCount() != 1 || mt.parameterType(0) != int.class) {
                throw interpreterException(new IllegalArgumentException("Bad constructor descriptor: " + ft));
            }
            return MethodHandles.arrayConstructor(mt.returnType());
        } else {
            try {
                return l.findConstructor(mt.returnType(), mt.changeReturnType(void.class));
            } catch (NoSuchMethodException | IllegalAccessException e) {
                throw interpreterException(e);
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

    static MethodHandle resolveToMethodHandle(MethodHandles.Lookup l, MethodRef d, CoreOp.InvokeOp.InvokeKind kind) {
        try {
            return d.resolveToHandle(l, kind);
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    static VarHandle resolveToVarHandle(MethodHandles.Lookup l, FieldRef d) {
        try {
            return d.resolveToHandle(l);
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    public static MethodType resolveToMethodType(MethodHandles.Lookup l, FunctionType ft) {
        try {
            return MethodRef.toNominalDescriptor(ft).resolveConstantDesc(l);
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
        }
    }

    public static Class<?> resolveToClass(MethodHandles.Lookup l, TypeElement d) {
        try {
            if (d instanceof JavaType jt) {
                return (Class<?>)jt.erasure().resolve(l);
            } else {
                throw new ReflectiveOperationException();
            }
        } catch (ReflectiveOperationException e) {
            throw interpreterException(e);
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
