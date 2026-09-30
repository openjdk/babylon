/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.extern.ExternalizedOp;

import java.lang.invoke.*;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toMap;

public class JavaLowInterpreter extends AbstractJavaInterpreter {
    public JavaLowInterpreter() {
    }

    @Override
    protected Env newEnv(MethodHandles.Lookup l) {
        return new JavaEnv(new HashMap<>(), l, new ArrayDeque<>());
    }

    @Override
    public OpEffect executeOp(Op op, Env e) {
        Object result;
        switch (op) {
            case CoreOp.VarOp o -> {
                Object init = o.isUninitialized() ? VarBox.UNINITIALIZED : e.valueOf(o.initOperand());
                result = new VarBox(init);
            }
            case CoreOp.VarAccessOp.VarLoadOp o -> {
                CoreOp.Var<?> variable = (CoreOp.Var<?>) e.valueOf(o.varOperand());
                Object value = variable.value();
                if (value == VarBox.UNINITIALIZED) {
                    throw new InterpreterException("Loading from uninitialized variable");
                }
                result = value;
            }
            case CoreOp.VarAccessOp.VarStoreOp o -> {
                VarBox variable = (VarBox) e.valueOf(o.varOperand());
                Object v = e.valueOf(o.storeOperand());
                variable.value = v;
                result = null;
            }
            case JavaOp.InvokeOp o -> {
                JavaEnv je = (JavaEnv) e;
                MethodType target = resolveToMethodType(je.l, o.opSignature());
                MethodHandles.Lookup il = switch (o.invokeKind()) {
                    case STATIC, INSTANCE -> je.l;
                    case SUPER -> je.l.in(target.parameterType(0));
                };
                MethodHandle mh = resolveToMethodHandle(il, o.invokeReference(), o.invokeKind());

                mh = mh.asType(target).asFixedArity();
                List<Object> operands = e.valuesOf(o.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ArithmeticOperation _ -> {
                JavaEnv je = (JavaEnv) e;
                MethodHandle mh = opHandle(je.l, externalizeOpName(op), op.opSignature());
                List<Object> operands = e.valuesOf(op.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ConvOp _ -> {
                JavaEnv je = (JavaEnv) e;
                MethodHandle mh = opHandle(je.l, externalizeOpName(op) + "_" + op.opSignature().returnType(), op.opSignature());
                List<Object> operands = e.valuesOf(op.operands());
                try {
                    result = mh.invokeWithArguments(operands.toArray());
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case CoreOp.ConstantOp o -> {
                if (o.resultType().equals(JavaType.J_L_CLASS)) {
                    try {
                        result = resolveToClass(((JavaEnv) e).l, (JavaType) o.value());
                    } catch (ReflectiveOperationException ex) {
                        return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                    }
                } else {
                    result = o.value();
                }
            }
            case JavaOp.AssertOp o -> {
                TerminatingOpEffect perdEffect = executeBody(o.predicateBody(), List.of(), e);
                boolean b = switch (perdEffect.terminatingOp()) {
                    case CoreOp.YieldOp _ when perdEffect.operands().getFirst() instanceof Boolean av -> av;
                    default -> throw new InternalError();
                };
                if (!b) {
                    Body detailsBody = o.detailsBody();
                    AssertionError ae;
                    if (detailsBody != null) {
                        TerminatingOpEffect messEffect = executeBody(detailsBody, List.of(), e);
                        Object message = switch (messEffect.terminatingOp()) {
                            case CoreOp.YieldOp _ -> messEffect.operands().getFirst();
                            default -> throw new InternalError();
                        };
                        ae = new AssertionError(message);
                    } else {
                        ae = new AssertionError();
                    }
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ae), e);
                }
                result = null;
            }
            case CoreOp.FuncCallOp o -> {
                String name = o.funcName();

                // Find top-level op
                Op top = o;
                while (top.ancestorBody() != null) {
                    top = top.ancestorOp();
                }

                // Ensure top-level op is a module and function name
                // is in the module's function table
                if (top instanceof CoreOp.ModuleOp mop) {
                    CoreOp.FuncOp funcOp = mop.functionTable().get(name);
                    if (funcOp == null) {
                        throw new InterpreterException("Function " + name + " cannot be resolved: not in module's function table");
                    }
                    try {
                        JavaEnv je = (JavaEnv) e;
                        result = interpret(funcOp, e.valuesOf(o.operands()), je.l);
                    } catch (InterpreterException ex) {
                        throw ex;
                    } catch (Throwable t) {
                        return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                    }
                } else {
                    throw new InterpreterException("Function " + name + " cannot be resolved: top level op is not a module");
                }
            }
            case CoreOp.QuotedOp o -> {
                SequencedMap<Value, Object> capturedValues = o.capturedValues().stream()
                        .collect(toMap(v -> v, e::valueOf, (v, _) -> v, LinkedHashMap::new));
                result = new Quoted<>(o.quotedOp(), capturedValues);
            }
            case JavaOp.LambdaOp o -> {
                // bind the instance on which the method interpreting the lambda body is called
                // ensuring the body of the lambda op is interpreted using the JavaLowInterpreter
                result = executeLambdaOp(o, e, lambdaBodyInterpreter.bindTo(this));
            }
            case CoreOp.TupleOp o -> {
                List<Object> values = o.operands().stream().map(e::valueOf).toList();
                result = values.toArray();
            }
            case CoreOp.TupleLoadOp o -> {
                Object[] arr = (Object[]) e.valueOf(o.operands().getFirst());
                try {
                    result = arr[o.index()];
                } catch (ArrayIndexOutOfBoundsException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case CoreOp.TupleWithOp o -> {
                Object[] arr = (Object[]) e.valueOf(o.operands().getFirst());
                Object[] newArr = Arrays.copyOf(arr, arr.length);
                try {
                    newArr[o.index()] = e.valueOf(o.operands().get(1));
                } catch (ArrayIndexOutOfBoundsException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = newArr;
            }
            case JavaOp.FieldAccessOp.FieldLoadOp o -> {
                JavaEnv je = (JavaEnv) e;
                VarHandle vh;
                try {
                    vh = resolveToVarHandle(je.l, o.fieldReference());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    if (o.operands().isEmpty()) {
                        result = vh.get();
                    } else {
                        Object v = e.valueOf(o.operands().get(0));
                        result = vh.get(v);
                    }
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.FieldAccessOp.FieldStoreOp o -> {
                JavaEnv je = (JavaEnv) e;
                VarHandle vh;
                try {
                    vh = resolveToVarHandle(je.l, o.fieldReference());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    if (o.operands().size() == 1) {
                        Object v = e.valueOf(o.operands().get(0));
                        vh.set(v);
                    } else {
                        Object r = e.valueOf(o.operands().get(0));
                        Object v = e.valueOf(o.operands().get(1));
                        vh.set(r, v);
                    }
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = null;
            }
            case JavaOp.InstanceOfOp o -> {
                JavaEnv je = (JavaEnv) e;
                Object obj = e.valueOf(o.operands().get(0));
                Class<?> c;
                try {
                    c = resolveToClass(je.l, o.targetType());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = c.isInstance(obj);
            }
            case JavaOp.CastOp o  -> {
                Class<?> c;
                try {
                    JavaEnv je = (JavaEnv) e;
                    c = resolveToClass(je.l, o.targetType());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    Object v = e.valueOf(o.operands().get(0));
                    result = c.cast(v);
                } catch (ClassCastException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.NewOp o  -> {
                Object[] values = o.operands().stream().map(e::valueOf).toArray();
                MethodHandle mh;
                try {
                    JavaEnv je = (JavaEnv) e;
                    mh = resolveToConstructorHandle(je.l, o.constructorReference());
                } catch (ReflectiveOperationException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                try {
                    result = mh.invokeWithArguments(values);
                } catch (Throwable t) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(t), e);
                }
            }
            case JavaOp.ArrayLengthOp o -> {
                Object a = e.valueOf(o.operands().get(0));
                try {
                    result = Array.getLength(a);
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.ArrayAccessOp.ArrayLoadOp o -> {
                Object a = e.valueOf(o.operands().get(0));
                Object index = e.valueOf(o.operands().get(1));
                try {
                    result = Array.get(a, (int) index);
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
            }
            case JavaOp.ArrayAccessOp.ArrayStoreOp o -> {
                Object a = e.valueOf(o.operands().get(0));
                Object index = e.valueOf(o.operands().get(1));
                Object v = e.valueOf(o.operands().get(2));
                try {
                    Array.set(a, (int) index, v);
                } catch (RuntimeException ex) {
                    return new TerminatingOpEffect(fakeThrowOp, List.of(ex), e);
                }
                result = null;
            }
            case JavaOp.ConcatOp o -> {
                result = o.operands().stream()
                        .map(e::valueOf)
                        .map(String::valueOf)
                        .collect(Collectors.joining());
            }
            default -> throw new UnsupportedOperationException(op.toString());
        }
        return new OpResultEffect(result, e);
    }

    @Override
    public BlockEffect executeTerminatingOp(Op.Terminating op, Env e) {
        return switch (op) {
            case CoreOp.BranchOp o -> {
                Block.Reference r = o.successors().getFirst();
                List<Object> arguments = e.valuesOf(r.arguments());
                yield new SuccessorEffect(r.targetBlock(), arguments, e);
            }
            case CoreOp.ConditionalBranchOp o -> {
                boolean p = (boolean) e.valueOf(o.predicateOperand());
                Block.Reference r = p ? o.trueBranch() : o.falseBranch();
                List<Object> arguments = e.valuesOf(r.arguments());
                yield new SuccessorEffect(r.targetBlock(), arguments, e);
            }
            case CoreOp.ReturnOp o -> {
                List<Object> operands = e.valuesOf(o.operands());
                yield new TerminatingOpEffect(o, operands, e);
            }
            case JavaOp.ThrowOp o -> e.onAbruptCompletion(o, new TerminatingOpEffect(o, e.valuesOf(o.operands()),e));
            case CoreOp.YieldOp o -> {
                if (o.ancestorBody().ancestorBody() == null) {
                    throw new IllegalStateException("Yielding to no parent body");
                }
                List<Object> operands = e.valuesOf(o.operands());
                yield new TerminatingOpEffect(o, operands, e);
            }
            case JavaOp.ExceptionRegionEnter o -> {
                JavaEnv je = (JavaEnv) e;
                List<CatchHandler> handlers = CatchHandler.of(o);
                je = je.registerCatchHandlers(handlers);
                yield new SuccessorEffect(o.startReference().targetBlock(), je.valuesOf(o.startReference().arguments()), je);
            }
            case JavaOp.ExceptionRegionExit o -> {
                JavaEnv je = (JavaEnv) e;
                List<CatchHandler> handlers = CatchHandler.of(o.enterOp());
                je = je.removeCatchHandlers(handlers);
                yield new SuccessorEffect(o.endReference().targetBlock(), je.valuesOf(o.endReference().arguments()), je);
            }
            default -> throw new UnsupportedOperationException(op.toString());
        };
    }

    static MethodType resolveToMethodType(MethodHandles.Lookup l, FunctionType ft) {
        try {
            return MethodRef.toNominalDescriptor(ft).resolveConstantDesc(l);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static MethodHandle resolveToMethodHandle(MethodHandles.Lookup l, MethodRef d, JavaOp.InvokeOp.InvokeKind kind) {
        try {
            return d.resolveToHandle(l, kind);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static VarHandle resolveToVarHandle(MethodHandles.Lookup l, FieldRef d) throws ReflectiveOperationException {
        return d.resolveToHandle(l);
    }

    static MethodHandle resolveToConstructorHandle(MethodHandles.Lookup l, MethodRef d) throws ReflectiveOperationException {
        return d.resolveToHandle(l, JavaOp.InvokeOp.InvokeKind.SUPER);
    }

    static String externalizeOpName(Op op) {
        return (op instanceof ExternalizedOp.Externalizable eop)
                ? eop.externalizeOpName()
                : op.getClass().getName();
    }

    static MethodHandle opHandle(MethodHandles.Lookup l, String opName, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft).erase();
        try {
            return MethodHandles.lookup().findStatic(ArithmeticAndConvOpImpls.class, opName, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
