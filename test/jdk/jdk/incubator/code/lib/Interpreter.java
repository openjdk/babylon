/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleProxies;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;

import static java.util.stream.Collectors.toMap;

// with the change to the hierachy, we have some code duplication
// see if we can improve that
public abstract class Interpreter {
    public Interpreter() {
    }

    public abstract OpEffect executeOp(Op op, Env env);

    public abstract BlockEffect executeTerminatingOp(Op.Terminating op, Env env);

    public TerminatingOpEffect executeBody(Body body, List<Object> args, Env env) {
        Block block = body.entryBlock();
        while (true) {
            // bind block parameters in new env
            env = env.bind(block.parameters(), args);
            switch (executeBlock(block, env)) {
                // pass control to ancestor op
                case TerminatingOpEffect e -> {
                    return e;
                }
                // pass control to successor block
                case SuccessorEffect e -> {
                    block = e.successor();
                    args = e.args();
                    env = e.e();
                }
            }
        }
    }

    public BlockEffect executeBlock(Block block, Env env) {
        for (var op = block.firstOp(); !(op instanceof Op.Terminating top); op = block.nextOp(op)) {
            switch (executeOp(op, env)) {
                // op completed abruptly, pass control to ancestor op
                case TerminatingOpEffect e -> {
                    return env.onAbruptCompletion(op, e);
                }
                // op completed normally, bind op result in new env, pass control to next op
                case OpResultEffect e -> env = env.bind(op.result(), e.result);
            }
        }

        return executeTerminatingOp(top, env);
    }

    public interface Env {
        Env bind(List<? extends Value> symbolicValues, List<Object> runtimeValues);

        Env bind(Value symbolicValue, Object runtimeValue);

        List<Object> valuesOf(List<? extends Value> symbolicValues);

        Object valueOf(Value symbolicValue);

        BlockEffect onAbruptCompletion(Op op, TerminatingOpEffect eff);
    }

    public sealed interface BlockEffect
            permits SuccessorEffect, TerminatingOpEffect {
    }

    public sealed interface OpEffect
            permits OpResultEffect, TerminatingOpEffect {
    }

    public record SuccessorEffect(Block successor, List<Object> args, Env e)
            implements BlockEffect {
    }

    public record TerminatingOpEffect(Op terminatingOp, List<Object> operands, Env e)
            implements BlockEffect, OpEffect {
    }

    public record OpResultEffect(Object result, Env e)
            implements OpEffect {
    }

    static <T extends Op & Op.Invokable> Object invoke(MethodHandles.Lookup l, T op, Object... args) {
        return invoke(l, op, Arrays.asList(args));
    }

    static <T extends Op & Op.Invokable> Object invoke(MethodHandles.Lookup l, T op, List<Object> argsAndCaptures) {
        return new JavaHighInterpreter(new JavaLowInterpreter()).interpret(op, argsAndCaptures, l);
    }

    /**
     * Exception thrown by the interpreter when execution fails.
     */
    @SuppressWarnings("serial")
    public static final class InterpreterException extends RuntimeException {
        InterpreterException(Throwable cause) {
            super(cause);
        }
        InterpreterException(String message) {
            super(message);
        }
    }

    public <T extends Op & Op.Invokable> Object interpret(T op, List<Object> argsAndCaptures, MethodHandles.Lookup l) {
        JavaLowInterpreter.validateTypes(op, argsAndCaptures, l);

        return interpret_(op, l,
                argsAndCaptures.subList(op.parameters().size(), argsAndCaptures.size()).toArray(),
                argsAndCaptures.subList(0, op.parameters().size()).toArray());
    }

    protected  <T extends Op & Op.Invokable> Object interpret_(T op, MethodHandles.Lookup l, Object[] captures, Object[] args) {
        Env e = newEnv(l);
        e = e.bind(op.capturedValues(), Arrays.asList(captures));
        var effect = executeBody(op.body(), Arrays.asList(args), e);
        switch (effect.terminatingOp()) {
            case CoreOp.ReturnOp rop -> {
                return rop.operands().isEmpty() ? null : effect.operands().getFirst();
            }
            case JavaOp.ThrowOp _ -> {
                JavaLowInterpreter.eraseAndThrow((Throwable) effect.operands().getFirst());
                throw new InternalError(); // @@@ shouldn't reach here
            }
            default -> throw new InternalError(effect.toString());
        }
    }

    protected Object interpretLambdaBody(JavaOp.LambdaOp lambdaOp, MethodHandles.Lookup l, Object[] captures, Object[] args) {
        return interpret_(lambdaOp, l, captures, args);
    }

    protected static final MethodHandle lambdaBodyInterpreter;
    static {
        try {
            lambdaBodyInterpreter = MethodHandles.lookup().findVirtual(Interpreter.class, "interpretLambdaBody",
                    MethodType.methodType(Object.class, JavaOp.LambdaOp.class, MethodHandles.Lookup.class, Object[].class, Object[].class));
        } catch (Throwable t) {
            throw new InternalError();
        }
    }

    protected OpEffect executeLambdaOp(JavaOp.LambdaOp o, Env env, MethodHandle lambdaBodyInterpreter) {
        JavaLowInterpreter.JavaEnv je = (JavaLowInterpreter.JavaEnv) env;
        Class<?> fi;
        try {
            fi = JavaLowInterpreter.resolveToClass(je.l, o.functionalInterface());
        } catch (ReflectiveOperationException ex) {
            return new TerminatingOpEffect(JavaLowInterpreter.fakeThrowOp, List.of(ex), env);
        }

        SequencedMap<Value, Object> capturedValuesAndArguments = o.capturedValues().stream()
                .collect(toMap(v -> v, env::valueOf, (v, _) -> v, LinkedHashMap::new));
        Object[] capturedArguments = capturedValuesAndArguments.sequencedValues().toArray(Object[]::new);

        MethodHandle fProxy = lambdaBodyInterpreter.bindTo(o).bindTo(je.l).bindTo(capturedArguments)
                .asCollector(Object[].class, o.parameters().size());
        Object fiInstance = MethodHandleProxies.asInterfaceInstance(fi, fProxy);

        Object result;
        // If a reflectable lambda proxy again to add method Quoted quoted()
        if (o.isReflectable()) {
            result = Proxy.newProxyInstance(je.l.lookupClass().getClassLoader(), new Class<?>[]{fi},
                    new InvocationHandler() {
                        private final Quoted<JavaOp.LambdaOp> quoted = new Quoted<>(o, capturedValuesAndArguments);

                        @Override
                        public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                            if (Objects.equals(method.getName(), "quoted") && method.getParameterCount() == 0) {
                                return __internal_quoted();
                            } else {
                                // Delegate to FI instance
                                return method.invoke(fiInstance, args);
                            }
                        }

                        private Quoted<JavaOp.LambdaOp> __internal_quoted() {
                            return quoted;
                        }
                    });
        } else {
            result = fiInstance;
        }
        return new OpResultEffect(result, env);
    }

    abstract Env newEnv(MethodHandles.Lookup l);
}
