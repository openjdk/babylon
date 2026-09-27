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

import jdk.incubator.code.Body;
import jdk.incubator.code.Op;

import jdk.incubator.code.Quoted;
import jdk.incubator.code.Value;
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

public class JavaHighInterpreter extends Interpreter {
    private final JavaLowInterpreter javaLowInterpreter;

    public JavaHighInterpreter(JavaLowInterpreter javaLowInterpreter) {
        this.javaLowInterpreter = javaLowInterpreter;
    }

    protected Env newEnv(MethodHandles.Lookup l) {
        return new JavaHighEnv(new HashMap<>(), l, new ArrayDeque<>());
    }

    static class JavaHighEnv extends JavaLowInterpreter.JavaEnv {
        private JavaHighEnv(Map<Value, Object> bindings, MethodHandles.Lookup l, Deque<List<JavaLowInterpreter.CatchHandler>> catchHandlers) {
            super(bindings, l, catchHandlers);
        }

        @Override
        protected Env newEnv(Map<Value, Object> m) {
            return new JavaHighEnv(m, l, catchHandlers);
        }

        @Override
        protected JavaLowInterpreter.JavaEnv newEnv(Deque<List<JavaLowInterpreter.CatchHandler>> catchBlocks) {
            return new JavaHighEnv(bindings, l, catchBlocks);
        }

        @Override
        public BlockEffect onAbruptCompletion(Op op, TerminatingOpEffect eff) {
            if (eff.terminatingOp() instanceof JavaOp.ThrowOp) {
                return super.onAbruptCompletion(op, eff);
            }
            return eff;
        }
    }

    @Override
    public OpEffect executeOp(Op op, Env e) {
        return switch (op) {
            case JavaOp.ForOp o -> executeForOp(o, e);
            case JavaOp.IfOp o -> executeIfOp(o, e);
            case JavaOp.TryOp o -> executeTryOp(o, e);
            case JavaOp.BreakOp o -> executeBreakOp(o, e);
            case JavaOp.LabeledOp o -> executeLabeledOp(o, e);
            case JavaOp.ContinueOp o -> executeContinueOp(o, e);
            case JavaOp.BlockOp o -> executeBlockOp(o, e);
            case JavaOp.LambdaOp o -> executeLambdaOp(o, e);
            default -> javaLowInterpreter.executeOp(op, e);
        };
    }

    // TODO labeled ops, sw
    OpEffect executeContinueOp(JavaOp.ContinueOp continueOp, Env e) {
        return new TerminatingOpEffect(continueOp, e.valuesOf(continueOp.operands()), e);
    }

    OpEffect executeBreakOp(JavaOp.BreakOp breakOp, Env e) {
        return new TerminatingOpEffect(breakOp, e.valuesOf(breakOp.operands()), e);
    }

    OpEffect executeLabeledOp(JavaOp.LabeledOp op, Env e) {
        TerminatingOpEffect effect = executeBody(op.body(), List.of(), e);
        if (effect.terminatingOp() instanceof JavaOp.BreakOp bop && bop.labelOperand().equals(op.labelIdentifier())) {
            return new OpResultEffect(null, e);
        }
        return processVoidEffect(effect, op.body(), e);
    }

    OpEffect executeBlockOp(JavaOp.BlockOp op, Env e) {
        TerminatingOpEffect effect = executeBody(op.body(), List.of(), e);
        return processVoidEffect(effect, op.body(), e);
    }

    @Override
    public BlockEffect executeTerminatingOp(Op.Terminating op, Env e) {
        return switch (op) {
            case JavaOp.StatementTargetingOp _ -> {
                List<Object> operands = e.valuesOf(op.operands());
                yield new TerminatingOpEffect(op, operands, e);
            }
            default -> javaLowInterpreter.executeTerminatingOp(op, e);
        };
    }


    OpEffect executeForOp(JavaOp.ForOp op, Env e) {
        var initEffect = executeBody(op.initBody(), List.of(), e);
        switch (initEffect.terminatingOp()) {
            case CoreOp.YieldOp _ -> {}
            default -> {
                return initEffect;
            }
        }
        // init body may yield nothing in case variables we initialize are defined outside the for operation
        Object loopVariables = initEffect.operands().isEmpty() ? null : initEffect.operands().getFirst();

        List<Object> args;
        if (loopVariables instanceof Object[] arr) {
            args = Arrays.asList(arr);
        } else if (loopVariables != null){
            args = List.of(loopVariables);
        } else {
            args = List.of();
        }

        loop:
        while (true) {
            var condEffect = executeBody(op.condBody(), args, e);
            var opt = processBooleanEffect(condEffect);
            if (opt.isEmpty()) {
                return condEffect;
            }
            var p = opt.get();
            if (!p)
                break loop;

            var loopEffect = executeBody(op.loopBody(), args, e);
            switch (loopEffect.terminatingOp()) {
                case JavaOp.ContinueOp _ -> {
                }
                case JavaOp.BreakOp _ -> {
                    break loop;
                }
                default -> { // can we have other kind ?
                    return loopEffect;
                }
            }

            var updateEffect = executeBody(op.updateBody(), args, e);
            switch (updateEffect.terminatingOp()) {
                case CoreOp.YieldOp _ -> {}
                default -> {
                    return updateEffect;
                }
            }
        }

        // Void/unit result
        return new OpResultEffect(op.result(), null);
    }

    OpEffect executeIfOp(JavaOp.IfOp op, Env e) {
        List<Body> bodies = op.bodies();
        Body action = null;
        for (int i = 0; action == null; i += 2) {
            if (i == bodies.size() - 1) {
                action = bodies.get(i);
            } else if (i > bodies.size() - 1) {
                // no action to execute and no else
                return new OpResultEffect(null, e);
            } else {
                Body pred = bodies.get(i);
                var condEffect = executeBody(pred, List.of(), e);
                var opt = processBooleanEffect(condEffect);
                if (opt.isEmpty()) {
                    return condEffect;
                }
                boolean p = opt.get();
                if (p) {
                    action = bodies.get(i + 1);
                }
            }
        }

        var bodyEffect = executeBody(action, List.of(), e);
        switch (bodyEffect.terminatingOp()) {
            case CoreOp.YieldOp _ -> {
            }
            default -> {
                return bodyEffect;
            }
        }

        // Void/unit result
        return new OpResultEffect(op.result(), null);
    }

    OpEffect executeTryOp(JavaOp.TryOp tryOp, Env e) {
        Throwable t = null;
        TerminatingOpEffect effect = null;

        // create resources
        List<Object> rArgs = new ArrayList<>();
        l:
        for (Body rb : tryOp.resourceBodies()) {
            var re = executeBody(rb, rArgs, e);
            switch (re.terminatingOp()) {
                case CoreOp.YieldOp _ -> {}
                case JavaOp.ThrowOp o -> {
                    t = ((Throwable) re.operands().getFirst());
                    effect = new TerminatingOpEffect(o, List.of(t), e);
                    break l;
                }
                default -> throw new InterpreterException("Resource body of TryOp terminate with unexpected operation");
            }
            rArgs.addAll(re.operands());
        }

        // try body
        if (t == null) {
            effect = executeBody(tryOp.body(), rArgs, e);
            if (effect.terminatingOp() instanceof JavaOp.ThrowOp)
                t = (Throwable) effect.operands().getFirst();
        }

        // close resources
        for (Object r : rArgs.reversed()) {
            if (r instanceof JavaLowInterpreter.VarBox vb) {
                r = vb.value();
            }
            try {
                ((AutoCloseable) r).close();
            } catch (ClassCastException cce) {
                throw new InterpreterException(cce);
            } catch (Exception ex) {
                if (t == null)  t = ex;
                else            t.addSuppressed(ex);
                effect = new TerminatingOpEffect(JavaLowInterpreter.fakeThrowOp, List.of(t), e);
            }
        }

        // catch body
        if (t != null) {
            JavaLowInterpreter.JavaEnv je = (JavaLowInterpreter.JavaEnv) e;
            Body catchBody = findCatchBody(je.l, tryOp, t);
            if (catchBody != null) {
                effect = executeBody(catchBody, List.of(t), e);
            }
        }

        // finally body
        if (tryOp.finallyBody() != null) {
            var finallyEffect = executeBody(tryOp.finallyBody(), List.of(), e);
            if (!(finallyEffect.terminatingOp() instanceof CoreOp.YieldOp)) {
                return finallyEffect;
            }
        }

        if (effect != null && !(effect.terminatingOp() instanceof CoreOp.YieldOp)) {
            return effect;
        }
        return new OpResultEffect(null, null);
    }

    private static Body findCatchBody(MethodHandles.Lookup l, JavaOp.TryOp tryOp, Throwable t) {
        for (int i = 0; i < tryOp.catchBodies().size(); i++) {
            Body catchBody = tryOp.catchBodies().get(i);
            JavaLowInterpreter.CatchHandler handler = new JavaLowInterpreter.CatchHandler(tryOp.catchTypes().get(i), catchBody.entryBlock());
            try {
                if (handler.matches(l, t)) {
                    return catchBody;
                }
            } catch (ReflectiveOperationException ex) {
                throw new InterpreterException(ex);
            }
        }
        return null;
    }

    static OpEffect processVoidEffect(TerminatingOpEffect eff, Body body, Env e) {
        if (eff.terminatingOp() instanceof CoreOp.YieldOp yop) {
            if (yop.ancestorBody() == body) {
                return new OpResultEffect(null, e);
            }
            throw new InterpreterException("YieldOp not from the body");
        }
        return eff;
    }

    static Optional<Boolean> processBooleanEffect(TerminatingOpEffect eff) {
        return switch (eff.terminatingOp()) {
            case CoreOp.YieldOp _ when !eff.operands().isEmpty()
                    && eff.operands().getFirst() instanceof Boolean b -> Optional.of(b);
            case CoreOp.YieldOp _ -> throw new InterpreterException("YieldOp witn no boolean operand");
            default -> Optional.empty(); // abrupt completion
        };
    }

    public <T extends Op & Op.Invokable> Object interpret(T op, List<Object> argsAndCaptures, MethodHandles.Lookup l) {
        JavaLowInterpreter.validateTypes(op, argsAndCaptures, l);

        return interpret_(op, l,
                argsAndCaptures.subList(op.parameters().size(), argsAndCaptures.size()).toArray(),
                argsAndCaptures.subList(0, op.parameters().size()).toArray());
    }

    private <T extends Op & Op.Invokable> Object interpret_(T op, MethodHandles.Lookup l, Object[] captures, Object[] args) {
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

    private static final MethodHandle interpretLambdaOpMH;
    static {
        try {
            interpretLambdaOpMH = MethodHandles.lookup().findVirtual(JavaHighInterpreter.class, "interpretLambdaOp",
                    MethodType.methodType(Object.class, JavaOp.LambdaOp.class, MethodHandles.Lookup.class, Object[].class, Object[].class));
        } catch (Throwable t) {
            throw new InternalError();
        }
    }

    private Object interpretLambdaOp(JavaOp.LambdaOp op, MethodHandles.Lookup l, Object[] captures, Object[] args) {
        return interpret_(op, l, captures, args);
    }

    private OpEffect executeLambdaOp(JavaOp.LambdaOp o, Env env) {
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

        MethodHandle fProxy = interpretLambdaOpMH.bindTo(this).bindTo(o).bindTo(je.l).bindTo(capturedArguments)
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
}
