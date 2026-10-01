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
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;

import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.invoke.MethodHandles;
import java.util.*;

// private, pkg-private, protected, public
// any uses of interpreters in HAT or cr-examples ?
class JavaHighInterpreter extends AbstractJavaInterpreter {
    private final JavaLowInterpreter javaLowInterpreter;

    public JavaHighInterpreter(JavaLowInterpreter javaLowInterpreter) {
        this.javaLowInterpreter = javaLowInterpreter;
    }

    @Override
    Env newEnv(MethodHandles.Lookup l) {
        return new JavaHighEnv(new HashMap<>(), l, new ArrayDeque<>());
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
            case JavaOp.LambdaOp o -> {
                // bind the instance on which the method interpreting the lambda body is called
                // ensuring the body of the lambda op is interpreted using the JavaHighInterpreter;
                yield executeLambdaOp(o, e, LAMBDA_BODY_INTERPRETER.bindTo(this));
            }
            case JavaOp.SwitchOp o -> executeSwitchOp(o, e);
            case JavaOp.ConditionalAndOp o -> executeConditionalAndOp(o, e);
            case JavaOp.ConditionalOrOp o -> executeConditionalOrOp(o, e);
            case JavaOp.PatternOps.MatchOp o -> executePatternMatchOp(o, e);
            case JavaOp.ConditionalExpressionOp o -> executeConditionalExpressionOp(o, e);
            default -> javaLowInterpreter.executeOp(op, e);
        };
    }

    @Override
    public BlockEffect executeTerminatingOp(Op.Terminating op, Env e) {
        return switch (op) {
            case JavaOp.StatementTargetingOp _ -> {
                List<Object> operands = e.valuesOf(op.operands());
                yield new TerminatingOpEffect(op, operands, e);
            }
            case JavaOp.YieldOp _ -> {
                if (op.ancestorBody().ancestorBody() == null) {
                    throw new IllegalStateException("Yielding to no parent body");
                }
                yield new TerminatingOpEffect(op, e.valuesOf(op.operands()), e);
            }
            case JavaOp.SwitchFallthroughOp _ -> new TerminatingOpEffect(op, List.of(), e);
            default -> javaLowInterpreter.executeTerminatingOp(op, e);
        };
    }

    private OpEffect executeConditionalExpressionOp(JavaOp.ConditionalExpressionOp o, Env e) {
        TerminatingOpEffect predEffect = executeBody(o.predicateBody(), List.of(), e);
        Optional<Boolean> p = processBooleanEffect(predEffect);
        if (p.isEmpty()) {
            return predEffect;
        }
        Body bodyToExecute = p.get() ? o.trueBody() : o.falseBody();
        TerminatingOpEffect actionEffect = executeBody(bodyToExecute, List.of(), e);
        if (!(actionEffect.terminatingOp() instanceof CoreOp.YieldOp)) {
            return actionEffect;
        }
        if (actionEffect.operands().isEmpty()) {
            throw new InterpreterException("Action body of ConditionalExpressionOp must terminate with an operand");
        }
        return new OpResultEffect(actionEffect.operands().getFirst(), e);
    }

    private OpEffect executeConditionalAndOp(JavaOp.ConditionalAndOp o, Env e) {
        for (Body body : o.bodies()) {
            TerminatingOpEffect effect = executeBody(body, List.of(), e);
            Optional<Boolean> b = processBooleanEffect(effect);
            if (b.isEmpty()) {
                return effect;
            }
            if (!b.get()) {
                return new OpResultEffect(false, e);
            }
        }
        return new OpResultEffect(true, e);
    }

    private OpEffect executeConditionalOrOp(JavaOp.ConditionalOrOp o, Env e) {
        for (Body body : o.bodies()) {
            TerminatingOpEffect effect = executeBody(body, List.of(), e);
            Optional<Boolean> b = processBooleanEffect(effect);
            if (b.isEmpty()) {
                return effect;
            }
            if (b.get()) {
                return new OpResultEffect(true, e);
            }
        }
        return new OpResultEffect(false, e);
    }

    private OpEffect executePatternMatchOp(JavaOp.PatternOps.MatchOp o, Env e) {
        if (!(o.patternBody().entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop)) {
            throw new InternalError("A pattern body must terminate with a YieldOp");
        }
        if (yop.operands().isEmpty()) {
            throw new InternalError("The YieldOp of a pattern body must have one operand");
        }
        if (!(yop.operands().getFirst() instanceof Op.Result opr) ||
                !(opr.op() instanceof JavaOp.PatternOps.PatternOp po)) {
            throw new InternalError("YieldOp of pattern body must have an operand that's the result of a PatternOp");
        }
        Deque<Object> values = new ArrayDeque<>();
        Deque<JavaOp.PatternOps.PatternOp> patterns = new ArrayDeque<>();
        values.addLast(e.valueOf(o.targetOperand()));
        patterns.addLast(po);
        ArrayList<Object> patternVariablesValues = new ArrayList<>();
        while (!patterns.isEmpty()) {
            JavaOp.PatternOps.PatternOp patternOp = patterns.removeFirst();
            Object value = values.removeFirst();
            if (patternOp instanceof JavaOp.PatternOps.TypePatternOp typePatternOp) {
                Object castedValue = cast(typePatternOp.targetType(), e, value);
                if (castedValue == null) {
                    return new OpResultEffect(false, e);
                }
                patternVariablesValues.add(castedValue); // we don't have a Var for match all pattern, for type pattern we always have it
            } else if (patternOp instanceof JavaOp.PatternOps.RecordPatternOp recordPatternOp) {
                Object castedValue = cast(recordPatternOp.targetType(), e, value);
                if (castedValue == null) {
                    return new OpResultEffect(false, e);
                }
                for (int i = 0; i < recordPatternOp.recordReference().components().size(); i++) {
                    MethodRef compGetter = recordPatternOp.recordReference().methodForComponent(i);
                    Object compValue;
                    try {
                        compValue = compGetter.resolveToMethod(((JavaLowInterpreter.JavaEnv) e).l).invoke(value);
                    } catch (ReflectiveOperationException ex) {
                        throw new InternalError(ex);
                    }
                    JavaOp.PatternOps.PatternOp compPattern = (JavaOp.PatternOps.PatternOp) recordPatternOp.operands().get(i).asResult().op();
                    patterns.addLast(compPattern);
                    values.addLast(compValue);
                }
            }
        }

        TerminatingOpEffect effect = executeBody(o.matchBody(), patternVariablesValues, e);
        if (!(effect.terminatingOp() instanceof CoreOp.YieldOp)) {
            return effect;
        }

        return new OpResultEffect(true, e);
    }

    private static Object cast(CodeType type, Env e, Object value) {
        if (!(type instanceof JavaType jt)) {
            throw new InterpreterException("The target type of a TypePatternOp must be an instance JavaType");
        }
        Class c;
        try {
            c = (Class) jt.resolve(((JavaLowInterpreter.JavaEnv) e).l);
        } catch (ReflectiveOperationException ex) {
            throw new InterpreterException(ex);
        }
        try {
            return c.cast(value);
        } catch (ClassCastException ex) {
            return null;
        }
    }

    private static boolean isDefaultLabel(Body body) {
        return body.blocks().size() == 1 &&
                body.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop &&
                yop.operands().getFirst() instanceof Op.Result opr &&
                opr.op() instanceof CoreOp.ConstantOp cop &&
                cop.value() instanceof Boolean b && b;
    }

    private OpEffect executeSwitchOp(JavaOp.SwitchOp op, Env e) {
        int i;
        int defLabelIndex = -1;
        for (i = 0; i < op.bodies().size(); i+=2) {
            if (isDefaultLabel(op.bodies().get(i))) {
                defLabelIndex = i;
                continue;
            }
            TerminatingOpEffect effect = executeBody(op.bodies().get(i), e.valuesOf(op.operands()), e);
            Optional<Boolean> b = processBooleanEffect(effect);
            if (b.isEmpty()) {
                return effect;
            }
            if (b.get()) {
                break;
            }
        }

        if (i >= op.bodies().size() && defLabelIndex != -1) {
            i = defLabelIndex;
        }
        i++;
        while (i < op.bodies().size()) {
            TerminatingOpEffect effect = executeBody(op.bodies().get(i), List.of(), e);
            if (effect.terminatingOp() instanceof CoreOp.YieldOp || effect.terminatingOp() instanceof JavaOp.YieldOp ||
                    effect.terminatingOp() instanceof JavaOp.BreakOp) {
                return new OpResultEffect(effect.operands().isEmpty() ? null : effect.operands().getFirst(), e);
            } else if (effect.terminatingOp() instanceof JavaOp.SwitchFallthroughOp) {
                i += 2;
            } else {
                return effect;
            }
        }

        return new OpResultEffect(null, e);
    }

    private OpEffect executeContinueOp(JavaOp.ContinueOp continueOp, Env e) {
        return new TerminatingOpEffect(continueOp, e.valuesOf(continueOp.operands()), e);
    }

    private OpEffect executeBreakOp(JavaOp.BreakOp breakOp, Env e) {
        return new TerminatingOpEffect(breakOp, e.valuesOf(breakOp.operands()), e);
    }

    private OpEffect executeLabeledOp(JavaOp.LabeledOp op, Env e) {
        TerminatingOpEffect effect = executeBody(op.body(), List.of(), e);
        if (effect.terminatingOp() instanceof JavaOp.BreakOp bop && bop.labelOperand().equals(op.labelIdentifier())) {
            return new OpResultEffect(null, e);
        }
        return processVoidEffect(effect, op.body(), e);
    }

    private OpEffect executeBlockOp(JavaOp.BlockOp op, Env e) {
        TerminatingOpEffect effect = executeBody(op.body(), List.of(), e);
        if (!(effect.terminatingOp() instanceof CoreOp.YieldOp)) {
            return effect;
        }
        return new OpResultEffect(effect.operands().isEmpty() ? null : effect.operands().getFirst(), e);
    }

    private OpEffect executeForOp(JavaOp.ForOp op, Env e) {
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

    private OpEffect executeIfOp(JavaOp.IfOp op, Env e) {
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

    private OpEffect executeTryOp(JavaOp.TryOp tryOp, Env e) {
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
            if (r instanceof VarBox vb) {
                r = vb.value();
            }
            try {
                ((AutoCloseable) r).close();
            } catch (ClassCastException cce) {
                throw new InterpreterException(cce);
            } catch (Exception ex) {
                if (t == null)  t = ex;
                else            t.addSuppressed(ex);
                effect = new TerminatingOpEffect(FAKE_THROW_OP, List.of(t), e);
            }
        }

        // catch body
        if (t != null) {
            JavaEnv je = (JavaEnv) e;
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
            CatchHandler handler = new CatchHandler(tryOp.catchTypes().get(i), catchBody.entryBlock());
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

    private static OpEffect processVoidEffect(TerminatingOpEffect eff, Body body, Env e) {
        if (eff.terminatingOp() instanceof CoreOp.YieldOp yop) {
            if (yop.ancestorBody() == body) {
                return new OpResultEffect(null, e);
            }
            throw new InterpreterException("YieldOp not from the body");
        }
        return eff;
    }

    private static Optional<Boolean> processBooleanEffect(TerminatingOpEffect eff) {
        return switch (eff.terminatingOp()) {
            case CoreOp.YieldOp _ when !eff.operands().isEmpty()
                    && eff.operands().getFirst() instanceof Boolean b -> Optional.of(b);
            case CoreOp.YieldOp _ -> throw new InterpreterException("YieldOp with no boolean operand");
            default -> Optional.empty(); // abrupt completion
        };
    }

    static class JavaHighEnv extends JavaEnv {
        private JavaHighEnv(Map<Value, Object> bindings, MethodHandles.Lookup l, Deque<List<CatchHandler>> catchHandlers) {
            super(bindings, l, catchHandlers);
        }

        @Override
        Env newEnv(Map<Value, Object> m) {
            return new JavaHighEnv(m, l, catchHandlers);
        }

        @Override
        JavaEnv newEnv(Deque<List<CatchHandler>> catchBlocks) {
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
}
