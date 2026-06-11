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

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;

import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandles;
import java.util.*;

public class JavaHighInterpreter extends JavaLowInterpreter {
    public JavaHighInterpreter() {
    }

    @Override
    protected Env newEnv(MethodHandles.Lookup l) {
        return new JavaHighEnv(new HashMap<>(), l, new ArrayDeque<>());
    }

    static class JavaHighEnv extends JavaLowInterpreter.JavaEnv {
        private JavaHighEnv(Map<Value, Object> bindings, MethodHandles.Lookup l, Deque<List<Block>> catchBlocks) {
            super(bindings, l, catchBlocks);
        }

        @Override
        protected Env newEnv(Map<Value, Object> m) {
            return new JavaHighEnv(m, l, catchBlocks);
        }

        @Override
        protected JavaEnv newEnv(Deque<List<Block>> catchBlocks) {
            return new JavaHighEnv(bindings, l, catchBlocks);
        }

        @Override
        public BlockEffect onAbruptCompletion(TerminatingOpEffect eff) {
            if (eff.terminatingOp() instanceof JavaOp.ThrowOp) {
                return super.onAbruptCompletion(eff);
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
            default -> super.executeOp(op, e);
        };
    }

    // TODO labeled ops, sw
    OpEffect executeContinueOp(JavaOp.ContinueOp continueOp, Env e) {
        return new TerminatingOpEffect(continueOp, e.valuesOf(continueOp.operands()), e);
    }

    OpEffect executeBreakOp(JavaOp.BreakOp breakOp, Env e) {
        return new TerminatingOpEffect(breakOp, e.valuesOf(breakOp.operands()), e);
    }

    OpEffect executeLabeledOp(JavaOp.LabeledOp labeledOp, Env e) {
        TerminatingOpEffect effect = executeBody(labeledOp.body(), List.of(), e);
        if (effect.terminatingOp() instanceof JavaOp.BreakOp bop && bop.labelOperand().equals(labeledOp.labelIdentifier())) {
            return new OpResultEffect(null, e);
        } else if (effect.terminatingOp().equals(labeledOp.body().entryBlock().terminatingOp())) {
            return new OpResultEffect(null, e);
        }
        return effect;
    }

    OpEffect executeBlockOp(JavaOp.BlockOp blockOp, Env e) {
        TerminatingOpEffect effect = executeBody(blockOp.body(), List.of(), e);
        if (effect.terminatingOp().equals(blockOp.body().entryBlock().terminatingOp())) {
            return new OpResultEffect(null, e);
        }
        return effect;
    }

    @Override
    public <O extends Op & Op.Terminating> BlockEffect executeTerminatingOp(O op, Env e) {
        return switch (op) {
            case JavaOp.StatementTargetOp _ -> {
                List<Object> operands = e.valuesOf(op.operands());
                yield new TerminatingOpEffect(op, operands, e);
            }
            default -> super.executeTerminatingOp(op, e);
        };
    }


    OpEffect executeForOp(JavaOp.ForOp op, Env e) {
        var initEffect = executeBody(op.initBody(), List.of(), e);
        Object loopVariables = switch (initEffect.terminatingOp()) {
            // init body may yield nothing in case variables we initialize are defined outside the for operation
            case CoreOp.YieldOp _ -> initEffect.operands().isEmpty() ? null : initEffect.operands().getFirst();
            default -> throw new InternalError();
        };
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
            boolean p = switch (condEffect.terminatingOp()) {
                case CoreOp.YieldOp _ -> (boolean) condEffect.operands().getFirst();
                default -> throw new InternalError();
            };
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
                case CoreOp.YieldOp _ -> {
                }
                default -> throw new InternalError();
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
            } else {
                Body pred = bodies.get(i);
                var condEffect = executeBody(pred, List.of(), e);
                boolean p = switch (condEffect.terminatingOp()) {
                    case CoreOp.YieldOp _ -> (boolean) condEffect.operands().getFirst();
                    default -> throw new InternalError();
                };
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
        List<Object> rArgs = new ArrayList<>();
        for (Body rb : tryOp.resourceBodies()) {
            var re = executeBody(rb, rArgs, e);
            // @@@ rb complete abruptly ?
            rArgs.addAll(re.operands());
        }

        var effect = executeBody(tryOp.body(), rArgs, e);

        Throwable rcT = null;
        for (Object r : rArgs.reversed()) {
            if (r instanceof VarBox vb) {
                r = vb.value();
            }
            try {
                ((AutoCloseable) r).close();
            } catch (Exception ex) {
                if (rcT == null) rcT = ex;
                else rcT.addSuppressed(ex);
            }
        }

        Throwable primaryT = null;
        if (effect.terminatingOp() instanceof JavaOp.ThrowOp) {
            primaryT = ((Throwable) effect.operands().getFirst());
            if (rcT != null) {
                primaryT.addSuppressed(rcT);
                for (Throwable t : rcT.getSuppressed()) {
                    primaryT.addSuppressed(t);
                }
            }
        } else if (rcT != null) {
            primaryT = rcT;
        }

        Body catchBody = null;
        if (primaryT != null) {
            JavaEnv je = (JavaEnv) e;
            catchBody = findCatchBody(je.l, tryOp, primaryT);
            if (catchBody != null) {
                effect = executeBody(catchBody, List.of(primaryT), e);
            }
        }

        if (tryOp.finallyBody() != null) {
            var finallyEffect = executeBody(tryOp.finallyBody(), List.of(), e);
            if (!(finallyEffect.terminatingOp() instanceof CoreOp.YieldOp)) {
                return finallyEffect;
            }
        }

        if (!(effect.terminatingOp() instanceof CoreOp.YieldOp)) {
            return effect;
        }
        if (rcT != null && catchBody == null) {
            return new TerminatingOpEffect(fakeThrowOp, List.of(rcT), e);
        }
        return new OpResultEffect(null, null);
    }

    private static Body findCatchBody(MethodHandles.Lookup l, JavaOp.TryOp tryOp, Throwable t) {
        Body cb = null;
        for (Body catchBody : tryOp.catchBodies()) {
            Class<?> c;
            try {
                c = resolveToClass(l, catchBody.entryBlock().parameters().getFirst().type());
            } catch (ReflectiveOperationException ex) {
                throw new InterpreterException(ex);
            }
            if (c.isInstance(t)) {
                cb = catchBody;
                break;
            }
        }
        return cb;
    }
}
