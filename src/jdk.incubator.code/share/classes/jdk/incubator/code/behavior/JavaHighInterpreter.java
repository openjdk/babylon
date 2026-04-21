package jdk.incubator.code.behavior;

import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;

public class JavaHighInterpreter extends JavaLowInterpreter {
    public JavaHighInterpreter() {
    }

    @Override
    public OpEffect executeOp(Op op, Env e) {
        return switch (op) {
            case JavaOp.ForOp o -> executeForOp(o, e);
            case JavaOp.IfOp o -> executeIfOp(o, e);
            default -> super.executeOp(op, e);
        };
    }

    @Override
    public <O extends Op & Op.Terminating> BlockEffect executeTerminatingOp(O op, Env e) {
        return switch (op) {
            case CoreOp.YieldOp _,
                 JavaOp.StatementTargetOp _,
                 JavaOp.ThrowOp _ -> {
                List<Object> operands = e.valuesOf(op.operands());
                yield new TerminatingOpEffect(op, operands, e);
            }
            default -> super.executeTerminatingOp(op, e);
        };
    }


    OpEffect executeForOp(JavaOp.ForOp op, Env e) {
        var initEffect = executeBody(op.initBody(), List.of(), e);
        Object loopVariables = switch (initEffect.terminatingOp()) {
            case CoreOp.YieldOp _ -> initEffect.operands().getFirst();
            default -> throw new InternalError();
        };

        loop:
        while (true) {
            var condEffect = executeBody(op.condBody(), List.of(loopVariables), e);
            boolean p = switch (condEffect.terminatingOp()) {
                case CoreOp.YieldOp _ -> (boolean) condEffect.operands().getFirst();
                default -> throw new InternalError();
            };
            if (!p)
                break loop;

            var loopEffect = executeBody(op.loopBody(), List.of(loopVariables), e);
            switch (loopEffect.terminatingOp()) {
                case JavaOp.ContinueOp _ -> {
                }
                case JavaOp.BreakOp _ -> {
                    break loop;
                }
                default -> {
                    return loopEffect;
                }
            }

            var updateEffect = executeBody(op.updateBody(), List.of(loopVariables), e);
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
}