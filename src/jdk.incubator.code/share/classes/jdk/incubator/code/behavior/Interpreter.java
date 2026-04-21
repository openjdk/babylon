package jdk.incubator.code.behavior;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.util.List;

public abstract class Interpreter {
    public Interpreter() {
    }

    public abstract OpEffect executeOp(Op op, Env env);

    public abstract <O extends Op & Op.Terminating> BlockEffect executeTerminatingOp(O op, Env env);

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
        var op = block.firstOp();
        for (; !(op instanceof Op.Terminating); op = block.nextOp(op)) {
            switch (executeOp(op, env)) {
                // op completed abruptly, pass control to ancestor op
                case TerminatingOpEffect e -> {
                    return e;
                }
                // op completed normally, bind op result in new env, pass control to next op
                case OpResultEffect e -> env = env.bind(op.result(), e.result);
            }
        }

        return executeTerminatingOp((Op & Op.Terminating) op, env);
    }

    public interface Env {
        Env bind(List<? extends Value> symbolicValues, List<Object> runtimeValues);

        Env bind(Value symbolicValue, Object runtimeValue);

        List<Object> valuesOf(List<? extends Value> symbolicValues);

        Object valueOf(Value symbolicValue);
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
}
