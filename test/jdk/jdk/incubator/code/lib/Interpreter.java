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

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;
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
                    return env.onAbruptCompletion(block, e);
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
        return new JavaHighInterpreter().interpret(op, argsAndCaptures, l);
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
}
