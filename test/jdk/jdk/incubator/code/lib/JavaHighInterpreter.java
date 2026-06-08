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

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.Arrays;
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
            case CoreOp.YieldOp _, JavaOp.StatementTargetOp _, JavaOp.ThrowOp _ -> {
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
}
