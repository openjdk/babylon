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

package oracle.code.triton;

import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.Patterns;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.CoreOp.VarAccessOp.VarLoadOp;
import java.lang.reflect.code.op.CoreOp.VarAccessOp.VarStoreOp;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayList;
import java.util.List;

import static java.lang.reflect.code.analysis.Patterns.*;

// @@@ Very basic, limited, and partially correct
public class SimpleCountedForLoopInfo {

    final ExtendedOp.JavaForOp fop;

    SimpleCountedForLoopInfo(ExtendedOp.JavaForOp fop) {
        this.fop = fop;

        if (fop.init().yieldType().equals(JavaType.VOID)) {
            throw new IllegalArgumentException("Loop variable externally initialized");
        }
        if (fop.loopBody().entryBlock().parameters().size() > 1) {
            throw new IllegalArgumentException("Two or more loop variables");
        }
    }

    public List<Op> startExpression() {
        /*
        ()Var<int> -> {
            %12 : int = constant @"0";
            %13 : Var<int> = var %12 @"i";
            yield %13;
        }
         */

        Patterns.OpPattern p = opP(CoreOp.YieldOp.class,
                opP(CoreOp.VarOp.class,
                        opResultP()));

        // match against yieldOp
        Op yieldOp = fop.init().entryBlock().ops().getLast();
        List<Value> matches = Patterns.match(null, yieldOp, p, (matchState, o) -> {
            return matchState.matchedOperands();
        });
        if (matches == null) {
            throw new IllegalArgumentException();
        }
        Op.Result initValue = (Op.Result) matches.get(0);

        return traverseOperands(initValue.op());
    }

    public List<Op> endExpression() {
        /*
        (%14 : Var<int>)boolean -> {
            %15 : int = var.load %14;
            %16 : int = var.load %2;
            %17 : boolean = lt %15 %16;
            yield %17;
        }
         */

        Patterns.OpPattern p = opP(CoreOp.YieldOp.class,
                opP(CoreOp.LtOp.class,
                        opP(VarLoadOp.class,
                                blockParameterP()),
                        opResultP()));

        // match against yieldOp
        Op yieldOp = fop.cond().entryBlock().ops().getLast();
        List<Value> matches = Patterns.match(null, yieldOp, p, (matchState, o) -> {
            return matchState.matchedOperands();
        });
        if (matches == null) {
            throw new IllegalArgumentException();
        }
        Op.Result endValue = (Op.Result) matches.get(1);

        return traverseOperands(endValue.op());
    }

    public List<Op> stepExpression() {
        /*
        (%18 : Var<int>)void -> {
            %19 : int = var.load %18;
            %20 : int = constant @"1";
            %21 : int = add %19 %20;
            var.store %18 %21;
            yield;
        }
         */

        Patterns.OpPattern p = opP(VarStoreOp.class,
                blockParameterP(),
                opP(CoreOp.AddOp.class,
                        opP(VarLoadOp.class, blockParameterP()),
                        opResultP()));

        // Match against last store op
        // @@@ Add Block.prevOp()
        Op storeOp = fop.update().entryBlock().ops().get(fop.update().entryBlock().ops().size() - 2);
        List<Value> matches = Patterns.match(null, storeOp, p, (matchState, r) -> {
            return matchState.matchedOperands();
        });
        if (matches == null) {
            throw new IllegalArgumentException();
        }
        Op.Result stepValue = (Op.Result) matches.get(2);

        return traverseOperands(stepValue.op());
    }

    static List<Op> traverseOperands(Op op) {
        List<Op> ops = new ArrayList<>();
        traverseOperands(ops, op);
        return ops;
    }

    // Hoist the expression
    // @@@ should be pure and independent of the loop variable
    static void traverseOperands(List<Op> ops, Op op) {
        for (Value operand : op.operands()) {
            if (operand.declaringBlock().parentBody() == op.ancestorBody()) {
                if (operand instanceof Op.Result r) {
                    traverseOperands(ops, r.op());
                }
            }
        }

        ops.add(op);
    }
}
