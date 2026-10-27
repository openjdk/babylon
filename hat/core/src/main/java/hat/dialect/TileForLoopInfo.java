/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.dialect;

import hat.codetypes.ConstantType;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;
import java.util.Map;

import static jdk.incubator.code.dialect.core.CoreOp.ConstantOp;
import static jdk.incubator.code.dialect.core.CoreOp.VarOp;

/**
 * Utility methods for analyzing counted loops with code reflection to be process by the Tile Transformer.
 *
 * <p>
 * The following methods in this class provide:
 * <ul>
 *     <li>A known starting constant value</li>
 *     <li>A loop bound that stays invariant in the loop body</li>
 *     <li>A fixed step constant</li>
 * </ul>
 *</p>
 *
 * <p>
 * The induction variable can't change in the loop body, thus, no storeOp for the induction variable.
 * The loop body does not contain breakOp, or continue conditions.
 * </p>
 *
 */
public class TileForLoopInfo {

    private TileForLoopInfo() {}

    public record LoopInfo(CoreOp.VarOp inductionVar, CoreOp.ConstantOp initValue, Value loopBound, Value step) {}

    private static Value checkConditionalLeftOperandIsSingleLtOp(JavaOp.ForOp forOp) {
        Op yieldConditionOp = forOp.condBody().entryBlock().ops().getLast();
        Value last = yieldConditionOp.operands().getLast();
        if (forOp.condBody().entryBlock().parameters().size() != 1
                || !(last.declaringElement() instanceof JavaOp.LtOp ltOp)
                || !(ltOp.lhsOperand().declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp)
                || !(varLoadOp.varOperand().equals(forOp.condBody().entryBlock().parameters().getFirst()))) {
            throw new IllegalArgumentException("For-loop is not a counted loop");
        }
        return ltOp.rhsOperand();
    }

    public static LoopInfo info(JavaOp.ForOp forOp) {

        // Induction var and init value
        Op yieldValue = forOp.initBody().entryBlock().ops().getLast();
        CoreOp.VarOp inductionVar = null;
        CoreOp.ConstantOp initConstant = null;
        if (yieldValue instanceof CoreOp.YieldOp yieldOp
                && yieldOp.operands().getFirst().declaringElement() instanceof CoreOp.VarOp v
                && v.operands().getFirst().declaringElement() instanceof CoreOp.ConstantOp c) {
            inductionVar = v;
            initConstant = c;
        }

        // Check the induction variable is not mutated
        boolean[] inductionWrite = new boolean[1];
        forOp.loopBody().entryBlock()
                .elements()
                .filter(e -> e instanceof CoreOp.VarAccessOp.VarStoreOp)
                .map(element -> (CoreOp.VarAccessOp.VarStoreOp) element)
                .filter(store -> store.operands().getFirst() instanceof Block.Parameter)
                .forEach(store -> inductionWrite[0] = true);

        if (inductionWrite[0]) {
            throw new IllegalArgumentException("For-loop is not a counted loop");
        }

        // Loop bound
        Value boundValue = checkConditionalLeftOperandIsSingleLtOp(forOp);

        // Step
        /*
         (%73 : Var<java.type:"int">)java.type:"void" -> {
            %74 : java.type:"int" = var.load %73 @loc="226:32";
            %75 : java.type:"int" = constant @loc="226:32" @1;
            %76 : java.type:"int" = add %74 %75 @loc="226:32";
            var.store %73 %76 @loc="226:32";
            yield @loc="226:9";
        }
        */
        List<Op> ops = forOp.updateBody().entryBlock().ops();
        if (!(ops.getLast() instanceof CoreOp.YieldOp)) {
            throw new IllegalStateException("YieldOp expected");
        }

        Op op = ops.get(ops.size() - 2);
        if (!(op instanceof CoreOp.VarAccessOp.VarStoreOp varStoreOp)) {
            throw new IllegalStateException("VarStoreOp expected");
        }
        Value incrementResult = varStoreOp.operands().getLast();
        JavaOp.AddOp addOp = getStepValue(forOp, incrementResult);

        // Final check
        // no loop exit (break) nor continues
        boolean b = forOp.loopBody().entryBlock().elements().anyMatch(e -> e instanceof JavaOp.BreakOp || e instanceof JavaOp.ContinueOp);
        if (b) {
            throw new IllegalArgumentException("For-loop is not a counted loop");
        }

        return new LoopInfo(inductionVar, initConstant, boundValue, addOp.rhsOperand());
    }

    private static JavaOp.AddOp getStepValue(JavaOp.ForOp forOp, Value incrementResult) {
        if (!(incrementResult.declaringElement() instanceof JavaOp.AddOp addOp)) {
            throw new IllegalStateException("AddOp expected");
        }
        if (!(addOp.rhsOperand().declaringElement() instanceof ConstantOp)) {
            throw new IllegalStateException("Constant expected as increment step");
        }
        if (forOp.updateBody().entryBlock().parameters().size() != 1) {
            throw new IllegalArgumentException("Expected one induction variable");
        }
        if (!(addOp.lhsOperand().declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp)) {
            throw new IllegalArgumentException("VarLoadOp expected");
        }
        return addOp;
    }

    private record BoundInfo(ConstantOp loopBound, VarOp varOpOnWrite, boolean check) {}

    private static BoundInfo check(Value value, Map<Value, CodeType> valueMapType) {
        return switch (value.declaringElement()) {
            case ConstantOp c -> new BoundInfo(c, null, false);
            case CoreOp.YieldOp yieldOp -> check(yieldOp.operands().getLast(), valueMapType);
            case JavaOp.LtOp compareOp -> check(compareOp.operands().getLast(), valueMapType);
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> check(varLoadOp.operands().getLast(), valueMapType);
            case VarOp varOp -> {
                CodeType codeType = valueMapType.get(varOp.operands().getFirst());
                if (codeType instanceof ConstantType c) {
                    yield new BoundInfo(CoreOp.constant(codeType, c.value()), varOp, true);
                }
                yield null;
            }
            case null, default -> null;
        };
    }

    private static boolean checkLoopBound(JavaOp.ForOp forOp, Map<Value, CodeType> valueTypeMap) {
        // Obtain loop bound
        Op yieldConditionOp = forOp.condBody().entryBlock().ops().getLast();
        BoundInfo check = check(yieldConditionOp.result(), valueTypeMap);

        if (check == null || check.loopBound == null) {
            throw new IllegalArgumentException("For-loop is not a counted loop");
        }

        // Check loopBound is not written in the loop body
        boolean[] loopBoundWrite = new boolean[1];
        if (check.check) {
            forOp.loopBody().entryBlock()
                    .elements()
                    .filter(e -> e instanceof CoreOp.VarAccessOp.VarStoreOp)
                    .map(element -> (CoreOp.VarAccessOp.VarStoreOp) element)
                    .filter(store -> store.operands().getFirst().equals(check.varOpOnWrite.result()))
                    .forEach(store -> loopBoundWrite[0] = true);
        }

        if (loopBoundWrite[0]) {
            throw new IllegalArgumentException("For-loop is not a counted loop");
        }
        return true;
    }

    public static boolean check(JavaOp.ForOp forOp, Map<Value, CodeType> valueTypeMap) {
        info(forOp);
        return checkLoopBound(forOp, valueTypeMap);
    }

}