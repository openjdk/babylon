/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package optkl;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.util.ops.StatementLikeOp;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;


public interface OpTkl {

    static Type classTypeToTypeOrThrow(MethodHandles.Lookup lookup, ClassType classType) {
        try {
            return classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static boolean isAssignable(MethodHandles.Lookup lookup, TypeElement typeElement, Class<?>... classes) {
        if (typeElement instanceof ClassType classType) {
            Type type = classTypeToTypeOrThrow(lookup, classType);
            return Arrays.stream(classes).anyMatch(clazz -> clazz.isAssignableFrom((Class<?>) type));
        }
        return false;
    }


    static Op.Result lhsResult(JavaOp.BinaryOp binaryOp) {
        return (Op.Result) binaryOp.operands().get(0);
    }

    static Op.Result rhsResult(JavaOp.BinaryOp binaryOp) {
        return (Op.Result) binaryOp.operands().get(1);
    }

    static List<Op> ops(JavaOp.JavaConditionalOp javaConditionalOp, int idx) {
        return javaConditionalOp.bodies().get(idx).entryBlock().ops();
    }

    static List<Op> lhsOps(JavaOp.JavaConditionalOp javaConditionalOp) {
        return ops(javaConditionalOp, 0);
    }

    static List<Op> rhsOps(JavaOp.JavaConditionalOp javaConditionalOp) {
        return ops(javaConditionalOp, 1);
    }

    static Op.Result result(JavaOp.BinaryTestOp binaryTestOp, int idx) {
        return (Op.Result) binaryTestOp.operands().get(idx);
    }

    static Op.Result lhsResult(JavaOp.BinaryTestOp binaryTestOp) {
        return result(binaryTestOp, 0);
    }

    static Op.Result rhsResult(JavaOp.BinaryTestOp binaryTestOp) {
        return result(binaryTestOp, 1);
    }

    static Op.Result result(JavaOp.ConvOp convOp) {
        return (Op.Result) convOp.operands().getFirst();
    }

    static Op.Result result(CoreOp.ReturnOp returnOp) {
        return (Op.Result) returnOp.operands().getFirst();
    }

    static Block entryBlockOfBodyN(Op op, int idx) {
        return op.bodies().get(idx).entryBlock();
    }

    static Block condBlock(JavaOp.ConditionalExpressionOp ternaryOp) {
        return entryBlockOfBodyN(ternaryOp, 0);
    }

    static Block thenBlock(JavaOp.ConditionalExpressionOp ternaryOp) {
        return entryBlockOfBodyN(ternaryOp, 1);
    }

    static Block elseBlock(JavaOp.ConditionalExpressionOp ternaryOp) {
        return entryBlockOfBodyN(ternaryOp, 2);
    }


    static Value operandNOrNull(Op op, int idx) {
        return op.operands().size() > idx ? op.operands().get(idx) : null;
    }

    static Block updateBlock(JavaOp.ForOp forOp) {return forOp.update().entryBlock();}
    static Block condBlock(JavaOp.ForOp forOp) {
        return forOp.cond().entryBlock();
    }
    static Block initBlock(JavaOp.ForOp forOp) {
        return forOp.init().entryBlock();
    }
    static Block condBlock(JavaOp.WhileOp whileOp) {
        return entryBlockOfBodyN(whileOp, 0);
    }

    static PrimitiveType asPrimitiveResultOrNull(Value v) {
        return (v instanceof Op.Result r && r.op().resultType() instanceof PrimitiveType primitiveType)?primitiveType:null;
    }

    static boolean isPrimitiveResult(Value v) {
        return (asPrimitiveResultOrNull(v) != null);
    }

    static Op.Result asResultOrThrow(Value value) {
        if (value instanceof Op.Result result) {
            return result;
        } else {
            throw new RuntimeException("Value not a result");
        }
    }

    static Stream<Op.Result> operandsAsResults(jdk.incubator.code.CodeElement<?, ?> codeElement) {
        return codeElement instanceof Op ?
                ((Op) codeElement).operands().stream().filter(o -> o instanceof Op.Result).map(o -> (Op.Result) o)
                : Stream.of();
    }

    static Op.Result operandNAsResult(jdk.incubator.code.CodeElement<?, ?> codeElement, int n) {
        return codeElement instanceof Op op && op.operands().size() > n && op.operands().get(n) instanceof Op.Result result ? result : null;
    }

    static Op.Result asResultOrNull(Value operand) {
        return operand instanceof Op.Result result ? result : null;
    }

    static Op asOpFromResultOrNull(Value operand) {
        return asResultOrNull(operand) instanceof Op.Result r && r.op() instanceof Op op ? op : null;
    }

    static Op opOfResultOrNull(Op.Result result) {
        return result.op() instanceof Op op ? op : null;
    }

    static CoreOp.VarAccessOp.VarLoadOp asVarLoadOrNull(Op op) {
        return op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp ? varLoadOp : null;
    }

    static boolean resultType(MethodHandles.Lookup lookup, CoreOp.VarAccessOp.VarLoadOp varLoadOp, Class<?>... classes) {
        return isAssignable(lookup, varLoadOp.resultType(), classes);
    }

    static Stream<Op> loopBodyStatements(Op.Loop op) {
        var list = new ArrayList<>(statements(op.loopBody().entryBlock()).toList());
        if (list.getLast() instanceof JavaOp.ContinueOp) {
            list.removeLast();
        }
        return list.stream();
    }

    static Op asStatementOpOrNull(CodeElement<?, ?> ce) {
        if (ce instanceof Op op) {
            return (
                    (
                            (op instanceof CoreOp.VarAccessOp.VarStoreOp && op.operands().get(1).uses().size() < 2)
                                    || (op instanceof CoreOp.VarOp || op.result().uses().isEmpty())
                                    || (op instanceof StatementLikeOp)
                    )
                            && !(op instanceof CoreOp.VarOp varOp && isParamVar(varOp))//..ParamVar.of(varOp) != null)
                            && !(op instanceof CoreOp.YieldOp)
            )
                    ? op
                    : null;
        } else {
            return null;
        }

    }

    static boolean isStatementOp(CodeElement<?, ?> ce) {
        return Objects.nonNull(asStatementOpOrNull(ce));
    }

    static Stream<Op> statements(Block block) {
        return block.ops().stream().filter(OpTkl::isStatementOp);
    }

    static boolean isParamVar(CoreOp.VarOp varOp) {
        return !varOp.isUninitialized()
                && varOp.operands().getFirst() instanceof Block.Parameter parameter
                && parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp;
    }
}


