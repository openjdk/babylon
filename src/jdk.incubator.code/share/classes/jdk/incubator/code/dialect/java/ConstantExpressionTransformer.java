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
package jdk.incubator.code.dialect.java;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;

import java.lang.invoke.MethodHandles;
import java.util.Optional;

import static jdk.incubator.code.dialect.core.CoreOp.constant;
import static jdk.incubator.code.dialect.java.JavaOp.ConstantExpressionEvaluator;

/**
 * A transformer that replaces every operation that models a constant expression with a constant operation
 * whose value is the result of evaluating that constant expression.
 */
public class ConstantExpressionTransformer implements CodeTransformer {
    private final ConstantExpressionEvaluator constantExpressionEvaluator;

    private ConstantExpressionTransformer(ConstantExpressionEvaluator constantExpressionEvaluator) {
        this.constantExpressionEvaluator = constantExpressionEvaluator;
    }

    @Override
    public Block.Builder acceptOp(Block.Builder b, Op op) {
        Optional<Object> v = constantExpressionEvaluator.evaluate(op.result());
        if (v.isPresent()) {
            Op.Result c = b.add(constant(op.resultType(), v.get()));
            b.context().mapValue(op.result(), c);
        } else {
            b.add(op);
        }
        return b;
    }

    /**
     * Transforms an operation, replacing an operation that models a constant expression with a constant operation
     * whose value is the result of evaluating that constant expression.
     *
     * @param l  the lookup to use for reflective access
     * @param op the operation to transform
     * @return the transformed operation
     */
    public static Op transform(MethodHandles.Lookup l, Op op) {
        return op.transform(CodeContext.create(), new ConstantExpressionTransformer(new ConstantExpressionEvaluator(l)));
    }
}
