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

import jdk.incubator.code.*;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @run junit TestCompareCodeItems
 */

public class TestCompareCodeItems {

    @Reflect
    IntBinaryOperator f = (a, b) -> {
        IntUnaryOperator u = v ->
                v + ((a > b) ? a : b + 42);
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += u.applyAsInt(i) + 42;
            if (sum instanceof byte _) {
                // Increment sum if it fits in a byte
                sum++;
            }
        }
        return sum;
    };

    @Test
    public void testCompareCodeElements() {
        JavaOp.LambdaOp op = Op.ofLambda(f).orElseThrow().op();

        testCompareCodeElements(op);
        op = op.transform(CodeContext.create(), CodeTransformer.LOWERING_TRANSFORMER);
        testCompareCodeElements(op);
    }

    void testCompareCodeElements(Op op) {
        List<CodeElement<?, ?>> l = op.elements().toList();

        for (int i = 0; i < l.size(); i++) {
            for (int j = 0; j < l.size(); j++) {
                CodeElement<?, ?> a = l.get(i);
                CodeElement<?, ?> b = l.get(j);
                Assertions.assertEquals(CodeElement.compare(a, b), Integer.compare(i, j));
            }
        }
    }

    @Test
    public void testCompareCodeElementsDifferentModels() {
        JavaOp.LambdaOp op1 = Op.ofLambda(f).orElseThrow().op();
        JavaOp.LambdaOp op2 = op1.transform(CodeContext.create(), CodeTransformer.COPYING_TRANSFORMER);

        testCompareCodeElements(op1, op2);
    }

    void testCompareCodeElements(Op op1, Op op2) {
        List<CodeElement<?, ?>> l1 = op1.elements().toList();
        List<CodeElement<?, ?>> l2 = op2.elements().toList();

        for (int i = 0; i < l1.size(); i++) {
            for (int j = 0; j < l2.size(); j++) {
                CodeElement<?, ?> a = l1.get(i);
                CodeElement<?, ?> b = l2.get(j);
                Assertions.assertThrows(IllegalArgumentException.class, () -> CodeElement.compare(a, b));
            }
        }
    }


    @Test
    public void testCompareValues() {
        JavaOp.LambdaOp op = Op.ofLambda(f).orElseThrow().op();

        testCompareValues(op);
        op = op.transform(CodeContext.create(), CodeTransformer.LOWERING_TRANSFORMER);
        testCompareValues(op);
        op = SSA.transform(op);
        testCompareValues(op);
    }

    void testCompareValues(Op op) {
        List<Value> l = op.elements().<Value>mapMulti((e, c) -> {
            if (e instanceof Block b) {
                b.parameters().forEach(c);
            } else if (e instanceof Op o) {
                if (o.result() != null) {
                    c.accept(o.result());
                }
            }
        }).toList();

        for (int i = 0; i < l.size(); i++) {
            for (int j = 0; j < l.size(); j++) {
                Value a = l.get(i);
                Value b = l.get(j);
                Assertions.assertEquals(Value.compare(a, b), CodeElement.compare(a.declaringElement(), b.declaringElement()));
            }
        }
    }
}
