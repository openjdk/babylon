/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestMethodRefLambda
 */

import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;

// The comment allow us to use non-static method source
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestMethodRefLambda {

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable {}

    interface QuotableFunction<T, R> extends Function<T, R>, Quotable {}

    interface QuotableBiFunction<T, U, R> extends BiFunction<T, U, R>, Quotable {}

    List<Quotable> methodRefLambdas() {
        return List.of(
                (QuotableIntUnaryOperator) TestMethodRefLambda::m1,
                (QuotableIntUnaryOperator) TestMethodRefLambda::m2,
                (QuotableFunction<Integer, Integer>) TestMethodRefLambda::m1,
                (QuotableFunction<Integer, Integer>) TestMethodRefLambda::m2,
                (QuotableIntUnaryOperator) this::m3,
                (QuotableBiFunction<TestMethodRefLambda, Integer, Integer>) TestMethodRefLambda::m4
        );
    }

    @ParameterizedTest
    @MethodSource("methodRefLambdas")
    public void testIsMethodReference(Quotable q) {
        Quoted quoted = Op.ofQuotable(q).get();
        JavaOp.LambdaOp lop = (JavaOp.LambdaOp) quoted.op();
        Assertions.assertTrue(lop.methodReference().isPresent());
    }

    static int m1(int i) {
        return i;
    }

    static Integer m2(Integer i) {
        return i;
    }

    int m3(int i) {
        return i;
    }

    static int m4(TestMethodRefLambda tl, int i) {
        return i;
    }
}
