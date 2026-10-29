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

import java.util.Arrays;
import java.util.List;

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.internal.ConstantValueAnalysis;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import static org.junit.jupiter.api.Assertions.*;

/*
 * @test
 * @summary Test coverage for constant expression forms specified in JLS 15.29
 * @modules jdk.incubator.code/jdk.incubator.code.internal
 * @run junit TestConstantValueAnalysis
 */
public class TestConstantValueAnalysis {

    @Reflect
    static int primitiveLiteral() {
        return 1;
    }

    @Reflect
    static String stringLiteral() {
        return "hello";
    }

    @Reflect
    static String textBlock() {
        return """
                hello
                """;
    }

    @Reflect
    static byte primitiveCast() {
        return (byte) (1 + 2);
    }

    @Reflect
    static String stringCast() {
        return (String) "hello";
    }

    @Reflect
    static int unaryNumeric() {
        return +(-~1);
    }

    @Reflect
    static boolean unaryLogical() {
        return !false;
    }

    @Reflect
    static int multiplicative() {
        return 1 * 2 / 3 % 4;
    }

    @Reflect
    static int additive() {
        return 1 + 2 - 3;
    }

    @Reflect
    static String stringAdditive() {
        return "hello" + 1;
    }

    @Reflect
    static int shifts() {
        return ((1 << 2) >> 1) >>> 1;
    }

    @Reflect
    static boolean relational() {
        return (1 < 2) & (1 <= 2) & (2 > 1) & (2 >= 1);
    }

    @Reflect
    static boolean equality() {
        return (1 == 1) & (1 != 2);
    }

    @Reflect
    static int bitwise() {
        return (1 & 2) | (3 ^ 4);
    }

    @Reflect
    static boolean logical() {
        return true & (false | (true ^ false));
    }

    @Reflect
    static boolean conditionalAnd() {
        return true && false;
    }

    @Reflect
    static boolean conditionalOr() {
        return false || true;
    }

    @Reflect
    static int conditional() {
        return true ? 1 : 2;
    }

    @Reflect
    static int parenthesized() {
        return (1 + 2) * 3;
    }

    @Reflect
    static int simpleConstantName() {
        final int constant = 1 + 2;
        return constant;
    }

    @Reflect
    static int qualifiedConstantName() {
        return Integer.BYTES;
    }

    @Test
    void testEvaluateConstantExpressions() {
        assertAll(Arrays.stream(getClass().getDeclaredMethods())
                        .filter(m -> m.isAnnotationPresent(Reflect.class))
                        .map(m -> (Executable) () -> {
                    Value v = ((CoreOp.ReturnOp)Op.ofMethod(m).orElseThrow().body().entryBlock().terminatingOp()).returnValue();
                    assertEquals(m.invoke(null), new ConstantValueAnalysis(_ -> List.of()).evaluate(v).orElseThrow(), m.getName());
                }));
    }
}
