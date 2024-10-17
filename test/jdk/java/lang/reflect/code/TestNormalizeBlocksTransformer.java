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
 * @run testng TestNormalizeBlocksTransformer
 */

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.NormalizeBlocksTransformer;
import java.lang.reflect.code.op.ExtendedOp;
import java.lang.reflect.code.parser.OpParser;
import java.util.stream.Stream;

public class TestNormalizeBlocksTransformer {
    static final String TEST1_INPUT = """
            func @"f" (%0 : int)int -> {
                %1 : int = invoke @"C::m()int";
                branch ^block_1;

              ^block_1:
                %2 : int = invoke %1 @"C::m(int)int";
                branch ^block_2(%2);

              ^block_2(%3: int):
                %4 : int = invoke %2 %3 @"C::m(int, int)int";
                branch ^block_3(%3);

              ^block_3(%5: int):
                %6 : int = invoke %4 %3 %5 @"C::m(int, int, int)int";
                branch ^block_4;

              ^block_4:
                return %6;
            };
            """;
    static final String TEST1_EXPECTED = """
            func @"f" (%0 : int)int -> {
                %1 : int = invoke @"C::m()int";
                %2 : int = invoke %1 @"C::m(int)int";
                %3 : int = invoke %2 %2 @"C::m(int, int)int";
                %4 : int = invoke %3 %2 %2 @"C::m(int, int, int)int";
                return %4;
            };
            """;

    static final String TEST2_INPUT = """
            func @"f" (%0 : java.lang.Object)void -> {
                %1 : Var<java.lang.Object> = var %0 @"o";
                exception.region.enter ^block_1 ^block_8 ^block_3;

              ^block_1:
                %3 : int = invoke @"A::try_()int";
                branch ^block_2;

              ^block_2:
                exception.region.exit ^block_6 ^block_3 ^block_8;

              ^block_3(%4 : java.lang.RuntimeException):
                exception.region.enter ^block_4 ^block_8;

              ^block_4:
                %6 : Var<java.lang.RuntimeException> = var %4 @"e";
                branch ^block_5;

              ^block_5:
                exception.region.exit ^block_6 ^block_8;

              ^block_6:
                %7 : int = invoke @"A::finally_()int";
                branch ^block_7;

              ^block_7:
                return;

              ^block_8(%8 : java.lang.Throwable):
                %9 : int = invoke @"A::finally_()int";
                throw %8;
            };
            """;
    static final String TEST2_EXPECTED = """
            func @"f" (%0 : java.lang.Object)void -> {
                %1 : Var<java.lang.Object> = var %0 @"o";
                exception.region.enter ^block_1 ^block_5 ^block_2;

              ^block_1:
                %3 : int = invoke @"A::try_()int";
                exception.region.exit ^block_4 ^block_2 ^block_5;

              ^block_2(%4 : java.lang.RuntimeException):
                exception.region.enter ^block_3 ^block_5;

              ^block_3:
                %6 : Var<java.lang.RuntimeException> = var %4 @"e";
                exception.region.exit ^block_4 ^block_5;

              ^block_4:
                %7 : int = invoke @"A::finally_()int";
                return;

              ^block_5(%8 : java.lang.Throwable):
                %9 : int = invoke @"A::finally_()int";
                throw %8;
            };""";

    static final String TEST3_INPUT = """
            func @"f" (%0 : int)int -> {
                %1 : int = constant @"0";
                %2 : boolean = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : int = constant @"1";
                branch ^block_1_1;

              ^block_1_1:
                branch ^block_3(%3);

              ^block_2:
                %4 : int = constant @"-1";
                branch ^block_2_1;

              ^block_2_1:
                branch ^block_3(%4);

              ^block_3(%5 : int):
                return %5;
            };""";
    static final String TEST3_EXPECTED = """
            func @"f" (%0 : int)int -> {
                %1 : int = constant @"0";
                %2 : boolean = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : int = constant @"1";
                branch ^block_3(%3);

              ^block_2:
                %4 : int = constant @"-1";
                branch ^block_3(%4);

              ^block_3(%5 : int):
                return %5;
            };
            """;

    static final String TEST4_INPUT = """
            func @"f" (%0 : int)int -> {
                %1 : int = constant @"0";
                %2 : boolean = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : int = constant @"1";
                branch ^block_1_1;

              ^block_1_1:
                branch ^block_3(%0, %3, %1);

              ^block_2:
                %4 : int = constant @"-1";
                branch ^block_2_1;

              ^block_2_1:
                branch ^block_3(%0, %4, %1);

              ^block_3(%unused_1 : int, %5 : int, %unused_2 : int):
                return %5;
            };""";
    static final String TEST4_EXPECTED = """
            func @"f" (%0 : int)int -> {
                %1 : int = constant @"0";
                %2 : boolean = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : int = constant @"1";
                branch ^block_3(%3);

              ^block_2:
                %4 : int = constant @"-1";
                branch ^block_3(%4);

              ^block_3(%5 : int):
                return %5;
            };
            """;

    static final String TEST5_INPUT = """
            func @"f" ()void -> {
                exception.region.enter ^block_1 ^block_4;

              ^block_1:
                invoke @"A::m()void";
                branch ^block_2;

              ^block_2:
                exception.region.exit ^block_3 ^block_4;

              ^block_3:
                branch ^block_5;

              ^block_4(%1 : java.lang.Throwable):
                branch ^block_5;

              ^block_5:
                return;
            };
            """;
    static final String TEST5_EXPECTED = """
            func @"f" ()void -> {
                exception.region.enter ^block_1 ^block_3;

              ^block_1:
                invoke @"A::m()void";
                exception.region.exit ^block_2 ^block_3;

              ^block_2:
                branch ^block_4;

              ^block_3(%1 : java.lang.Throwable):
                branch ^block_4;

              ^block_4:
                return;
            };
            """;

    @DataProvider
    static Object[][] testModels() {
        return new Object[][]{
                parse(TEST1_INPUT, TEST1_EXPECTED),
                parse(TEST2_INPUT, TEST2_EXPECTED),
                parse(TEST3_INPUT, TEST3_EXPECTED),
                parse(TEST4_INPUT, TEST4_EXPECTED),
                parse(TEST5_INPUT, TEST5_EXPECTED),
        };
    }

    static Object[] parse(String... models) {
        return Stream.of(models).map(s -> OpParser.fromString(ExtendedOp.FACTORY, s).getFirst())
                .toArray(Object[]::new);
    }

    @Test(dataProvider = "testModels")
    public void test(Op input, Op expected) {
        Op actual = NormalizeBlocksTransformer.transform(input);
        Assert.assertEquals(actual.toText(), expected.toText());
    }
}
