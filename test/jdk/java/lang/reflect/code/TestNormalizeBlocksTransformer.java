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
 * @run junit TestNormalizeBlocksTransformer
 */

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.NormalizeBlocksTransformer;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.extern.OpParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

public class TestNormalizeBlocksTransformer {
    static final String TEST1_INPUT = """
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = invoke @java.ref:"C::m():int";
                branch ^block_1;

              ^block_1:
                %2 : java.type:"int" = invoke %1 @java.ref:"C::m(int):int";
                branch ^block_2(%2);

              ^block_2(%3: java.type:"int"):
                %4 : java.type:"int" = invoke %2 %3 @java.ref:"C::m(int, int):int";
                branch ^block_3(%3);

              ^block_3(%5: java.type:"int"):
                %6 : java.type:"int" = invoke %4 %3 %5 @java.ref:"C::m(int, int, int):int";
                branch ^block_4;

              ^block_4:
                return %6;
            };
            """;
    static final String TEST1_EXPECTED = """
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = invoke @java.ref:"C::m():int";
                %2 : java.type:"int" = invoke %1 @java.ref:"C::m(int):int";
                %3 : java.type:"int" = invoke %2 %2 @java.ref:"C::m(int, int):int";
                %4 : java.type:"int" = invoke %3 %2 %2 @java.ref:"C::m(int, int, int):int";
                return %4;
            };
            """;

    static final String TEST2_INPUT = """
            func @"f" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                exception.region.enter ^block_1 ^block_8 ^block_3;

              ^block_1:
                %3 : java.type:"int" = invoke @java.ref:"A::try_():int";
                branch ^block_2;

              ^block_2:
                exception.region.exit ^block_6 ^block_3 ^block_8;

              ^block_3(%4 : java.type:"java.lang.RuntimeException"):
                exception.region.enter ^block_4 ^block_8;

              ^block_4:
                %6 : Var<java.type:"java.lang.RuntimeException"> = var %4 @"e";
                branch ^block_5;

              ^block_5:
                exception.region.exit ^block_6 ^block_8;

              ^block_6:
                %7 : java.type:"int" = invoke @java.ref:"A::finally_():int";
                branch ^block_7;

              ^block_7:
                return;

              ^block_8(%8 : java.type:"java.lang.Throwable"):
                %9 : java.type:"int" = invoke @java.ref:"A::finally_():int";
                throw %8;
            };
            """;
    static final String TEST2_EXPECTED = """
            func @"f" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                exception.region.enter ^block_1 ^block_5 ^block_2;

              ^block_1:
                %3 : java.type:"int" = invoke @java.ref:"A::try_():int";
                exception.region.exit ^block_4 ^block_2 ^block_5;

              ^block_2(%4 : java.type:"java.lang.RuntimeException"):
                exception.region.enter ^block_3 ^block_5;

              ^block_3:
                %6 : Var<java.type:"java.lang.RuntimeException"> = var %4 @"e";
                exception.region.exit ^block_4 ^block_5;

              ^block_4:
                %7 : java.type:"int" = invoke @java.ref:"A::finally_():int";
                return;

              ^block_5(%8 : java.type:"java.lang.Throwable"):
                %9 : java.type:"int" = invoke @java.ref:"A::finally_():int";
                throw %8;
            };""";

    static final String TEST3_INPUT = """
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = constant @0;
                %2 : java.type:"boolean" = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : java.type:"int" = constant @1;
                branch ^block_1_1;

              ^block_1_1:
                branch ^block_3(%3);

              ^block_2:
                %4 : java.type:"int" = constant @-1;
                branch ^block_2_1;

              ^block_2_1:
                branch ^block_3(%4);

              ^block_3(%5 : java.type:"int"):
                return %5;
            };""";
    static final String TEST3_EXPECTED = """
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = constant @0;
                %2 : java.type:"boolean" = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : java.type:"int" = constant @1;
                branch ^block_3(%3);

              ^block_2:
                %4 : java.type:"int" = constant @-1;
                branch ^block_3(%4);

              ^block_3(%5 : java.type:"int"):
                return %5;
            };
            """;

    static final String TEST4_INPUT = """
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = constant @0;
                %2 : java.type:"boolean" = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : java.type:"int" = constant @1;
                branch ^block_1_1;

              ^block_1_1:
                branch ^block_3(%0, %3, %1);

              ^block_2:
                %4 : java.type:"int" = constant @-1;
                branch ^block_2_1;

              ^block_2_1:
                branch ^block_3(%0, %4, %1);

              ^block_3(%unused_1 : java.type:"int", %5 : java.type:"int", %unused_2 : java.type:"int"):
                return %5;
            };""";
    static final String TEST4_EXPECTED = """
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = constant @0;
                %2 : java.type:"boolean" = gt %0 %1;
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : java.type:"int" = constant @1;
                branch ^block_3(%3);

              ^block_2:
                %4 : java.type:"int" = constant @-1;
                branch ^block_3(%4);

              ^block_3(%5 : java.type:"int"):
                return %5;
            };
            """;

    static final String TEST5_INPUT = """
            func @"f" ()java.type:"void" -> {
                exception.region.enter ^block_1 ^block_4;

              ^block_1:
                invoke @java.ref:"A::m():void";
                branch ^block_2;

              ^block_2:
                exception.region.exit ^block_3 ^block_4;

              ^block_3:
                branch ^block_5;

              ^block_4(%1 : java.type:"java.lang.Throwable"):
                branch ^block_5;

              ^block_5:
                return;
            };
            """;
    static final String TEST5_EXPECTED = """
            func @"f" ()java.type:"void" -> {
                exception.region.enter ^block_1 ^block_3;

              ^block_1:
                invoke @java.ref:"A::m():void";
                exception.region.exit ^block_2 ^block_3;

              ^block_2:
                branch ^block_4;

              ^block_3(%1 : java.type:"java.lang.Throwable"):
                branch ^block_4;

              ^block_4:
                return;
            };
            """;

    static final String TEST6_INPUT = """
            func @"m" (%0 : java.type:"java.lang.Object")java.type:"int" -> {
                %1 : java.type:"java.lang.Object" = constant @null;
                %2 : java.type:"boolean" = invoke %0 %1 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : java.type:"java.lang.NullPointerException" = new @java.ref:"java.lang.NullPointerException::()";
                throw %3;

              ^block_2:
                %4 : java.type:"boolean" = instanceof %0 @java.type:"java.util.List";
                cbranch %4 ^block_3 ^block_5;

              ^block_3:
                %5 : java.type:"java.util.List" = cast %0 @java.type:"java.util.List";
                branch ^block_4;

              ^block_4:
                %6 : java.type:"boolean" = constant @true;
                branch ^block_6(%6);

              ^block_5:
                %7 : java.type:"boolean" = constant @false;
                branch ^block_6(%7);

              ^block_6(%8 : java.type:"boolean"):
                cbranch %8 ^block_7 ^block_8;

              ^block_7:
                %9 : java.type:"int" = constant @1;
                branch ^block_21(%9);

              ^block_8:
                %10 : java.type:"boolean" = instanceof %0 @java.type:"java.lang.String";
                cbranch %10 ^block_9 ^block_11;

              ^block_9:
                %11 : java.type:"java.lang.String" = cast %0 @java.type:"java.lang.String";
                branch ^block_10;

              ^block_10:
                %12 : java.type:"boolean" = constant @true;
                branch ^block_12(%12);

              ^block_11:
                %13 : java.type:"boolean" = constant @false;
                branch ^block_12(%13);

              ^block_12(%14 : java.type:"boolean"):
                cbranch %14 ^block_13 ^block_14;

              ^block_13:
                %15 : java.type:"int" = constant @2;
                branch ^block_21(%15);

              ^block_14:
                %16 : java.type:"boolean" = instanceof %0 @java.type:"java.util.Map";
                cbranch %16 ^block_15 ^block_17;

              ^block_15:
                %17 : java.type:"java.util.Map" = cast %0 @java.type:"java.util.Map";
                branch ^block_16;

              ^block_16:
                %18 : java.type:"boolean" = constant @true;
                branch ^block_18(%18);

              ^block_17:
                %19 : java.type:"boolean" = constant @false;
                branch ^block_18(%19);

              ^block_18(%20 : java.type:"boolean"):
                cbranch %20 ^block_19 ^block_20;

              ^block_19:
                %21 : java.type:"int" = constant @3;
                branch ^block_21(%21);

              ^block_20:
                %22 : java.type:"int" = constant @-1;
                branch ^block_21(%22);

              ^block_21(%23 : java.type:"int"):
                return %23;
            };
            """;
    static final String TEST6_EXPECTED = """
            func @"m" (%0 : java.type:"java.lang.Object")java.type:"int" -> {
                %1 : java.type:"java.lang.Object" = constant @null;
                %2 : java.type:"boolean" = invoke %0 %1 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                cbranch %2 ^block_1 ^block_2;

              ^block_1:
                %3 : java.type:"java.lang.NullPointerException" = new @java.ref:"java.lang.NullPointerException::()";
                throw %3;

              ^block_2:
                %4 : java.type:"boolean" = instanceof %0 @java.type:"java.util.List";
                cbranch %4 ^block_3 ^block_4;

              ^block_3:
                %5 : java.type:"java.util.List" = cast %0 @java.type:"java.util.List";
                %6 : java.type:"int" = constant @1;
                branch ^block_9(%6);

              ^block_4:
                %7 : java.type:"boolean" = instanceof %0 @java.type:"java.lang.String";
                cbranch %7 ^block_5 ^block_6;

              ^block_5:
                %8 : java.type:"java.lang.String" = cast %0 @java.type:"java.lang.String";
                %9 : java.type:"int" = constant @2;
                branch ^block_9(%9);

              ^block_6:
                %10 : java.type:"boolean" = instanceof %0 @java.type:"java.util.Map";
                cbranch %10 ^block_7 ^block_8;

              ^block_7:
                %11 : java.type:"java.util.Map" = cast %0 @java.type:"java.util.Map";
                %12 : java.type:"int" = constant @3;
                branch ^block_9(%12);

              ^block_8:
                %13 : java.type:"int" = constant @-1;
                branch ^block_9(%13);

              ^block_9(%14 : java.type:"int"):
                return %14;
            };
            """;
    static Object[][] testModels() {
        return new Object[][]{
                parse(TEST1_INPUT, TEST1_EXPECTED),
                parse(TEST2_INPUT, TEST2_EXPECTED),
                parse(TEST3_INPUT, TEST3_EXPECTED),
                parse(TEST4_INPUT, TEST4_EXPECTED),
                parse(TEST5_INPUT, TEST5_EXPECTED),
                parse(TEST6_INPUT, TEST6_EXPECTED),
        };
    }

    static Object[] parse(String... models) {
        return Stream.of(models).map(s -> OpParser.fromText(JavaOp.JAVA_DIALECT_FACTORY, s).getFirst())
                .toArray(Object[]::new);
    }

    @ParameterizedTest
    @MethodSource("testModels")
    public void test(Op input, Op expected) {
        Op actual = NormalizeBlocksTransformer.transform(input);
        Assertions.assertEquals(expected.toText(), actual.toText());
    }
}
