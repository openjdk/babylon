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

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.ConstantExpressionTransformer;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestStringConstantExpressionInterning
 */
public class TestStringConstantExpressionInterning {

    @Reflect
    static boolean t_strLiteral() {
        return "A" + 1 == "A1";
    }

    static final String B = "B";
    @Reflect
    static boolean t_classVariable() {
        return B + 1 == "B1";
    }

    static final String C;
    static {
        C = "C";
    }
    @Reflect
    static boolean f_classVariable2() {
        return C + 1 == "C1";
    }

    static final String D = "D".toLowerCase();
    @Reflect
    static boolean f_classVariable3() {
        return D == "d";
    }

    final String E = "E";
    @Reflect
    boolean t_instanceVariable() {
        return E + 1 == "E1";
    }

    @Reflect
    static boolean t_localVariable() {
        final String s = "A";
        return s + 1 == "A1";
    }

    @Reflect
    static boolean t_case7() {
        // @@@ we keep the conditional expr in the model
        String s = 1 < 2 ? "A" + 1 : LocalDate.now().toString();
        return s == "A1";
    }

    @Reflect
    static boolean t_case8() {
        int i = 1;
        String s = switch(i) {
            case 1 -> "A" + 2 + 4;
            default -> "B";
        };
        return s == "A24";
    }

    static Stream<Method> cases() {
        return Arrays.stream(TestStringConstantExpressionInterning.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class));
    }

    @ParameterizedTest
    @MethodSource("cases")
    void test(Method m) throws Throwable {
        FuncOp op = Op.ofMethod(m).get();
        // expected value is encoded in the method name because evaluation API not fully aligned with JLS
        Object expected = m.getName().startsWith("t");
        MethodHandles.Lookup l = MethodHandles.lookup();

        FuncOp lop = op.transform(CodeTransformer.LOWERING_TRANSFORMER);
        FuncOp cfop = ConstantExpressionTransformer.transform(lop);
        List<Object> args = new ArrayList<>();
        if (!m.accessFlags().contains(AccessFlag.STATIC)) {
            args.add(this);
        }
        Assertions.assertEquals(expected, Interpreter.invoke(l, cfop, args));

        Assertions.assertEquals(expected, BytecodeGenerator.generate(l, op).invokeWithArguments(args.toArray()));
    }


    static Stream<Arguments> models() {
        return Stream.of(
                Arguments.of(
                        func("f1", CoreType.functionType(JavaType.BOOLEAN)).body(b -> {
                            Op.Result c1 = b.add(constant(JavaType.J_L_STRING, "A"));
                            Op.Result c2 = b.add(constant(JavaType.J_L_STRING, "A"));
                            b.add(return_(b.add(JavaOp.eq(c1, c2))));
                        }),
                        true
                ),
                Arguments.of(
                        func("f2", CoreType.functionType(JavaType.BOOLEAN)).body(b -> {
                            Op.Result c1 = b.add(constant(JavaType.J_L_STRING, "A1"));
                            Op.Result c2 = b.add(JavaOp.concat(
                                    b.add(constant(JavaType.J_L_STRING, "A")),
                                    b.add(constant(JavaType.INT, 1))));
                            b.add(return_(b.add(JavaOp.eq(c1, c2))));
                        }), true
                ),
                Arguments.of(
                        func("f3", CoreType.functionType(JavaType.BOOLEAN)).body(b -> {
                            // by using ConstantOp we are modeling a literal
                            Op.Result c1 = b.add(constant(JavaType.J_L_STRING, LocalDate.now().toString()));
                            Op.Result c2 = b.add(constant(JavaType.J_L_STRING, LocalDate.now().toString()));
                            b.add(return_(b.add(JavaOp.eq(c1, c2))));
                        }), true
                ),
                Arguments.of(
                        func("f4", CoreType.functionType(JavaType.BOOLEAN)).body(b -> {
                            Body.Builder p = Body.Builder.of(b.parentBody(), CoreType.functionType(JavaType.BOOLEAN));
                            Block.Builder pb = p.entryBlock();
                            pb.add(core_yield(pb.add(constant(JavaType.BOOLEAN, true))));

                            Body.Builder t = Body.Builder.of(b.parentBody(), CoreType.functionType(JavaType.J_L_STRING));
                            Block.Builder tb = t.entryBlock();
                            tb.add(core_yield(tb.add(JavaOp.concat(
                                    tb.add(constant(JavaType.J_L_STRING, "A")),
                                    tb.add(constant(JavaType.INT, 1))
                            ))));

                            Body.Builder f = Body.Builder.of(b.parentBody(), CoreType.functionType(JavaType.J_L_STRING));
                            Block.Builder fb = f.entryBlock();
                            fb.add(core_yield(fb.add(constant(JavaType.J_L_STRING, null))));

                            Op.Result c1 = b.add(JavaOp.conditionalExpression(p, t, f));
                            Op.Result c2 = b.add(constant(JavaType.J_L_STRING, "A1"));
                            b.add(return_(b.add(JavaOp.eq(c1, c2))));
                        }), true
                ),
                Arguments.of(
                        func("f5", CoreType.functionType(JavaType.BOOLEAN)).body(b -> {
                            Op.Result c1 = b.add(JavaOp.concat(
                                    b.add(constant(JavaType.J_L_STRING, "A")),
                                    b.add(constant(JavaType.INT, 1))));
                            Op.Result c2 = b.add(JavaOp.cast(JavaType.J_L_STRING, b.add(JavaOp.concat(
                                    b.add(constant(JavaType.J_L_STRING, "A")),
                                    b.add(constant(JavaType.INT, 1))))));
                            b.add(return_(b.add(JavaOp.eq(c1, c2))));
                        }), true
                )
        );
    }

    @ParameterizedTest
    @MethodSource("models")
    void testManuallyBuiltModel(FuncOp op, Object expected) {
        op = ConstantExpressionTransformer.transform(op);
        Object actual = Interpreter.invoke(MethodHandles.lookup(), op);
        Assertions.assertEquals(expected, actual, op.funcName());
    }

}
