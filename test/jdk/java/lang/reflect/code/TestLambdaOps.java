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
 * @run junit TestLambdaOps
 */

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreOp.*;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.LambdaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.INT;
import static jdk.incubator.code.dialect.java.JavaType.type;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestLambdaOps {
    static class Builder {
        static final MethodRef ACCEPT_METHOD = MethodRef.method(type(Builder.class), "accept",
                INT, CoreOp.QuotedOp.QUOTED_TYPE);

        static int accept(Quoted l) {
            Assertions.assertEquals(l.capturedValues().size(), 1);
            Assertions.assertEquals(l.capturedValues().values().iterator().next(), 1);

            List<Object> arguments = new ArrayList<>();
            arguments.add(42);
            arguments.addAll(l.capturedValues().values());
            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) l.op(),
                    arguments);
            return r;
        }
    }

    @Test
    public void testQuotedWithCapture() {
        // functional type = (int)int
        FuncOp f = func("f", functionType(INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    // op type = ()Quoted<LambdaOp>
                    QuotedOp qop = quoted(block.parentBody(), qblock -> {
                        return JavaOp.lambda(qblock.parentBody(),
                                functionType(INT, INT), type(IntUnaryOperator.class))
                                .body(lblock -> {
                                    Block.Parameter li = lblock.parameters().get(0);

                                    lblock.op(return_(
                                            // capture i from function's body
                                            lblock.op(JavaOp.add(i, li))
                                    ));
                                });
                    });
                    Op.Result lquoted = block.op(qop);

                    Op.Result or = block.op(JavaOp.invoke(Builder.ACCEPT_METHOD, lquoted));
                    block.op(return_(or));
                });

        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assertions.assertEquals(43, ir);
    }

    static final MethodRef INT_UNARY_OPERATOR_METHOD = MethodRef.method(
            IntUnaryOperator.class, "applyAsInt",
            int.class, int.class);

    @Test
    public void testWithCapture() {
        // functional type = (int)int
        FuncOp f = func("f", functionType(INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    // op type = ()IntUnaryOperator
                    //   captures i
                    LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                            functionType(INT, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Block.Parameter li = lblock.parameters().get(0);

                                lblock.op(return_(
                                        lblock.op(JavaOp.add(i, li))));
                            });
                    Op.Result fi = block.op(lambda);

                    Op.Result fortyTwo = block.op(constant(INT, 42));
                    Op.Result or = block.op(JavaOp.invoke(INT_UNARY_OPERATOR_METHOD, fi, fortyTwo));
                    block.op(return_(or));
                });

        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assertions.assertEquals(43, ir);
    }

    static int f(int i) {
        IntUnaryOperator fi = li -> {
            return i + li;
        };

        int fortyTwo = 42;
        int or = fi.applyAsInt(fortyTwo);
        return or;
    }

    @Test
    public void testQuotableModel() {
        Quotable quotable = (Runnable & Quotable) () -> {};
        Op qop = Op.ofQuotable(quotable).get().op();
        Op top = qop.ancestorOp().ancestorOp();
        Assertions.assertTrue(top instanceof CoreOp.FuncOp);

        CoreOp.FuncOp fop = (CoreOp.FuncOp) top;
        Assertions.assertEquals(fop.invokableType().returnType(), type(Quoted.class));
    }

    @FunctionalInterface
    public interface QuotableIntSupplier extends IntSupplier, Quotable {
    }

    @CodeReflection
    static QuotableIntSupplier quote(int i) {
        QuotableIntSupplier s = () -> i;
        return s;
    }

    @Test
    public void testQuote() {
        FuncOp g = getFuncOp("quote");
        System.out.println(g.toText());

        {
            QuotableIntSupplier op = (QuotableIntSupplier) Interpreter.invoke(MethodHandles.lookup(), g, 42);
            Assertions.assertEquals(42, op.getAsInt());

            Quoted q = Op.ofQuotable(op).get();
            System.out.println(q.op().toText());
            Assertions.assertEquals(1, q.capturedValues().size());
            Assertions.assertEquals(42, ((Var<?>)q.capturedValues().values().iterator().next()).value());

            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(),
                    new ArrayList<>(q.capturedValues().sequencedValues()));
            Assertions.assertEquals(42, r);

            r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(),
                    List.of(CoreOp.Var.of(0)));
            Assertions.assertEquals(0, r);
        }

        {
            QuotableIntSupplier op = quote(42);
            Assertions.assertEquals(42, op.getAsInt());

            Quoted q = Op.ofQuotable(op).get();
            System.out.println(q.op().toText());
            System.out.print(q.capturedValues().values());
            Assertions.assertEquals(1, q.capturedValues().size());
            Assertions.assertEquals(42, ((Var<?>)q.capturedValues().values().iterator().next()).value());

            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(),
                    new ArrayList<>(q.capturedValues().sequencedValues()));
            Assertions.assertEquals(42, r);

            r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(),
                    List.of(CoreOp.Var.of(0)));
            Assertions.assertEquals(0, r);
        }
    }


    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable {}

    interface QuotableFunction<T, R> extends Function<T, R>, Quotable {}

    interface QuotableBiFunction<T, U, R> extends BiFunction<T, U, R>, Quotable {}

    Iterator<Quotable> methodRefLambdas() {
        return List.of(
                (QuotableIntUnaryOperator) TestLambdaOps::m1,
                (QuotableIntUnaryOperator) TestLambdaOps::m2,
                (QuotableFunction<Integer, Integer>) TestLambdaOps::m1,
                (QuotableFunction<Integer, Integer>) TestLambdaOps::m2,
                (QuotableIntUnaryOperator) this::m3,
                (QuotableBiFunction<TestLambdaOps, Integer, Integer>) TestLambdaOps::m4
        ).iterator();
    }

    @ParameterizedTest
    @MethodSource("methodRefLambdas")
    public void testIsMethodReference(Quotable q) {
        Quoted quoted = Op.ofQuotable(q).get();
        LambdaOp lop = (LambdaOp) quoted.op();
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

    static int m4(TestLambdaOps tl, int i) {
        return i;
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestLambdaOps.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
