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
 * @run testng TestLambdaOps
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.op.CoreOps.FuncOp;
import java.lang.reflect.code.op.CoreOps.LambdaOp;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOps.*;
import static java.lang.reflect.code.op.CoreOps.constant;
import static java.lang.reflect.code.type.FunctionType.functionType;
import static java.lang.reflect.code.type.JavaType.INT;
import static java.lang.reflect.code.type.JavaType.type;

@Test
public class TestLambdaOps {
    static class Builder {
        static final MethodRef ACCEPT_METHOD = MethodRef.method(type(Builder.class), "accept",
                INT, CoreOps.QuotedOp.QUOTED_TYPE);

        static int accept(Quoted l) {
            Assert.assertEquals(1, l.capturedValues().size());
            Assert.assertEquals(1, l.capturedValues().values().iterator().next());

            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) l.op(),
                    l.capturedValues(), 42);
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
                        return lambda(qblock.parentBody(),
                                functionType(INT, INT), type(IntUnaryOperator.class))
                                .body(lblock -> {
                                    Block.Parameter li = lblock.parameters().get(0);

                                    lblock.op(_return(
                                            // capture i from function's body
                                            lblock.op(add(i, li))
                                    ));
                                });
                    });
                    Op.Result lquoted = block.op(qop);

                    Op.Result or = block.op(invoke(Builder.ACCEPT_METHOD, lquoted));
                    block.op(_return(or));
                });

        f.writeTo(System.out);

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
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
                    LambdaOp lambda = lambda(block.parentBody(),
                            functionType(INT, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Block.Parameter li = lblock.parameters().get(0);

                                lblock.op(_return(
                                        lblock.op(add(i, li))));
                            });
                    Op.Result fi = block.op(lambda);

                    Op.Result fortyTwo = block.op(constant(INT, 42));
                    Op.Result or = block.op(invoke(INT_UNARY_OPERATOR_METHOD, fi, fortyTwo));
                    block.op(_return(or));
                });

        f.writeTo(System.out);

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
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
        Op qop = quotable.quoted().op();
        Op top = qop.ancestorBody().parentOp().ancestorBody().parentOp();
        Assert.assertTrue(top instanceof CoreOps.FuncOp);

        CoreOps.FuncOp fop = (CoreOps.FuncOp) top;
        Assert.assertEquals(type(Quoted.class), fop.invokableType().returnType());
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
        g.writeTo(System.out);

        {
            QuotableIntSupplier op = (QuotableIntSupplier) Interpreter.invoke(MethodHandles.lookup(), g, 42);
            Assert.assertEquals(op.getAsInt(), 42);

            Quoted q = op.quoted();
            q.op().writeTo(System.out);
            Assert.assertEquals(q.capturedValues().size(), 1);
            Assert.assertEquals(((Var<?>)q.capturedValues().values().iterator().next()).value(), 42);

            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(), q.capturedValues(), List.of());
            Assert.assertEquals(r, 42);

            Map<Value, Object> cvs = Map.of(
                    q.capturedValues().keySet().iterator().next(),
                    CoreOps.Var.of(0)
            );
            r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(), cvs, List.of());
            Assert.assertEquals(r, 0);
        }

        {
            QuotableIntSupplier op = quote(42);
            Assert.assertEquals(op.getAsInt(), 42);

            Quoted q = op.quoted();
            q.op().writeTo(System.out);
            System.out.print(q.capturedValues().values());
            Assert.assertEquals(q.capturedValues().size(), 1);
            Assert.assertEquals(((Var<?>)q.capturedValues().values().iterator().next()).value(), 42);

            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(), q.capturedValues(), List.of());
            Assert.assertEquals(r, 42);

            Map<Value, Object> cvs = Map.of(
                    q.capturedValues().keySet().iterator().next(),
                    CoreOps.Var.of(0)
            );
            r = (int) Interpreter.invoke(MethodHandles.lookup(), (LambdaOp) q.op(), cvs, List.of());
            Assert.assertEquals(r, 0);
        }
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestLambdaOps.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
