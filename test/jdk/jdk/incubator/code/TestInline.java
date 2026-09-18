/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.Inliner;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.IntBinaryOperator;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.INT;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestInline
 */

public class TestInline {

    @Test
    public void testInline() {
        @Reflect
        IntBinaryOperator q = (int a, int b) -> a + b;
        JavaOp.LambdaOp cop = Op.ofLambda(q).get().op();

        // functional type = (int)int
        FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.add(constant(INT, 42));

                    var cb = Inliner.inline(fblock, cop, List.of(i, fortyTwo));
                    cb.add(return_(cb.parameters().getFirst()));
                });

        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assertions.assertEquals(43, ir);
    }

    @Test
    public void testInlineLowerMultipleReturn() {
        @Reflect
        IntBinaryOperator q = (int a, int b) ->  {
            if (a < 10) {
                return a + b;
            }
            return a - b;
        };
        JavaOp.LambdaOp cop = Op.ofLambda(q).get().op();
        System.out.println(cop.toText());
        JavaOp.LambdaOp lcop = cop.transform(CodeContext.create(), CodeTransformer.LOWERING_TRANSFORMER);
        System.out.println(lcop.toText());

        // functional type = (int)int
        FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.add(constant(INT, 42));

                    var cb = Inliner.inline(fblock, lcop, List.of(i, fortyTwo));
                    cb.add(return_(cb.parameters().getFirst()));
                });
        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assertions.assertEquals(43, ir);
    }

    @Test
    public void testInlineMultipleReturnLower() {
        @Reflect
        IntBinaryOperator q = (int a, int b) ->  {
            if (a < 10) {
                return a + b;
            }
            return a - b;
        };
        JavaOp.LambdaOp cop = Op.ofLambda(q).get().op();
        System.out.println(cop.toText());

        FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.add(constant(INT, 42));

                    Inliner.inlineWithContinuation(fblock, cop, List.of(i, fortyTwo), (block) -> {
                        Value returnValue = block.parameters().isEmpty()
                                ? null
                                : block.parameters().getFirst();
                        block.add(returnValue != null ? return_(returnValue) : return_());
                    });
                });
        System.out.println(f.toText());

        f = f.transform(CodeTransformer.LOWERING_TRANSFORMER);
        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assertions.assertEquals(43, ir);
    }

    @Test
    public void testInlineReturnVoid() {
        @Reflect
        Consumer<int[]> q = (int[] a) -> {
            a[0] = 42;
            return;
        };
        JavaOp.LambdaOp cop = Op.ofLambda(q).get().op();

        // functional type = (int)int
        FuncOp f = func("f", functionType(JavaType.VOID, JavaType.type(int[].class)))
                .body(fblock -> {
                    Block.Parameter a = fblock.parameters().get(0);

                    var cb = Inliner.inline(fblock, cop, List.of(a));
                    cb.add(return_());
                });

        System.out.println(f.toText());

        int[] a = new int[1];
        Interpreter.invoke(MethodHandles.lookup(), f, a);
        Assertions.assertEquals(42, a[0]);
    }

    @Test
    public void testInlineMultipleReturnVoid() {
        @Reflect
        Consumer<int[]> q = (int[] a) -> {
            if (a.length == 1) {
                a[0] = 42;
                return;
            } else if (a.length == 2) {
                a[1] = 42;
                return;
            }
        };
        JavaOp.LambdaOp cop = Op.ofLambda(q).get().op();

        // functional type = (int)int
        FuncOp f = func("f", functionType(JavaType.VOID, JavaType.type(int[].class)))
                .body(fblock -> {
                    Block.Parameter a = fblock.parameters().get(0);

                    Inliner.inlineWithContinuation(fblock, cop, List.of(a),
                            continueBlock -> continueBlock.add(return_()));
                });

        System.out.println(f.toText());

        int[] a = new int[1];
        Interpreter.invoke(MethodHandles.lookup(), f, a);
        Assertions.assertEquals(42, a[0]);
    }


    @Reflect
    static void n() {
    }
    @Reflect
    static int m(int i) {
        n();
        return i; // this make sure context needs to be set properly after inlining, for the transformation to work
    }

    @Test
    void testInlineInTransformation() throws NoSuchMethodException {
        FuncOp m = Op.ofMethod(this.getClass().getDeclaredMethod("m", int.class)).get();
        FuncOp n = Op.ofMethod(this.getClass().getDeclaredMethod("n")).get();
        m.transform((b, o) -> {
            if (o instanceof JavaOp.InvokeOp iop && iop.invokeReference().name().equals("n")) {
                return Inliner.inline(b, n, List.of());
            }
            b.add(o);
            return b;
        });
    }

    @Test
    public void testNoInlinableReturns() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder block = body.entryBlock();

        @Reflect
        Runnable noReturn = () -> {
            while (true) {}
        };
        JavaOp.LambdaOp noReturnLambda = Op.ofLambda(noReturn).get().op();
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> Inliner.inline(block, noReturnLambda, List.of()));
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> Inliner.inlineWithContinuation(block, noReturnLambda, List.of(), (_) -> { }));

        @Reflect
        Runnable noInlinableReturn = () -> {
            if (true) {
                return;
            } else {
                return;
            }
        };
        JavaOp.LambdaOp noInlinableReturnLambda = Op.ofLambda(noInlinableReturn).get().op();
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> Inliner.inline(block, noInlinableReturnLambda, List.of()));
    }

    @Test
    public void testNonTargetingReturn() {
        @Reflect
        Runnable nonTargetingReturn = () -> {
            Runnable nested = () -> { return; };
        };
        JavaOp.LambdaOp nonTargetingReturnLambda = Op.ofLambda(nonTargetingReturn).get().op();

        {
            FuncOp f = func("f", FUNCTION_TYPE_VOID).body(block -> {
                block = Inliner.inline(block, nonTargetingReturnLambda, List.of());
                block.add(JavaOp.throw_
                        (block.add(JavaOp.new_(MethodRef.constructor(RuntimeException.class)))));
            });

            List<Op> returnOps = f.elements().filter(e -> e instanceof ReturnOp)
                    .map(e -> (Op) e)
                    .toList();
            Assertions.assertEquals(1, returnOps.size());
            Assertions.assertInstanceOf(JavaOp.LambdaOp.class, returnOps.getFirst().ancestorOp());
        }

        {
            FuncOp f = func("f", FUNCTION_TYPE_VOID).body(block -> {
                Inliner.inlineWithContinuation(block, nonTargetingReturnLambda, List.of(), (continueBlock) -> {
                    continueBlock.add(JavaOp.throw_(
                            continueBlock.add(JavaOp.new_(MethodRef.constructor(RuntimeException.class)))));
                });
            });

            List<Op> returnOps = f.elements().filter(e -> e instanceof ReturnOp)
                    .map(e -> (Op) e)
                    .toList();
            Assertions.assertEquals(1, returnOps.size());
            Assertions.assertInstanceOf(JavaOp.LambdaOp.class, returnOps.getFirst().ancestorOp());
        }
    }


}
