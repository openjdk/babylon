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
 * @run testng TestTry
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.Stream;

public class TestTry {

    @CodeReflection
    public static void catching(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
        } catch (IllegalStateException e) {
            consume(e);
            c.accept(1);
            c.accept(-1);
        } catch (IllegalArgumentException e) {
            consume(e);
            c.accept(2);
            c.accept(-1);
        }
        c.accept(3);
        c.accept(-1);
    }

    @Test
    public void testCatching() {
        CoreOp.FuncOp f = getFuncOp("catching");

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lf, c),
                TestTry::catching);

        test.accept(i -> {
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalArgumentException();
        });
        test.accept(i -> {
            if (i == 0) throw new NullPointerException();
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
            if (i == 1) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalArgumentException();
            if (i == 2) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 3) throw new IllegalStateException();
        });
    }


    @CodeReflection
    public static void catchThrowable(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
        } catch (IllegalStateException e) {
            consume(e);
            c.accept(1);
            c.accept(-1);
        } catch (Throwable e) {
            consume(e);
            c.accept(2);
            c.accept(-1);
        }
        c.accept(3);
        c.accept(-1);
    }

    @Test
    public void testCatchThrowable() {
        CoreOp.FuncOp f = getFuncOp("catchThrowable");

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lf, c),
                TestTry::catchThrowable);

        test.accept(i -> {
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
        });
        test.accept(i -> {
            if (i == 0) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
            if (i == 1) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 0) throw new RuntimeException();
            if (i == 2) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 3) throw new IllegalStateException();
        });
    }


    @CodeReflection
    public static void catchNested(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
            try {
                c.accept(1);
                c.accept(-1);
            } catch (IllegalStateException e) {
                consume(e);
                c.accept(2);
                c.accept(-1);
            }
            c.accept(3);
            c.accept(-1);
        } catch (IllegalArgumentException e) {
            consume(e);
            c.accept(4);
            c.accept(-1);
        }
        c.accept(5);
        c.accept(-1);
    }

    @Test
    public void testCatchNested() {
        CoreOp.FuncOp f = getFuncOp("catchNested");

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lf, c),
                TestTry::catchNested);

        test.accept(i -> {
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalArgumentException();
        });
        test.accept(i -> {
            if (i == 1) throw new IllegalStateException();
        });
        test.accept(i -> {
            if (i == 1) throw new IllegalArgumentException();
        });
        test.accept(i -> {
            if (i == 1) throw new IllegalStateException();
            if (i == 2) throw new IllegalArgumentException();
        });
        test.accept(i -> {
            if (i == 1) throw new IllegalStateException();
            if (i == 2) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 3) throw new IllegalArgumentException();
        });
        test.accept(i -> {
            if (i == 3) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 3) throw new IllegalArgumentException();
            if (i == 4) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 5) throw new RuntimeException();
        });
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestTry.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

    static void consume(Throwable e) {
    }

    static Consumer<IntConsumer> testConsumer(Consumer<IntConsumer> actualR, Consumer<IntConsumer> expectedR) {
        return c -> {
            List<Integer> actual = new ArrayList<>();
            IntConsumer actualC = actual::add;
            Throwable actualT = null;
            try {
                actualR.accept(actualC.andThen(c));
            } catch (Interpreter.InterpreterException e) {
                throw e;
            } catch (Throwable t) {
                actualT = t;
                if (t instanceof AssertionError) {
                    t.printStackTrace();
                }
            }

            List<Integer> expected = new ArrayList<>();
            IntConsumer expectedC = expected::add;
            Throwable expectedT = null;
            try {
                expectedR.accept(expectedC.andThen(c));
            } catch (Throwable t) {
                expectedT = t;
            }

            Assert.assertEquals(
                    actualT != null ? actualT.getClass() : null,
                    expectedT != null ? expectedT.getClass() : null);
            Assert.assertEquals(actual, expected);
        };
    }
}
