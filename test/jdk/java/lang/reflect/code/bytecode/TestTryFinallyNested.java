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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.bytecode.BytecodeLower;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestTryFinallyNested
 */

public class TestTryFinallyNested {
    @CodeReflection
    public static void tryCatchFinally(IntConsumer c, int i) {
        try {
            try {
                if (i == 0) {
                    return;
                }
                c.accept(0);
            } catch (IllegalStateException e) {
                if (i == 1) {
                    return;
                }
                c.accept(1);
            } finally {
                if (i == 2) {
                    return;
                }
                c.accept(2);
            }
            if (i == 3) {
                return;
            }
            c.accept(3);
        } catch (IllegalStateException e) {
            if (i == 4) {
                return;
            }
            c.accept(4);
        } finally {
            if (i == 5) {
                return;
            }
            c.accept(5);
        }
        c.accept(6);
    }

    @Test
    public void testCatchFinally() {
        CoreOps.FuncOp f = getFuncOp("tryCatchFinally");

        MethodHandle mh = generate(f);

        for (int ra = -1; ra < 6; ra++) {
            int fra = ra;

            Consumer<IntConsumer> test = testConsumer(
                    c -> invoke(mh, c, fra),
                    c -> tryCatchFinally(c, fra)
            );

            test.accept(i -> {});
            for (int ea = 0; ea < 6; ea++) {
                int fea = ea;
                test.accept(i -> {
                    if (i == fea) throw new IllegalStateException();
                });
                test.accept(i -> {
                    if (i == fea) throw new RuntimeException();
                });
            }
        }
    }

    @CodeReflection
    public static void tryForLoop(IntConsumer c) {
        for (int i = 0; i < 8; i++) {
            c.accept(0);
            try {
                if (i == 4) {
                    continue;
                } else if (i == 5) {
                    break;
                }
                c.accept(1);
            } finally {
                c.accept(2);
            }
            c.accept(3);
        }
        c.accept(4);
    }

    @Test
    public void testTryForLoop() {
        CoreOps.FuncOp f = getFuncOp("tryForLoop");

        MethodHandle mh = generate(f);

        Consumer<IntConsumer> test = testConsumer(
                asConsumer(mh),
                TestTryFinallyNested::tryForLoop
        );

        test.accept(i -> { });
    }


    @CodeReflection
    public static void tryLabeledForLoop(IntConsumer c) {
        a: for (int i = 0; i < 8; i++) {
            c.accept(0);
            b: {
                try {
                    if (i == 4) {
                        continue a;
                    } else if (i == 5) {
                        break b;
                    } else if (i == 6) {
                        break a;
                    }
                    c.accept(1);
                } finally {
                    c.accept(2);
                }
                c.accept(3);
            }
            c.accept(4);
        }
        c.accept(5);
    }

    @Test
    public void testTryLabeledForLoop() {
        CoreOps.FuncOp f = getFuncOp("tryLabeledForLoop");

        MethodHandle mh = generate(f);

        Consumer<IntConsumer> test = testConsumer(
                asConsumer(mh),
                TestTryFinallyNested::tryLabeledForLoop
        );

        test.accept(i -> { });
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

    static MethodHandle generate(CoreOps.FuncOp f) {
        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        lf.writeTo(System.out);

        lf = SSA.transform(lf);
        lf.writeTo(System.out);

        CoreOps.FuncOp bcf = BytecodeLower.lowerToBytecodeDialect(MethodHandles.lookup(), lf);
        bcf.writeTo(System.out);

        return BytecodeGenerator.generate(MethodHandles.lookup(), bcf);
    }

    static <T> Consumer<T> asConsumer(MethodHandle mh) {
        return c -> {
            try {
                mh.invoke(c);
            } catch (Throwable e) {
                throw erase(e);
            }
        };
    }

    static void invoke(MethodHandle mh, IntConsumer c, int i) {
        try {
            mh.invoke(c, i);
        } catch (Throwable e) {
            throw erase(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <E extends Throwable> E erase(Throwable e) throws E {
        return (E) e;
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestTryFinallyNested.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
