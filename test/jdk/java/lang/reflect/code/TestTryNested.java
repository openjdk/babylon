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
 * @run testng TestTryNested
 */

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.CodeReflection;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.Stream;

public class TestTryNested {
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
        CoreOp.FuncOp f = getFuncOp("tryCatchFinally");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        for (int ra = -1; ra < 6; ra++) {
            int fra = ra;

            Consumer<IntConsumer> test = testConsumer(
                    c -> Interpreter.invoke(MethodHandles.lookup(), lf, c, fra),
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
    public static void tryCatchBreak(IntConsumer c, int i) {
        a: try {
            try {
                if (i == 0) {
                    break a;
                }
                c.accept(0);
            } catch (IllegalStateException e) {
                if (i == 1) {
                    break a;
                }
                c.accept(1);
            }
            if (i == 2) {
                break a;
            }
            c.accept(2);
        } catch (IllegalStateException e) {
            if (i == 3) {
                break a;
            }
            c.accept(3);
        }
        c.accept(4);
    }

    @Test
    public void testCatchBreak() {
        CoreOp.FuncOp f = getFuncOp("tryCatchBreak");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        for (int ra = -1; ra < 4; ra++) {
            int fra = ra;

            Consumer<IntConsumer> test = testConsumer(
                    c -> Interpreter.invoke(MethodHandles.lookup(), lf, c, fra),
                    c -> tryCatchBreak(c, fra)
            );

            test.accept(i -> {});
            for (int ea = 0; ea <= 4; ea++) {
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
    public static void tryCatchFinallyBreak(IntConsumer c, int i) {
        a: try {
            try {
                if (i == 0) {
                    break a;
                }
                c.accept(0);
            } catch (IllegalStateException e) {
                if (i == 1) {
                    break a;
                }
                c.accept(1);
            } finally {
                if (i == 2) {
                    break a;
                }
                c.accept(2);
            }
            if (i == 3) {
                break a;
            }
            c.accept(3);
        } catch (IllegalStateException e) {
            if (i == 4) {
                break a;
            }
            c.accept(4);
        } finally {
            if (i == 5) {
                break a;
            }
            c.accept(5);
        }
        c.accept(6);
    }

    @Test
    public void testCatchFinallyBreak() {
        CoreOp.FuncOp f = getFuncOp("tryCatchFinallyBreak");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        for (int ra = -1; ra < 6; ra++) {
            int fra = ra;

            Consumer<IntConsumer> test = testConsumer(
                    c -> Interpreter.invoke(MethodHandles.lookup(), lf, c, fra),
                    c -> tryCatchFinallyBreak(c, fra)
            );

            test.accept(i -> {});
            for (int ea = 0; ea <= 6; ea++) {
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
            } catch (IllegalStateException e) {
                c.accept(2);
            }
            c.accept(3);
        }
        c.accept(4);
    }

    @Test
    public void testTryForLoop() {
        CoreOp.FuncOp f = getFuncOp("tryForLoop");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lf, c),
                TestTryNested::tryForLoop
        );

        test.accept(i -> { });
        for (int ea = 0; ea <= 4; ea++) {
            int fea = ea;
            test.accept(i -> {
                if (i == fea) throw new IllegalStateException();
            });
            test.accept(i -> {
                if (i == fea) throw new RuntimeException();
            });
        }
    }

    @CodeReflection
    public static void tryForLoopFinally(IntConsumer c) {
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
    public void testTryForLoopFinally() {
        CoreOp.FuncOp f = getFuncOp("tryForLoopFinally");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lf, c),
                TestTryNested::tryForLoopFinally
        );

        test.accept(i -> { });
        for (int ea = 0; ea <= 4; ea++) {
            int fea = ea;
            test.accept(i -> {
                if (i == fea) throw new IllegalStateException();
            });
            test.accept(i -> {
                if (i == fea) throw new RuntimeException();
            });
        }
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
        CoreOp.FuncOp f = getFuncOp("tryLabeledForLoop");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lf, c),
                TestTryNested::tryLabeledForLoop
        );

        test.accept(i -> { });
    }


    @CodeReflection
    public static void tryLambda(IntConsumer c, int i) {
        try {
            c.accept(0);
            Runnable r = () -> {
                if (i == 0) {
                    c.accept(1);
                    return;
                } else {
                    c.accept(2);
                }
                c.accept(3);
            };
            r.run();
            c.accept(4);
        } finally {
            c.accept(5);
        }
    }

    @Test
    public void testTryLambda() {
        CoreOp.FuncOp f = getFuncOp("tryLambda");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        for (int ra = 0; ra < 2; ra++) {
            final int fra = ra;
            Consumer<IntConsumer> test = testConsumer(
                    c -> Interpreter.invoke(MethodHandles.lookup(), lf, c, fra),
                    c -> tryLambda(c, fra)
            );
            test.accept(i -> { });
        }
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestTryNested.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
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
