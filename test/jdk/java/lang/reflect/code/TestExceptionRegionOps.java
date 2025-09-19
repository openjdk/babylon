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
 * @run junit TestExceptionRegionOps
 */

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

public class TestExceptionRegionOps {

    public void testF(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
        } catch (IllegalStateException e) {
            c.accept(1);
            c.accept(-1);
        } catch (IllegalArgumentException e) {
            c.accept(2);
            c.accept(-1);
        }
        c.accept(3);
        c.accept(-1);
    }

    @Test
    public void test() {
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, type(IntConsumer.class)))
                .body(fbody -> {
                    var fblock = fbody.entryBlock();
                    var catchER1ISE = fblock.block(type(IllegalStateException.class));
                    var catchER1IAE = fblock.block(type(IllegalArgumentException.class));
                    var enterER1 = fblock.block();
                    var end = fblock.block();

                    //
                    var c = fblock.parameters().get(0);
                    fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchER1IAE.successor(), catchER1ISE.successor()));

                    // Start of exception region
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, 0))));
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, -1))));
                    // End of exception region
                    enterER1.op(exceptionRegionExit(end.successor(),
                        catchER1ISE.successor(), catchER1IAE.successor()));

                    // First catch block for exception region
                    catchER1ISE.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1ISE.op(constant(INT, 1))));
                    catchER1ISE.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1ISE.op(constant(INT, -1))));
                    catchER1ISE.op(branch(end.successor()));

                    // Second catch for exception region
                    catchER1IAE.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1IAE.op(constant(INT, 2))));
                    catchER1IAE.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1IAE.op(constant(INT, -1))));
                    catchER1IAE.op(branch(end.successor()));

                    //
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, 3))));
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, -1))));
                    end.op(CoreOp.return_());
                });

        System.out.println(f.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), f, c),
                this::testF);

        test.accept(i -> {});
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


    public void testCatchThrowableF(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
        } catch (IllegalStateException e) {
            c.accept(1);
            c.accept(-1);
        } catch (Throwable e) {
            c.accept(2);
            c.accept(-1);
        }
        c.accept(3);
        c.accept(-1);
    }

    @Test
    public void testCatchThrowable() {
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, type(IntConsumer.class)))
                .body(fbody -> {
                    var fblock = fbody.entryBlock();
                    var catchER1ISE = fblock.block(type(IllegalStateException.class));
                    var catchER1T = fblock.block(type(Throwable.class));
                    var enterER1 = fblock.block();
                    var end = fblock.block();

                    //
                    var c = fblock.parameters().get(0);
                    fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchER1T.successor(), catchER1ISE.successor()));

                    // Start of exception region
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, 0))));
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, -1))));
                    // End of exception region
                    enterER1.op(exceptionRegionExit(end.successor(),
                        catchER1ISE.successor(), catchER1T.successor()));

                    // First catch block for exception region
                    catchER1ISE.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1ISE.op(constant(INT, 1))));
                    catchER1ISE.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1ISE.op(constant(INT, -1))));
                    catchER1ISE.op(branch(end.successor()));

                    // Second catch for exception region
                    catchER1T.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1T.op(constant(INT, 2))));
                    catchER1T.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1T.op(constant(INT, -1))));
                    catchER1T.op(branch(end.successor()));

                    //
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, 3))));
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, -1))));
                    end.op(return_());
                });

        System.out.println(f.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), f, c),
                this::testCatchThrowableF);

        test.accept(i -> {});
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


    public void testNestedF(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
            try {
                c.accept(1);
                c.accept(-1);
            } catch (IllegalStateException e) {
                c.accept(2);
                c.accept(-1);
            }
            c.accept(3);
            c.accept(-1);
        } catch (IllegalArgumentException e) {
            c.accept(4);
            c.accept(-1);
        }
        c.accept(5);
        c.accept(-1);
    }

    @Test
    public void testNested() {
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, type(IntConsumer.class)))
                .body(fbody -> {
                    var fblock = fbody.entryBlock();
                    var catchER1 = fblock.block(type(IllegalArgumentException.class));
                    var catchER2 = fblock.block(type(IllegalStateException.class));
                    var enterER1 = fblock.block();
                    var enterER2 = fblock.block();
                    var b3 = fblock.block();
                    var end = fblock.block();

                    //
                    var c = fblock.parameters().get(0);
                    fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchER1.successor()));

                    // Start of first exception region
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, 0))));
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, -1))));
                    enterER1.op(exceptionRegionEnter(
                            enterER2.successor(),
                            catchER2.successor()));

                    // Start of second exception region
                    enterER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER2.op(constant(INT, 1))));
                    enterER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER2.op(constant(INT, -1))));
                    // End of second exception region
                    enterER2.op(exceptionRegionExit(b3.successor(),
                        catchER2.successor()));

                    // Catch block for second exception region
                    catchER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER2.op(constant(INT, 2))));
                    catchER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER2.op(constant(INT, -1))));
                    catchER2.op(branch(b3.successor()));

                    b3.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b3.op(constant(INT, 3))));
                    b3.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b3.op(constant(INT, -1))));
                    // End of first exception region
                    b3.op(exceptionRegionExit(end.successor(),
                        catchER1.successor()));

                    // Catch block for first exception region
                    catchER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1.op(constant(INT, 4))));
                    catchER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchER1.op(constant(INT, -1))));
                    catchER1.op(branch(end.successor()));

                    //
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, 5))));
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, -1))));
                    end.op(CoreOp.return_());
                });

        System.out.println(f.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), f, c),
                this::testNestedF);

        test.accept(i -> {});
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

    public void testCatchFinallyF(IntConsumer c) {
        try {
            c.accept(0);
            c.accept(-1);
        } catch (IllegalStateException e) {
            c.accept(1);
            c.accept(-1);
        } finally {
            c.accept(2);
            c.accept(-1);
        }
        c.accept(3);
        c.accept(-1);
    }

    @Test
    public void testCatchFinally() {
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, JavaType.type(IntConsumer.class)))
                .body(fbody -> {
                    var fblock = fbody.entryBlock();
                    var catchRE = fblock.block(type(IllegalStateException.class));
                    var catchAll = fblock.block(type(Throwable.class));
                    var enterER1 = fblock.block();
                    var exitER1 = fblock.block();
                    var enterER2 = fblock.block();
                    var exitER2 = fblock.block();
                    var end = fblock.block();

                    //
                    var c = fblock.parameters().get(0);
                    fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchAll.successor(), catchRE.successor()));

                    // Start of exception region
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, 0))));
                    enterER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER1.op(constant(INT, -1))));
                    // End of exception region
                    enterER1.op(exceptionRegionExit(exitER1.successor(),
                        catchRE.successor(), catchAll.successor()));
                    // Inline finally
                    exitER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, exitER1.op(constant(INT, 2))));
                    exitER1.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, exitER1.op(constant(INT, -1))));
                    exitER1.op(branch(end.successor()));

                    // Catch block for RuntimeException
                    catchRE.op(exceptionRegionEnter(
                            enterER2.successor(),
                            catchAll.successor()));
                    // Start of exception region
                    enterER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER2.op(constant(INT, 1))));
                    enterER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, enterER2.op(constant(INT, -1))));
                    // End of exception region
                    enterER2.op(exceptionRegionExit(exitER2.successor(),
                        catchAll.successor()));
                    // Inline finally
                    exitER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, exitER2.op(constant(INT, 2))));
                    exitER2.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, exitER2.op(constant(INT, -1))));
                    exitER2.op(branch(end.successor()));

                    // Catch all block for finally
                    catchAll.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchAll.op(constant(INT, 2))));
                    catchAll.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, catchAll.op(constant(INT, -1))));
                    catchAll.op(throw_(catchAll.parameters().get(0)));

                    //
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, 3))));
                    end.op(JavaOp.invoke(INT_CONSUMER_ACCEPT_METHOD, c, end.op(constant(INT, -1))));
                    end.op(CoreOp.return_());
                });

        System.out.println(f.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), f, c),
                this::testCatchFinallyF
                );

        test.accept(i -> {});
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
        });
        test.accept(i -> {
            if (i == 0) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 2) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 0) throw new IllegalStateException();
            if (i == 1) throw new RuntimeException();
        });
        test.accept(i -> {
            if (i == 3) throw new RuntimeException();
        });
    }

    static final MethodRef INT_CONSUMER_ACCEPT_METHOD = MethodRef.method(type(IntConsumer.class), "accept",
            VOID, INT);

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
            }

            List<Integer> expected = new ArrayList<>();
            IntConsumer expectedC = expected::add;
            Throwable expectedT = null;
            try {
                expectedR.accept(expectedC.andThen(c));
            } catch (Throwable t) {
                expectedT = t;
            }

            Assertions.assertEquals(
                    expectedT != null ? expectedT.getClass() : null, actualT != null ? actualT.getClass() : null
            );
            Assertions.assertEquals(expected, actual);
        };
    }
}
