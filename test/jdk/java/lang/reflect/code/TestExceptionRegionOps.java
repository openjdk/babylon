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
 * @run testng TestExceptionRegionOps
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

import static java.lang.reflect.code.op.CoreOps._return;
import static java.lang.reflect.code.op.CoreOps._throw;
import static java.lang.reflect.code.op.CoreOps.branch;
import static java.lang.reflect.code.op.CoreOps.constant;
import static java.lang.reflect.code.op.CoreOps.exceptionRegionEnter;
import static java.lang.reflect.code.op.CoreOps.exceptionRegionExit;
import static java.lang.reflect.code.op.CoreOps.func;
import static java.lang.reflect.code.type.FunctionType.*;
import static java.lang.reflect.code.type.JavaType.*;
import static java.lang.reflect.code.type.JavaType.VOID;

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
        CoreOps.FuncOp f = func("f", functionType(VOID, type(IntConsumer.class)))
                .body(fbody -> {
                    var fblock = fbody.entryBlock();
                    var catchER1ISE = fblock.block(type(IllegalStateException.class));
                    var catchER1IAE = fblock.block(type(IllegalArgumentException.class));
                    var enterER1 = fblock.block();
                    var end = fblock.block();

                    //
                    var c = fblock.parameters().get(0);
                    var er1 = fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchER1ISE.successor(), catchER1IAE.successor()));

                    // Start of exception region
                    enterER1.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 0))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        // End of exception region
                        b.op(exceptionRegionExit(er1, end.successor()));
                    });

                    // First catch block for exception region
                    catchER1ISE.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 1))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    // Second catch for exception region
                    catchER1IAE.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 2))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    //
                    end.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 3))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(_return());
                    });
                });

        f.writeTo(System.out);

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
        CoreOps.FuncOp f = func("f", functionType(VOID, type(IntConsumer.class)))
                .body(fbody -> {
                    var fblock = fbody.entryBlock();
                    var catchER1ISE = fblock.block(type(IllegalStateException.class));
                    var catchER1T = fblock.block(type(Throwable.class));
                    var enterER1 = fblock.block();
                    var end = fblock.block();

                    //
                    var c = fblock.parameters().get(0);
                    var er1 = fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchER1ISE.successor(), catchER1T.successor()));

                    // Start of exception region
                    enterER1.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 0))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        // End of exception region
                        b.op(exceptionRegionExit(er1, end.successor()));
                    });

                    // First catch block for exception region
                    catchER1ISE.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 1))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    // Second catch for exception region
                    catchER1T.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 2))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    //
                    end.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 3))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(_return());
                    });
                });

        f.writeTo(System.out);

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
        CoreOps.FuncOp f = func("f", functionType(VOID, type(IntConsumer.class)))
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
                    var er1 = fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchER1.successor()));

                    // Start of first exception region
                    enterER1.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 0))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                    });
                    var er2 = enterER1.op(exceptionRegionEnter(
                            enterER2.successor(),
                            catchER2.successor()));

                    // Start of second exception region
                    enterER2.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 1))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        // End of second exception region
                        b.op(exceptionRegionExit(er2, b3.successor()));
                    });

                    // Catch block for second exception region
                    catchER2.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 2))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(b3.successor()));
                    });

                    b3.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 3))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        // End of first exception region
                        b.op(exceptionRegionExit(er1, end.successor()));
                    });

                    // Catch block for first exception region
                    catchER1.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 4))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    //
                    end.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 5))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(_return());
                    });
                });

        f.writeTo(System.out);

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
        CoreOps.FuncOp f = func("f", functionType(VOID, JavaType.type(IntConsumer.class)))
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
                    var er1 = fblock.op(exceptionRegionEnter(
                            enterER1.successor(),
                            catchRE.successor(), catchAll.successor()));

                    // Start of exception region
                    enterER1.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 0))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        // End of exception region
                        b.op(exceptionRegionExit(er1, exitER1.successor()));
                    });
                    // Inline finally
                    exitER1.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 2))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    // Catch block for RuntimeException
                    var er2 = catchRE.op(exceptionRegionEnter(
                            enterER2.successor(),
                            catchAll.successor()));
                    // Start of exception region
                    enterER2.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 1))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        // End of exception region
                        b.op(exceptionRegionExit(er2, exitER2.successor()));
                    });
                    // Inline finally
                    exitER2.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 2))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(branch(end.successor()));
                    });

                    // Catch all block for finally
                    catchAll.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 2))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(_throw(catchAll.parameters().get(0)));
                    });

                    //
                    end.ops(b -> {
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, 3))));
                        b.op(CoreOps.invoke(INT_CONSUMER_ACCEPT_METHOD, c, b.op(constant(INT, -1))));
                        b.op(_return());
                    });
                });

        f.writeTo(System.out);

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

    static final MethodDesc INT_CONSUMER_ACCEPT_METHOD = MethodDesc.method(type(IntConsumer.class), "accept",
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

            Assert.assertEquals(
                    actualT != null ? actualT.getClass() : null,
                    expectedT != null ? expectedT.getClass() : null);
            Assert.assertEquals(actual, expected);
        };
    }
}
