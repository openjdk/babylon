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

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @library ../
 * @run junit/othervm -Djdk.invoke.MethodHandle.dumpClassFiles=true TestStatementTarget
 * @run main Unreflect TestStatementTarget
 * @run junit/othervm -Djdk.invoke.MethodHandle.dumpClassFiles=true TestStatementTarget
 */
public final class TestStatementTarget {

    public enum Ev {
        ENTER, CONTINUE, BREAK, RETURN, THROW, EXIT, METHOD_EXIT,
        L0_TRY_ENTER, L0_FIN_ENTER, L0_LOOP_ENTER, L0_FIN_EXIT, L0_CATCH_ENTER, L0_SYNC_ENTER,
        L1_TRY_ENTER, L1_FIN_ENTER, L1_LOOP_ENTER, L1_FIN_EXIT, L1_CATCH_ENTER, L1_TRY_THROW,
        L1_TWR_BODY_ENTER, L1_TWR_R0_OPEN, L1_TWR_R0_CLOSE, L1_TWR_R1_OPEN, L1_TWR_R1_CLOSE,
        L2_TRY_ENTER, L2_FIN_ENTER, L2_LOOP_ENTER, L2_FIN_EXIT, L2_SYNC_ENTER,
        L2_TWR_BODY_ENTER, L2_TWR_R0_OPEN, L2_TWR_R0_CLOSE, L2_TWR_R1_OPEN, L2_TWR_R1_CLOSE, L2_TWR_R2_OPEN,
        L2_TWR_R2_CLOSE, L2_TWR_R3_OPEN, L2_TWR_R3_CLOSE,
        L3_TRY_ENTER, L3_FIN_ENTER, L3_LOOP_ENTER, L3_FIN_EXIT,
        L3_TWR_BODY_ENTER, L3_TWR_R0_OPEN, L3_TWR_R0_CLOSE, L3_TWR_R1_OPEN, L3_TWR_R1_CLOSE,
        L4_TRY_ENTER, L4_FIN_ENTER, L4_LOOP_ENTER, L4_FIN_EXIT,
        L4_TWR_BODY_ENTER, L4_TWR_R0_OPEN, L4_TWR_R0_CLOSE,
        L5_TWR_BODY_ENTER, L5_TWR_R0_OPEN, L5_TWR_R0_CLOSE
    }

    public static final class Resource implements AutoCloseable {
        private final List<Ev> log;
        private final boolean closeFailure;
        private final Ev closeEvent;

        Resource(List<Ev> log, boolean closeFailure, Ev closeEvent) {
            this.log = log;
            this.closeFailure = closeFailure;
            this.closeEvent = closeEvent;
        }

        @Override
        public void close() {
            log.add(closeEvent);
            if (closeFailure) {
                throw new IllegalStateException("close-" + closeEvent);
            }
        }
    }

    public static AutoCloseable open(List<Ev> log, boolean closeFailure, boolean acquisitionFailure, Ev openEvent, Ev closeEvent) {
        log.add(openEvent);
        if (acquisitionFailure) {
            throw new IllegalArgumentException("open-" + openEvent);
        }
        return new Resource(log, closeFailure, closeEvent);
    }

    @Reflect
    public static void finallyContinueExternal(int mode, List<Ev> log) throws Exception {
        outer: for (int i_outer = 0; i_outer < 2; i_outer++) {
            log.add(Ev.L0_LOOP_ENTER);
            try {
                log.add(Ev.L1_TRY_ENTER);
                if (mode == 2) throw new IllegalArgumentException("f0");
                if (mode == 3) return;
            } finally {
                log.add(Ev.L1_FIN_ENTER);
                log.add(Ev.ENTER);
                if (mode == 1) {
                    if (i_outer == 0) {
                        log.add(Ev.CONTINUE);
                        continue outer;
                    }
                }
                if (mode == 4) {
                    log.add(Ev.THROW);
                    throw new IllegalStateException("body");
                }
                log.add(Ev.EXIT);
                log.add(Ev.L1_FIN_EXIT);
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void catchContinueExternal(int mode, List<Ev> log) throws Exception {
        outer: for (int i_outer = 0; i_outer < 2; i_outer++) {
            log.add(Ev.L0_LOOP_ENTER);
            try {
                log.add(Ev.L1_TRY_THROW);
                throw new IllegalStateException("enter-catch");
            } catch (IllegalStateException ex) {
                log.add(Ev.L1_CATCH_ENTER);
                log.add(Ev.ENTER);
                if (mode == 1) {
                    if (i_outer == 0) {
                        log.add(Ev.CONTINUE);
                        continue outer;
                    }
                }
                if (mode == 2) {
                    log.add(Ev.THROW);
                    throw new IllegalStateException("body");
                }
                log.add(Ev.EXIT);
            } finally {
                log.add(Ev.L1_FIN_ENTER);
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void twrContinueOuterTry(int mode, List<Ev> log) throws Exception {
        try {
            log.add(Ev.L0_TRY_ENTER);
            loop: for (int i_loop = 0; i_loop < 2; i_loop++) {
                log.add(Ev.L1_LOOP_ENTER);
                try (var _ = open(log, mode == 2 || mode == 3, false, Ev.L2_TWR_R0_OPEN, Ev.L2_TWR_R0_CLOSE)) {
                    log.add(Ev.L2_TWR_BODY_ENTER);
                    log.add(Ev.ENTER);
                    if (mode == 1) {
                        if (i_loop == 0) {
                            log.add(Ev.CONTINUE);
                            continue loop;
                        }
                    }
                    if (mode == 3) {
                        log.add(Ev.THROW);
                        throw new IllegalStateException("body");
                    }
                    log.add(Ev.EXIT);
                }
            }
        } finally {
            log.add(Ev.L0_FIN_ENTER);
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void nestedTwrContinueWithinOuterTwr(int mode, List<Ev> log) throws Exception {
        try (var _ = open(log, mode == 2 || mode == 3, false, Ev.L1_TWR_R0_OPEN, Ev.L1_TWR_R0_CLOSE);
             var _ = open(log, false, mode == 4, Ev.L1_TWR_R1_OPEN, Ev.L1_TWR_R1_CLOSE)) {
            log.add(Ev.L1_TWR_BODY_ENTER);
            loop: for (int i_loop = 0; i_loop < 2; i_loop++) {
                log.add(Ev.L2_LOOP_ENTER);
                try (var _ = open(log, mode == 5 || mode == 6, false, Ev.L3_TWR_R0_OPEN, Ev.L3_TWR_R0_CLOSE)) {
                    log.add(Ev.L3_TWR_BODY_ENTER);
                    log.add(Ev.ENTER);
                    if (mode == 1 && i_loop == 0) {
                        log.add(Ev.CONTINUE);
                        continue loop;
                    }
                    if (mode == 6) {
                        log.add(Ev.THROW);
                        throw new IllegalStateException("inner TWR body");
                    }
                    log.add(Ev.EXIT);
                }
            }
            if (mode == 3) {
                log.add(Ev.THROW);
                throw new IllegalStateException("outer TWR body");
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void nestedFinallyTwrBreakExternal(int mode, List<Ev> log) throws Exception {
        outer: for (int i_outer = 0; i_outer < 2; i_outer++) {
            log.add(Ev.L0_LOOP_ENTER);
            try {
                log.add(Ev.L1_TRY_ENTER);
                if (mode == 3) throw new IllegalArgumentException("f0");
                if (mode == 4) return;
            } finally {
                log.add(Ev.L1_FIN_ENTER);
                try {
                    log.add(Ev.L2_TRY_ENTER);
                    if (mode == 3) throw new IllegalArgumentException("f1");
                    if (mode == 4) return;
                } finally {
                    log.add(Ev.L2_FIN_ENTER);
                    try (var _ = open(log, mode == 2 || mode == 5, false, Ev.L3_TWR_R0_OPEN, Ev.L3_TWR_R0_CLOSE);
                         var _ = open(log, false, mode == 6, Ev.L3_TWR_R1_OPEN, Ev.L3_TWR_R1_CLOSE)) {
                        log.add(Ev.L3_TWR_BODY_ENTER);
                        log.add(Ev.ENTER);
                        if (mode == 1) {
                            if (i_outer == 0) {
                                log.add(Ev.BREAK);
                                break outer;
                            }
                        }
                        if (mode == 5) {
                            log.add(Ev.THROW);
                            throw new IllegalStateException("body");
                        }
                        log.add(Ev.EXIT);
                    }
                    log.add(Ev.L2_FIN_EXIT);
                }
                log.add(Ev.L1_FIN_EXIT);
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void catchTwrContinueExternal(int mode, List<Ev> log) throws Exception {
        outer: for (int i_outer = 0; i_outer < 2; i_outer++) {
            log.add(Ev.L0_LOOP_ENTER);
            try {
                log.add(Ev.L1_TRY_THROW);
                throw new IllegalStateException("enter-catch");
            } catch (IllegalStateException ex) {
                log.add(Ev.L1_CATCH_ENTER);
                try (var _ = open(log, mode == 2 || mode == 3, false, Ev.L2_TWR_R0_OPEN, Ev.L2_TWR_R0_CLOSE)) {
                    log.add(Ev.L2_TWR_BODY_ENTER);
                    log.add(Ev.ENTER);
                    if (mode == 1) {
                        if (i_outer == 0) {
                            log.add(Ev.CONTINUE);
                            continue outer;
                        }
                    }
                    if (mode == 3) {
                        log.add(Ev.THROW);
                        throw new IllegalStateException("body");
                    }
                    log.add(Ev.EXIT);
                }
            } finally {
                log.add(Ev.L1_FIN_ENTER);
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void tryTwrThrow(int mode, List<Ev> log) throws Exception {
        try {
            log.add(Ev.L0_TRY_ENTER);
            try (var _ = open(log, mode == 2 || mode == 3, false, Ev.L1_TWR_R0_OPEN, Ev.L1_TWR_R0_CLOSE);
                 var _ = open(log, false, mode == 4, Ev.L1_TWR_R1_OPEN, Ev.L1_TWR_R1_CLOSE)) {
                log.add(Ev.L1_TWR_BODY_ENTER);
                log.add(Ev.ENTER);
                if (mode == 1) {
                    log.add(Ev.THROW);
                    throw new IllegalStateException("body");
                }
                if (mode == 3) {
                    log.add(Ev.THROW);
                    throw new IllegalStateException("body");
                }
                log.add(Ev.EXIT);
            }
        } catch (IllegalStateException ex) {
            log.add(Ev.L0_CATCH_ENTER);
        } finally {
            log.add(Ev.L0_FIN_ENTER);
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void finallySynchronizedTwrContinueExternal(int mode, List<Ev> log) throws Exception {
        outer: for (int i_outer = 0; i_outer < 2; i_outer++) {
            log.add(Ev.L0_LOOP_ENTER);
            try {
                log.add(Ev.L1_TRY_ENTER);
                if (mode == 3) throw new IllegalArgumentException("f0");
                if (mode == 4) return;
            } finally {
                log.add(Ev.L1_FIN_ENTER);
                synchronized (TestStatementTarget.class) {
                    log.add(Ev.L2_SYNC_ENTER);
                    try (var _ = open(log, mode == 2 || mode == 5, false, Ev.L3_TWR_R0_OPEN, Ev.L3_TWR_R0_CLOSE)) {
                        log.add(Ev.L3_TWR_BODY_ENTER);
                        log.add(Ev.ENTER);
                        if (mode == 1) {
                            if (i_outer == 0) {
                                log.add(Ev.CONTINUE);
                                continue outer;
                            }
                        }
                        if (mode == 5) {
                            log.add(Ev.THROW);
                            throw new IllegalStateException("body");
                        }
                        log.add(Ev.EXIT);
                    }
                }
                log.add(Ev.L1_FIN_EXIT);
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    @Reflect
    public static void deepThroughTwr(int mode, List<Ev> log) throws Exception {
        outer: for (int i_outer = 0; i_outer < 2; i_outer++) {
            log.add(Ev.L0_LOOP_ENTER);
            try {
                log.add(Ev.L1_TRY_ENTER);
                try {
                    log.add(Ev.L2_TRY_ENTER);
                    try {
                        log.add(Ev.L3_TRY_ENTER);
                        inner: for (int i_inner = 0; i_inner < 2; i_inner++) {
                            log.add(Ev.L4_LOOP_ENTER);
                            try (var _ = open(log, mode == 5 || mode == 6, false, Ev.L5_TWR_R0_OPEN, Ev.L5_TWR_R0_CLOSE)) {
                                log.add(Ev.L5_TWR_BODY_ENTER);
                                log.add(Ev.ENTER);
                                if (i_outer == 0) {
                                    if (i_inner == 0) {
                                        if (mode == 1) {
                                            log.add(Ev.CONTINUE);
                                            continue outer;
                                        }
                                        if (mode == 2) {
                                            log.add(Ev.BREAK);
                                            break outer;
                                        }
                                        if (mode == 3) {
                                            log.add(Ev.RETURN);
                                            return;
                                        }
                                        if (mode == 4) {
                                            log.add(Ev.THROW);
                                            throw new IllegalStateException("body");
                                        }
                                    }
                                }
                                if (mode == 6) {
                                    log.add(Ev.THROW);
                                    throw new IllegalStateException("body");
                                }
                                log.add(Ev.EXIT);
                            }
                        }
                    } finally {
                        log.add(Ev.L3_FIN_ENTER);
                    }
                } finally {
                    log.add(Ev.L2_FIN_ENTER);
                }
            } finally {
                log.add(Ev.L1_FIN_ENTER);
            }
        }
        log.add(Ev.METHOD_EXIT);
    }

    static Stream<Method> reflectMethods() {
        return Stream.of(TestStatementTarget.class.getDeclaredMethods())
                .filter(method -> method.isAnnotationPresent(Reflect.class))
                .sorted((a, b) -> a.getName().compareTo(b.getName()));
    }

    @ParameterizedTest
    @MethodSource("reflectMethods")
    public void testGeneratedBytecode(Method method) {
        CoreOp.FuncOp model = Op.ofMethod(method).orElseThrow();
        MethodHandle generated = BytecodeGenerator.generate(MethodHandles.lookup(), model);
        for (int mode = 0; mode < 7; mode++) {
            var expectedTrace = new ArrayList<Ev>();
            var actualTrace = new ArrayList<Ev>();
            try {
                method.invoke(null, mode, expectedTrace);
                generated.invoke(mode, actualTrace);
            } catch (Throwable expectedThrowable) {
                int fmode = mode;
                var actualThrowable = assertThrows(Throwable.class, () -> generated.invoke(fmode, actualTrace));
                if (expectedThrowable instanceof InvocationTargetException) {
                    expectedThrowable = expectedThrowable.getCause();
                }
                while (expectedThrowable != null) {
                    assertEquals(expectedThrowable.toString(), actualThrowable.toString());
                    expectedThrowable = expectedThrowable.getCause();
                    actualThrowable = actualThrowable.getCause();
                }
            }
            assertEquals(expectedTrace, actualTrace);
        }
    }
}
