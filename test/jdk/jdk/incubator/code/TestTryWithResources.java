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

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestTryWithResources
 * @run main Unreflect TestTryWithResources
 * @run junit TestTryWithResources
 */

import java.io.Closeable;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.function.Consumer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestTryWithResources {

    record Resource(Consumer<String> log, String suffix, boolean throwOnClose) implements Closeable {

        Resource {
            log.accept("open" + suffix);
        }

        @Override
        public void close() throws IOException {
            log.accept("close" + suffix);
            if (throwOnClose) {
                log.accept("throwClose" + suffix);
                throw new IOException();
            }
        }
    }

    @Reflect
    public static void tryWithResources(Consumer<String> log, boolean throwInBody, boolean throwOnClose1, boolean throwOnClose2, boolean throwOnClose3) throws IOException {
        var r2 = new Resource(log, "2", throwOnClose2);
        try {
            try (var _ = new Resource(log, "1", throwOnClose1)) {
                log.accept("outerBody");
                try (var _ = r2;
                     var _ = new Resource(log, "3", throwOnClose3)) {
                    log.accept("innerBody");
                    if (throwInBody) {
                        log.accept("throwBody");
                        throw new IllegalStateException("body");
                    }
                } finally {
                    log.accept("innerFinally");
                }
            } finally {
                log.accept("outerFinally");
            }
        } finally {
            log.accept("end");
        }
    }

    @Test
    public void testTryWithResources() throws Throwable {
        Method m = TestTryWithResources.class.getDeclaredMethod("tryWithResources", Consumer.class, boolean.class, boolean.class, boolean.class, boolean.class);
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), Op.ofMethod(m).orElseThrow());
        for (int i = 0; i < 16; i++) {
            boolean throwInBody = (i & 1) != 0;
            boolean throwOnClose1 = (i & 2) != 0;
            boolean throwOnClose2 = (i & 4) != 0;
            boolean throwOnClose3 = (i & 8) != 0;
            var expected = new ArrayList<String>();
            var actual = new ArrayList<String>();
            try {
                tryWithResources(expected::add, throwInBody, throwOnClose1, throwOnClose2, throwOnClose3);
                mh.invoke((Consumer<String>)actual::add, throwInBody, throwOnClose1, throwOnClose2, throwOnClose3);
            } catch (Throwable t) {
                assertEquals(t.getSuppressed().length,
                        assertThrowsExactly(t.getClass(), () -> mh.invoke((Consumer<String>)actual::add, throwInBody, throwOnClose1, throwOnClose2, throwOnClose3)).getSuppressed().length);
            }
            assertIterableEquals(expected, actual);
        }
    }
}
