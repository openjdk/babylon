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
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestTryWithResources {
    static StringBuilder log;

    record Resource(boolean throwOnClose) implements Closeable {

        Resource {
            log.append("open;");
        }

        @Override
        public void close() throws IOException {
            log.append("close;");
            if (throwOnClose) {
                throw new IOException();
            }
        }
    }

    @Reflect
    public static void tryWithResources(boolean throwInBody, boolean throwOnClose) throws IOException {
        log = new StringBuilder();
        try {
            try (var r = new Resource(throwOnClose)) {
                log.append("body;");
                if (throwInBody) {
                    throw new IllegalStateException("body");
                }
            } finally {
                log.append("innerFinally;");
            }
        } finally {
            log.append("outerFinally;");
        }
    }

    @Test
    public void testTryWithResources() throws Throwable {
        Method m = TestTryWithResources.class.getDeclaredMethod("tryWithResources", boolean.class, boolean.class);
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), Op.ofMethod(m).orElseThrow());

        assertNull(mh.invoke(false, false));
        assertEquals("open;body;close;innerFinally;outerFinally;", log.toString());
        assertThrows(IllegalStateException.class, () -> mh.invoke(true,  false));
        assertEquals("open;body;close;innerFinally;outerFinally;", log.toString());
        assertThrows(IOException.class, () -> mh.invoke(false, true));
        assertEquals("open;body;close;innerFinally;outerFinally;", log.toString());
        assertThrows(IllegalStateException.class, () -> mh.invoke(true,  true));
        assertEquals("open;body;close;innerFinally;outerFinally;", log.toString());
    }
}
