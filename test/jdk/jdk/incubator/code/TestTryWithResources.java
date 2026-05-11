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

    record Resource(String suffix, boolean throwOnClose) implements Closeable {

        Resource {
            log.append("open").append(suffix).append(';');
        }

        @Override
        public void close() throws IOException {
            log.append("close").append(suffix).append(';');
            if (throwOnClose) {
                log.append("throwClose").append(suffix).append(';');
                throw new IOException();
            }
        }
    }

    @Reflect
    public static void tryWithResources(boolean throwInBody, boolean throwOnClose1, boolean throwOnClose2, boolean throwOnClose3) throws IOException {
        log = new StringBuilder();
        try {
            try (var _ = new Resource("1", throwOnClose1)) {
                log.append("outerBody;");
                try (var _ = new Resource("2", throwOnClose2);
                     var _ = new Resource("3", throwOnClose3)) {
                    log.append("innerBody;");
                    if (throwInBody) {
                        log.append("throwBody;");
                        throw new IllegalStateException("body");
                    }
                } finally {
                    log.append("innerFinally;");
                }
            } finally {
                log.append("outerFinally;");
            }
        } finally {
            log.append("end;");
        }
    }

    @Test
    public void testTryWithResources() throws Throwable {
        Method m = TestTryWithResources.class.getDeclaredMethod("tryWithResources", boolean.class, boolean.class, boolean.class, boolean.class);
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), Op.ofMethod(m).orElseThrow());

        mh.invoke(false, false, false, false);
        assertEquals("open1;outerBody;open2;open3;innerBody;close3;close2;innerFinally;close1;outerFinally;end;", log.toString());

        assertEquals(0, assertThrows(IOException.class, () -> mh.invoke(false, false, true, false)).getSuppressed().length);
        assertEquals("open1;outerBody;open2;open3;innerBody;close3;close2;throwClose2;innerFinally;close1;outerFinally;end;", log.toString());

        assertEquals(1, assertThrows(IOException.class, () -> mh.invoke(false, false, true, true)).getSuppressed().length);
        assertEquals("open1;outerBody;open2;open3;innerBody;close3;throwClose3;close2;throwClose2;innerFinally;close1;outerFinally;end;", log.toString());

        assertEquals(3, assertThrows(IllegalStateException.class, () -> mh.invoke(true, true, true, true)).getSuppressed().length);
        assertEquals("open1;outerBody;open2;open3;innerBody;throwBody;close3;throwClose3;close2;throwClose2;innerFinally;close1;throwClose1;outerFinally;end;", log.toString());
    }
}
