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

import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestTryWithResources {

    static final class Resource implements AutoCloseable {
        final boolean throwOnClose;

        Resource(boolean throwOnClose) {
            this.throwOnClose = throwOnClose;
        }

        @Override
        public void close() throws IOException {
            if (throwOnClose) {
                throw new IOException();
            }
        }
    }

    @Reflect
    public static int tryWithResources(boolean throwInBody, boolean throwOnClose1, boolean throwOnClose2) throws Exception {
        try (var resource1 = new Resource(throwOnClose1);
             var resource2 = new Resource(throwOnClose2)) {
            if (throwInBody) {
                throw new IllegalStateException();
            }
            return 1;
        } catch (IOException ignored) {
            return 2;
        } catch (IllegalStateException ignored) {
            return 3;
        }
    }

    @Test
    public void testTryWithResources() throws Throwable {
        Optional<Method> om = Stream.of(TestTryWithResources.class.getDeclaredMethods()).filter(m -> m.getName().equals("tryWithResources")).findFirst();
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), Op.ofMethod(om.orElseThrow()).orElseThrow());
        Assertions.assertEquals(1, (int) mh.invoke(false, false, false));
        Assertions.assertEquals(2, (int) mh.invoke(false, false, true));
        Assertions.assertEquals(2, (int) mh.invoke(false, true, false));
        Assertions.assertEquals(3, (int) mh.invoke(true, false, false));
        Assertions.assertEquals(3, (int) mh.invoke(true, false, true));
        Assertions.assertEquals(3, (int) mh.invoke(true, true, false));
        Assertions.assertEquals(3, (int) mh.invoke(true, true, true));
    }
}
