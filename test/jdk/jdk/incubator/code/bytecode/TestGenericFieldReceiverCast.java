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
 * @library ../
 * @modules jdk.incubator.code
 * @run junit TestGenericFieldReceiverCast
 * @run main Unreflect TestGenericFieldReceiverCast
 * @run junit TestGenericFieldReceiverCast
 */


import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestGenericFieldReceiverCast {

    static class Base {
    }

    static class Child extends Base {
        int value = 42;
    }

    static class TestBase<B extends Base> {
        final B b;

        TestBase(B b) {
            this.b = b;
        }
    }

    static final class TestClass extends TestBase<Child> {
        TestClass() {
            super(new Child());
        }

        @Reflect
        int value() {
            return b.value;
        }
    }

    @Test
    public void test() throws Throwable {
        Assertions.assertEquals(42, new TestClass().value());
    }
}
