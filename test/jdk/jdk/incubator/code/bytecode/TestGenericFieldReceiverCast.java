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

    static class Wrapper<T extends Number> {
        T n;

        Wrapper(T n) {
            this.n = n;
        }
    }

    static final class TestClass extends Wrapper<Integer> {
        TestClass() {
            super(42);
        }

        @Reflect
        int test() {
            return n.compareTo(0);
        }

        @Reflect
        static int test(TestClass t) {
            return t.n.compareTo(0);
        }

        @Reflect
        int testCA() {
            return (n++).compareTo(0);
        }

        @Reflect
        static int testCA(TestClass t) {
            return (t.n++).compareTo(0);
        }
    }

    @Test
    public void test() throws Throwable {
        Assertions.assertTrue(new TestClass().test() > 0);
        Assertions.assertTrue(TestClass.test(new TestClass()) > 0);
        Assertions.assertTrue(new TestClass().testCA() > 0);
        Assertions.assertTrue(TestClass.testCA(new TestClass()) > 0);
    }
}
