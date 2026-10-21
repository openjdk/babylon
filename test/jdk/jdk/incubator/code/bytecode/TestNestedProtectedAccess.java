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

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import resolve.Parent;

/*
 * @test
 * @modules jdk.incubator.code
 * @library ../lib/
 * @enablePreview
 * @run junit/othervm
 *      -Dtest.query=junit-select:class:TestNestedProtectedAccess$Nested
 *      TestNestedProtectedAccess
 * @run junit/othervm
 *      -Dtest.query=junit-select:class:TestNestedProtectedAccess$ExtNested
 *      TestNestedProtectedAccess
 */
public class TestNestedProtectedAccess extends Parent {

    public static class Child extends TestNestedProtectedAccess {}

    public static class Nested {

        @Reflect
        public int accessProtectedMethod() {
            return number();
        }

        @Test
        public void testProtectedMethodAccess() throws Throwable {
            Method accessMethod = Nested.class.getDeclaredMethod("accessProtectedMethod");
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(accessMethod).orElseThrow());
            Assertions.assertEquals(accessProtectedMethod(), (int) handle.invokeExact(this));
        }

        @Reflect
        public int accessProtectedField() {
            return number;
        }

        @Test
        public void testProtectedFieldAccess() throws Throwable {
            Method accessMethod = Nested.class.getDeclaredMethod("accessProtectedField");
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(accessMethod).orElseThrow());
            Assertions.assertEquals(accessProtectedField(), (int) handle.invokeExact(this));
        }

        @Reflect
        public void writeProtectedField(int i) {
            number = i;
        }

        @Test
        public void testProtectedFieldWrite() throws Throwable {
            Method writeMethod = Nested.class.getDeclaredMethod("writeProtectedField", int.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(writeMethod).orElseThrow());
            handle.invokeExact(this, 43);
            Assertions.assertEquals(43, accessProtectedField());
        }

        @Reflect
        public int accessInstanceProtectedMethod(Child ch) {
            return ch.instanceNumber();
        }

        @Test
        public void testInstanceProtectedMethodAccess() throws Throwable {
            Method accessMethod = Nested.class.getDeclaredMethod("accessInstanceProtectedMethod", Child.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(accessMethod).orElseThrow());
            Child ch = new Child();
            Assertions.assertEquals(accessInstanceProtectedMethod(ch), (int) handle.invokeExact(this, ch));
        }

        @Reflect
        public int accessInstanceProtectedField(Child ch) {
            return ch.instanceNumber;
        }

        @Test
        public void testInstanceProtectedFieldAccess() throws Throwable {
            Method accessMethod = Nested.class.getDeclaredMethod("accessInstanceProtectedField", Child.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(accessMethod).orElseThrow());
            Child ch = new Child();
            Assertions.assertEquals(accessInstanceProtectedField(ch), (int) handle.invokeExact(this, ch));
        }

        @Reflect
        public void writeInstanceProtectedField(Child ch, int i) {
            ch.instanceNumber = i;
        }

        @Test
        public void testInstanceProtectedFieldWrite() throws Throwable {
            Method writeMethod = Nested.class.getDeclaredMethod("writeInstanceProtectedField", Child.class, int.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(writeMethod).orElseThrow());
            Child ch = new Child();
            handle.invokeExact(this, ch, 43);
            Assertions.assertEquals(43, accessInstanceProtectedField(ch));
        }
    }

    public static class ExtNested extends Parent {

        @Reflect
        public int accessInstanceProtectedMethod(Child ch) {
            return ch.instanceNumber();
        }

        @Test
        public void testInstanceProtectedMethodAccess() throws Throwable {
            Method accessMethod = ExtNested.class.getDeclaredMethod("accessInstanceProtectedMethod", Child.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(accessMethod).orElseThrow());
            Child ch = new Child();
            Assertions.assertEquals(accessInstanceProtectedMethod(ch), (int) handle.invokeExact(this, ch));
        }

        @Reflect
        public int accessInstanceProtectedField(Child ch) {
            return ch.instanceNumber;
        }

        @Test
        public void testInstanceProtectedFieldAccess() throws Throwable {
            Method accessMethod = ExtNested.class.getDeclaredMethod("accessInstanceProtectedField", Child.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(accessMethod).orElseThrow());
            Child ch = new Child();
            Assertions.assertEquals(accessInstanceProtectedField(ch), (int) handle.invokeExact(this, ch));
        }

        @Reflect
        public void writeInstanceProtectedField(Child ch, int i) {
            ch.instanceNumber = i;
        }

        @Test
        public void testInstanceProtectedFieldWrite() throws Throwable {
            Method writeMethod = ExtNested.class.getDeclaredMethod("writeInstanceProtectedField", Child.class, int.class);
            MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                             Op.ofMethod(writeMethod).orElseThrow());
            Child ch = new Child();
            handle.invokeExact(this, ch, 43);
            Assertions.assertEquals(43, accessInstanceProtectedField(ch));
        }
    }
}
