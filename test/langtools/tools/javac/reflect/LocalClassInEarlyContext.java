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
 * @summary Test for local class creation in early construction context
 * @modules jdk.incubator.code
 */
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.extern.OpWriter;

public class LocalClassInEarlyContext {
    static class TestNoCapture {
        TestNoCapture() {
            class Foo {
                void t() { }
            }
            check((@Reflect Runnable)() -> new Foo(), EXPECTED);
            super();
        }

        static final String EXPECTED =
                """
                %0 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %1 : java.type:"LocalClassInEarlyContext$TestNoCapture::$1Foo" = new @java.ref:"LocalClassInEarlyContext$TestNoCapture::$1Foo::()";
                    return;
                };
                """;
    }

    static class TestCapture {
        TestCapture(int i, String s) {
            final long L = 42L; // this should NOT be captured
            class Foo {
                long t() {
                    return i + s.length() + L;
                }
            }
            check((@Reflect Runnable)() -> new Foo(), EXPECTED);
            super();
        }

        static final String EXPECTED =
                """
                %0 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %1 : java.type:"int" = var.load %2;
                    %3 : java.type:"java.lang.String" = var.load %4;
                    %5 : java.type:"LocalClassInEarlyContext$TestCapture::$1Foo" = new %1 %3 @java.ref:"LocalClassInEarlyContext$TestCapture::$1Foo::(int, java.lang.String)";
                    return;
                };
                """;
    }

    public static void main(String[] args) {
        new TestNoCapture();
        new TestCapture(1, "");
    }

    static void check(Runnable r, String expected) {
        expected = expected.trim();
        var quoted = Op.ofLambda(r).get();
        var found = OpWriter.toText(quoted.op(), OpWriter.LocationOption.DROP_LOCATION);
        if (!found.equals(expected)) {
            throw new AssertionError("Model mismatch. Found:\n" + found + "\nExpected:\n" + expected);
        }
    }
}
