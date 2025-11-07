/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package hat.test.engine;

import hat.buffer.Float2;
import hat.buffer.Float4;

public class HATAsserts {

    public static void assertEquals(int expected, int actual) {
        if (expected != actual) {
            throw new HATAssertionError("Expected: " + expected + " != actual: " + actual);
        }
    }

    public static void assertEquals(long expected, long actual) {
        if (expected != actual) {
            throw new HATAssertionError("Expected: " + expected + " != actual: " + actual);
        }
    }

    public static void assertEquals(float expected, float actual, float delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new HATAssertionError("Expected: " + expected + " != actual: " + actual);
        }
    }

    public static void assertEquals(double expected, double actual, double delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new HATAssertionError("Expected: " + expected + " != actual: " + actual);
        }
    }

    public static void assertEquals(Float4 expected, Float4 actual, float delta) {
        float[] arrayExpected = expected.toArray();
        float[] arrayActual = actual.toArray();
        for (int i = 0; i < 4; i++) {
            var expectedValue = arrayExpected[i];
            var actualValue = arrayActual[i];
            if (Math.abs(expectedValue - actualValue) > delta) {
                throw new HATAssertionError("Expected: " + expectedValue + " != actual: " + actualValue);
            }
        }
    }

    public static void assertEquals(Float2 expected, Float2 actual, float delta) {
        float[] arrayExpected = expected.toArray();
        float[] arrayActual = actual.toArray();
        for (int i = 0; i < 2; i++) {
            var expectedValue = arrayExpected[i];
            var actualValue = arrayActual[i];
            if (Math.abs(expectedValue - actualValue) > delta) {
                throw new HATAssertionError("Expected: " + expectedValue + " != actual: " + actualValue);
            }
        }
    }

    public static void assertTrue(boolean isCorrect) {
        if (!isCorrect) {
            throw new HATAssertionError("Expected: " + isCorrect);
        }
    }
}
