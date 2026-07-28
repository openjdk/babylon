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

 import java.lang.reflect.Method;
 import java.util.Arrays;
 import jdk.incubator.code.Reflect;

/*
 * @test
 * @summary Verify reflected bridge generation
 * @modules jdk.incubator.code
 * @run main BridgeTest
 */
public class BridgeTest {

    interface Value<T> {
        T value();
    }

    static final class StringValue implements Value<String> {
        @Override
        @Reflect
        public String value() {
            return "string value";
        }
    }

    public static void main(String[] args) throws Exception {
        var val = new StringValue().value();
        if (!"string value".equals(val)) {
            throw new AssertionError("Unexpected bridge invocation result: " + val);
        }
        Method bridge = Arrays.stream(StringValue.class.getDeclaredMethods())
                              .filter(Method::isBridge)
                              .findFirst()
                              .orElseThrow();
        if (bridge.getReturnType() != Object.class) {
            throw new AssertionError("Unexpected bridge return type: " + bridge.getReturnType());
        }
    }
}
