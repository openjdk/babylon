/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
 * @run junit TestAttributeSerialization
 */

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;

public class TestAttributeSerialization {

    static class TestOp extends Op {
        final Object attributeValue;

        TestOp(ExternalizedOp opdef) {
            this((Object) null);
        }

        TestOp(TestOp that, CopyContext cc) {
            super(that, cc);
            this.attributeValue = that.attributeValue;
        }

        @Override
        public TestOp transform(CopyContext cc, OpTransformer ot) {
            return new TestOp(this, cc);
        }

        TestOp(Object attributeValue) {
            super("test-op", List.of());
            this.attributeValue = attributeValue;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("a", attributeValue);
        }
    }


    static Object[][] attributes() {
        return new Object[][] {
                { new int[] {1, 2, 3}, "[1, 2, 3]"},
                { new int[][] { {1}, {2}, {3}}, "[[1], [2], [3]]"},
                { new Object[] {1, new int[] {1, 2, 3}, 3}, "[1, [1, 2, 3], 3]"},
        };
    }

    @ParameterizedTest
    @MethodSource("attributes")
    public void testAttributes(Object a, String s) {
        TestOp op = new TestOp(a);
        String serOp = op.toText();
        Assertions.assertTrue(serOp.contains(s), serOp);
    }

}
