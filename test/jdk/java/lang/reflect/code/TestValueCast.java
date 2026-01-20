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
 * @run junit TestValueCast
 * @run junit/othervm -Dbabylon.ssa=cytron TestValueCast
 */

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.function.IntBinaryOperator;
import java.util.stream.Stream;

public class TestValueCast {

    public JavaOp.LambdaOp f() {
        IntBinaryOperator ibo = (@Reflect IntBinaryOperator) (a, b) -> {
            if (a > b) {
                a *= 2;
                return a % b;
            }
            return a + b;
        };
        return Op.ofLambda(ibo).get().op();
    }

    static Stream<Value> values(CodeElement<?, ?> r) {
        return r.elements().mapMulti((e, c) -> {
            switch (e) {
                case Block block -> block.parameters().forEach(c);
                case Op op -> c.accept(op.result());
                case Body _ -> { }
            }
        });
    }

    @Test
    public void testCast() {
        JavaOp.LambdaOp f = f();
        Stream<Value> stream = values(f);
        stream.forEach(v -> {
            switch (v) {
                case Op.Result r -> Assertions.assertEquals(r, v.result());
                case Block.Parameter p -> Assertions.assertEquals(p, v.parameter());
            }
        });
    }

    @Test
    public void testExceptions() {
        JavaOp.LambdaOp f = f();
        Stream<Value> stream = values(f);
        stream.forEach(v -> {
            switch (v) {
                case Op.Result r -> Assertions.assertThrows(IllegalStateException.class, () -> v.parameter());
                case Block.Parameter p -> Assertions.assertThrows(IllegalStateException.class, () -> v.result());
            }
        });
    }
}
