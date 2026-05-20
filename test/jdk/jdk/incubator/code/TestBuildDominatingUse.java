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
 * @run junit TestBuildDominatingUse
 */

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;

public class TestBuildDominatingUse {

    @Test
    public void testUndominatedOperandUse() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();
        Block.Builder block1 = entryBlock.block();

        Op.Result zeroValue = block1.op(CoreOp.constant(JavaType.INT, 0));
        block1.op(return_());

        // Use of operation result as operand is not dominated by its declaration
        entryBlock.op(JavaOp.neg(zeroValue));
        entryBlock.op(branch(block1.reference()));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testUndominatedParameterUse() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();
        Block.Builder block1 = entryBlock.block(JavaType.INT);
        Block.Parameter p = block1.parameters().getFirst();

        block1.op(return_());

        // Use of block parameter as block argument is not dominated by its declaration
        entryBlock.op(branch(block1.reference(p)));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testUndominatedOperandNestedUse() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();
        Block.Builder block1 = entryBlock.block();

        Op.Result zeroValue = block1.op(CoreOp.constant(JavaType.INT, 0));
        block1.op(return_());

        entryBlock.op(JavaOp.lambda(body, FUNCTION_TYPE_VOID, JavaType.type(Runnable.class))
                .body(block2 -> {
                    // Use of operation result as operand is not dominated by its declaration
                    block2.op(JavaOp.neg(zeroValue));
                    block2.op(return_());
                }));
        entryBlock.op(branch(block1.reference()));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testUndominatedParameterNestedUse() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();
        Block.Builder block1 = entryBlock.block(JavaType.INT);
        Block.Parameter p = block1.parameters().getFirst();

        block1.op(return_());

        entryBlock.op(JavaOp.lambda(body, FUNCTION_TYPE_VOID, JavaType.type(Runnable.class))
                .body(block2 -> {
                    Block.Builder block3 = block2.block(JavaType.INT);
                    block3.op(return_());

                    // Use of block parameter as block argument is not dominated by its declaration
                    block2.op(branch(block3.reference(p)));
                }));
        Op.Result zero = entryBlock.op(constant(JavaType.INT, 0));
        entryBlock.op(branch(block1.reference(zero)));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }
}
