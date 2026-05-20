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

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.*;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestBuildUnreferencedBlocks
 */

public class TestBuildUnreferencedBlocks {

    @Reflect
    public static int f() {
        int sum = 0;
        for (int i = 0; i < 1; i++) {
            sum += i;
        }
        return sum;
    }

    @Test
    public void testRemoveNestedBlocks() throws Throwable {
        Method m = TestBuildUnreferencedBlocks.class.getDeclaredMethod("f");
        CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();

        CoreOp.FuncOp g = CoreOp.func("g", CoreType.functionType(JavaType.VOID)).body(b -> {
            b.op(CoreOp.return_());

            Block.Builder unreferencedBlock1 = b.block();
            unreferencedBlock1.transformBody(f.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);

            Block.Builder unreferencedBlock2 = b.block();
            unreferencedBlock2.transformBody(f.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(1, g.body().entryBlock().ops().size());
    }


    @Test
    public void testRemoveBlocks() throws Throwable {
        Method m = TestBuildUnreferencedBlocks.class.getDeclaredMethod("f");
        CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();

        CoreOp.FuncOp g = CoreOp.func("g", CoreType.functionType(JavaType.VOID)).body(b -> {
            b.op(CoreOp.return_());

            CoreOp.FuncOp fLowered = f.transform(CodeTransformer.LOWERING_TRANSFORMER);

            Block.Builder unreferencedBlock1 = b.block();
            unreferencedBlock1.transformBody(fLowered.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);

            Block.Builder unreferencedBlock2 = b.block();
            unreferencedBlock2.transformBody(fLowered.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(1, g.body().entryBlock().ops().size());
    }


    @Test
    public void testNestedUsesInUnreferencedBlock() {
        CoreOp.FuncOp g = CoreOp.func("f", CoreType.functionType(JavaType.VOID)).body(b -> {
            Op.Result zeroValue = b.op(CoreOp.constant(JavaType.INT, 0));
            Op.Result oneValue = b.op(CoreOp.constant(JavaType.INT, 1));
            b.op(CoreOp.return_());

            JavaType.type(Runnable.class);

            Block.Builder unreferencedBlock1 = b.block();
            JavaOp.LambdaOp lop = JavaOp.lambda(b.parentBody(), FunctionType.FUNCTION_TYPE_VOID, JavaType.type(Runnable.class)).body(lb -> {
                lb.op(JavaOp.neg(zeroValue));
                lb.op(JavaOp.neg(oneValue));
                lb.op(CoreOp.return_());
            });
            unreferencedBlock1.op(lop);
            unreferencedBlock1.op(CoreOp.return_());
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(3, g.body().entryBlock().ops().size());

        Op.Result zeroValue = g.body().entryBlock().ops().get(0).result();
        Assertions.assertTrue(zeroValue.uses().isEmpty());

        Op.Result oneValue = g.body().entryBlock().ops().get(1).result();
        Assertions.assertTrue(oneValue.uses().isEmpty());
    }

    @Test
    public void testUsesInUnreferencedBlocks() {
        CoreOp.FuncOp g = CoreOp.func("f", CoreType.functionType(JavaType.VOID)).body(b -> {
            Op.Result zeroValue = b.op(CoreOp.constant(JavaType.INT, 0));
            Op.Result oneValue = b.op(CoreOp.constant(JavaType.INT, 1));
            b.op(CoreOp.return_());

            Block.Builder unreferencedBlock1 = b.block();
            unreferencedBlock1.op(JavaOp.neg(zeroValue));
            unreferencedBlock1.op(JavaOp.neg(oneValue));
            unreferencedBlock1.op(CoreOp.return_());

            Block.Builder unreferencedBlock2 = b.block();
            unreferencedBlock2.op(JavaOp.neg(zeroValue));
            unreferencedBlock2.op(JavaOp.neg(oneValue));
            unreferencedBlock2.op(CoreOp.return_());
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(3, g.body().entryBlock().ops().size());

        Op.Result zeroValue = g.body().entryBlock().ops().get(0).result();
        Assertions.assertTrue(zeroValue.uses().isEmpty());

        Op.Result oneValue = g.body().entryBlock().ops().get(1).result();
        Assertions.assertTrue(oneValue.uses().isEmpty());
    }

    @Test
    public void testResultUseFromUnreferencedBlock() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();

        Block.Builder unreferencedBlock = entryBlock.block();
        Op.Result zeroValue = unreferencedBlock.op(CoreOp.constant(JavaType.INT, 0));
        unreferencedBlock.op(return_());

        // Uses operation result declared in unreferenced block
        entryBlock.op(JavaOp.neg(zeroValue));
        entryBlock.op(return_());

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testParameterUseFromUnreferencedBlock() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();

        Block.Builder unreferencedBlock = entryBlock.block(JavaType.INT);
        unreferencedBlock.op(return_());

        // Uses block parameter declared in unreferenced block
        entryBlock.op(JavaOp.neg(unreferencedBlock.parameters().getFirst()));
        entryBlock.op(return_());

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }
}
