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
import java.util.ArrayList;
import java.util.List;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestBuildUnreachableBlocks
 */

public class TestBuildUnreachableBlocks {

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
        Method m = TestBuildUnreachableBlocks.class.getDeclaredMethod("f");
        CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();

        CoreOp.FuncOp g = CoreOp.func("g", CoreType.functionType(JavaType.VOID)).body(b -> {
            b.op(CoreOp.return_());

            Block.Builder unreachableBlock1 = b.block();
            unreachableBlock1.transformBody(f.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);

            Block.Builder unreachableBlock2 = b.block();
            unreachableBlock2.transformBody(f.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(1, g.body().entryBlock().ops().size());
    }


    @Test
    public void testRemoveBlocks() throws Throwable {
        Method m = TestBuildUnreachableBlocks.class.getDeclaredMethod("f");
        CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();

        CoreOp.FuncOp g = CoreOp.func("g", CoreType.functionType(JavaType.VOID)).body(b -> {
            b.op(CoreOp.return_());

            CoreOp.FuncOp fLowered = f.transform(CodeTransformer.LOWERING_TRANSFORMER);

            Block.Builder unreachableBlock1 = b.block();
            unreachableBlock1.transformBody(fLowered.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);

            Block.Builder unreachableBlock2 = b.block();
            unreachableBlock2.transformBody(fLowered.body(), List.of(), CodeTransformer.COPYING_TRANSFORMER);
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(1, g.body().entryBlock().ops().size());
    }

    @Test
    public void testNoUnreachablePredecessors() {
        CoreOp.FuncOp f = CoreOp.func("f", CoreType.functionType(JavaType.VOID)).body(b -> {
            Block.Builder reachableBlock1 = b.block();

            b.op(branch(reachableBlock1.reference()));

            reachableBlock1.op(CoreOp.return_());

            Block.Builder unreachableBlock1 = b.block();
            unreachableBlock1.op(branch(reachableBlock1.reference()));

            Block.Builder unreachableBlock2 = b.block();
            unreachableBlock2.op(branch(reachableBlock1.reference()));
        });

        Assertions.assertEquals(2, f.body().blocks().size());

        Block reachableBlock = f.body().blocks().get(1);
        Assertions.assertEquals(1, reachableBlock.predecessors().size());
    }

    @Test
    public void testUnobservableUnreachableBlocks() {
        List<Op.Result> results = new ArrayList<>();
        CoreOp.FuncOp f = CoreOp.func("f", CoreType.functionType(JavaType.VOID)).body(b -> {
            b.op(CoreOp.return_());

            Block.Builder unreachableBlock1 = b.block();
            results.add(unreachableBlock1.op(return_()));

            Block.Builder unreachableBlock2 = b.block();
            results.add(unreachableBlock2.op(return_()));
        });

        Assertions.assertEquals(1, f.body().blocks().size());
        Assertions.assertEquals(1, f.body().entryBlock().ops().size());

        for (Op.Result r : results) {
            // Declaring block, an unreachable block, is not observable
            Assertions.assertThrows(IllegalStateException.class, r::declaringBlock);
        }
    }

    @Test
    public void testNestedUsesInUnreachableBlock() {
        CoreOp.FuncOp g = CoreOp.func("f", CoreType.functionType(JavaType.VOID)).body(b -> {
            Op.Result zeroValue = b.op(CoreOp.constant(JavaType.INT, 0));
            Op.Result oneValue = b.op(CoreOp.constant(JavaType.INT, 1));
            b.op(CoreOp.return_());

            JavaType.type(Runnable.class);

            Block.Builder unreachableBlock1 = b.block();
            JavaOp.LambdaOp lop = JavaOp.lambda(b.parentBody(), FunctionType.FUNCTION_TYPE_VOID, JavaType.type(Runnable.class)).body(lb -> {
                lb.op(JavaOp.neg(zeroValue));
                lb.op(JavaOp.neg(oneValue));
                lb.op(CoreOp.return_());
            });
            unreachableBlock1.op(lop);
            unreachableBlock1.op(CoreOp.return_());
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(3, g.body().entryBlock().ops().size());

        Op.Result zeroValue = g.body().entryBlock().ops().get(0).result();
        Assertions.assertTrue(zeroValue.uses().isEmpty());

        Op.Result oneValue = g.body().entryBlock().ops().get(1).result();
        Assertions.assertTrue(oneValue.uses().isEmpty());
    }

    @Test
    public void testUsesInUnreachableBlocks() {
        CoreOp.FuncOp g = CoreOp.func("f", CoreType.functionType(JavaType.VOID)).body(b -> {
            Op.Result zeroValue = b.op(CoreOp.constant(JavaType.INT, 0));
            Op.Result oneValue = b.op(CoreOp.constant(JavaType.INT, 1));
            b.op(CoreOp.return_());

            Block.Builder unreachableBlock1 = b.block();
            unreachableBlock1.op(JavaOp.neg(zeroValue));
            unreachableBlock1.op(JavaOp.neg(oneValue));
            unreachableBlock1.op(CoreOp.return_());

            Block.Builder unreachableBlock2 = b.block();
            unreachableBlock2.op(JavaOp.neg(zeroValue));
            unreachableBlock2.op(JavaOp.neg(oneValue));
            unreachableBlock2.op(CoreOp.return_());
        });

        Assertions.assertEquals(1, g.body().blocks().size());
        Assertions.assertEquals(3, g.body().entryBlock().ops().size());

        Op.Result zeroValue = g.body().entryBlock().ops().get(0).result();
        Assertions.assertTrue(zeroValue.uses().isEmpty());

        Op.Result oneValue = g.body().entryBlock().ops().get(1).result();
        Assertions.assertTrue(oneValue.uses().isEmpty());
    }

    @Test
    public void testResultUseFromUnreachableBlock() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();

        Block.Builder unreachableBlock = entryBlock.block();
        Op.Result zeroValue = unreachableBlock.op(CoreOp.constant(JavaType.INT, 0));
        unreachableBlock.op(return_());

        // Uses operation result declared in unreachable block
        entryBlock.op(JavaOp.neg(zeroValue));
        entryBlock.op(return_());

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testParameterUseFromUnreachableBlock() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        Block.Builder entryBlock = body.entryBlock();

        Block.Builder unreachableBlock = entryBlock.block(JavaType.INT);
        unreachableBlock.op(return_());

        // Uses block parameter declared in unreachable block
        entryBlock.op(JavaOp.neg(unreachableBlock.parameters().getFirst()));
        entryBlock.op(return_());

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }
}
