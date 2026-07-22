/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestBuildOp
 */
public class TestBuildOp {

    @Reflect
    static List<Integer> f(int i) {
        return new ArrayList<>(i);
    }

    @Test
    void testCopyFromMethod() throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp f = Op.ofMethod(m).get();

        assertOpIsCopiedWhenAddedToBlock(f);
    }

    @Test
    void testCopyFromLambda() {
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i / 2;
        Quoted<?> quoted = Op.ofLambda(q).get();
        assert quoted.capturedValues().isEmpty();

        assertOpIsCopiedWhenAddedToBlock(quoted.op());
    }

    @Test
    void testCopyFromOp() {
        CoreOp.ConstantOp constant = CoreOp.constant(JavaType.INT, 7);
        constant.buildAsRoot();

        assertOpIsCopiedWhenAddedToBlock(constant);
    }

    @Test
    void testBuildAsRoot() {
        CoreOp.FuncOp funcOp = CoreOp.func("f", FunctionType.FUNCTION_TYPE_VOID).body(b -> {
            b.add(CoreOp.return_());
        });

        Assertions.assertFalse(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isPlacedInBlock());
        funcOp.buildAsRoot();

        Assertions.assertTrue(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isPlacedInBlock());
        funcOp.buildAsRoot();
    }

    @Test
    void testBuiltLambdaRoot() {
        IntBinaryOperator q = (@Reflect IntBinaryOperator)(int a, int b) -> a + b;
        Quoted<?> quoted = Op.ofLambda(q).orElseThrow();

        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) quoted.op().ancestorOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorOp();

        Assertions.assertTrue(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isPlacedInBlock());
    }

    @Test
    void testOpPlaced() {
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        Op.Result r = body.entryBlock().add(CoreOp.constant(JavaType.DOUBLE, 1d));
        body.entryBlock().add(CoreOp.return_());

        Assertions.assertThrows(IllegalStateException.class, () -> r.op().buildAsRoot());
        Assertions.assertTrue(r.op().isPlacedInBlock());
        Assertions.assertFalse(r.op().isRoot());

        CoreOp.func("f", body);
        Assertions.assertTrue(r.op().isPlacedInBlock());
    }

    @Test
    void testSetLocation() {
        CoreOp.ConstantOp cop = CoreOp.constant(JavaType.LONG, 1L);
        cop.setLocation(Op.Location.NO_LOCATION);
        cop.buildAsRoot();

        Assertions.assertThrows(IllegalStateException.class, () -> cop.setLocation(Op.Location.NO_LOCATION));

        IntBinaryOperator q = (@Reflect IntBinaryOperator)(int a, int b) -> a + b;
        Quoted<?> quoted = Op.ofLambda(q).orElseThrow();
        Assertions.assertThrows(IllegalStateException.class, () -> quoted.op().setLocation(Op.Location.NO_LOCATION));
    }

    @Test
    void testBuildAsRootForOpWithOpenBody() {
        Body.Builder outer = Body.Builder.of(null, CoreType.functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter p = outer.entryBlock().parameters().getFirst();

        Body.Builder inner = Body.Builder.of(outer, CoreType.FUNCTION_TYPE_VOID);
        inner.entryBlock().add(JavaOp.neg(p));
        inner.entryBlock().add(CoreOp.return_());
        JavaOp.LambdaOp lambdaOp = JavaOp.lambda(JavaType.type(Runnable.class), inner);

        Assertions.assertThrowsExactly(IllegalStateException.class, lambdaOp::buildAsRoot);
    }

    @Test
    void testBuildAsRootForOpWithOperands() {
        Body.Builder body = Body.Builder.of(null, CoreType.functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter p = body.entryBlock().parameters().getFirst();
        Op op = JavaOp.neg(p);
        Assertions.assertThrowsExactly(IllegalStateException.class, op::buildAsRoot);
    }

    @Test
    void testBuildAsRootForOpWithSuccessor() {
        Body.Builder body = Body.Builder.of(null, CoreType.functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter p = body.entryBlock().parameters().getFirst();
        Block.Builder block2 = body.entryBlock().block();
        Op op = CoreOp.branch(block2.reference());
        Assertions.assertThrowsExactly(IllegalStateException.class, op::buildAsRoot);
    }

    void assertOpIsCopiedWhenAddedToBlock(Op op) {
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        body.entryBlock().add(op);
        body.entryBlock().add(CoreOp.return_());
        CoreOp.FuncOp funcOp = CoreOp.func("t", body);
        boolean b = funcOp.body().entryBlock().ops().stream().allMatch(o -> o != op);
        Assertions.assertTrue(b);
    }
}
