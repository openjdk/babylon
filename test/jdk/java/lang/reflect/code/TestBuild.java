/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.dialect.java.JavaOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.*;
import jdk.incubator.code.analysis.SSA;
import java.util.function.IntBinaryOperator;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;
import static jdk.incubator.code.dialect.java.JavaType.INT;
import static jdk.incubator.code.dialect.java.JavaType.type;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestBuild
 * @run testng/othervm -Dbabylon.ssa=cytron TestBuild
 */

public class TestBuild {

    public JavaOp.LambdaOp f() {
        IntBinaryOperator ibo = (IntBinaryOperator & Quotable) (a, b) -> a + b;
        Quotable iboq = (Quotable) ibo;
        return SSA.transform((JavaOp.LambdaOp) Op.ofQuotable(iboq).get().op());
    }

    @Test
    public void testBoundValueAsOperand() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableType());
        var block = body.entryBlock();

        var a = f.body().entryBlock().parameters().get(0);
        var b = f.body().entryBlock().parameters().get(1);
        // Passing bound values as operands to a new unbound operation
        var addop = JavaOp.add(a, b);

        Assert.assertThrows(IllegalStateException.class, () -> block.op(addop));
    }

    @Test
    public void testBoundValueAsHeaderArgument() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableType());
        var block = body.entryBlock();
        var anotherBlock = block.block(INT, INT);

        var a = f.body().entryBlock().parameters().get(0);
        var b = f.body().entryBlock().parameters().get(1);
        // Passing bound values as header arguments of a header
        // that is the successor of a terminal operation
        var brop = branch(anotherBlock.successor(a, b));

        Assert.assertThrows(IllegalStateException.class, () -> block.op(brop));
    }

    @Test
    public void testUnmappedBoundValue() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableType());
        var block = body.entryBlock();

        var freturnOp = f.body().entryBlock().terminatingOp();
        // Unmapped bound value that is operand of the bound return op
        Assert.assertThrows(IllegalArgumentException.class, () -> block.op(freturnOp));
    }

    @Test
    public void testMappingToBoundValue() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableType());
        var block = body.entryBlock();

        var result = f.body().entryBlock().firstOp().result();
        // Mapping to a bound value
        Assert.assertThrows(IllegalArgumentException.class, () -> block.context().mapValue(result, result));
    }

    @Test
    public void testMappedBoundValue() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableType());
        var block = body.entryBlock();

        var a = block.parameters().get(0);
        var b = block.parameters().get(1);
        var result = block.op(JavaOp.add(a, b));
        // Map the bound value used as the operand to the bound return op to
        // the above value
        block.context().mapValue(f.body().entryBlock().firstOp().result(), result);

        var freturnOp = f.body().entryBlock().terminatingOp();
        // No error since values (operands) are mapped
        block.op(freturnOp);
    }

    @Test
    public void testPartiallyConstructedValueAccess() {
        var body = Body.Builder.of(null, functionType(INT, INT, INT));
        var block = body.entryBlock();

        Block.Parameter a = block.parameters().get(0);
        Block.Parameter b = block.parameters().get(1);
        Op.Result result = block.op(JavaOp.add(a, b));

        // Access the declaring block of values before the block and its body are
        // constructed
        Assert.assertThrows(IllegalStateException.class, a::declaringBlock);
        Assert.assertThrows(IllegalStateException.class, result::declaringBlock);
        // Access to parent block/body of operation result before they are constructed
        Assert.assertThrows(IllegalStateException.class, result.op()::ancestorBlock);
        Assert.assertThrows(IllegalStateException.class, result.op()::ancestorBody);
        // Access to set of users before constructed
        Assert.assertThrows(IllegalStateException.class, a::uses);

        block.op(return_(result));

        var f = func("f", body);

        Assert.assertNotNull(a.declaringBlock());
        Assert.assertNotNull(result.declaringBlock());
        Assert.assertNotNull(result.op().ancestorBlock());
        Assert.assertNotNull(result.op().ancestorBody());
        Assert.assertNotNull(a.uses());
    }

    @Test
    public void testPartiallyConstructedHeaderAccess() {
        var body = Body.Builder.of(null, functionType(INT, INT, INT));
        var block = body.entryBlock();
        var anotherBlock = block.block(INT, INT);

        var a = block.parameters().get(0);
        var b = block.parameters().get(1);
        Block.Reference successor = anotherBlock.successor(a, b);
        // Access to target block before constructed
        Assert.assertThrows(IllegalStateException.class, successor::targetBlock);
        block.op(branch(anotherBlock.successor(a, b)));

        a = anotherBlock.parameters().get(0);
        b = anotherBlock.parameters().get(1);
        var result = anotherBlock.op(JavaOp.add(a, b));
        anotherBlock.op(return_(result));

        var f = func("f", body);

        Assert.assertNotNull(successor.targetBlock());
    }

    @Test
    public void testValueUseFromOtherModel() {
        var abody = Body.Builder.of(null, functionType(INT, INT, INT));
        var ablock = abody.entryBlock();
        var aa = ablock.parameters().get(0);
        var ab = ablock.parameters().get(1);

        var bbody = Body.Builder.of(null, abody.bodyType());
        var bblock = bbody.entryBlock();

        // Operation uses values from another model
        var addOp = JavaOp.add(aa, ab);
        Assert.assertThrows(IllegalStateException.class, () -> bblock.op(addOp));
    }

    @Test
    public void testHeaderFromOtherBody() {
        var abody = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var ablock = abody.entryBlock().block();

        var bbody = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var bblock = bbody.entryBlock();

        // Operation uses header with target block from another model
        var brOp = branch(ablock.successor());
        Assert.assertThrows(IllegalStateException.class, () -> bblock.op(brOp));
    }

    @Test
    public void testHeaderFromEntryBlock() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        Assert.assertThrows(IllegalStateException.class, block::successor);
    }

    @Test
    public void testBuiltBodyBuilder() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.op(return_());
        func("f", body);

        // Body is built
        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testBodyBuilderWithBuiltAncestor() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.op(return_());
        func("f", body);

        // ancestor body is built
        Assert.assertThrows(IllegalStateException.class, () -> Body.Builder.of(body, FUNCTION_TYPE_VOID));
    }

    @Test
    public void testBodyBuilderWithUnbuiltChildren() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.op(return_());

        Body.Builder.of(body, FUNCTION_TYPE_VOID);

        // Great-grandchild body is not built
        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testMistmatchedBody() {
        var body1 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block1 = body1.entryBlock();

        var anotherBody = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        var body2 = Body.Builder.of(anotherBody, FUNCTION_TYPE_VOID);
        var block2 = body2.entryBlock();
        block2.op(return_());
        var lambdaOp = JavaOp.lambda(type(Runnable.class), body2);

        // Op's grandparent body is not parent body of block1
        Assert.assertThrows(IllegalStateException.class, () -> block1.op(lambdaOp));
    }

    @Test
    public void testAppendAfterTerminatingOperation() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.op(return_());

        // Append operation after terminating operation
        Assert.assertThrows(IllegalStateException.class, () -> block.op(return_()));
    }

    @Test
    public void testNoTerminatingOperation() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.op(constant(INT, 0));

        // No terminating operation
        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testUnreferencedBlocksRemoved() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.op(return_());

        // Create empty blocks
        block.block();
        block.block();
        block.block();

        FuncOp f = func("f", body);
        Assert.assertEquals(f.body().blocks().size(), 1);
    }

    @Test
    public void testEmptyEntryBlock() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();

        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testNonEmptyEntryBlockNoTerminatingOp() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        // No terminating op
        block.op(constant(INT, 0));

        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testEmptyBlockWithPredecessor() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var entryBlock = body.entryBlock();
        // Create empty block
        var block = entryBlock.block();
        // Branch to empty block
        entryBlock.op(branch(block.successor()));

        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testNonEmptyBlockNoTerminatingOp() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var entryBlock = body.entryBlock();
        // Create empty block
        var block = entryBlock.block();
        // Branch to empty block
        entryBlock.op(branch(block.successor()));
        // No terminating op
        block.op(constant(INT, 0));

        Assert.assertThrows(IllegalStateException.class, () -> func("f", body));
    }
}
