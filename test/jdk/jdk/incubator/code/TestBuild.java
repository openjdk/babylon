/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.SSA;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.AccessFlag;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntBinaryOperator;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestBuild
 * @run junit/othervm -Dbabylon.ssa=cytron TestBuild
 */

public class TestBuild {

    public JavaOp.LambdaOp f() {
        IntBinaryOperator ibo = (@Reflect IntBinaryOperator) (a, b) -> a + b;
        return SSA.transform(Op.ofLambda(ibo).get().op());
    }

    @Test
    public void testBuiltValueAsOperand() {
        JavaOp.LambdaOp f = f();

        var a = f.body().entryBlock().parameters().get(0);
        var b = f.body().entryBlock().parameters().get(1);
        // Passing built values as operands to a new unplaced operation
        Assertions.assertThrows(IllegalArgumentException.class, () -> JavaOp.add(a, b));
    }

    @Test
    public void testBuiltValueAsBlockArgument() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableSignature());
        var block = body.entryBlock();
        var anotherBlock = block.block(INT, INT);

        var a = f.body().entryBlock().parameters().get(0);
        var b = f.body().entryBlock().parameters().get(1);
        // Passing built values as block arguments of a block reference
        Assertions.assertThrows(IllegalArgumentException.class, () -> branch(anotherBlock.reference(a, b)));
    }

    @Test
    public void testUnmappedBuiltValue() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableSignature());
        var block = body.entryBlock();

        var freturnOp = f.body().entryBlock().terminatingOp();
        // Unmapped built value that is operand of the built return op
        Assertions.assertThrows(IllegalArgumentException.class, () -> block.add(freturnOp));
    }

    @Test
    public void testMappingToBuiltValue() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableSignature());
        var block = body.entryBlock();

        var result = f.body().entryBlock().firstOp().result();
        // Mapping to a built value
        Assertions.assertThrows(IllegalArgumentException.class, () -> block.context().mapValue(result, result));
    }

    @Test
    public void testMappedBuiltValue() {
        JavaOp.LambdaOp f = f();

        var body = Body.Builder.of(null, f.invokableSignature());
        var block = body.entryBlock();

        var a = block.parameters().get(0);
        var b = block.parameters().get(1);
        var result = block.add(JavaOp.add(a, b));
        // Map the built value used as the operand to the built return op to
        // the above value
        block.context().mapValue(f.body().entryBlock().firstOp().result(), result);

        var freturnOp = f.body().entryBlock().terminatingOp();
        // No error since values (operands) are mapped
        block.add(freturnOp);
    }

    @Test
    public void testUnbuiltBlocksViaValue() {
        var body = Body.Builder.of(null, functionType(INT, INT, INT));
        var block = body.entryBlock();

        Block.Parameter a = block.parameters().get(0);
        Block.Parameter b = block.parameters().get(1);
        Op.Result result = block.add(JavaOp.add(a, b));

        // Declaring block is being built
        Assertions.assertThrows(IllegalStateException.class, a::declaringBlock);
        Assertions.assertThrows(IllegalStateException.class, result::declaringBlock);
        // Access to parent block/body of operation result before they are built
        Assertions.assertThrows(IllegalStateException.class, result.op()::ancestorBlock);
        Assertions.assertThrows(IllegalStateException.class, result.op()::ancestorBody);
        // Access to set of users while declaring block is being built
        Assertions.assertThrows(IllegalStateException.class, a::uses);

        block.add(return_(result));

        func("f", body);

        Assertions.assertNotNull(a.declaringBlock());
        Assertions.assertNotNull(result.declaringBlock());
        Assertions.assertNotNull(result.op().ancestorBlock());
        Assertions.assertNotNull(result.op().ancestorBody());
        Assertions.assertNotNull(a.uses());
    }

    @Test
    public void testUnbuiltBlocksViaReference() {
        var body = Body.Builder.of(null, functionType(INT, INT, INT));
        var block = body.entryBlock();
        var anotherBlock = block.block(INT, INT);

        var a = block.parameters().get(0);
        var b = block.parameters().get(1);
        Block.Reference successor = anotherBlock.reference(a, b);
        // Target block is being built
        Assertions.assertThrows(IllegalStateException.class, successor::targetBlock);
        block.add(branch(anotherBlock.reference(a, b)));

        a = anotherBlock.parameters().get(0);
        b = anotherBlock.parameters().get(1);
        var result = anotherBlock.add(JavaOp.add(a, b));
        anotherBlock.add(return_(result));

        func("f", body);

        Assertions.assertNotNull(successor.targetBlock());
    }

    @Test
    public void testValueUseFromOtherModel() {
        var abody = Body.Builder.of(null, functionType(INT, INT, INT));
        var ablock = abody.entryBlock();
        var aa = ablock.parameters().get(0);
        var ab = ablock.parameters().get(1);

        var bbody = Body.Builder.of(null, abody.bodySignature());
        var bblock = bbody.entryBlock();

        // Operation uses values from another model
        var addOp = JavaOp.add(aa, ab);
        Assertions.assertThrows(IllegalStateException.class, () -> bblock.add(addOp));
    }

    @Test
    public void testReferenceFromOtherBody() {
        var abody = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var ablock = abody.entryBlock().block();

        var bbody = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var bblock = bbody.entryBlock();

        // Operation uses header with target block from another model
        var brOp = branch(ablock.reference());
        Assertions.assertThrows(IllegalStateException.class, () -> bblock.add(brOp));
    }

    @Test
    public void testReferenceFromEntryBlock() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        Assertions.assertThrows(IllegalStateException.class, block::reference);
    }

    @Test
    public void testBuiltBodyBuilder() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.add(return_());
        func("f", body);

        // Body is built
        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testBodyBuilderWithBuiltAncestor() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.add(return_());
        func("f", body);

        // ancestor body is built
        Assertions.assertThrows(IllegalStateException.class, () -> Body.Builder.of(body, FUNCTION_TYPE_VOID));
    }

    @Test
    public void testBodyBuilderWithUnbuiltChildren() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.add(return_());

        Body.Builder.of(body, FUNCTION_TYPE_VOID);

        // Great-grandchild body is not built
        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testBodyBuilderWithUnplacedOperation() {
        var body1 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block1 = body1.entryBlock();
        block1.add(return_());

        var body2 = Body.Builder.of(body1, FUNCTION_TYPE_VOID);
        var block2 = body2.entryBlock();
        block2.add(return_());
        CoreOp.func("f", body2);

        // Great-grandchild body is child of unplaced operation
        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body1));
    }

    @Test
    public void testFailedBodyBuilder() {
        var body1 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block1 = body1.entryBlock();
        block1.add(return_());

        var body2 = Body.Builder.of(body1, FUNCTION_TYPE_VOID);
        var block2 = body2.entryBlock();
        // body2 finishes in failure
        Assertions.assertThrows(IllegalStateException.class, () -> CoreOp.func("f", body2));

        // Great-grandchild body finishes in failure
        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body1));
    }

    @Test
    public void testMistmatchedBody() {
        var body1 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block1 = body1.entryBlock();

        var anotherBody = Body.Builder.of(null, FUNCTION_TYPE_VOID);

        var body2 = Body.Builder.of(anotherBody, FUNCTION_TYPE_VOID);
        var block2 = body2.entryBlock();
        block2.add(return_());
        var lambdaOp = JavaOp.lambda(type(Runnable.class), body2);

        // lambdaOp's grandparent body is not parent body of block1
        Assertions.assertThrows(IllegalStateException.class, () -> block1.add(lambdaOp));
    }

    @Test
    public void testIsolatedBody() {
        var body1 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block1 = body1.entryBlock();

        var body2 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block2 = body2.entryBlock();
        block2.add(return_());
        var lambdaOp = JavaOp.lambda(type(Runnable.class), body2);

        Assertions.assertDoesNotThrow(() -> block1.add(lambdaOp));
        block1.add(return_());
        Assertions.assertDoesNotThrow(() -> func("f", body1));
    }

    @Test
    public void testValueUseInIsolatedBody() {
        var body1 = Body.Builder.of(null, functionType(VOID, INT));
        var block1 = body1.entryBlock();
        var p = block1.parameters().get(0);
        block1.add(return_());

        var body2 = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block2 = body2.entryBlock();
        // Value p is not reachable from block2
        Assertions.assertThrows(IllegalStateException.class, () -> block2.add(JavaOp.neg(p)));
    }

    @Test
    public void testAppendAfterTerminatingOperation() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.add(return_());

        // Append operation after terminating operation
        Assertions.assertThrows(IllegalStateException.class, () -> block.add(return_()));
    }

    @Test
    public void testNoTerminatingOperation() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.add(constant(INT, 0));

        // No terminating operation
        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testUnreferencedBlocksRemoved() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        block.add(return_());

        // Create empty blocks
        block.block();
        block.block();
        block.block();

        FuncOp f = func("f", body);
        Assertions.assertEquals(1, f.body().blocks().size());
    }

    @Test
    public void testEmptyEntryBlock() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testNonEmptyEntryBlockNoTerminatingOp() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var block = body.entryBlock();
        // No terminating op
        block.add(constant(INT, 0));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testEmptyBlockWithPredecessor() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var entryBlock = body.entryBlock();
        // Create empty block
        var block = entryBlock.block();
        // Branch to empty block
        entryBlock.add(branch(block.reference()));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    public void testNonEmptyBlockNoTerminatingOp() {
        var body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var entryBlock = body.entryBlock();
        // Create empty block
        var block = entryBlock.block();
        // Branch to empty block
        entryBlock.add(branch(block.reference()));
        // No terminating op
        block.add(constant(INT, 0));

        Assertions.assertThrows(IllegalStateException.class, () -> func("f", body));
    }

    @Test
    void testBuilderInoperableAfterBuildFinishes() {
        var bodyBuilder = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        var entryBlockBuilder = bodyBuilder.entryBlock();
        var blockBuilder = entryBlockBuilder.block();
        blockBuilder.add(return_());
        entryBlockBuilder.add(branch(blockBuilder.reference()));

        var fop = func("f", bodyBuilder);

        for (Object r : List.of(bodyBuilder, blockBuilder)) {
            for (Method m : r.getClass().getDeclaredMethods()) {
                if (m.accessFlags().contains(AccessFlag.STATIC) || !m.accessFlags().contains(AccessFlag.PUBLIC)) {
                    continue;
                }
                List<Object> args = new ArrayList<>();
                for (Class<?> parameterType : m.getParameterTypes()) {
                    Object arg;
                    if (parameterType == Value[].class) {
                        arg = new Value[]{};
                    } else if (parameterType == CodeType[].class) {
                        arg = new CodeType[]{};
                    } else if (parameterType == CodeType.class) {
                        arg = INT;
                    } else if (parameterType == List.class) {
                        arg = List.of();
                    } else if (parameterType == CodeContext.class) {
                        arg = CodeContext.create();
                    } else if (parameterType == CodeTransformer.class) {
                        arg = CodeTransformer.COPYING_TRANSFORMER;
                    } else if (parameterType == Body.class) {
                        arg = fop.body();
                    } else if (parameterType == Op.class) {
                        arg = fop;
                    } else if (parameterType == Object.class) {
                        arg = null;
                    } else {
                        throw new AssertionError("Unhandled parameter type " + parameterType + ", in the method " + m);
                    }
                    args.add(arg);
                }
                var wrapperException = Assertions.assertThrowsExactly(InvocationTargetException.class,
                        () -> m.invoke(r, args.toArray()));
                Assertions.assertInstanceOf(IllegalStateException.class, wrapperException.getCause());
            }
        }
    }
}
