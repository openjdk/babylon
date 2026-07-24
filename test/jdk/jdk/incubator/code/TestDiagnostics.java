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

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.*;
import static org.junit.jupiter.api.Assertions.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestDiagnostics
 */
public class TestDiagnostics {
    @Test
    public void testBlock() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        body.entryBlock().add(constant(JavaType.INT, 1));

        assertEquals("""
                Block has no terminating operation as the last operation
                ^block_0:
                ^~~~~~~~ missing terminal op
                  %0 : java.type:"int" = constant @1;
                """, assertThrows(IllegalStateException.class, () -> func("f", body)).getMessage());
    }

    @Test
    public void testBlockParameter() {
        Body.Builder body = Body.Builder.of(null, functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter parameter = body.entryBlock().parameters().getFirst();
        body.entryBlock().add(return_());
        func("f", body);

        assertEquals("""
                A new operation cannot directly use a value from a completed code model
                ^block_0(%0 : java.type:"int"):
                         ^~ value from completed model
                  return;
                """, assertThrows(IllegalArgumentException.class, () -> JavaOp.neg(parameter)).getMessage());
    }

    @Test
    public void testOperation() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Op operation = body.entryBlock().add(constant(JavaType.INT, 1)).op();

        assertEquals("""
                Cannot access the operation's parent block while it is still being built
                ^block_0:
                  %0 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ operation in block being built
                """, assertThrows(IllegalStateException.class, operation::parent).getMessage());
    }

    @Test
    public void testOperationResult() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        body.entryBlock().add(constant(JavaType.INT, 1));
        body.entryBlock().add(return_());
        CoreOp.FuncOp input = func("f", body);
        Op.Result value = input.body().entryBlock().ops().getFirst().result();

        assertEquals("""
                No output value is mapped to this input value in the code context
                ^block_0:
                  %0 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ unmapped input value
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().getValue(value)).getMessage());
    }

    @Test
    public void testSuccessor() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder entry = body.entryBlock();
        Block.Builder target = entry.block();
        Op.Result value = entry.add(constant(JavaType.INT, 1));
        entry.add(branch(target.reference(value)));
        target.add(return_());

        assertEquals("""
                Block reference argument count is 1 but target block parameter count is 0
                ^block_0:
                  %0 : java.type:"int" = constant @1;
                  branch ^block_1(%0);
                  ^~~~~~~~~~~~~~~~~~~~ reference with wrong arity
                ^block_1:
                ^~~~~~~~ target block
                  return;
                """, assertThrows(IllegalStateException.class, () -> func("f", body)).getMessage());
    }

    @Test
    public void testNestedValueName() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        body.entryBlock().add(constant(JavaType.INT, 1));
        CoreOp.QuotedOp quoted = CoreOp.quoted(body, _ -> constant(JavaType.INT, 2));
        body.entryBlock().add(quoted);
        body.entryBlock().add(return_());
        func("f", body);

        assertEquals("""
                No output value is mapped to this input value in the code context
                ^block_1:
                  %2 : java.type:"int" = constant @2;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ unmapped input value
                  yield %2;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().getValue(quoted.quotedOp().result())).getMessage());
    }

    @Test
    public void testBuiltOutputValueMapping() {
        Op.Result value = constantModel().body().entryBlock().firstOp().result();
        assertEquals("""
                Output value's declaring block is built
                ^block_0:
                  %0 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ output value
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().mapValue(value, value)).getMessage());
    }

    @Test
    public void testMissingBlockMapping() {
        Block block = constantModel().body().entryBlock();
        assertEquals("""
                No mapping for input block
                ^block_0:
                ^~~~~~~~ input block
                  %0 : java.type:"int" = constant @1;
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().getBlock(block)).getMessage());
    }

    @Test
    public void testMissingReferenceMapping() {
        Block.Reference reference = builtReference();
        assertEquals("""
                No mapping for input block reference
                ^block_1:
                ^~~~~~~~ reference target
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().getReference(reference)).getMessage());
    }

    @Test
    public void testBuiltReferenceTargetMapping() {
        Block.Reference reference = builtReference();
        assertEquals("""
                Output block reference's target block is built
                ^block_1:
                ^~~~~~~~ reference target
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().mapReference(reference, reference)).getMessage());
    }

    @Test
    public void testBuiltReferenceArgumentMapping() {
        Body.Builder outer = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder target = outer.entryBlock().block();
        Body.Builder inner = Body.Builder.of(outer, FUNCTION_TYPE_VOID);
        Op.Result argument = inner.entryBlock().add(constant(JavaType.INT, 1));
        Block.Reference reference = target.reference(argument);
        inner.entryBlock().add(return_());
        func("nested", inner);

        assertEquals("""
                Output block reference argument's declaring block is built
                ^block_2:
                  %0 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ reference argument
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().mapReference(reference, reference)).getMessage());
    }

    @Test
    public void testMissingReferenceTargetMapping() {
        Block.Reference reference = builtReference();
        assertEquals("""
                No mapping for input reference target block
                ^block_1:
                ^~~~~~~~ reference target
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> CodeContext.create().getReferenceOrCreate(reference)).getMessage());
    }

    @Test
    public void testEntryBlockReference() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        assertEquals("""
                Entry block cannot be referenced and targeted as a successor
                ^block_0:
                ^~~~~~~~ entry block
                """, assertThrows(IllegalStateException.class,
                body.entryBlock()::reference).getMessage());
    }

    @Test
    public void testBuiltBlockArgument() {
        Op.Result value = constantModel().body().entryBlock().firstOp().result();
        Block.Builder target = Body.Builder.of(null, FUNCTION_TYPE_VOID).entryBlock().block();
        assertEquals("""
                Argument's declaring block is built
                ^block_0:
                  %0 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ argument
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> target.reference(value)).getMessage());
    }

    @Test
    public void testAppendAfterTerminator() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        body.entryBlock().add(return_());
        assertEquals("""
                Operation cannot be appended, the block has a terminating operation
                ^block_0:
                  return;
                  ^~~~~~~ terminating operation
                """, assertThrows(IllegalStateException.class,
                () -> body.entryBlock().add(return_())).getMessage());
    }

    @Test
    public void testMismatchedOperationBody() {
        Body.Builder destination = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Body.Builder otherAncestor = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        CoreOp.QuotedOp quoted = CoreOp.quoted(otherAncestor, _ -> constant(JavaType.INT, 1));
        assertEquals("""
                Body of operation is connected to a different ancestor body
                ^block_1:
                ^~~~~~~~ operation body
                  %0 : java.type:"int" = constant @1;
                  yield %0;
                """, assertThrows(IllegalStateException.class,
                () -> destination.entryBlock().add(quoted)).getMessage());
    }

    @Test
    public void testUnreachableOperand() {
        Body.Builder source = Body.Builder.of(null, functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter value = source.entryBlock().parameters().getFirst();
        Body.Builder destination = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        assertEquals("""
                Cannot append an operation because its operand belongs to another code model
                ^block_0(%0 : java.type:"int"):
                         ^~ operand from another model
                ^block_0:
                ^~~~~~~~ operation appended here
                """, assertThrows(IllegalStateException.class,
                () -> destination.entryBlock().add(JavaOp.neg(value))).getMessage());
    }

    @Test
    public void testForeignReferenceTarget() {
        Body.Builder source = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder foreignTarget = source.entryBlock().block();
        Body.Builder destination = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        assertEquals("""
                Target of block reference is not a sibling of this block
                ^block_1:
                ^~~~~~~~ reference target
                ^block_0:
                ^~~~~~~~ append block
                """, assertThrows(IllegalStateException.class,
                () -> destination.entryBlock().add(branch(foreignTarget.reference()))).getMessage());
    }

    @Test
    public void testUnreachableReferenceArgument() {
        Body.Builder source = Body.Builder.of(null, functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter argument = source.entryBlock().parameters().getFirst();
        Body.Builder destination = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder target = destination.entryBlock().block();
        assertEquals("""
                Cannot append a block reference because its argument belongs to another code model
                ^block_0(%0 : java.type:"int"):
                         ^~ argument from another model
                ^block_0:
                ^~~~~~~~ referencing operation appended here
                """, assertThrows(IllegalStateException.class,
                () -> destination.entryBlock().add(branch(target.reference(argument)))).getMessage());
    }

    @Test
    public void testUnreachableOperationResult() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder dead = body.entryBlock().block();
        Op.Result value = dead.add(constant(JavaType.INT, 1));
        dead.add(return_());
        body.entryBlock().add(JavaOp.neg(value));
        body.entryBlock().add(return_());
        assertEquals("""
                Use of an operation result is not dominated by the result
                ^block_1:
                  %1 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ declaration
                  return;
                ^block_0:
                  %0 : java.type:"int" = neg %1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ invalid use
                  return;
                """, assertThrows(IllegalStateException.class, () -> func("f", body)).getMessage());
    }

    @Test
    public void testUnreachableBlockParameter() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder dead = body.entryBlock().block(JavaType.INT);
        Block.Parameter value = dead.parameters().getFirst();
        dead.add(return_());
        body.entryBlock().add(JavaOp.neg(value));
        body.entryBlock().add(return_());
        assertEquals("""
                Use of block parameter is not dominated by the parameter
                ^block_1(%1 : java.type:"int"):
                         ^~ value declared here
                  return;
                ^block_0:
                  %0 : java.type:"int" = neg %1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ invalid use
                  return;
                """, assertThrows(IllegalStateException.class, () -> func("f", body)).getMessage());
    }

    @Test
    public void testBlockParameterDominance() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder target = body.entryBlock().block(JavaType.INT);
        Block.Parameter value = target.parameters().getFirst();
        Op.Result argument = body.entryBlock().add(constant(JavaType.INT, 1));
        body.entryBlock().add(JavaOp.neg(value));
        body.entryBlock().add(branch(target.reference(argument)));
        target.add(return_());
        assertEquals("""
                Use of value is not dominated by value
                ^block_1(%2 : java.type:"int"):
                         ^~ value declared here
                  return;
                ^block_0:
                  %0 : java.type:"int" = constant @1;
                  %1 : java.type:"int" = neg %2;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ invalid use
                  branch ^block_1(%0);
                """, assertThrows(IllegalStateException.class, () -> func("f", body)).getMessage());
    }

    @Test
    public void testOperationResultDominance() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder target = body.entryBlock().block();
        Op.Result value = target.add(constant(JavaType.INT, 1));
        target.add(return_());
        body.entryBlock().add(JavaOp.neg(value));
        body.entryBlock().add(branch(target.reference()));
        assertEquals("""
                Use of value is not dominated by value
                ^block_1:
                  %1 : java.type:"int" = constant @1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ value declared here
                  return;
                ^block_0:
                  %0 : java.type:"int" = neg %1;
                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ invalid use
                  branch ^block_1;
                """, assertThrows(IllegalStateException.class, () -> func("f", body)).getMessage());
    }

    @Test
    public void testMissingQuotedValue() {
        Body.Builder body = Body.Builder.of(null, functionType(JavaType.INT, JavaType.INT));
        Block.Parameter parameter = body.entryBlock().parameters().getFirst();
        Op.Result result = body.entryBlock().add(JavaOp.neg(parameter));
        body.entryBlock().add(return_(result));
        func("f", body);
        assertEquals("""
                Cannot create a quoted value because no runtime value was provided for this operand
                ^block_0(%0 : java.type:"int"):
                         ^~ operand without runtime value
                  %1 : java.type:"int" = neg %0;
                  return %1;
                """, assertThrows(IllegalArgumentException.class,
                () -> new Quoted<>(result.op(), Map.of())).getMessage());
    }

    @Test
    public void testMissingQuotedCapturedValue() {
        Body.Builder body = Body.Builder.of(null, functionType(JavaType.VOID, JavaType.INT));
        Block.Parameter parameter = body.entryBlock().parameters().getFirst();
        CoreOp.QuotedOp quoted = CoreOp.quoted(body, _ -> JavaOp.neg(parameter));
        body.entryBlock().add(quoted);
        body.entryBlock().add(return_());
        func("f", body);
        assertEquals("""
                Cannot create a quoted value because no runtime value was provided for this captured value
                ^block_0(%0 : java.type:"int"):
                         ^~ captured value without runtime value
                  %1 : java.type:"jdk.incubator.code.Quoted<jdk.incubator.code.Op>" = quoted;
                  return;
                """, assertThrows(IllegalArgumentException.class,
                () -> new Quoted<>(quoted, Map.of())).getMessage());
    }

    private static CoreOp.FuncOp constantModel() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        body.entryBlock().add(constant(JavaType.INT, 1));
        body.entryBlock().add(return_());
        return func("f", body);
    }

    private static Block.Reference builtReference() {
        Body.Builder body = Body.Builder.of(null, FUNCTION_TYPE_VOID);
        Block.Builder target = body.entryBlock().block();
        Block.Reference reference = target.reference();
        body.entryBlock().add(branch(reference));
        target.add(return_());
        func("f", body);
        return reference;
    }
}
