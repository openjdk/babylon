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

import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.classfile.ClassFile;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.constant.*;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.StringConcatFactory;
import java.util.List;

/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.internal
 * @library ../../lib
 * @enablePreview
 * @run junit TestLiftCustomBytecode
 */

public class TestLiftCustomBytecode {

    @Test
    public void testBackJumps() throws Throwable {
        CoreOp.FuncOp f = getFuncOp(ClassFile.of().build(ClassDesc.of("BackJumps"), clb ->
                clb.withMethodBody("backJumps", MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_int), ClassFile.ACC_STATIC, cob -> {
                    Label l1 = cob.newLabel();
                    Label l2 = cob.newLabel();
                    Label l3 = cob.newLabel();
                    Label l4 = cob.newLabel();
                    // Code wrapped in back jumps requires multiple passes and block skipping
                    cob.goto_(l1)
                       .labelBinding(l2)
                       .goto_(l3)
                       .labelBinding(l4)
                       .iload(0)
                       .ireturn()
                       .labelBinding(l1)
                       .goto_(l2)
                       .labelBinding(l3)
                       .goto_(l4);
                })), "backJumps");

        Assertions.assertEquals(42, (int) Interpreter.invoke(MethodHandles.lookup(), f, 42));
    }

    @Test
    public void testDeepStackJump() throws Throwable {
        CoreOp.FuncOp f = getFuncOp(ClassFile.of().build(ClassDesc.of("DeepStackJump"), clb ->
                clb.withMethodBody("deepStackJump", MethodTypeDesc.of(ConstantDescs.CD_long), ClassFile.ACC_STATIC, cob -> {
                    Label l = cob.newLabel();
                    cob.lconst_1().iconst_1().iconst_2()
                       .goto_(l)
                       .labelBinding(l)
                       .iadd().i2l().ladd()
                       .lreturn();
                })), "deepStackJump");

        Assertions.assertEquals(4, (long) Interpreter.invoke(MethodHandles.lookup(), f));
    }

    public record TestRecord(int i, String s) {
    }

    @Test
    public void testObjectMethodsIndy() throws Throwable {
        byte[] testRecord = TestRecord.class.getResourceAsStream("TestLiftCustomBytecode$TestRecord.class").readAllBytes();
        CoreOp.FuncOp toString = getFuncOp(testRecord, "toString");
        CoreOp.FuncOp hashCode = getFuncOp(testRecord, "hashCode");
        CoreOp.FuncOp equals = getFuncOp(testRecord, "equals");

        TestRecord tr1 = new TestRecord(1, "hi"), tr2 = new TestRecord(2, "bye"), tr3 = new TestRecord(1, "hi");
        MethodHandles.Lookup lookup = MethodHandles.lookup();
        Assertions.assertEquals(tr1.toString(), (String)Interpreter.invoke(lookup, toString, tr1));
        Assertions.assertEquals(tr2.toString(), (String)Interpreter.invoke(lookup, toString, tr2));
        Assertions.assertEquals(tr1.hashCode(), (int)Interpreter.invoke(lookup, hashCode, tr1));
        Assertions.assertEquals(tr2.hashCode(), (int)Interpreter.invoke(lookup, hashCode, tr2));
        Assertions.assertTrue((boolean)Interpreter.invoke(lookup, equals, tr1, tr1));
        Assertions.assertFalse((boolean)Interpreter.invoke(lookup, equals, tr1, tr2));
        Assertions.assertTrue((boolean)Interpreter.invoke(lookup, equals, tr1, tr3));
        Assertions.assertFalse((boolean)Interpreter.invoke(lookup, equals, tr1, "hello"));
    }

    @Test
    public void testConstantBootstrapsCondy() throws Throwable {
        byte[] testCondy = ClassFile.of().build(ClassDesc.of("TestCondy"), clb ->
                clb.withMethodBody("condyMethod", MethodTypeDesc.of(ConstantDescs.CD_Class), ClassFile.ACC_STATIC, cob ->
                        cob.ldc(DynamicConstantDesc.ofNamed(
                                ConstantDescs.ofConstantBootstrap(ConstantDescs.CD_ConstantBootstraps, "primitiveClass", ConstantDescs.CD_Class),
                                int.class.descriptorString(),
                                ConstantDescs.CD_Class))
                           .areturn()));

        CoreOp.FuncOp primitiveInteger = getFuncOp(testCondy, "condyMethod");

        MethodHandles.Lookup lookup = MethodHandles.lookup();
        Assertions.assertEquals(int.class, (Class)Interpreter.invoke(lookup, primitiveInteger));
    }

    @Test
    public void testStringMakeConcat() throws Throwable {
        byte[] testStringMakeConcat = ClassFile.of().build(ClassDesc.of("TestStringMakeConcat"), clb ->
                clb.withMethodBody("concatMethod", MethodTypeDesc.of(ConstantDescs.CD_String), ClassFile.ACC_STATIC, cob ->
                        cob.ldc("A").ldc("B").ldc("C")
                           .invokedynamic(DynamicCallSiteDesc.of(
                                ConstantDescs.ofCallsiteBootstrap(StringConcatFactory.class.describeConstable().get(), "makeConcat", ConstantDescs.CD_CallSite),
                                MethodTypeDesc.of(ConstantDescs.CD_String, ConstantDescs.CD_String, ConstantDescs.CD_String, ConstantDescs.CD_String)))
                           .areturn()));

        CoreOp.FuncOp concatMethod = getFuncOp(testStringMakeConcat, "concatMethod");

        MethodHandles.Lookup lookup = MethodHandles.lookup();
        Assertions.assertEquals("ABC", (String)Interpreter.invoke(lookup, concatMethod));
    }

    @Test
    public void testStandaloneNumericCompare() throws Throwable {
        var lookup = MethodHandles.lookup();
        for (var opcode : List.of(Opcode.LCMP, Opcode.FCMPL, Opcode.FCMPG, Opcode.DCMPL, Opcode.DCMPG)) {
            var compare = OperatorInstruction.of(opcode);
            var kind = compare.typeKind();
            var signature = MethodTypeDesc.of(ConstantDescs.CD_int, kind.upperBound(), kind.upperBound());
            byte[] data = ClassFile.of().build(ClassDesc.of("StandaloneNumericCompare"), clb -> clb
                    .withMethodBody("compare", signature, ClassFile.ACC_STATIC, cob -> cob
                            .loadLocal(kind, 0)
                            .loadLocal(kind, kind.slotSize())
                            .with(compare).ireturn()));
            var hidden = lookup.defineHiddenClass(data, true);
            var original = hidden.findStatic(hidden.lookupClass(), "compare", signature.resolveConstantDesc(lookup));
            var model = getFuncOp(data, "compare");
            var generated = BytecodeGenerator.generate(lookup, model);
            var values = switch (kind) {
                case LONG -> List.of(Long.MIN_VALUE, 0L, Long.MAX_VALUE);
                // @@@ Float.NaN is excluded
                case FLOAT -> List.of(Float.NEGATIVE_INFINITY, -Float.MIN_VALUE, -0.0f, 0.0f, Float.MIN_VALUE, Float.POSITIVE_INFINITY);
                // @@@ Double.NaN is excluded
                default -> List.of(Double.NEGATIVE_INFINITY, -Double.MIN_VALUE, -0.0, 0.0, Double.MIN_VALUE, Double.POSITIVE_INFINITY);
            };
            for (Object a : values) {
                for (Object b : values) {
                    Object expected = original.invokeWithArguments(a, b);
                    Assertions.assertEquals(expected, Interpreter.invoke(lookup, model, a, b));
                    Assertions.assertEquals(expected, generated.invokeWithArguments(a, b));
                }
            }
        }
    }

    static CoreOp.FuncOp getFuncOp(byte[] classdata, String method) {
        CoreOp.FuncOp flift = BytecodeLift.lift(classdata, method);
        System.out.println(flift.toText());
        return flift;
    }
}
