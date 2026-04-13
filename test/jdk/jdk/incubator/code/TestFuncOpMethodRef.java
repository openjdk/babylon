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
 * @modules jdk.incubator.code/jdk.incubator.code.internal
 * @library lib
 * @run junit TestFuncOpMethodRef
 */

import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.DialectFactory;
import jdk.incubator.code.internal.OpBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.LinkedHashMap;
import java.util.Map;

import static jdk.incubator.code.dialect.core.CoreOp.*;

public class TestFuncOpMethodRef {
    // create, use, write, parse, use

    @Test
    void test() {
        FuncOp f = func("f", CoreType.FUNCTION_TYPE_VOID).body(b -> {
            b.op(return_());
        });
        Assertions.assertTrue(f.mref().isEmpty());

        ModuleOp mop = OpBuilder.createBuilderFunctions(new LinkedHashMap<>(Map.of(f.funcName(), f)),
                b -> b.op(JavaOp.fieldLoad(
                        FieldRef.field(JavaOp.class, "JAVA_DIALECT_FACTORY", DialectFactory.class))));
        CoreOp.FuncOp cf = (CoreOp.FuncOp) Interpreter.invoke(MethodHandles.lookup(),
                mop.transform(CodeTransformer.LOWERING_TRANSFORMER).functionTable().get(f.funcName()));
        Assertions.assertTrue(cf.mref().isEmpty());
    }

    private final JavaType thisType = JavaType.type(this.getClass().describeConstable().get());

    @Test
    void test2() {
        MethodRef mr = MethodRef.method(thisType, "f", FunctionType.FUNCTION_TYPE_VOID);
        FuncOp f = func(mr.refType(), mr.name(), mr.type()).body(b -> {
            b.op(return_());
        });
        Assertions.assertTrue(f.mref().isPresent());
        Assertions.assertEquals(mr, f.mref().get());

        ModuleOp mop = OpBuilder.createBuilderFunctions(new LinkedHashMap<>(Map.of(f.funcName(), f)),
                b -> b.op(JavaOp.fieldLoad(
                        FieldRef.field(JavaOp.class, "JAVA_DIALECT_FACTORY", DialectFactory.class))));
        CoreOp.FuncOp cf = (CoreOp.FuncOp) Interpreter.invoke(MethodHandles.lookup(),
                mop.transform(CodeTransformer.LOWERING_TRANSFORMER).functionTable().get(f.funcName()));
        Assertions.assertTrue(cf.mref().isPresent());

        Assertions.assertEquals(f.mref().get(), cf.mref().get());
    }

    @Test
    void test3() {
        MethodRef mr = MethodRef.method(thisType, "f", FunctionType.FUNCTION_TYPE_VOID);
        FuncOp f = func(mr).body(b -> {
            b.op(return_());
        });
        Assertions.assertTrue(f.mref().isPresent());
        Assertions.assertEquals(mr, f.mref().get());

        ModuleOp mop = OpBuilder.createBuilderFunctions(new LinkedHashMap<>(Map.of(f.funcName(), f)),
                b -> b.op(JavaOp.fieldLoad(
                        FieldRef.field(JavaOp.class, "JAVA_DIALECT_FACTORY", DialectFactory.class))));
        CoreOp.FuncOp cf = (CoreOp.FuncOp) Interpreter.invoke(MethodHandles.lookup(),
                mop.transform(CodeTransformer.LOWERING_TRANSFORMER).functionTable().get(f.funcName()));
        Assertions.assertTrue(cf.mref().isPresent());

        Assertions.assertEquals(f.mref().get(), cf.mref().get());
    }
}