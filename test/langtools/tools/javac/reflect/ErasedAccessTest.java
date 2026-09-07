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

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.CodeModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.TypeCheckInstruction;
import java.lang.constant.ClassDesc;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.List;

/*
 * @test
 * @summary Smoke test for timing of synthetic erasure casts
 * @modules jdk.incubator.code
 * @build ErasedAccessTest
 * @run main ErasedAccessTest
 * @run main CodeReflectionTester ErasedAccessTest
 */

public class ErasedAccessTest {
    @Reflect
    @IR("""
            func @"method_typeTest" (%0 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"boolean" -> {
                %1 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %0 @"xs";
                %2 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %1;
                %3 : java.type:"java.lang.String" = invoke %2 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                %4 : java.type:"boolean" = instanceof %3 @java.type:"java.lang.Object";
                return %4;
            };
            """)
    static boolean method_typeTest(Box<String> xs) {
        return xs.get() instanceof Object;
    }

    @Reflect
    @IR("""
            func @"method_typeTestCond" (%0 : java.type:"boolean", %1 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"boolean" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"c";
                %3 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %1 @"xs";
                %4 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %7 : java.type:"java.lang.String" = invoke %6 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                        %8 : java.type:"java.lang.String" = cast %7 @java.type:"java.lang.String";
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %10 : java.type:"java.lang.String" = invoke %9 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                        %11 : java.type:"java.lang.String" = cast %10 @java.type:"java.lang.String";
                        yield %11;
                    };
                %12 : java.type:"boolean" = instanceof %4 @java.type:"java.lang.Object";
                return %12;
            };
            """)
    static boolean method_typeTestCond(boolean c, Box<String> xs) {
        return (c ? xs.get() : xs.get()) instanceof Object;
    }

    @Reflect
    @IR("""
            func @"method_chainedCall" (%0 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"int" -> {
                %1 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %0 @"xs";
                %2 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %1;
                %3 : java.type:"java.lang.String" = invoke %2 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                %4 : java.type:"java.lang.String" = cast %3 @java.type:"java.lang.String";
                %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.String::hashCode():int";
                return %5;
            };
            """)
    static int method_chainedCall(Box<String> xs) {
        return xs.get().hashCode();
    }

    @Reflect
    @IR("""
            func @"method_chainedCallCond" (%0 : java.type:"boolean", %1 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"int" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"c";
                %3 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %1 @"xs";
                %4 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %7 : java.type:"java.lang.String" = invoke %6 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                        %8 : java.type:"java.lang.String" = cast %7 @java.type:"java.lang.String";
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %10 : java.type:"java.lang.String" = invoke %9 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                        %11 : java.type:"java.lang.String" = cast %10 @java.type:"java.lang.String";
                        yield %11;
                    };
                %12 : java.type:"int" = invoke %4 @java.ref:"java.lang.String::hashCode():int";
                return %12;
            };
            """)
    static int method_chainedCallCond(boolean c, Box<String> xs) {
        return (c ? xs.get() : xs.get()).hashCode();
    }

    @Reflect
    @IR("""
            func @"method_exec" (%0 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"void" -> {
                %1 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %0 @"xs";
                %2 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %1;
                %3 : java.type:"java.lang.String" = invoke %2 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                return;
            };
            """)
    static void method_exec(Box<String> xs) {
        xs.get();
    }

    @Reflect
    @IR("""
            func @"field_typeTest" (%0 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"boolean" -> {
                %1 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %0 @"xs";
                %2 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %1;
                %3 : java.type:"java.lang.String" = field.load %2 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                %4 : java.type:"boolean" = instanceof %3 @java.type:"java.lang.Object";
                return %4;
            };
            """)
    static boolean field_typeTest(Box<String> xs) {
        return xs.x instanceof Object;
    }

    @Reflect
    @IR("""
            func @"field_typeTestCond" (%0 : java.type:"boolean", %1 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"boolean" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"c";
                %3 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %1 @"xs";
                %4 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %7 : java.type:"java.lang.String" = field.load %6 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                        %8 : java.type:"java.lang.String" = cast %7 @java.type:"java.lang.String";
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %10 : java.type:"java.lang.String" = field.load %9 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                        %11 : java.type:"java.lang.String" = cast %10 @java.type:"java.lang.String";
                        yield %11;
                    };
                %12 : java.type:"boolean" = instanceof %4 @java.type:"java.lang.Object";
                return %12;
            };
            """)
    static boolean field_typeTestCond(boolean c, Box<String> xs) {
        return (c ? xs.x : xs.x) instanceof Object;
    }

    @Reflect
    @IR("""
            func @"field_chainedCall" (%0 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"int" -> {
                %1 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %0 @"xs";
                %2 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %1;
                %3 : java.type:"java.lang.String" = field.load %2 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                %4 : java.type:"java.lang.String" = cast %3 @java.type:"java.lang.String";
                %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.String::hashCode():int";
                return %5;
            };
            """)
    static int field_chainedCall(Box<String> xs) {
        return xs.x.hashCode();
    }

    @Reflect
    @IR("""
            func @"field_chainedCallCond" (%0 : java.type:"boolean", %1 : java.type:"ErasedAccessTest$Box<java.lang.String>")java.type:"int" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"c";
                %3 : Var<java.type:"ErasedAccessTest$Box<java.lang.String>"> = var %1 @"xs";
                %4 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %7 : java.type:"java.lang.String" = field.load %6 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                        %8 : java.type:"java.lang.String" = cast %7 @java.type:"java.lang.String";
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %10 : java.type:"java.lang.String" = field.load %9 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                        %11 : java.type:"java.lang.String" = cast %10 @java.type:"java.lang.String";
                        yield %11;
                    };
                %12 : java.type:"int" = invoke %4 @java.ref:"java.lang.String::hashCode():int";
                return %12;
            };
            """)
    static int field_chainedCallCond(boolean c, Box<String> xs) {
        return (c ? xs.x : xs.x).hashCode();
    }

    static class Box<X> {
        X x;
        Box(X x) { this.x = x; }
        X get() { return x; }
    }

    static final String TEST_CLASSES_DIR = System.getProperty("test.classes", ".");

    public static void main(String[] args) throws ReflectiveOperationException, IOException {
        ClassModel mod = ClassFile.of().parse(Path.of(TEST_CLASSES_DIR, "ErasedAccessTest.class"));
        for (Method m : ErasedAccessTest.class.getDeclaredMethods()) {
            if (m.isAnnotationPresent(Reflect.class)) {
                FuncOp model = Op.ofMethod(m).get();
                System.out.println(model.toText());
                CodeModel bytecode = mod.methods().stream()
                        .filter(mm -> mm.methodName().stringValue().equals(m.getName()))
                        .findAny()
                        .flatMap(MethodModel::code)
                        .orElseThrow(() -> new AssertionError("Code model not found for method " + m.getName()));
                List<ClassDesc> modelCasts = modelCasts(model);
                List<ClassDesc> bytecodeCasts = bytecodeCasts(bytecode);
                if (!modelCasts.equals(bytecodeCasts)) {
                    throw new AssertionError("Casts do not match for method " + m.getName() +
                            "\nbytecode casts = " + bytecodeCasts +
                            "\nmodel casts = " + modelCasts);
                }
            }
        }
    }

    static List<ClassDesc> bytecodeCasts(CodeModel m) {
        return m.elementStream()
                .filter(i -> i instanceof TypeCheckInstruction tci && tci.opcode() == Opcode.CHECKCAST)
                .map(TypeCheckInstruction.class::cast)
                .map(i -> i.type().asSymbol())
                .toList();
    }

    static List<ClassDesc> modelCasts(FuncOp model) {
        return model.elements()
                .filter(JavaOp.CastOp.class::isInstance)
                .map(JavaOp.CastOp.class::cast)
                .map(c -> ((JavaType) c.targetType()).toNominalDescriptor())
                .toList();
    }
}
