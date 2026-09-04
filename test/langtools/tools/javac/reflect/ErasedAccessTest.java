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
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

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
                %3 : java.type:"java.lang.Object" = invoke %2 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
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
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %9 : java.type:"java.lang.String" = invoke %8 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                        yield %9;
                    };
                %10 : java.type:"boolean" = instanceof %4 @java.type:"java.lang.Object";
                return %10;
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
                %4 : java.type:"int" = invoke %3 @java.ref:"java.lang.String::hashCode():int";
                return %4;
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
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %9 : java.type:"java.lang.String" = invoke %8 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
                        yield %9;
                    };
                %10 : java.type:"int" = invoke %4 @java.ref:"java.lang.String::hashCode():int";
                return %10;
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
                %3 : java.type:"java.lang.Object" = invoke %2 @java.ref:"ErasedAccessTest$Box::get():java.lang.Object";
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
                %3 : java.type:"java.lang.Object" = field.load %2 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
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
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %9 : java.type:"java.lang.String" = field.load %8 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                        yield %9;
                    };
                %10 : java.type:"boolean" = instanceof %4 @java.type:"java.lang.Object";
                return %10;
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
                %4 : java.type:"int" = invoke %3 @java.ref:"java.lang.String::hashCode():int";
                return %4;
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
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"ErasedAccessTest$Box<java.lang.String>" = var.load %3;
                        %9 : java.type:"java.lang.String" = field.load %8 @java.ref:"ErasedAccessTest$Box::x:java.lang.Object";
                        yield %9;
                    };
                %10 : java.type:"int" = invoke %4 @java.ref:"java.lang.String::hashCode():int";
                return %10;
            };
            """)
    static int field_chainedCallCond(boolean c, Box<String> xs) {
        return (c ? xs.x : xs.x).hashCode();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static Box<String> pollutedBox() {
        return (Box<String>)new Box(1);
    }

    static class Box<X> {
        X x;
        Box(X x) { this.x = x; }
        X get() { return x; }
    }

    public static void main(String[] args) throws ReflectiveOperationException {
        for (Method m : ErasedAccessTest.class.getDeclaredMethods()) {
            if (m.isAnnotationPresent(Reflect.class)) {
                FuncOp model = Op.ofMethod(m).get();
                System.out.println(model.toText());
                boolean reflectionResult = invokeReflection(m);
                boolean modelResult = invokeModel(model);
                if (reflectionResult != modelResult) {
                    if (reflectionResult) {
                        throw new AssertionError("Unexpected ClassCastException when executing model");
                    } else {
                        throw new AssertionError("Missing ClassCastException when executing model");
                    }
                }
            }
        }
    }

    static boolean invokeReflection(Method m) {
        Object[] invokeArgs = m.getParameterTypes().length == 1 ?
                new Object[] { pollutedBox() } :
                new Object[] { true, pollutedBox() };
        try {
            m.invoke(null, invokeArgs);
            return true;
        } catch (ReflectiveOperationException e) {
            if (!(e.getCause() instanceof ClassCastException)) {
                throw new AssertionError("unexpected exception", e);
            }
            return false;
        }
    }

    static boolean invokeModel(FuncOp funcOp) {
        MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                funcOp);
        Object[] invokeArgs = funcOp.parameters().size() == 1 ?
                new Object[] { pollutedBox() } :
                new Object[] { true, pollutedBox() };
        try {
            handle.invokeWithArguments(invokeArgs);
            return true;
        } catch (ClassCastException e) {
            return false;
        } catch (Throwable ex) {
            throw new AssertionError("unexpected exception", ex);
        }
    }
}
