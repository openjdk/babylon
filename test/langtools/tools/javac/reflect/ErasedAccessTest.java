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
import java.io.Serializable;
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

    static class Unbounded<X, T extends Throwable> {
        X x;
        T t;

        X getX() {
            return x;
        }
        T getT() {
            return t;
        }
    }

    static class UnboundedInteger extends Unbounded<Integer, WrongThreadException> {

        @IR("""
                func @"testInstanceof" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %4 : java.type:"boolean" = instanceof %3 @java.type:"java.lang.Integer";
                    %5 : Var<java.type:"boolean"> = var %4 @"f_s_s";
                    %6 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %7 : java.type:"java.lang.Integer" = field.load %6 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %8 : java.type:"boolean" = instanceof %7 @java.type:"java.lang.Integer";
                    %9 : Var<java.type:"boolean"> = var %8 @"f_q_s";
                    %10 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %11 : java.type:"boolean" = instanceof %10 @java.type:"java.lang.Integer";
                    %12 : Var<java.type:"boolean"> = var %11 @"m_s_s";
                    %13 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = invoke %13 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %15 : java.type:"boolean" = instanceof %14 @java.type:"java.lang.Integer";
                    %16 : Var<java.type:"boolean"> = var %15 @"m_q_s";
                    return;
                };
                """)
        @Reflect
        void testInstanceof(UnboundedInteger test) {
            // simple field name
            boolean f_s_s = x instanceof Integer;

            // qualified field name
            boolean f_q_s = test.x instanceof Integer;

            // simple method name
            boolean m_s_s = getX() instanceof Integer;

            // qualified method name
            boolean m_q_s = test.getX() instanceof Integer;
        }

        @IR("""
                func @"testInstanceofCond" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger", %2 : java.type:"boolean")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"boolean"> = var %2 @"cond";
                    %5 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = var.load %4;
                            yield %6;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                            yield %8;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %9 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                            yield %10;
                        };
                    %11 : java.type:"boolean" = instanceof %5 @java.type:"java.lang.Object";
                    %12 : Var<java.type:"boolean"> = var %11 @"f_s_o";
                    %13 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %14 : java.type:"boolean" = var.load %4;
                            yield %14;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %15 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %16 : java.type:"java.lang.Integer" = field.load %15 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %17 : java.type:"java.lang.Integer" = cast %16 @java.type:"java.lang.Integer";
                            yield %17;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %18 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %19 : java.type:"java.lang.Integer" = field.load %18 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %20 : java.type:"java.lang.Integer" = cast %19 @java.type:"java.lang.Integer";
                            yield %20;
                        };
                    %21 : java.type:"boolean" = instanceof %13 @java.type:"java.lang.Object";
                    %22 : Var<java.type:"boolean"> = var %21 @"f_q_o";
                    %23 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %24 : java.type:"boolean" = var.load %4;
                            yield %24;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %25 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %26 : java.type:"java.lang.Integer" = cast %25 @java.type:"java.lang.Integer";
                            yield %26;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %27 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %28 : java.type:"java.lang.Integer" = cast %27 @java.type:"java.lang.Integer";
                            yield %28;
                        };
                    %29 : java.type:"boolean" = instanceof %23 @java.type:"java.lang.Object";
                    %30 : Var<java.type:"boolean"> = var %29 @"m_s_o";
                    %31 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %32 : java.type:"boolean" = var.load %4;
                            yield %32;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %33 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %34 : java.type:"java.lang.Integer" = invoke %33 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %35 : java.type:"java.lang.Integer" = cast %34 @java.type:"java.lang.Integer";
                            yield %35;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %36 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %37 : java.type:"java.lang.Integer" = invoke %36 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %38 : java.type:"java.lang.Integer" = cast %37 @java.type:"java.lang.Integer";
                            yield %38;
                        };
                    %39 : java.type:"boolean" = instanceof %31 @java.type:"java.lang.Object";
                    %40 : Var<java.type:"boolean"> = var %39 @"m_q_o";
                    return;
                };
                """)
        @Reflect
        void testInstanceofCond(UnboundedInteger test, boolean cond) {
            // simple field name
            boolean f_s_o = (cond ? x : x) instanceof Object;

            // qualified field name
            boolean f_q_o = (cond ? test.x : test.x) instanceof Object;

            // simple method name
            boolean m_s_o = (cond ? getX() : getX()) instanceof Object;

            // qualified method name
            boolean m_q_o = (cond ? test.getX() : test.getX()) instanceof Object;
        }

        @IR("""
                func @"testExec" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %4 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %5 : java.type:"java.lang.Integer" = invoke %4 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    return;
                };
                """)
        @Reflect
        void testExec(UnboundedInteger test) {
            // simple method name
            getX();

            // qualified method name
            test.getX();
        }

        @IR("""
                func @"testChainedCall" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %4 : java.type:"java.lang.Integer" = cast %3 @java.type:"java.lang.Integer";
                    %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.Integer::hashCode():int";
                    %6 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %7 : java.type:"java.lang.Integer" = field.load %6 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                    %9 : java.type:"int" = invoke %8 @java.ref:"java.lang.Integer::hashCode():int";
                    %10 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %11 : java.type:"java.lang.Integer" = cast %10 @java.type:"java.lang.Integer";
                    %12 : java.type:"int" = invoke %11 @java.ref:"java.lang.Integer::hashCode():int";
                    %13 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = invoke %13 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %15 : java.type:"java.lang.Integer" = cast %14 @java.type:"java.lang.Integer";
                    %16 : java.type:"int" = invoke %15 @java.ref:"java.lang.Integer::hashCode():int";
                    return;
                };
                """)
        @Reflect
        void testChainedCall(UnboundedInteger test) {
            // simple field name
            x.hashCode();

            // qualified field name
            test.x.hashCode();

            // simple method name
            getX().hashCode();

            // qualified method name
            test.getX().hashCode();
        }

        @IR("""
                func @"testChainedCallCond" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger", %2 : java.type:"boolean")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"boolean"> = var %2 @"cond";
                    %5 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = var.load %4;
                            yield %6;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                            yield %8;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %9 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                            yield %10;
                        };
                    %11 : java.type:"int" = invoke %5 @java.ref:"java.lang.Integer::hashCode():int";
                    %12 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %13 : java.type:"boolean" = var.load %4;
                            yield %13;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %14 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %15 : java.type:"java.lang.Integer" = field.load %14 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %16 : java.type:"java.lang.Integer" = cast %15 @java.type:"java.lang.Integer";
                            yield %16;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %17 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %18 : java.type:"java.lang.Integer" = field.load %17 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %19 : java.type:"java.lang.Integer" = cast %18 @java.type:"java.lang.Integer";
                            yield %19;
                        };
                    %20 : java.type:"int" = invoke %12 @java.ref:"java.lang.Integer::hashCode():int";
                    %21 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %22 : java.type:"boolean" = var.load %4;
                            yield %22;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %23 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %24 : java.type:"java.lang.Integer" = cast %23 @java.type:"java.lang.Integer";
                            yield %24;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %25 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %26 : java.type:"java.lang.Integer" = cast %25 @java.type:"java.lang.Integer";
                            yield %26;
                        };
                    %27 : java.type:"int" = invoke %21 @java.ref:"java.lang.Integer::hashCode():int";
                    %28 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %29 : java.type:"boolean" = var.load %4;
                            yield %29;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %30 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %31 : java.type:"java.lang.Integer" = invoke %30 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %32 : java.type:"java.lang.Integer" = cast %31 @java.type:"java.lang.Integer";
                            yield %32;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %33 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %34 : java.type:"java.lang.Integer" = invoke %33 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %35 : java.type:"java.lang.Integer" = cast %34 @java.type:"java.lang.Integer";
                            yield %35;
                        };
                    %36 : java.type:"int" = invoke %28 @java.ref:"java.lang.Integer::hashCode():int";
                    return;
                };
                """)
        @Reflect
        void testChainedCallCond(UnboundedInteger test, boolean cond) {
            // simple field name
            (cond ? x : x).hashCode();

            // qualified field name
            (cond ? test.x : test.x).hashCode();

            // simple method name
            (cond ? getX() : getX()).hashCode();

            // qualified method name
            (cond ? test.getX() : test.getX()).hashCode();
        }

        @IR("""
                func @"testAssign" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : Var<java.type:"java.lang.Number"> = var @"n";
                    %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                    %6 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    var.store %3 %6;
                    %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %8 : java.type:"java.lang.Number" = cast %7 @java.type:"java.lang.Number";
                    var.store %4 %8;
                    %9 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                    var.store %5 %10;
                    %11 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %12 : java.type:"java.lang.Integer" = field.load %11 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    var.store %3 %12;
                    %13 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = field.load %13 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %15 : java.type:"java.lang.Number" = cast %14 @java.type:"java.lang.Number";
                    var.store %4 %15;
                    %16 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %17 : java.type:"java.lang.Integer" = field.load %16 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %18 : java.type:"java.lang.Integer" = cast %17 @java.type:"java.lang.Integer";
                    var.store %5 %18;
                    %19 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    var.store %3 %19;
                    %20 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %21 : java.type:"java.lang.Number" = cast %20 @java.type:"java.lang.Number";
                    var.store %4 %21;
                    %22 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %23 : java.type:"java.lang.Integer" = cast %22 @java.type:"java.lang.Integer";
                    var.store %5 %23;
                    %24 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %25 : java.type:"java.lang.Integer" = invoke %24 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    var.store %3 %25;
                    %26 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %27 : java.type:"java.lang.Integer" = invoke %26 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %28 : java.type:"java.lang.Number" = cast %27 @java.type:"java.lang.Number";
                    var.store %4 %28;
                    %29 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %30 : java.type:"java.lang.Integer" = invoke %29 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %31 : java.type:"java.lang.Integer" = cast %30 @java.type:"java.lang.Integer";
                    var.store %5 %31;
                    return;
                };
                """)
        @Reflect
        void testAssign(UnboundedInteger test) {
            Object o; Number n; Integer i;

            // simple field name
            o = x;
            n = x;
            i = x;

            // qualified field name
            o = test.x;
            n = test.x;
            i = test.x;

            // simple method name
            o = getX();
            n = getX();
            i = getX();

            // qualified method name
            o = test.getX();
            n = test.getX();
            i = test.getX();
        }

        @IR("""
                func @"testArrayInit" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object[]"> = var @"o";
                    %4 : Var<java.type:"java.lang.Number[]"> = var @"n";
                    %5 : Var<java.type:"java.lang.Integer[]"> = var @"i";
                    %6 : java.type:"int" = constant @1;
                    %7 : java.type:"java.lang.Object[]" = new %6 @java.ref:"java.lang.Object[]::(int)";
                    %8 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %9 : java.type:"int" = constant @0;
                    array.store %7 %9 %8;
                    var.store %3 %7;
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"java.lang.Number[]" = new %10 @java.ref:"java.lang.Number[]::(int)";
                    %12 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %13 : java.type:"java.lang.Number" = cast %12 @java.type:"java.lang.Number";
                    %14 : java.type:"int" = constant @0;
                    array.store %11 %14 %13;
                    var.store %4 %11;
                    %15 : java.type:"int" = constant @1;
                    %16 : java.type:"java.lang.Integer[]" = new %15 @java.ref:"java.lang.Integer[]::(int)";
                    %17 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %18 : java.type:"java.lang.Integer" = cast %17 @java.type:"java.lang.Integer";
                    %19 : java.type:"int" = constant @0;
                    array.store %16 %19 %18;
                    var.store %5 %16;
                    %20 : java.type:"int" = constant @1;
                    %21 : java.type:"java.lang.Object[]" = new %20 @java.ref:"java.lang.Object[]::(int)";
                    %22 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %23 : java.type:"java.lang.Integer" = field.load %22 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %24 : java.type:"int" = constant @0;
                    array.store %21 %24 %23;
                    var.store %3 %21;
                    %25 : java.type:"int" = constant @1;
                    %26 : java.type:"java.lang.Number[]" = new %25 @java.ref:"java.lang.Number[]::(int)";
                    %27 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %28 : java.type:"java.lang.Integer" = field.load %27 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %29 : java.type:"java.lang.Number" = cast %28 @java.type:"java.lang.Number";
                    %30 : java.type:"int" = constant @0;
                    array.store %26 %30 %29;
                    var.store %4 %26;
                    %31 : java.type:"int" = constant @1;
                    %32 : java.type:"java.lang.Integer[]" = new %31 @java.ref:"java.lang.Integer[]::(int)";
                    %33 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %34 : java.type:"java.lang.Integer" = field.load %33 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %35 : java.type:"java.lang.Integer" = cast %34 @java.type:"java.lang.Integer";
                    %36 : java.type:"int" = constant @0;
                    array.store %32 %36 %35;
                    var.store %5 %32;
                    %37 : java.type:"int" = constant @1;
                    %38 : java.type:"java.lang.Object[]" = new %37 @java.ref:"java.lang.Object[]::(int)";
                    %39 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %40 : java.type:"int" = constant @0;
                    array.store %38 %40 %39;
                    var.store %3 %38;
                    %41 : java.type:"int" = constant @1;
                    %42 : java.type:"java.lang.Number[]" = new %41 @java.ref:"java.lang.Number[]::(int)";
                    %43 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %44 : java.type:"java.lang.Number" = cast %43 @java.type:"java.lang.Number";
                    %45 : java.type:"int" = constant @0;
                    array.store %42 %45 %44;
                    var.store %4 %42;
                    %46 : java.type:"int" = constant @1;
                    %47 : java.type:"java.lang.Integer[]" = new %46 @java.ref:"java.lang.Integer[]::(int)";
                    %48 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %49 : java.type:"java.lang.Integer" = cast %48 @java.type:"java.lang.Integer";
                    %50 : java.type:"int" = constant @0;
                    array.store %47 %50 %49;
                    var.store %5 %47;
                    %51 : java.type:"int" = constant @1;
                    %52 : java.type:"java.lang.Object[]" = new %51 @java.ref:"java.lang.Object[]::(int)";
                    %53 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %54 : java.type:"java.lang.Integer" = invoke %53 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %55 : java.type:"int" = constant @0;
                    array.store %52 %55 %54;
                    var.store %3 %52;
                    %56 : java.type:"int" = constant @1;
                    %57 : java.type:"java.lang.Number[]" = new %56 @java.ref:"java.lang.Number[]::(int)";
                    %58 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %59 : java.type:"java.lang.Integer" = invoke %58 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %60 : java.type:"java.lang.Number" = cast %59 @java.type:"java.lang.Number";
                    %61 : java.type:"int" = constant @0;
                    array.store %57 %61 %60;
                    var.store %4 %57;
                    %62 : java.type:"int" = constant @1;
                    %63 : java.type:"java.lang.Integer[]" = new %62 @java.ref:"java.lang.Integer[]::(int)";
                    %64 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %65 : java.type:"java.lang.Integer" = invoke %64 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %66 : java.type:"java.lang.Integer" = cast %65 @java.type:"java.lang.Integer";
                    %67 : java.type:"int" = constant @0;
                    array.store %63 %67 %66;
                    var.store %5 %63;
                    return;
                };
                """)
        @Reflect
        void testArrayInit(UnboundedInteger test) {
            Object[] o; Number[] n; Integer[] i;

            // simple field name
            o = new Object[] { x };
            n = new Number[] { x };
            i = new Integer[] { x };

            // qualified field name
            o = new Object[] { test.x };
            n = new Number[] { test.x };
            i = new Integer[] { test.x };

            // simple method name
            o = new Object[] { getX() };
            n = new Number[] { getX() };
            i = new Integer[] { getX() };

            // qualified method name
            o = new Object[] { test.getX() };
            n = new Number[] { test.getX() };
            i = new Integer[] { test.getX() };
        }

        @IR("""
                func @"testCast" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : Var<java.type:"java.lang.Number"> = var @"n";
                    %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                    %6 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    var.store %3 %6;
                    %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %8 : java.type:"java.lang.Number" = cast %7 @java.type:"java.lang.Number";
                    var.store %4 %8;
                    %9 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                    var.store %5 %10;
                    %11 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %12 : java.type:"java.lang.Integer" = field.load %11 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    var.store %3 %12;
                    %13 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = field.load %13 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %15 : java.type:"java.lang.Number" = cast %14 @java.type:"java.lang.Number";
                    var.store %4 %15;
                    %16 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %17 : java.type:"java.lang.Integer" = field.load %16 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %18 : java.type:"java.lang.Integer" = cast %17 @java.type:"java.lang.Integer";
                    var.store %5 %18;
                    %19 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    var.store %3 %19;
                    %20 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %21 : java.type:"java.lang.Number" = cast %20 @java.type:"java.lang.Number";
                    var.store %4 %21;
                    %22 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %23 : java.type:"java.lang.Integer" = cast %22 @java.type:"java.lang.Integer";
                    var.store %5 %23;
                    %24 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %25 : java.type:"java.lang.Integer" = invoke %24 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    var.store %3 %25;
                    %26 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %27 : java.type:"java.lang.Integer" = invoke %26 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %28 : java.type:"java.lang.Number" = cast %27 @java.type:"java.lang.Number";
                    var.store %4 %28;
                    %29 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %30 : java.type:"java.lang.Integer" = invoke %29 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %31 : java.type:"java.lang.Integer" = cast %30 @java.type:"java.lang.Integer";
                    var.store %5 %31;
                    return;
                };
                """)
        @Reflect
        void testCast(UnboundedInteger test) {
            Object o; Number n; Integer i;

            // simple field name
            o = (Object) x;
            n = (Number) x;
            i = (Integer) x;

            // qualified field name
            o = (Object) test.x;
            n = (Number) test.x;
            i = (Integer) test.x;

            // simple method name
            o = (Object) getX();
            n = (Number) getX();
            i = (Integer) getX();

            // qualified method name
            o = (Object) test.getX();
            n = (Number) test.getX();
            i = (Integer) test.getX();
        }

        @IR("""
                func @"testIntersectionCast" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %5 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %4 @java.type:"java.lang.Comparable";
                    %6 : java.type:"java.io.Serializable" = cast %5 @java.type:"java.io.Serializable";
                    %7 : java.type:"java.lang.Number" = cast %6 @java.type:"java.lang.Number";
                    var.store %3 %7;
                    %8 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %9 : java.type:"java.lang.Integer" = field.load %8 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %10 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %9 @java.type:"java.lang.Comparable";
                    %11 : java.type:"java.io.Serializable" = cast %10 @java.type:"java.io.Serializable";
                    %12 : java.type:"java.lang.Number" = cast %11 @java.type:"java.lang.Number";
                    var.store %3 %12;
                    %13 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %14 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %13 @java.type:"java.lang.Comparable";
                    %15 : java.type:"java.io.Serializable" = cast %14 @java.type:"java.io.Serializable";
                    %16 : java.type:"java.lang.Number" = cast %15 @java.type:"java.lang.Number";
                    var.store %3 %16;
                    %17 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %18 : java.type:"java.lang.Integer" = invoke %17 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %19 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %18 @java.type:"java.lang.Comparable";
                    %20 : java.type:"java.io.Serializable" = cast %19 @java.type:"java.io.Serializable";
                    %21 : java.type:"java.lang.Number" = cast %20 @java.type:"java.lang.Number";
                    var.store %3 %21;
                    return;
                };
                """)
        @Reflect
        void testIntersectionCast(UnboundedInteger test) {
            Object o;

            // simple field name
            o = (Number & Comparable<Integer> & Serializable) x;

            // qualified field name
            o = (Number & Comparable<Integer> & Serializable) test.x;

            // simple method name
            o = (Number & Comparable<Integer> & Serializable) getX();

            // qualified method name
            o = (Number & Comparable<Integer> & Serializable) test.getX();
        }

        void o(Object o) { }
        void n(Number n) { }
        void i(Integer i) { }

        @IR("""
                func @"testMethod" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    invoke %0 %3 @java.ref:"ErasedAccessTest$UnboundedInteger::o(java.lang.Object):void";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %5 : java.type:"java.lang.Number" = cast %4 @java.type:"java.lang.Number";
                    invoke %0 %5 @java.ref:"ErasedAccessTest$UnboundedInteger::n(java.lang.Number):void";
                    %6 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %7 : java.type:"java.lang.Integer" = cast %6 @java.type:"java.lang.Integer";
                    invoke %0 %7 @java.ref:"ErasedAccessTest$UnboundedInteger::i(java.lang.Integer):void";
                    %8 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %9 : java.type:"java.lang.Integer" = field.load %8 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    invoke %0 %9 @java.ref:"ErasedAccessTest$UnboundedInteger::o(java.lang.Object):void";
                    %10 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %11 : java.type:"java.lang.Integer" = field.load %10 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %12 : java.type:"java.lang.Number" = cast %11 @java.type:"java.lang.Number";
                    invoke %0 %12 @java.ref:"ErasedAccessTest$UnboundedInteger::n(java.lang.Number):void";
                    %13 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = field.load %13 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %15 : java.type:"java.lang.Integer" = cast %14 @java.type:"java.lang.Integer";
                    invoke %0 %15 @java.ref:"ErasedAccessTest$UnboundedInteger::i(java.lang.Integer):void";
                    %16 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    invoke %0 %16 @java.ref:"ErasedAccessTest$UnboundedInteger::o(java.lang.Object):void";
                    %17 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %18 : java.type:"java.lang.Number" = cast %17 @java.type:"java.lang.Number";
                    invoke %0 %18 @java.ref:"ErasedAccessTest$UnboundedInteger::n(java.lang.Number):void";
                    %19 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %20 : java.type:"java.lang.Integer" = cast %19 @java.type:"java.lang.Integer";
                    invoke %0 %20 @java.ref:"ErasedAccessTest$UnboundedInteger::i(java.lang.Integer):void";
                    %21 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %22 : java.type:"java.lang.Integer" = invoke %21 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    invoke %0 %22 @java.ref:"ErasedAccessTest$UnboundedInteger::o(java.lang.Object):void";
                    %23 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %24 : java.type:"java.lang.Integer" = invoke %23 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %25 : java.type:"java.lang.Number" = cast %24 @java.type:"java.lang.Number";
                    invoke %0 %25 @java.ref:"ErasedAccessTest$UnboundedInteger::n(java.lang.Number):void";
                    %26 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %27 : java.type:"java.lang.Integer" = invoke %26 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %28 : java.type:"java.lang.Integer" = cast %27 @java.type:"java.lang.Integer";
                    invoke %0 %28 @java.ref:"ErasedAccessTest$UnboundedInteger::i(java.lang.Integer):void";
                    return;
                };
                """)
        @Reflect
        void testMethod(UnboundedInteger test) {
            // simple field name
            o(x);
            n(x);
            i(x);

            // qualified field name
            o(test.x);
            n(test.x);
            i(test.x);

            // simple method name
            o(getX());
            n(getX());
            i(getX());

            // qualified method name
            o(test.getX());
            n(test.getX());
            i(test.getX());
        }

        @IR("""
                func @"testWidening" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"long"> = var @"l";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %5 : java.type:"java.lang.Integer" = cast %4 @java.type:"java.lang.Integer";
                    %6 : java.type:"int" = invoke %5 @java.ref:"java.lang.Integer::intValue():int";
                    %7 : java.type:"long" = conv %6;
                    var.store %3 %7;
                    %8 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %9 : java.type:"java.lang.Integer" = field.load %8 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                    %11 : java.type:"int" = invoke %10 @java.ref:"java.lang.Integer::intValue():int";
                    %12 : java.type:"long" = conv %11;
                    var.store %3 %12;
                    %13 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %14 : java.type:"java.lang.Integer" = cast %13 @java.type:"java.lang.Integer";
                    %15 : java.type:"int" = invoke %14 @java.ref:"java.lang.Integer::intValue():int";
                    %16 : java.type:"long" = conv %15;
                    var.store %3 %16;
                    %17 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %18 : java.type:"java.lang.Integer" = invoke %17 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %19 : java.type:"java.lang.Integer" = cast %18 @java.type:"java.lang.Integer";
                    %20 : java.type:"int" = invoke %19 @java.ref:"java.lang.Integer::intValue():int";
                    %21 : java.type:"long" = conv %20;
                    var.store %3 %21;
                    return;
                };
                """)
        @Reflect
        void testWidening(UnboundedInteger test) {
            long l;

            // simple field name
            l = x;

            // qualified field name
            l = test.x;

            // simple method name
            l = getX();

            // qualified method name
            l = test.getX();
        }

        @IR("""
                func @"testAssert" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    assert
                        ()java.type:"boolean" -> {
                            %3 : java.type:"boolean" = constant @false;
                            yield %3;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %5 : java.type:"java.lang.Integer" = cast %4 @java.type:"java.lang.Integer";
                            yield %5;
                        };
                    assert
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = constant @false;
                            yield %6;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %7 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                            %8 : java.type:"java.lang.Integer" = field.load %7 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %9 : java.type:"java.lang.Integer" = cast %8 @java.type:"java.lang.Integer";
                            yield %9;
                        };
                    assert
                        ()java.type:"boolean" -> {
                            %10 : java.type:"boolean" = constant @false;
                            yield %10;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %11 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %12 : java.type:"java.lang.Integer" = cast %11 @java.type:"java.lang.Integer";
                            yield %12;
                        };
                    assert
                        ()java.type:"boolean" -> {
                            %13 : java.type:"boolean" = constant @false;
                            yield %13;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %14 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                            %15 : java.type:"java.lang.Integer" = invoke %14 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %16 : java.type:"java.lang.Integer" = cast %15 @java.type:"java.lang.Integer";
                            yield %16;
                        };
                    return;
                };
                """)
        @Reflect
        void testAssert(UnboundedInteger test) {
            // simple field name
            assert false : x;

            // qualified field name
            assert false : test.x;

            // simple method name
            assert false : getX();

            // qualified method name
            assert false : test.getX();
        }

        @IR("""
                func @"testSynchronized" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %4 : java.type:"java.lang.Integer" = cast %3 @java.type:"java.lang.Integer";
                            yield %4;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %5 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                            %6 : java.type:"java.lang.Integer" = field.load %5 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %7 : java.type:"java.lang.Integer" = cast %6 @java.type:"java.lang.Integer";
                            yield %7;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %8 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %9 : java.type:"java.lang.Integer" = cast %8 @java.type:"java.lang.Integer";
                            yield %9;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %10 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                            %11 : java.type:"java.lang.Integer" = invoke %10 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %12 : java.type:"java.lang.Integer" = cast %11 @java.type:"java.lang.Integer";
                            yield %12;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    return;
                };
                """)
        @Reflect
        @SuppressWarnings("identity")
        void testSynchronized(UnboundedInteger test) {
            // simple field name
            synchronized (x) { };

            // qualified field name
            synchronized (test.x) { };

            // simple method name
            synchronized (getX()) { };

            // qualified method name
            synchronized (test.getX()) { };
        }

        @IR("""
                func @"testYield" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger", %2 : java.type:"int")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"int"> = var %2 @"s";
                    %5 : Var<java.type:"java.lang.Object"> = var @"o";
                    %6 : Var<java.type:"java.lang.Number"> = var @"n";
                    %7 : Var<java.type:"java.lang.Integer"> = var @"i";
                    %8 : java.type:"int" = var.load %4;
                    %9 : java.type:"java.lang.Object" = java.switch.expression %8
                        (%10 : java.type:"int")java.type:"boolean" -> {
                            %11 : java.type:"int" = constant @0;
                            %12 : java.type:"boolean" = eq %10 %11;
                            yield %12;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %13 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %14 : java.type:"java.lang.Integer" = cast %13 @java.type:"java.lang.Integer";
                            yield %14;
                        }
                        ()java.type:"boolean" -> {
                            %15 : java.type:"boolean" = constant @true;
                            yield %15;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %16 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %17 : java.type:"java.lang.Integer" = cast %16 @java.type:"java.lang.Integer";
                            yield %17;
                        };
                    var.store %5 %9;
                    %18 : java.type:"int" = var.load %4;
                    %19 : java.type:"java.lang.Number" = java.switch.expression %18
                        (%20 : java.type:"int")java.type:"boolean" -> {
                            %21 : java.type:"int" = constant @0;
                            %22 : java.type:"boolean" = eq %20 %21;
                            yield %22;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %23 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %24 : java.type:"java.lang.Integer" = cast %23 @java.type:"java.lang.Integer";
                            yield %24;
                        }
                        ()java.type:"boolean" -> {
                            %25 : java.type:"boolean" = constant @true;
                            yield %25;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %26 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %27 : java.type:"java.lang.Integer" = cast %26 @java.type:"java.lang.Integer";
                            yield %27;
                        };
                    var.store %6 %19;
                    %28 : java.type:"int" = var.load %4;
                    %29 : java.type:"java.lang.Integer" = java.switch.expression %28
                        (%30 : java.type:"int")java.type:"boolean" -> {
                            %31 : java.type:"int" = constant @0;
                            %32 : java.type:"boolean" = eq %30 %31;
                            yield %32;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %33 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %34 : java.type:"java.lang.Integer" = cast %33 @java.type:"java.lang.Integer";
                            yield %34;
                        }
                        ()java.type:"boolean" -> {
                            %35 : java.type:"boolean" = constant @true;
                            yield %35;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %36 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %37 : java.type:"java.lang.Integer" = cast %36 @java.type:"java.lang.Integer";
                            yield %37;
                        };
                    var.store %7 %29;
                    %38 : java.type:"int" = var.load %4;
                    %39 : java.type:"java.lang.Object" = java.switch.expression %38
                        (%40 : java.type:"int")java.type:"boolean" -> {
                            %41 : java.type:"int" = constant @0;
                            %42 : java.type:"boolean" = eq %40 %41;
                            yield %42;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %43 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %44 : java.type:"java.lang.Integer" = field.load %43 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %45 : java.type:"java.lang.Integer" = cast %44 @java.type:"java.lang.Integer";
                            yield %45;
                        }
                        ()java.type:"boolean" -> {
                            %46 : java.type:"boolean" = constant @true;
                            yield %46;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %47 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %48 : java.type:"java.lang.Integer" = field.load %47 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %49 : java.type:"java.lang.Integer" = cast %48 @java.type:"java.lang.Integer";
                            yield %49;
                        };
                    var.store %5 %39;
                    %50 : java.type:"int" = var.load %4;
                    %51 : java.type:"java.lang.Number" = java.switch.expression %50
                        (%52 : java.type:"int")java.type:"boolean" -> {
                            %53 : java.type:"int" = constant @0;
                            %54 : java.type:"boolean" = eq %52 %53;
                            yield %54;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %55 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %56 : java.type:"java.lang.Integer" = field.load %55 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %57 : java.type:"java.lang.Integer" = cast %56 @java.type:"java.lang.Integer";
                            yield %57;
                        }
                        ()java.type:"boolean" -> {
                            %58 : java.type:"boolean" = constant @true;
                            yield %58;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %59 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %60 : java.type:"java.lang.Integer" = field.load %59 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %61 : java.type:"java.lang.Integer" = cast %60 @java.type:"java.lang.Integer";
                            yield %61;
                        };
                    var.store %6 %51;
                    %62 : java.type:"int" = var.load %4;
                    %63 : java.type:"java.lang.Integer" = java.switch.expression %62
                        (%64 : java.type:"int")java.type:"boolean" -> {
                            %65 : java.type:"int" = constant @0;
                            %66 : java.type:"boolean" = eq %64 %65;
                            yield %66;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %67 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %68 : java.type:"java.lang.Integer" = field.load %67 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %69 : java.type:"java.lang.Integer" = cast %68 @java.type:"java.lang.Integer";
                            yield %69;
                        }
                        ()java.type:"boolean" -> {
                            %70 : java.type:"boolean" = constant @true;
                            yield %70;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %71 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %72 : java.type:"java.lang.Integer" = field.load %71 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                            %73 : java.type:"java.lang.Integer" = cast %72 @java.type:"java.lang.Integer";
                            yield %73;
                        };
                    var.store %7 %63;
                    %74 : java.type:"int" = var.load %4;
                    %75 : java.type:"java.lang.Object" = java.switch.expression %74
                        (%76 : java.type:"int")java.type:"boolean" -> {
                            %77 : java.type:"int" = constant @0;
                            %78 : java.type:"boolean" = eq %76 %77;
                            yield %78;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %79 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %80 : java.type:"java.lang.Integer" = cast %79 @java.type:"java.lang.Integer";
                            yield %80;
                        }
                        ()java.type:"boolean" -> {
                            %81 : java.type:"boolean" = constant @true;
                            yield %81;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %82 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %83 : java.type:"java.lang.Integer" = cast %82 @java.type:"java.lang.Integer";
                            yield %83;
                        };
                    var.store %5 %75;
                    %84 : java.type:"int" = var.load %4;
                    %85 : java.type:"java.lang.Number" = java.switch.expression %84
                        (%86 : java.type:"int")java.type:"boolean" -> {
                            %87 : java.type:"int" = constant @0;
                            %88 : java.type:"boolean" = eq %86 %87;
                            yield %88;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %89 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %90 : java.type:"java.lang.Integer" = cast %89 @java.type:"java.lang.Integer";
                            yield %90;
                        }
                        ()java.type:"boolean" -> {
                            %91 : java.type:"boolean" = constant @true;
                            yield %91;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %92 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %93 : java.type:"java.lang.Integer" = cast %92 @java.type:"java.lang.Integer";
                            yield %93;
                        };
                    var.store %6 %85;
                    %94 : java.type:"int" = var.load %4;
                    %95 : java.type:"java.lang.Integer" = java.switch.expression %94
                        (%96 : java.type:"int")java.type:"boolean" -> {
                            %97 : java.type:"int" = constant @0;
                            %98 : java.type:"boolean" = eq %96 %97;
                            yield %98;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %99 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %100 : java.type:"java.lang.Integer" = cast %99 @java.type:"java.lang.Integer";
                            yield %100;
                        }
                        ()java.type:"boolean" -> {
                            %101 : java.type:"boolean" = constant @true;
                            yield %101;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %102 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %103 : java.type:"java.lang.Integer" = cast %102 @java.type:"java.lang.Integer";
                            yield %103;
                        };
                    var.store %7 %95;
                    %104 : java.type:"int" = var.load %4;
                    %105 : java.type:"java.lang.Object" = java.switch.expression %104
                        (%106 : java.type:"int")java.type:"boolean" -> {
                            %107 : java.type:"int" = constant @0;
                            %108 : java.type:"boolean" = eq %106 %107;
                            yield %108;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %109 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %110 : java.type:"java.lang.Integer" = invoke %109 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %111 : java.type:"java.lang.Integer" = cast %110 @java.type:"java.lang.Integer";
                            yield %111;
                        }
                        ()java.type:"boolean" -> {
                            %112 : java.type:"boolean" = constant @true;
                            yield %112;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %113 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %114 : java.type:"java.lang.Integer" = invoke %113 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %115 : java.type:"java.lang.Integer" = cast %114 @java.type:"java.lang.Integer";
                            yield %115;
                        };
                    var.store %5 %105;
                    %116 : java.type:"int" = var.load %4;
                    %117 : java.type:"java.lang.Number" = java.switch.expression %116
                        (%118 : java.type:"int")java.type:"boolean" -> {
                            %119 : java.type:"int" = constant @0;
                            %120 : java.type:"boolean" = eq %118 %119;
                            yield %120;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %121 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %122 : java.type:"java.lang.Integer" = invoke %121 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %123 : java.type:"java.lang.Integer" = cast %122 @java.type:"java.lang.Integer";
                            yield %123;
                        }
                        ()java.type:"boolean" -> {
                            %124 : java.type:"boolean" = constant @true;
                            yield %124;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %125 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %126 : java.type:"java.lang.Integer" = invoke %125 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %127 : java.type:"java.lang.Integer" = cast %126 @java.type:"java.lang.Integer";
                            yield %127;
                        };
                    var.store %6 %117;
                    %128 : java.type:"int" = var.load %4;
                    %129 : java.type:"java.lang.Integer" = java.switch.expression %128
                        (%130 : java.type:"int")java.type:"boolean" -> {
                            %131 : java.type:"int" = constant @0;
                            %132 : java.type:"boolean" = eq %130 %131;
                            yield %132;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %133 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %134 : java.type:"java.lang.Integer" = invoke %133 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %135 : java.type:"java.lang.Integer" = cast %134 @java.type:"java.lang.Integer";
                            yield %135;
                        }
                        ()java.type:"boolean" -> {
                            %136 : java.type:"boolean" = constant @true;
                            yield %136;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %137 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %138 : java.type:"java.lang.Integer" = invoke %137 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                            %139 : java.type:"java.lang.Integer" = cast %138 @java.type:"java.lang.Integer";
                            yield %139;
                        };
                    var.store %7 %129;
                    return;
                };
                """)
        @Reflect
        void testYield(UnboundedInteger test, int s) {
            Object o; Number n; Integer i;

            // simple field name
            o = switch (s) {
                case 0 -> x;
                default -> x;
            };

            n = switch (s) {
                case 0 -> x;
                default -> x;
            };

            i = switch (s) {
                case 0 -> x;
                default -> x;
            };

            // qualified field name
            o = switch (s) {
                case 0 -> test.x;
                default -> test.x;
            };

            n = switch (s) {
                case 0 -> test.x;
                default -> test.x;
            };

            i = switch (s) {
                case 0 -> test.x;
                default -> test.x;
            };

            // simple method name
            o = switch (s) {
                case 0 -> getX();
                default -> getX();
            };

            n = switch (s) {
                case 0 -> getX();
                default -> getX();
            };

            i = switch (s) {
                case 0 -> getX();
                default -> getX();
            };

            // qualified method name
            o = switch (s) {
                case 0 -> test.getX();
                default -> test.getX();
            };

            n = switch (s) {
                case 0 -> test.getX();
                default -> test.getX();
            };

            i = switch (s) {
                case 0 -> test.getX();
                default -> test.getX();
            };
        }

        @IR("""
                func @"testThrows" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger", %2 : java.type:"boolean")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"boolean"> = var %2 @"cond";
                    java.if
                        ()java.type:"boolean" -> {
                            %5 : java.type:"boolean" = var.load %4;
                            yield %5;
                        }
                        ()java.type:"void" -> {
                            %6 : java.type:"java.lang.WrongThreadException" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::t:java.lang.Throwable";
                            %7 : java.type:"java.lang.WrongThreadException" = cast %6 @java.type:"java.lang.WrongThreadException";
                            throw %7;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.if
                        ()java.type:"boolean" -> {
                            %8 : java.type:"boolean" = var.load %4;
                            yield %8;
                        }
                        ()java.type:"void" -> {
                            %9 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %10 : java.type:"java.lang.WrongThreadException" = field.load %9 @java.ref:"ErasedAccessTest$UnboundedInteger::t:java.lang.Throwable";
                            %11 : java.type:"java.lang.WrongThreadException" = cast %10 @java.type:"java.lang.WrongThreadException";
                            throw %11;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.if
                        ()java.type:"boolean" -> {
                            %12 : java.type:"boolean" = var.load %4;
                            yield %12;
                        }
                        ()java.type:"void" -> {
                            %13 : java.type:"java.lang.WrongThreadException" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getT():java.lang.Throwable";
                            %14 : java.type:"java.lang.WrongThreadException" = cast %13 @java.type:"java.lang.WrongThreadException";
                            throw %14;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.if
                        ()java.type:"boolean" -> {
                            %15 : java.type:"boolean" = var.load %4;
                            yield %15;
                        }
                        ()java.type:"void" -> {
                            %16 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %3;
                            %17 : java.type:"java.lang.WrongThreadException" = invoke %16 @java.ref:"ErasedAccessTest$UnboundedInteger::getT():java.lang.Throwable";
                            %18 : java.type:"java.lang.WrongThreadException" = cast %17 @java.type:"java.lang.WrongThreadException";
                            throw %18;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    return;
                };
                """)
        @Reflect
        void testThrows(UnboundedInteger test, boolean cond) {
            // simple field name
            if (cond) {
                throw t;
            }

            // qualified field name
            if (cond) {
                throw test.t;
            }

            // simple method name
            if (cond) {
                throw getT();
            }

            // qualified method name
            if (cond) {
                throw test.getT();
            }
        }

        @IR("""
                func @"testSwitchSelector" (%0 : java.type:"ErasedAccessTest$UnboundedInteger", %1 : java.type:"ErasedAccessTest$UnboundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$UnboundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %5 : java.type:"java.lang.Integer" = cast %4 @java.type:"java.lang.Integer";
                    java.switch.statement %5
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = constant @true;
                            yield %6;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                    %9 : java.type:"java.lang.Object" = java.switch.expression %8
                        ()java.type:"boolean" -> {
                            %10 : java.type:"boolean" = constant @true;
                            yield %10;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %11 : java.type:"java.lang.Object" = constant @null;
                            yield %11;
                        };
                    var.store %3 %9;
                    %12 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %13 : java.type:"java.lang.Integer" = field.load %12 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %14 : java.type:"java.lang.Integer" = cast %13 @java.type:"java.lang.Integer";
                    java.switch.statement %14
                        ()java.type:"boolean" -> {
                            %15 : java.type:"boolean" = constant @true;
                            yield %15;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %16 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %17 : java.type:"java.lang.Integer" = field.load %16 @java.ref:"ErasedAccessTest$UnboundedInteger::x:java.lang.Object";
                    %18 : java.type:"java.lang.Integer" = cast %17 @java.type:"java.lang.Integer";
                    %19 : java.type:"java.lang.Object" = java.switch.expression %18
                        ()java.type:"boolean" -> {
                            %20 : java.type:"boolean" = constant @true;
                            yield %20;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %21 : java.type:"java.lang.Object" = constant @null;
                            yield %21;
                        };
                    var.store %3 %19;
                    %22 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %23 : java.type:"java.lang.Integer" = cast %22 @java.type:"java.lang.Integer";
                    java.switch.statement %23
                        ()java.type:"boolean" -> {
                            %24 : java.type:"boolean" = constant @true;
                            yield %24;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %25 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %26 : java.type:"java.lang.Integer" = cast %25 @java.type:"java.lang.Integer";
                    %27 : java.type:"java.lang.Object" = java.switch.expression %26
                        ()java.type:"boolean" -> {
                            %28 : java.type:"boolean" = constant @true;
                            yield %28;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %29 : java.type:"java.lang.Object" = constant @null;
                            yield %29;
                        };
                    var.store %3 %27;
                    %30 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %31 : java.type:"java.lang.Integer" = invoke %30 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %32 : java.type:"java.lang.Integer" = cast %31 @java.type:"java.lang.Integer";
                    java.switch.statement %32
                        ()java.type:"boolean" -> {
                            %33 : java.type:"boolean" = constant @true;
                            yield %33;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %34 : java.type:"ErasedAccessTest$UnboundedInteger" = var.load %2;
                    %35 : java.type:"java.lang.Integer" = invoke %34 @java.ref:"ErasedAccessTest$UnboundedInteger::getX():java.lang.Object";
                    %36 : java.type:"java.lang.Integer" = cast %35 @java.type:"java.lang.Integer";
                    %37 : java.type:"java.lang.Object" = java.switch.expression %36
                        ()java.type:"boolean" -> {
                            %38 : java.type:"boolean" = constant @true;
                            yield %38;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %39 : java.type:"java.lang.Object" = constant @null;
                            yield %39;
                        };
                    var.store %3 %37;
                    return;
                };
                """)
        @Reflect
        void testSwitchSelector(UnboundedInteger test) {
            Object o;

            // simple field name
            switch (x) {default -> { }}
            o = switch (x) { default -> null; };

            // qualified field name
            switch (test.x) {default -> { }}
            o = switch (test.x) { default -> null; };

            // simple method name
            switch (getX()) {default -> { }}
            o = switch (getX()) { default -> null; };

            // qualified method name
            switch (test.getX()) {default -> { }}
            o = switch (test.getX()) { default -> null; };
        }
    }

    // the part below is just copied from the above with minor adaptations in the expected IRs

    static class Bounded<X extends Number> {
        X x;

        X getX() {
            return x;
        }
    }

    static class BoundedInteger extends Bounded<Integer> {

        @IR("""
                func @"testInstanceof" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %4 : java.type:"boolean" = instanceof %3 @java.type:"java.lang.Integer";
                    %5 : Var<java.type:"boolean"> = var %4 @"f_s_s";
                    %6 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %7 : java.type:"java.lang.Integer" = field.load %6 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %8 : java.type:"boolean" = instanceof %7 @java.type:"java.lang.Integer";
                    %9 : Var<java.type:"boolean"> = var %8 @"f_q_s";
                    %10 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %11 : java.type:"boolean" = instanceof %10 @java.type:"java.lang.Integer";
                    %12 : Var<java.type:"boolean"> = var %11 @"m_s_s";
                    %13 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = invoke %13 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %15 : java.type:"boolean" = instanceof %14 @java.type:"java.lang.Integer";
                    %16 : Var<java.type:"boolean"> = var %15 @"m_q_s";
                    return;
                };
                """)
        @Reflect
        void testInstanceof(BoundedInteger test) {
            // simple field name
            boolean f_s_s = x instanceof Integer;

            // qualified field name
            boolean f_q_s = test.x instanceof Integer;

            // simple method name
            boolean m_s_s = getX() instanceof Integer;

            // qualified method name
            boolean m_q_s = test.getX() instanceof Integer;
        }

        @IR("""
                func @"testInstanceofCond" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger", %2 : java.type:"boolean")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"boolean"> = var %2 @"cond";
                    %5 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = var.load %4;
                            yield %6;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                            yield %8;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %9 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                            yield %10;
                        };
                    %11 : java.type:"boolean" = instanceof %5 @java.type:"java.lang.Object";
                    %12 : Var<java.type:"boolean"> = var %11 @"f_s_o";
                    %13 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %14 : java.type:"boolean" = var.load %4;
                            yield %14;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %15 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %16 : java.type:"java.lang.Integer" = field.load %15 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %17 : java.type:"java.lang.Integer" = cast %16 @java.type:"java.lang.Integer";
                            yield %17;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %18 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %19 : java.type:"java.lang.Integer" = field.load %18 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %20 : java.type:"java.lang.Integer" = cast %19 @java.type:"java.lang.Integer";
                            yield %20;
                        };
                    %21 : java.type:"boolean" = instanceof %13 @java.type:"java.lang.Object";
                    %22 : Var<java.type:"boolean"> = var %21 @"f_q_o";
                    %23 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %24 : java.type:"boolean" = var.load %4;
                            yield %24;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %25 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %26 : java.type:"java.lang.Integer" = cast %25 @java.type:"java.lang.Integer";
                            yield %26;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %27 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %28 : java.type:"java.lang.Integer" = cast %27 @java.type:"java.lang.Integer";
                            yield %28;
                        };
                    %29 : java.type:"boolean" = instanceof %23 @java.type:"java.lang.Object";
                    %30 : Var<java.type:"boolean"> = var %29 @"m_s_o";
                    %31 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %32 : java.type:"boolean" = var.load %4;
                            yield %32;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %33 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %34 : java.type:"java.lang.Integer" = invoke %33 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %35 : java.type:"java.lang.Integer" = cast %34 @java.type:"java.lang.Integer";
                            yield %35;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %36 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %37 : java.type:"java.lang.Integer" = invoke %36 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %38 : java.type:"java.lang.Integer" = cast %37 @java.type:"java.lang.Integer";
                            yield %38;
                        };
                    %39 : java.type:"boolean" = instanceof %31 @java.type:"java.lang.Object";
                    %40 : Var<java.type:"boolean"> = var %39 @"m_q_o";
                    return;
                };
                """)
        @Reflect
        void testInstanceofCond(BoundedInteger test, boolean cond) {
            // simple field name
            boolean f_s_o = (cond ? x : x) instanceof Object;

            // qualified field name
            boolean f_q_o = (cond ? test.x : test.x) instanceof Object;

            // simple method name
            boolean m_s_o = (cond ? getX() : getX()) instanceof Object;

            // qualified method name
            boolean m_q_o = (cond ? test.getX() : test.getX()) instanceof Object;
        }

        @IR("""
                func @"testExec" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %4 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %5 : java.type:"java.lang.Integer" = invoke %4 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    return;
                };
                """)
        @Reflect
        void testExec(BoundedInteger test) {
            // simple method name
            getX();

            // qualified method name
            test.getX();
        }

        @IR("""
                func @"testChainedCall" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %4 : java.type:"java.lang.Integer" = cast %3 @java.type:"java.lang.Integer";
                    %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.Integer::hashCode():int";
                    %6 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %7 : java.type:"java.lang.Integer" = field.load %6 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                    %9 : java.type:"int" = invoke %8 @java.ref:"java.lang.Integer::hashCode():int";
                    %10 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %11 : java.type:"java.lang.Integer" = cast %10 @java.type:"java.lang.Integer";
                    %12 : java.type:"int" = invoke %11 @java.ref:"java.lang.Integer::hashCode():int";
                    %13 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %14 : java.type:"java.lang.Integer" = invoke %13 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %15 : java.type:"java.lang.Integer" = cast %14 @java.type:"java.lang.Integer";
                    %16 : java.type:"int" = invoke %15 @java.ref:"java.lang.Integer::hashCode():int";
                    return;
                };
                """)
        @Reflect
        void testChainedCall(BoundedInteger test) {
            // simple field name
            x.hashCode();

            // qualified field name
            test.x.hashCode();

            // simple method name
            getX().hashCode();

            // qualified method name
            test.getX().hashCode();
        }

        @IR("""
                func @"testChainedCallCond" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger", %2 : java.type:"boolean")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"boolean"> = var %2 @"cond";
                    %5 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = var.load %4;
                            yield %6;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                            yield %8;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %9 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                            yield %10;
                        };
                    %11 : java.type:"int" = invoke %5 @java.ref:"java.lang.Integer::hashCode():int";
                    %12 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %13 : java.type:"boolean" = var.load %4;
                            yield %13;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %14 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %15 : java.type:"java.lang.Integer" = field.load %14 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %16 : java.type:"java.lang.Integer" = cast %15 @java.type:"java.lang.Integer";
                            yield %16;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %17 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %18 : java.type:"java.lang.Integer" = field.load %17 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %19 : java.type:"java.lang.Integer" = cast %18 @java.type:"java.lang.Integer";
                            yield %19;
                        };
                    %20 : java.type:"int" = invoke %12 @java.ref:"java.lang.Integer::hashCode():int";
                    %21 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %22 : java.type:"boolean" = var.load %4;
                            yield %22;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %23 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %24 : java.type:"java.lang.Integer" = cast %23 @java.type:"java.lang.Integer";
                            yield %24;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %25 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %26 : java.type:"java.lang.Integer" = cast %25 @java.type:"java.lang.Integer";
                            yield %26;
                        };
                    %27 : java.type:"int" = invoke %21 @java.ref:"java.lang.Integer::hashCode():int";
                    %28 : java.type:"java.lang.Integer" = java.cexpression
                        ()java.type:"boolean" -> {
                            %29 : java.type:"boolean" = var.load %4;
                            yield %29;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %30 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %31 : java.type:"java.lang.Integer" = invoke %30 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %32 : java.type:"java.lang.Integer" = cast %31 @java.type:"java.lang.Integer";
                            yield %32;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %33 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %34 : java.type:"java.lang.Integer" = invoke %33 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %35 : java.type:"java.lang.Integer" = cast %34 @java.type:"java.lang.Integer";
                            yield %35;
                        };
                    %36 : java.type:"int" = invoke %28 @java.ref:"java.lang.Integer::hashCode():int";
                    return;
                };
                """)
        @Reflect
        void testChainedCallCond(BoundedInteger test, boolean cond) {
            // simple field name
            (cond ? x : x).hashCode();

            // qualified field name
            (cond ? test.x : test.x).hashCode();

            // simple method name
            (cond ? getX() : getX()).hashCode();

            // qualified method name
            (cond ? test.getX() : test.getX()).hashCode();
        }

        @IR("""
                func @"testAssign" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : Var<java.type:"java.lang.Number"> = var @"n";
                    %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                    %6 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %3 %6;
                    %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %4 %7;
                    %8 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %9 : java.type:"java.lang.Integer" = cast %8 @java.type:"java.lang.Integer";
                    var.store %5 %9;
                    %10 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %11 : java.type:"java.lang.Integer" = field.load %10 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %3 %11;
                    %12 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %13 : java.type:"java.lang.Integer" = field.load %12 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %4 %13;
                    %14 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %15 : java.type:"java.lang.Integer" = field.load %14 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %16 : java.type:"java.lang.Integer" = cast %15 @java.type:"java.lang.Integer";
                    var.store %5 %16;
                    %17 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %3 %17;
                    %18 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %4 %18;
                    %19 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %20 : java.type:"java.lang.Integer" = cast %19 @java.type:"java.lang.Integer";
                    var.store %5 %20;
                    %21 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %22 : java.type:"java.lang.Integer" = invoke %21 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %3 %22;
                    %23 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %24 : java.type:"java.lang.Integer" = invoke %23 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %4 %24;
                    %25 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %26 : java.type:"java.lang.Integer" = invoke %25 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %27 : java.type:"java.lang.Integer" = cast %26 @java.type:"java.lang.Integer";
                    var.store %5 %27;
                    return;
                };
                """)
        @Reflect
        void testAssign(BoundedInteger test) {
            Object o; Number n; Integer i;

            // simple field name
            o = x;
            n = x;
            i = x;

            // qualified field name
            o = test.x;
            n = test.x;
            i = test.x;

            // simple method name
            o = getX();
            n = getX();
            i = getX();

            // qualified method name
            o = test.getX();
            n = test.getX();
            i = test.getX();
        }

        @IR("""
                func @"testArrayInit" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object[]"> = var @"o";
                    %4 : Var<java.type:"java.lang.Number[]"> = var @"n";
                    %5 : Var<java.type:"java.lang.Integer[]"> = var @"i";
                    %6 : java.type:"int" = constant @1;
                    %7 : java.type:"java.lang.Object[]" = new %6 @java.ref:"java.lang.Object[]::(int)";
                    %8 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %9 : java.type:"int" = constant @0;
                    array.store %7 %9 %8;
                    var.store %3 %7;
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"java.lang.Number[]" = new %10 @java.ref:"java.lang.Number[]::(int)";
                    %12 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %13 : java.type:"int" = constant @0;
                    array.store %11 %13 %12;
                    var.store %4 %11;
                    %14 : java.type:"int" = constant @1;
                    %15 : java.type:"java.lang.Integer[]" = new %14 @java.ref:"java.lang.Integer[]::(int)";
                    %16 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %17 : java.type:"java.lang.Integer" = cast %16 @java.type:"java.lang.Integer";
                    %18 : java.type:"int" = constant @0;
                    array.store %15 %18 %17;
                    var.store %5 %15;
                    %19 : java.type:"int" = constant @1;
                    %20 : java.type:"java.lang.Object[]" = new %19 @java.ref:"java.lang.Object[]::(int)";
                    %21 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %22 : java.type:"java.lang.Integer" = field.load %21 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %23 : java.type:"int" = constant @0;
                    array.store %20 %23 %22;
                    var.store %3 %20;
                    %24 : java.type:"int" = constant @1;
                    %25 : java.type:"java.lang.Number[]" = new %24 @java.ref:"java.lang.Number[]::(int)";
                    %26 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %27 : java.type:"java.lang.Integer" = field.load %26 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %28 : java.type:"int" = constant @0;
                    array.store %25 %28 %27;
                    var.store %4 %25;
                    %29 : java.type:"int" = constant @1;
                    %30 : java.type:"java.lang.Integer[]" = new %29 @java.ref:"java.lang.Integer[]::(int)";
                    %31 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %32 : java.type:"java.lang.Integer" = field.load %31 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %33 : java.type:"java.lang.Integer" = cast %32 @java.type:"java.lang.Integer";
                    %34 : java.type:"int" = constant @0;
                    array.store %30 %34 %33;
                    var.store %5 %30;
                    %35 : java.type:"int" = constant @1;
                    %36 : java.type:"java.lang.Object[]" = new %35 @java.ref:"java.lang.Object[]::(int)";
                    %37 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %38 : java.type:"int" = constant @0;
                    array.store %36 %38 %37;
                    var.store %3 %36;
                    %39 : java.type:"int" = constant @1;
                    %40 : java.type:"java.lang.Number[]" = new %39 @java.ref:"java.lang.Number[]::(int)";
                    %41 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %42 : java.type:"int" = constant @0;
                    array.store %40 %42 %41;
                    var.store %4 %40;
                    %43 : java.type:"int" = constant @1;
                    %44 : java.type:"java.lang.Integer[]" = new %43 @java.ref:"java.lang.Integer[]::(int)";
                    %45 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %46 : java.type:"java.lang.Integer" = cast %45 @java.type:"java.lang.Integer";
                    %47 : java.type:"int" = constant @0;
                    array.store %44 %47 %46;
                    var.store %5 %44;
                    %48 : java.type:"int" = constant @1;
                    %49 : java.type:"java.lang.Object[]" = new %48 @java.ref:"java.lang.Object[]::(int)";
                    %50 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %51 : java.type:"java.lang.Integer" = invoke %50 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %52 : java.type:"int" = constant @0;
                    array.store %49 %52 %51;
                    var.store %3 %49;
                    %53 : java.type:"int" = constant @1;
                    %54 : java.type:"java.lang.Number[]" = new %53 @java.ref:"java.lang.Number[]::(int)";
                    %55 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %56 : java.type:"java.lang.Integer" = invoke %55 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %57 : java.type:"int" = constant @0;
                    array.store %54 %57 %56;
                    var.store %4 %54;
                    %58 : java.type:"int" = constant @1;
                    %59 : java.type:"java.lang.Integer[]" = new %58 @java.ref:"java.lang.Integer[]::(int)";
                    %60 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %61 : java.type:"java.lang.Integer" = invoke %60 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %62 : java.type:"java.lang.Integer" = cast %61 @java.type:"java.lang.Integer";
                    %63 : java.type:"int" = constant @0;
                    array.store %59 %63 %62;
                    var.store %5 %59;
                    return;
                };
                """)
        @Reflect
        void testArrayInit(BoundedInteger test) {
            Object[] o; Number[] n; Integer[] i;

            // simple field name
            o = new Object[] { x };
            n = new Number[] { x };
            i = new Integer[] { x };

            // qualified field name
            o = new Object[] { test.x };
            n = new Number[] { test.x };
            i = new Integer[] { test.x };

            // simple method name
            o = new Object[] { getX() };
            n = new Number[] { getX() };
            i = new Integer[] { getX() };

            // qualified method name
            o = new Object[] { test.getX() };
            n = new Number[] { test.getX() };
            i = new Integer[] { test.getX() };
        }

        @IR("""
                func @"testCast" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : Var<java.type:"java.lang.Number"> = var @"n";
                    %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                    %6 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %3 %6;
                    %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %4 %7;
                    %8 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %9 : java.type:"java.lang.Integer" = cast %8 @java.type:"java.lang.Integer";
                    var.store %5 %9;
                    %10 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %11 : java.type:"java.lang.Integer" = field.load %10 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %3 %11;
                    %12 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %13 : java.type:"java.lang.Integer" = field.load %12 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    var.store %4 %13;
                    %14 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %15 : java.type:"java.lang.Integer" = field.load %14 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %16 : java.type:"java.lang.Integer" = cast %15 @java.type:"java.lang.Integer";
                    var.store %5 %16;
                    %17 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %3 %17;
                    %18 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %4 %18;
                    %19 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %20 : java.type:"java.lang.Integer" = cast %19 @java.type:"java.lang.Integer";
                    var.store %5 %20;
                    %21 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %22 : java.type:"java.lang.Integer" = invoke %21 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %3 %22;
                    %23 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %24 : java.type:"java.lang.Integer" = invoke %23 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    var.store %4 %24;
                    %25 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %26 : java.type:"java.lang.Integer" = invoke %25 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %27 : java.type:"java.lang.Integer" = cast %26 @java.type:"java.lang.Integer";
                    var.store %5 %27;
                    return;
                };
                """)
        @Reflect
        void testCast(BoundedInteger test) {
            Object o; Number n; Integer i;

            // simple field name
            o = (Object) x;
            n = (Number) x;
            i = (Integer) x;

            // qualified field name
            o = (Object) test.x;
            n = (Number) test.x;
            i = (Integer) test.x;

            // simple method name
            o = (Object) getX();
            n = (Number) getX();
            i = (Integer) getX();

            // qualified method name
            o = (Object) test.getX();
            n = (Number) test.getX();
            i = (Integer) test.getX();
        }

        @IR("""
                func @"testIntersectionCast" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %5 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %4 @java.type:"java.lang.Comparable";
                    %6 : java.type:"java.io.Serializable" = cast %5 @java.type:"java.io.Serializable";
                    %7 : java.type:"java.lang.Number" = cast %6 @java.type:"java.lang.Number";
                    var.store %3 %7;
                    %8 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %9 : java.type:"java.lang.Integer" = field.load %8 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %10 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %9 @java.type:"java.lang.Comparable";
                    %11 : java.type:"java.io.Serializable" = cast %10 @java.type:"java.io.Serializable";
                    %12 : java.type:"java.lang.Number" = cast %11 @java.type:"java.lang.Number";
                    var.store %3 %12;
                    %13 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %14 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %13 @java.type:"java.lang.Comparable";
                    %15 : java.type:"java.io.Serializable" = cast %14 @java.type:"java.io.Serializable";
                    %16 : java.type:"java.lang.Number" = cast %15 @java.type:"java.lang.Number";
                    var.store %3 %16;
                    %17 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %18 : java.type:"java.lang.Integer" = invoke %17 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %19 : java.type:"java.lang.Comparable<java.lang.Integer>" = cast %18 @java.type:"java.lang.Comparable";
                    %20 : java.type:"java.io.Serializable" = cast %19 @java.type:"java.io.Serializable";
                    %21 : java.type:"java.lang.Number" = cast %20 @java.type:"java.lang.Number";
                    var.store %3 %21;
                    return;
                };
                """)
        @Reflect
        void testIntersectionCast(BoundedInteger test) {
            Object o;

            // simple field name
            o = (Number & Comparable<Integer> & Serializable) x;

            // qualified field name
            o = (Number & Comparable<Integer> & Serializable) test.x;

            // simple method name
            o = (Number & Comparable<Integer> & Serializable) getX();

            // qualified method name
            o = (Number & Comparable<Integer> & Serializable) test.getX();
        }

        void o(Object o) { }
        void n(Number n) { }
        void i(Integer i) { }

        @IR("""
                func @"testMethod" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    invoke %0 %3 @java.ref:"ErasedAccessTest$BoundedInteger::o(java.lang.Object):void";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    invoke %0 %4 @java.ref:"ErasedAccessTest$BoundedInteger::n(java.lang.Number):void";
                    %5 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %6 : java.type:"java.lang.Integer" = cast %5 @java.type:"java.lang.Integer";
                    invoke %0 %6 @java.ref:"ErasedAccessTest$BoundedInteger::i(java.lang.Integer):void";
                    %7 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %8 : java.type:"java.lang.Integer" = field.load %7 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    invoke %0 %8 @java.ref:"ErasedAccessTest$BoundedInteger::o(java.lang.Object):void";
                    %9 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %10 : java.type:"java.lang.Integer" = field.load %9 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    invoke %0 %10 @java.ref:"ErasedAccessTest$BoundedInteger::n(java.lang.Number):void";
                    %11 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %12 : java.type:"java.lang.Integer" = field.load %11 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %13 : java.type:"java.lang.Integer" = cast %12 @java.type:"java.lang.Integer";
                    invoke %0 %13 @java.ref:"ErasedAccessTest$BoundedInteger::i(java.lang.Integer):void";
                    %14 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    invoke %0 %14 @java.ref:"ErasedAccessTest$BoundedInteger::o(java.lang.Object):void";
                    %15 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    invoke %0 %15 @java.ref:"ErasedAccessTest$BoundedInteger::n(java.lang.Number):void";
                    %16 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %17 : java.type:"java.lang.Integer" = cast %16 @java.type:"java.lang.Integer";
                    invoke %0 %17 @java.ref:"ErasedAccessTest$BoundedInteger::i(java.lang.Integer):void";
                    %18 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %19 : java.type:"java.lang.Integer" = invoke %18 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    invoke %0 %19 @java.ref:"ErasedAccessTest$BoundedInteger::o(java.lang.Object):void";
                    %20 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %21 : java.type:"java.lang.Integer" = invoke %20 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    invoke %0 %21 @java.ref:"ErasedAccessTest$BoundedInteger::n(java.lang.Number):void";
                    %22 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %23 : java.type:"java.lang.Integer" = invoke %22 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %24 : java.type:"java.lang.Integer" = cast %23 @java.type:"java.lang.Integer";
                    invoke %0 %24 @java.ref:"ErasedAccessTest$BoundedInteger::i(java.lang.Integer):void";
                    return;
                };
                """)
        @Reflect
        void testMethod(BoundedInteger test) {
            // simple field name
            o(x);
            n(x);
            i(x);

            // qualified field name
            o(test.x);
            n(test.x);
            i(test.x);

            // simple method name
            o(getX());
            n(getX());
            i(getX());

            // qualified method name
            o(test.getX());
            n(test.getX());
            i(test.getX());
        }

        @IR("""
                func @"testWidening" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"long"> = var @"l";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %5 : java.type:"java.lang.Integer" = cast %4 @java.type:"java.lang.Integer";
                    %6 : java.type:"int" = invoke %5 @java.ref:"java.lang.Integer::intValue():int";
                    %7 : java.type:"long" = conv %6;
                    var.store %3 %7;
                    %8 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %9 : java.type:"java.lang.Integer" = field.load %8 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %10 : java.type:"java.lang.Integer" = cast %9 @java.type:"java.lang.Integer";
                    %11 : java.type:"int" = invoke %10 @java.ref:"java.lang.Integer::intValue():int";
                    %12 : java.type:"long" = conv %11;
                    var.store %3 %12;
                    %13 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %14 : java.type:"java.lang.Integer" = cast %13 @java.type:"java.lang.Integer";
                    %15 : java.type:"int" = invoke %14 @java.ref:"java.lang.Integer::intValue():int";
                    %16 : java.type:"long" = conv %15;
                    var.store %3 %16;
                    %17 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %18 : java.type:"java.lang.Integer" = invoke %17 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %19 : java.type:"java.lang.Integer" = cast %18 @java.type:"java.lang.Integer";
                    %20 : java.type:"int" = invoke %19 @java.ref:"java.lang.Integer::intValue():int";
                    %21 : java.type:"long" = conv %20;
                    var.store %3 %21;
                    return;
                };
                """)
        @Reflect
        void testWidening(BoundedInteger test) {
            long l;

            // simple field name
            l = x;

            // qualified field name
            l = test.x;

            // simple method name
            l = getX();

            // qualified method name
            l = test.getX();
        }

        @IR("""
                func @"testAssert" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    assert
                        ()java.type:"boolean" -> {
                            %3 : java.type:"boolean" = constant @false;
                            yield %3;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %5 : java.type:"java.lang.Integer" = cast %4 @java.type:"java.lang.Integer";
                            yield %5;
                        };
                    assert
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = constant @false;
                            yield %6;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %7 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                            %8 : java.type:"java.lang.Integer" = field.load %7 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %9 : java.type:"java.lang.Integer" = cast %8 @java.type:"java.lang.Integer";
                            yield %9;
                        };
                    assert
                        ()java.type:"boolean" -> {
                            %10 : java.type:"boolean" = constant @false;
                            yield %10;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %11 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %12 : java.type:"java.lang.Integer" = cast %11 @java.type:"java.lang.Integer";
                            yield %12;
                        };
                    assert
                        ()java.type:"boolean" -> {
                            %13 : java.type:"boolean" = constant @false;
                            yield %13;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %14 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                            %15 : java.type:"java.lang.Integer" = invoke %14 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %16 : java.type:"java.lang.Integer" = cast %15 @java.type:"java.lang.Integer";
                            yield %16;
                        };
                    return;
                };
                """)
        @Reflect
        void testAssert(BoundedInteger test) {
            // simple field name
            assert false : x;

            // qualified field name
            assert false : test.x;

            // simple method name
            assert false : getX();

            // qualified method name
            assert false : test.getX();
        }

        @IR("""
                func @"testSynchronized" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %3 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %4 : java.type:"java.lang.Integer" = cast %3 @java.type:"java.lang.Integer";
                            yield %4;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %5 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                            %6 : java.type:"java.lang.Integer" = field.load %5 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %7 : java.type:"java.lang.Integer" = cast %6 @java.type:"java.lang.Integer";
                            yield %7;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %8 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %9 : java.type:"java.lang.Integer" = cast %8 @java.type:"java.lang.Integer";
                            yield %9;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    java.synchronized
                        ()java.type:"java.lang.Integer" -> {
                            %10 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                            %11 : java.type:"java.lang.Integer" = invoke %10 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %12 : java.type:"java.lang.Integer" = cast %11 @java.type:"java.lang.Integer";
                            yield %12;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    return;
                };
                """)
        @Reflect
        @SuppressWarnings("identity")
        void testSynchronized(BoundedInteger test) {
            // simple field name
            synchronized (x) { };

            // qualified field name
            synchronized (test.x) { };

            // simple method name
            synchronized (getX()) { };

            // qualified method name
            synchronized (test.getX()) { };
        }

        @IR("""
                func @"testYield" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger", %2 : java.type:"int")java.type:"void" -> {
                    %3 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %4 : Var<java.type:"int"> = var %2 @"s";
                    %5 : Var<java.type:"java.lang.Object"> = var @"o";
                    %6 : Var<java.type:"java.lang.Number"> = var @"n";
                    %7 : Var<java.type:"java.lang.Integer"> = var @"i";
                    %8 : java.type:"int" = var.load %4;
                    %9 : java.type:"java.lang.Object" = java.switch.expression %8
                        (%10 : java.type:"int")java.type:"boolean" -> {
                            %11 : java.type:"int" = constant @0;
                            %12 : java.type:"boolean" = eq %10 %11;
                            yield %12;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %13 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %14 : java.type:"java.lang.Integer" = cast %13 @java.type:"java.lang.Integer";
                            yield %14;
                        }
                        ()java.type:"boolean" -> {
                            %15 : java.type:"boolean" = constant @true;
                            yield %15;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %16 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %17 : java.type:"java.lang.Integer" = cast %16 @java.type:"java.lang.Integer";
                            yield %17;
                        };
                    var.store %5 %9;
                    %18 : java.type:"int" = var.load %4;
                    %19 : java.type:"java.lang.Number" = java.switch.expression %18
                        (%20 : java.type:"int")java.type:"boolean" -> {
                            %21 : java.type:"int" = constant @0;
                            %22 : java.type:"boolean" = eq %20 %21;
                            yield %22;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %23 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %24 : java.type:"java.lang.Integer" = cast %23 @java.type:"java.lang.Integer";
                            yield %24;
                        }
                        ()java.type:"boolean" -> {
                            %25 : java.type:"boolean" = constant @true;
                            yield %25;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %26 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %27 : java.type:"java.lang.Integer" = cast %26 @java.type:"java.lang.Integer";
                            yield %27;
                        };
                    var.store %6 %19;
                    %28 : java.type:"int" = var.load %4;
                    %29 : java.type:"java.lang.Integer" = java.switch.expression %28
                        (%30 : java.type:"int")java.type:"boolean" -> {
                            %31 : java.type:"int" = constant @0;
                            %32 : java.type:"boolean" = eq %30 %31;
                            yield %32;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %33 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %34 : java.type:"java.lang.Integer" = cast %33 @java.type:"java.lang.Integer";
                            yield %34;
                        }
                        ()java.type:"boolean" -> {
                            %35 : java.type:"boolean" = constant @true;
                            yield %35;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %36 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %37 : java.type:"java.lang.Integer" = cast %36 @java.type:"java.lang.Integer";
                            yield %37;
                        };
                    var.store %7 %29;
                    %38 : java.type:"int" = var.load %4;
                    %39 : java.type:"java.lang.Object" = java.switch.expression %38
                        (%40 : java.type:"int")java.type:"boolean" -> {
                            %41 : java.type:"int" = constant @0;
                            %42 : java.type:"boolean" = eq %40 %41;
                            yield %42;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %43 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %44 : java.type:"java.lang.Integer" = field.load %43 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %45 : java.type:"java.lang.Integer" = cast %44 @java.type:"java.lang.Integer";
                            yield %45;
                        }
                        ()java.type:"boolean" -> {
                            %46 : java.type:"boolean" = constant @true;
                            yield %46;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %47 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %48 : java.type:"java.lang.Integer" = field.load %47 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %49 : java.type:"java.lang.Integer" = cast %48 @java.type:"java.lang.Integer";
                            yield %49;
                        };
                    var.store %5 %39;
                    %50 : java.type:"int" = var.load %4;
                    %51 : java.type:"java.lang.Number" = java.switch.expression %50
                        (%52 : java.type:"int")java.type:"boolean" -> {
                            %53 : java.type:"int" = constant @0;
                            %54 : java.type:"boolean" = eq %52 %53;
                            yield %54;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %55 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %56 : java.type:"java.lang.Integer" = field.load %55 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %57 : java.type:"java.lang.Integer" = cast %56 @java.type:"java.lang.Integer";
                            yield %57;
                        }
                        ()java.type:"boolean" -> {
                            %58 : java.type:"boolean" = constant @true;
                            yield %58;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %59 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %60 : java.type:"java.lang.Integer" = field.load %59 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %61 : java.type:"java.lang.Integer" = cast %60 @java.type:"java.lang.Integer";
                            yield %61;
                        };
                    var.store %6 %51;
                    %62 : java.type:"int" = var.load %4;
                    %63 : java.type:"java.lang.Integer" = java.switch.expression %62
                        (%64 : java.type:"int")java.type:"boolean" -> {
                            %65 : java.type:"int" = constant @0;
                            %66 : java.type:"boolean" = eq %64 %65;
                            yield %66;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %67 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %68 : java.type:"java.lang.Integer" = field.load %67 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %69 : java.type:"java.lang.Integer" = cast %68 @java.type:"java.lang.Integer";
                            yield %69;
                        }
                        ()java.type:"boolean" -> {
                            %70 : java.type:"boolean" = constant @true;
                            yield %70;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %71 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %72 : java.type:"java.lang.Integer" = field.load %71 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                            %73 : java.type:"java.lang.Integer" = cast %72 @java.type:"java.lang.Integer";
                            yield %73;
                        };
                    var.store %7 %63;
                    %74 : java.type:"int" = var.load %4;
                    %75 : java.type:"java.lang.Object" = java.switch.expression %74
                        (%76 : java.type:"int")java.type:"boolean" -> {
                            %77 : java.type:"int" = constant @0;
                            %78 : java.type:"boolean" = eq %76 %77;
                            yield %78;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %79 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %80 : java.type:"java.lang.Integer" = cast %79 @java.type:"java.lang.Integer";
                            yield %80;
                        }
                        ()java.type:"boolean" -> {
                            %81 : java.type:"boolean" = constant @true;
                            yield %81;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %82 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %83 : java.type:"java.lang.Integer" = cast %82 @java.type:"java.lang.Integer";
                            yield %83;
                        };
                    var.store %5 %75;
                    %84 : java.type:"int" = var.load %4;
                    %85 : java.type:"java.lang.Number" = java.switch.expression %84
                        (%86 : java.type:"int")java.type:"boolean" -> {
                            %87 : java.type:"int" = constant @0;
                            %88 : java.type:"boolean" = eq %86 %87;
                            yield %88;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %89 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %90 : java.type:"java.lang.Integer" = cast %89 @java.type:"java.lang.Integer";
                            yield %90;
                        }
                        ()java.type:"boolean" -> {
                            %91 : java.type:"boolean" = constant @true;
                            yield %91;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %92 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %93 : java.type:"java.lang.Integer" = cast %92 @java.type:"java.lang.Integer";
                            yield %93;
                        };
                    var.store %6 %85;
                    %94 : java.type:"int" = var.load %4;
                    %95 : java.type:"java.lang.Integer" = java.switch.expression %94
                        (%96 : java.type:"int")java.type:"boolean" -> {
                            %97 : java.type:"int" = constant @0;
                            %98 : java.type:"boolean" = eq %96 %97;
                            yield %98;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %99 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %100 : java.type:"java.lang.Integer" = cast %99 @java.type:"java.lang.Integer";
                            yield %100;
                        }
                        ()java.type:"boolean" -> {
                            %101 : java.type:"boolean" = constant @true;
                            yield %101;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %102 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %103 : java.type:"java.lang.Integer" = cast %102 @java.type:"java.lang.Integer";
                            yield %103;
                        };
                    var.store %7 %95;
                    %104 : java.type:"int" = var.load %4;
                    %105 : java.type:"java.lang.Object" = java.switch.expression %104
                        (%106 : java.type:"int")java.type:"boolean" -> {
                            %107 : java.type:"int" = constant @0;
                            %108 : java.type:"boolean" = eq %106 %107;
                            yield %108;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %109 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %110 : java.type:"java.lang.Integer" = invoke %109 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %111 : java.type:"java.lang.Integer" = cast %110 @java.type:"java.lang.Integer";
                            yield %111;
                        }
                        ()java.type:"boolean" -> {
                            %112 : java.type:"boolean" = constant @true;
                            yield %112;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %113 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %114 : java.type:"java.lang.Integer" = invoke %113 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %115 : java.type:"java.lang.Integer" = cast %114 @java.type:"java.lang.Integer";
                            yield %115;
                        };
                    var.store %5 %105;
                    %116 : java.type:"int" = var.load %4;
                    %117 : java.type:"java.lang.Number" = java.switch.expression %116
                        (%118 : java.type:"int")java.type:"boolean" -> {
                            %119 : java.type:"int" = constant @0;
                            %120 : java.type:"boolean" = eq %118 %119;
                            yield %120;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %121 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %122 : java.type:"java.lang.Integer" = invoke %121 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %123 : java.type:"java.lang.Integer" = cast %122 @java.type:"java.lang.Integer";
                            yield %123;
                        }
                        ()java.type:"boolean" -> {
                            %124 : java.type:"boolean" = constant @true;
                            yield %124;
                        }
                        ()java.type:"java.lang.Number" -> {
                            %125 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %126 : java.type:"java.lang.Integer" = invoke %125 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %127 : java.type:"java.lang.Integer" = cast %126 @java.type:"java.lang.Integer";
                            yield %127;
                        };
                    var.store %6 %117;
                    %128 : java.type:"int" = var.load %4;
                    %129 : java.type:"java.lang.Integer" = java.switch.expression %128
                        (%130 : java.type:"int")java.type:"boolean" -> {
                            %131 : java.type:"int" = constant @0;
                            %132 : java.type:"boolean" = eq %130 %131;
                            yield %132;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %133 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %134 : java.type:"java.lang.Integer" = invoke %133 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %135 : java.type:"java.lang.Integer" = cast %134 @java.type:"java.lang.Integer";
                            yield %135;
                        }
                        ()java.type:"boolean" -> {
                            %136 : java.type:"boolean" = constant @true;
                            yield %136;
                        }
                        ()java.type:"java.lang.Integer" -> {
                            %137 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %3;
                            %138 : java.type:"java.lang.Integer" = invoke %137 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                            %139 : java.type:"java.lang.Integer" = cast %138 @java.type:"java.lang.Integer";
                            yield %139;
                        };
                    var.store %7 %129;
                    return;
                };
                """)
        @Reflect
        void testYield(BoundedInteger test, int s) {
            Object o; Number n; Integer i;

            // simple field name
            o = switch (s) {
                case 0 -> x;
                default -> x;
            };

            n = switch (s) {
                case 0 -> x;
                default -> x;
            };

            i = switch (s) {
                case 0 -> x;
                default -> x;
            };

            // qualified field name
            o = switch (s) {
                case 0 -> test.x;
                default -> test.x;
            };

            n = switch (s) {
                case 0 -> test.x;
                default -> test.x;
            };

            i = switch (s) {
                case 0 -> test.x;
                default -> test.x;
            };

            // simple method name
            o = switch (s) {
                case 0 -> getX();
                default -> getX();
            };

            n = switch (s) {
                case 0 -> getX();
                default -> getX();
            };

            i = switch (s) {
                case 0 -> getX();
                default -> getX();
            };

            // qualified method name
            o = switch (s) {
                case 0 -> test.getX();
                default -> test.getX();
            };

            n = switch (s) {
                case 0 -> test.getX();
                default -> test.getX();
            };

            i = switch (s) {
                case 0 -> test.getX();
                default -> test.getX();
            };
        }

        @IR("""
                func @"testSwitchSelector" (%0 : java.type:"ErasedAccessTest$BoundedInteger", %1 : java.type:"ErasedAccessTest$BoundedInteger")java.type:"void" -> {
                    %2 : Var<java.type:"ErasedAccessTest$BoundedInteger"> = var %1 @"test";
                    %3 : Var<java.type:"java.lang.Object"> = var @"o";
                    %4 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %5 : java.type:"java.lang.Integer" = cast %4 @java.type:"java.lang.Integer";
                    java.switch.statement %5
                        ()java.type:"boolean" -> {
                            %6 : java.type:"boolean" = constant @true;
                            yield %6;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %7 : java.type:"java.lang.Integer" = field.load %0 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %8 : java.type:"java.lang.Integer" = cast %7 @java.type:"java.lang.Integer";
                    %9 : java.type:"java.lang.Object" = java.switch.expression %8
                        ()java.type:"boolean" -> {
                            %10 : java.type:"boolean" = constant @true;
                            yield %10;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %11 : java.type:"java.lang.Object" = constant @null;
                            yield %11;
                        };
                    var.store %3 %9;
                    %12 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %13 : java.type:"java.lang.Integer" = field.load %12 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %14 : java.type:"java.lang.Integer" = cast %13 @java.type:"java.lang.Integer";
                    java.switch.statement %14
                        ()java.type:"boolean" -> {
                            %15 : java.type:"boolean" = constant @true;
                            yield %15;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %16 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %17 : java.type:"java.lang.Integer" = field.load %16 @java.ref:"ErasedAccessTest$BoundedInteger::x:java.lang.Number";
                    %18 : java.type:"java.lang.Integer" = cast %17 @java.type:"java.lang.Integer";
                    %19 : java.type:"java.lang.Object" = java.switch.expression %18
                        ()java.type:"boolean" -> {
                            %20 : java.type:"boolean" = constant @true;
                            yield %20;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %21 : java.type:"java.lang.Object" = constant @null;
                            yield %21;
                        };
                    var.store %3 %19;
                    %22 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %23 : java.type:"java.lang.Integer" = cast %22 @java.type:"java.lang.Integer";
                    java.switch.statement %23
                        ()java.type:"boolean" -> {
                            %24 : java.type:"boolean" = constant @true;
                            yield %24;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %25 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %26 : java.type:"java.lang.Integer" = cast %25 @java.type:"java.lang.Integer";
                    %27 : java.type:"java.lang.Object" = java.switch.expression %26
                        ()java.type:"boolean" -> {
                            %28 : java.type:"boolean" = constant @true;
                            yield %28;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %29 : java.type:"java.lang.Object" = constant @null;
                            yield %29;
                        };
                    var.store %3 %27;
                    %30 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %31 : java.type:"java.lang.Integer" = invoke %30 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %32 : java.type:"java.lang.Integer" = cast %31 @java.type:"java.lang.Integer";
                    java.switch.statement %32
                        ()java.type:"boolean" -> {
                            %33 : java.type:"boolean" = constant @true;
                            yield %33;
                        }
                        ()java.type:"void" -> {
                            yield;
                        };
                    %34 : java.type:"ErasedAccessTest$BoundedInteger" = var.load %2;
                    %35 : java.type:"java.lang.Integer" = invoke %34 @java.ref:"ErasedAccessTest$BoundedInteger::getX():java.lang.Number";
                    %36 : java.type:"java.lang.Integer" = cast %35 @java.type:"java.lang.Integer";
                    %37 : java.type:"java.lang.Object" = java.switch.expression %36
                        ()java.type:"boolean" -> {
                            %38 : java.type:"boolean" = constant @true;
                            yield %38;
                        }
                        ()java.type:"java.lang.Object" -> {
                            %39 : java.type:"java.lang.Object" = constant @null;
                            yield %39;
                        };
                    var.store %3 %37;
                    return;
                };
                """)
        @Reflect
        void testSwitchSelector(BoundedInteger test) {
            Object o;

            // simple field name
            switch (x) {default -> { }}
            o = switch (x) { default -> null; };

            // qualified field name
            switch (test.x) {default -> { }}
            o = switch (test.x) { default -> null; };

            // simple method name
            switch (getX()) {default -> { }}
            o = switch (getX()) { default -> null; };

            // qualified method name
            switch (test.getX()) {default -> { }}
            o = switch (test.getX()) { default -> null; };
        }
    }


    static final String TEST_CLASSES_DIR = System.getProperty("test.classes", ".");
    static final Class<?>[] TEST_CLASSES = new Class<?>[] { UnboundedInteger.class, BoundedInteger.class };

    public static void main(String[] args) throws ReflectiveOperationException, IOException {
        for (Class<?> testClass : TEST_CLASSES) {
            ClassModel mod = ClassFile.of().parse(Path.of(TEST_CLASSES_DIR, testClass.getName() + ".class"));
            for (Method m : testClass.getDeclaredMethods()) {
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
