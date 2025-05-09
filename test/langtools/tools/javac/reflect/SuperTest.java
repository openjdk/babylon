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

import jdk.incubator.code.CodeReflection;

/*
 * @test
 * @summary Smoke test for code reflection with super qualified expressions.
 * @modules jdk.incubator.code
 * @build SuperTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SuperTest
 */

public class SuperTest extends SuperClass implements SuperInterface {
    static int sf;
    int f;

    @Override
    public void get() {}
    static void sget() {}

    @CodeReflection
    @IR("""
            func @"superClassFieldAccess" (%0 : java.type:"SuperTest")java.type:"void" -> {
                %1 : java.type:"int" = field.load %0 @"SuperClass::f:int";
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = constant @"1";
                field.store %0 %3 @"SuperClass::f:int";
                %4 : java.type:"int" = field.load %0 @"SuperClass::f:int";
                var.store %2 %4;
                %5 : java.type:"int" = constant @"1";
                field.store %0 %5 @"SuperClass::f:int";
                %6 : java.type:"int" = field.load @"SuperClass::sf:int";
                var.store %2 %6;
                %7 : java.type:"int" = constant @"1";
                field.store %7 @"SuperClass::sf:int";
                %8 : java.type:"int" = field.load @"SuperClass::sf:int";
                var.store %2 %8;
                %9 : java.type:"int" = constant @"1";
                field.store %9 @"SuperClass::sf:int";
                return;
            };
            """)
    public void superClassFieldAccess() {
        int i = super.f;
        super.f = 1;
        i = SuperTest.super.f;
        SuperTest.super.f = 1;

        i = super.sf;
        super.sf = 1;
        i = SuperTest.super.sf;
        SuperTest.super.sf = 1;
    }

    @CodeReflection
    @IR("""
            func @"superClassMethodInvocation" (%0 : java.type:"SuperTest")java.type:"void" -> {
                invoke %0 @"SuperClass::get():void" @invoke.kind="SUPER";
                invoke %0 @"SuperClass::get():void" @invoke.kind="SUPER";
                invoke @"SuperClass::sget():void";
                invoke @"SuperClass::sget():void";
                return;
            };
            """)
    public void superClassMethodInvocation() {
        super.get();
        SuperTest.super.get();

        super.sget();
        SuperTest.super.sget();
    }

    @CodeReflection
    @IR("""
            func @"superInterfaceMethodInvocation" (%0 : java.type:"SuperTest")java.type:"void" -> {
                invoke %0 @"SuperInterface::get():void" @invoke.kind="SUPER";
                return;
            };
            """)
    public void superInterfaceMethodInvocation() {
        SuperInterface.super.get();
    }
}

class SuperClass {
    static int sf;
    int f;

    void get() {}
    static void sget() {}
}

interface SuperInterface {
    default void get() {}
}
