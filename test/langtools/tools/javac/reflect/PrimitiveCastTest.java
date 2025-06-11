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

/*
 * @test
 * @summary Smoke test for code reflection with primitive casts.
 * @modules jdk.incubator.code
 * @build PrimitiveCastTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester PrimitiveCastTest
 */

import jdk.incubator.code.CodeReflection;

public class PrimitiveCastTest {

    @CodeReflection
    @IR("""
            func @"testFromDouble" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"double")java.type:"void" -> {
                %2 : Var<java.type:"double"> = var %1 @"v";
                %3 : java.type:"double" = var.load %2;
                %4 : Var<java.type:"double"> = var %3 @"d";
                %5 : java.type:"double" = var.load %2;
                %6 : java.type:"float" = conv %5;
                %7 : Var<java.type:"float"> = var %6 @"f";
                %8 : java.type:"double" = var.load %2;
                %9 : java.type:"long" = conv %8;
                %10 : Var<java.type:"long"> = var %9 @"l";
                %11 : java.type:"double" = var.load %2;
                %12 : java.type:"int" = conv %11;
                %13 : Var<java.type:"int"> = var %12 @"i";
                %14 : java.type:"double" = var.load %2;
                %15 : java.type:"short" = conv %14;
                %16 : Var<java.type:"short"> = var %15 @"s";
                %17 : java.type:"double" = var.load %2;
                %18 : java.type:"char" = conv %17;
                %19 : Var<java.type:"char"> = var %18 @"c";
                %20 : java.type:"double" = var.load %2;
                %21 : java.type:"byte" = conv %20;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromDouble(double v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }

    @CodeReflection
    @IR("""
            func @"testFromFloat" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"float")java.type:"void" -> {
                %2 : Var<java.type:"float"> = var %1 @"v";
                %3 : java.type:"float" = var.load %2;
                %4 : java.type:"double" = conv %3;
                %5 : Var<java.type:"double"> = var %4 @"d";
                %6 : java.type:"float" = var.load %2;
                %7 : Var<java.type:"float"> = var %6 @"f";
                %8 : java.type:"float" = var.load %2;
                %9 : java.type:"long" = conv %8;
                %10 : Var<java.type:"long"> = var %9 @"l";
                %11 : java.type:"float" = var.load %2;
                %12 : java.type:"int" = conv %11;
                %13 : Var<java.type:"int"> = var %12 @"i";
                %14 : java.type:"float" = var.load %2;
                %15 : java.type:"short" = conv %14;
                %16 : Var<java.type:"short"> = var %15 @"s";
                %17 : java.type:"float" = var.load %2;
                %18 : java.type:"char" = conv %17;
                %19 : Var<java.type:"char"> = var %18 @"c";
                %20 : java.type:"float" = var.load %2;
                %21 : java.type:"byte" = conv %20;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromFloat(float v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }

    @CodeReflection
    @IR("""
            func @"testFromLong" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"long")java.type:"void" -> {
                %2 : Var<java.type:"long"> = var %1 @"v";
                %3 : java.type:"long" = var.load %2;
                %4 : java.type:"double" = conv %3;
                %5 : Var<java.type:"double"> = var %4 @"d";
                %6 : java.type:"long" = var.load %2;
                %7 : java.type:"float" = conv %6;
                %8 : Var<java.type:"float"> = var %7 @"f";
                %9 : java.type:"long" = var.load %2;
                %10 : Var<java.type:"long"> = var %9 @"l";
                %11 : java.type:"long" = var.load %2;
                %12 : java.type:"int" = conv %11;
                %13 : Var<java.type:"int"> = var %12 @"i";
                %14 : java.type:"long" = var.load %2;
                %15 : java.type:"short" = conv %14;
                %16 : Var<java.type:"short"> = var %15 @"s";
                %17 : java.type:"long" = var.load %2;
                %18 : java.type:"char" = conv %17;
                %19 : Var<java.type:"char"> = var %18 @"c";
                %20 : java.type:"long" = var.load %2;
                %21 : java.type:"byte" = conv %20;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromLong(long v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }

    @CodeReflection
    @IR("""
            func @"testFromInt" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"v";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"double" = conv %3;
                %5 : Var<java.type:"double"> = var %4 @"d";
                %6 : java.type:"int" = var.load %2;
                %7 : java.type:"float" = conv %6;
                %8 : Var<java.type:"float"> = var %7 @"f";
                %9 : java.type:"int" = var.load %2;
                %10 : java.type:"long" = conv %9;
                %11 : Var<java.type:"long"> = var %10 @"l";
                %12 : java.type:"int" = var.load %2;
                %13 : Var<java.type:"int"> = var %12 @"i";
                %14 : java.type:"int" = var.load %2;
                %15 : java.type:"short" = conv %14;
                %16 : Var<java.type:"short"> = var %15 @"s";
                %17 : java.type:"int" = var.load %2;
                %18 : java.type:"char" = conv %17;
                %19 : Var<java.type:"char"> = var %18 @"c";
                %20 : java.type:"int" = var.load %2;
                %21 : java.type:"byte" = conv %20;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromInt(int v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }

    @CodeReflection
    @IR("""
            func @"testFromShort" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"short")java.type:"void" -> {
                %2 : Var<java.type:"short"> = var %1 @"v";
                %3 : java.type:"short" = var.load %2;
                %4 : java.type:"double" = conv %3;
                %5 : Var<java.type:"double"> = var %4 @"d";
                %6 : java.type:"short" = var.load %2;
                %7 : java.type:"float" = conv %6;
                %8 : Var<java.type:"float"> = var %7 @"f";
                %9 : java.type:"short" = var.load %2;
                %10 : java.type:"long" = conv %9;
                %11 : Var<java.type:"long"> = var %10 @"l";
                %12 : java.type:"short" = var.load %2;
                %13 : java.type:"int" = conv %12;
                %14 : Var<java.type:"int"> = var %13 @"i";
                %15 : java.type:"short" = var.load %2;
                %16 : Var<java.type:"short"> = var %15 @"s";
                %17 : java.type:"short" = var.load %2;
                %18 : java.type:"char" = conv %17;
                %19 : Var<java.type:"char"> = var %18 @"c";
                %20 : java.type:"short" = var.load %2;
                %21 : java.type:"byte" = conv %20;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromShort(short v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }

    @CodeReflection
    @IR("""
            func @"testFromChar" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"char")java.type:"void" -> {
                %2 : Var<java.type:"char"> = var %1 @"v";
                %3 : java.type:"char" = var.load %2;
                %4 : java.type:"double" = conv %3;
                %5 : Var<java.type:"double"> = var %4 @"d";
                %6 : java.type:"char" = var.load %2;
                %7 : java.type:"float" = conv %6;
                %8 : Var<java.type:"float"> = var %7 @"f";
                %9 : java.type:"char" = var.load %2;
                %10 : java.type:"long" = conv %9;
                %11 : Var<java.type:"long"> = var %10 @"l";
                %12 : java.type:"char" = var.load %2;
                %13 : java.type:"int" = conv %12;
                %14 : Var<java.type:"int"> = var %13 @"i";
                %15 : java.type:"char" = var.load %2;
                %16 : java.type:"short" = conv %15;
                %17 : Var<java.type:"short"> = var %16 @"s";
                %18 : java.type:"char" = var.load %2;
                %19 : Var<java.type:"char"> = var %18 @"c";
                %20 : java.type:"char" = var.load %2;
                %21 : java.type:"byte" = conv %20;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromChar(char v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }

    @CodeReflection
    @IR("""
            func @"testFromByte" (%0 : java.type:"PrimitiveCastTest", %1 : java.type:"byte")java.type:"void" -> {
                %2 : Var<java.type:"byte"> = var %1 @"v";
                %3 : java.type:"byte" = var.load %2;
                %4 : java.type:"double" = conv %3;
                %5 : Var<java.type:"double"> = var %4 @"d";
                %6 : java.type:"byte" = var.load %2;
                %7 : java.type:"float" = conv %6;
                %8 : Var<java.type:"float"> = var %7 @"f";
                %9 : java.type:"byte" = var.load %2;
                %10 : java.type:"long" = conv %9;
                %11 : Var<java.type:"long"> = var %10 @"l";
                %12 : java.type:"byte" = var.load %2;
                %13 : java.type:"int" = conv %12;
                %14 : Var<java.type:"int"> = var %13 @"i";
                %15 : java.type:"byte" = var.load %2;
                %16 : java.type:"short" = conv %15;
                %17 : Var<java.type:"short"> = var %16 @"s";
                %18 : java.type:"byte" = var.load %2;
                %19 : java.type:"char" = conv %18;
                %20 : Var<java.type:"char"> = var %19 @"c";
                %21 : java.type:"byte" = var.load %2;
                %22 : Var<java.type:"byte"> = var %21 @"b";
                return;
            };
            """)
    @SuppressWarnings("cast")
    void testFromByte(byte v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
    }
}
