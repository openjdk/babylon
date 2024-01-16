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
 * @build PrimitiveCastTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester PrimitiveCastTest
 */

import java.lang.runtime.CodeReflection;

public class PrimitiveCastTest {

    @CodeReflection
    @IR("""
            func @"testFromDouble" (%0 : PrimitiveCastTest, %1 : double)void -> {
                %2 : Var<double> = var %1 @"v";
                %3 : double = var.load %2;
                %4 : Var<double> = var %3 @"d";
                %5 : double = var.load %2;
                %6 : float = conv %5;
                %7 : Var<float> = var %6 @"f";
                %8 : double = var.load %2;
                %9 : long = conv %8;
                %10 : Var<long> = var %9 @"l";
                %11 : double = var.load %2;
                %12 : int = conv %11;
                %13 : Var<int> = var %12 @"i";
                %14 : double = var.load %2;
                %15 : short = conv %14;
                %16 : Var<short> = var %15 @"s";
                %17 : double = var.load %2;
                %18 : char = conv %17;
                %19 : Var<char> = var %18 @"c";
                %20 : double = var.load %2;
                %21 : byte = conv %20;
                %22 : Var<byte> = var %21 @"b";
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
            func @"testFromFloat" (%0 : PrimitiveCastTest, %1 : float)void -> {
                %2 : Var<float> = var %1 @"v";
                %3 : float = var.load %2;
                %4 : double = conv %3;
                %5 : Var<double> = var %4 @"d";
                %6 : float = var.load %2;
                %7 : Var<float> = var %6 @"f";
                %8 : float = var.load %2;
                %9 : long = conv %8;
                %10 : Var<long> = var %9 @"l";
                %11 : float = var.load %2;
                %12 : int = conv %11;
                %13 : Var<int> = var %12 @"i";
                %14 : float = var.load %2;
                %15 : short = conv %14;
                %16 : Var<short> = var %15 @"s";
                %17 : float = var.load %2;
                %18 : char = conv %17;
                %19 : Var<char> = var %18 @"c";
                %20 : float = var.load %2;
                %21 : byte = conv %20;
                %22 : Var<byte> = var %21 @"b";
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
            func @"testFromLong" (%0 : PrimitiveCastTest, %1 : long)void -> {
                %2 : Var<long> = var %1 @"v";
                %3 : long = var.load %2;
                %4 : double = conv %3;
                %5 : Var<double> = var %4 @"d";
                %6 : long = var.load %2;
                %7 : float = conv %6;
                %8 : Var<float> = var %7 @"f";
                %9 : long = var.load %2;
                %10 : Var<long> = var %9 @"l";
                %11 : long = var.load %2;
                %12 : int = conv %11;
                %13 : Var<int> = var %12 @"i";
                %14 : long = var.load %2;
                %15 : short = conv %14;
                %16 : Var<short> = var %15 @"s";
                %17 : long = var.load %2;
                %18 : char = conv %17;
                %19 : Var<char> = var %18 @"c";
                %20 : long = var.load %2;
                %21 : byte = conv %20;
                %22 : Var<byte> = var %21 @"b";
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
            func @"testFromInt" (%0 : PrimitiveCastTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"v";
                %3 : int = var.load %2;
                %4 : double = conv %3;
                %5 : Var<double> = var %4 @"d";
                %6 : int = var.load %2;
                %7 : float = conv %6;
                %8 : Var<float> = var %7 @"f";
                %9 : int = var.load %2;
                %10 : long = conv %9;
                %11 : Var<long> = var %10 @"l";
                %12 : int = var.load %2;
                %13 : Var<int> = var %12 @"i";
                %14 : int = var.load %2;
                %15 : short = conv %14;
                %16 : Var<short> = var %15 @"s";
                %17 : int = var.load %2;
                %18 : char = conv %17;
                %19 : Var<char> = var %18 @"c";
                %20 : int = var.load %2;
                %21 : byte = conv %20;
                %22 : Var<byte> = var %21 @"b";
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
            func @"testFromShort" (%0 : PrimitiveCastTest, %1 : short)void -> {
                %2 : Var<short> = var %1 @"v";
                %3 : short = var.load %2;
                %4 : double = conv %3;
                %5 : Var<double> = var %4 @"d";
                %6 : short = var.load %2;
                %7 : float = conv %6;
                %8 : Var<float> = var %7 @"f";
                %9 : short = var.load %2;
                %10 : long = conv %9;
                %11 : Var<long> = var %10 @"l";
                %12 : short = var.load %2;
                %13 : int = conv %12;
                %14 : Var<int> = var %13 @"i";
                %15 : short = var.load %2;
                %16 : Var<short> = var %15 @"s";
                %17 : short = var.load %2;
                %18 : char = conv %17;
                %19 : Var<char> = var %18 @"c";
                %20 : short = var.load %2;
                %21 : byte = conv %20;
                %22 : Var<byte> = var %21 @"b";
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
            func @"testFromChar" (%0 : PrimitiveCastTest, %1 : char)void -> {
                %2 : Var<char> = var %1 @"v";
                %3 : char = var.load %2;
                %4 : double = conv %3;
                %5 : Var<double> = var %4 @"d";
                %6 : char = var.load %2;
                %7 : float = conv %6;
                %8 : Var<float> = var %7 @"f";
                %9 : char = var.load %2;
                %10 : long = conv %9;
                %11 : Var<long> = var %10 @"l";
                %12 : char = var.load %2;
                %13 : int = conv %12;
                %14 : Var<int> = var %13 @"i";
                %15 : char = var.load %2;
                %16 : short = conv %15;
                %17 : Var<short> = var %16 @"s";
                %18 : char = var.load %2;
                %19 : Var<char> = var %18 @"c";
                %20 : char = var.load %2;
                %21 : byte = conv %20;
                %22 : Var<byte> = var %21 @"b";
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
            func @"testFromByte" (%0 : PrimitiveCastTest, %1 : byte)void -> {
                %2 : Var<byte> = var %1 @"v";
                %3 : byte = var.load %2;
                %4 : double = conv %3;
                %5 : Var<double> = var %4 @"d";
                %6 : byte = var.load %2;
                %7 : float = conv %6;
                %8 : Var<float> = var %7 @"f";
                %9 : byte = var.load %2;
                %10 : long = conv %9;
                %11 : Var<long> = var %10 @"l";
                %12 : byte = var.load %2;
                %13 : int = conv %12;
                %14 : Var<int> = var %13 @"i";
                %15 : byte = var.load %2;
                %16 : short = conv %15;
                %17 : Var<short> = var %16 @"s";
                %18 : byte = var.load %2;
                %19 : char = conv %18;
                %20 : Var<char> = var %19 @"c";
                %21 : byte = var.load %2;
                %22 : Var<byte> = var %21 @"b";
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
