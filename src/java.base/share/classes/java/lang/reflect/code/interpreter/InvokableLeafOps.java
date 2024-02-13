/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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

package java.lang.reflect.code.interpreter;

final class InvokableLeafOps {

    public static String add(String a, String b) {
        return a.concat(b);
    }


    public static boolean eq(Object a, Object b) {
        return a == b;
    }

    public static boolean neq(Object a, Object b) {
        return a != b;
    }


    public static boolean not(boolean l) {
        return !l;
    }

    // int

    public static int neg(int l) {
        return -l;
    }

    public static int add(int l, int r) {
        return l + r;
    }

    public static int sub(int l, int r) {
        return l - r;
    }

    public static int mul(int l, int r) {
        return l * r;
    }

    public static int div(int l, int r) {
        return l / r;
    }

    public static int mod(int l, int r) {
        return l % r;
    }

    public static int or(int l, int r) {
        return l | r;
    }

    public static int and(int l, int r) {
        return l & r;
    }

    public static int xor(int l, int r) {
        return l ^ r;
    }

    public static boolean eq(int l, int r) {
        return l == r;
    }

    public static boolean neq(int l, int r) {
        return l != r;
    }

    public static boolean gt(int l, int r) {
        return l > r;
    }

    public static boolean ge(int l, int r) {
        return l >= r;
    }

    public static boolean lt(int l, int r) {
        return l < r;
    }

    public static boolean le(int l, int r) {
        return l <= r;
    }

   // long

    public static long neg(long l) {
        return -l;
    }

    public static long add(long l, long r) {
        return l + r;
    }

    public static long sub(long l, long r) {
        return l - r;
    }

    public static long mul(long l, long r) {
        return l * r;
    }

    public static long div(long l, long r) {
        return l / r;
    }

    public static long mod(long l, long r) {
        return l % r;
    }

    public static long or(long l, long r) {
        return l | r;
    }

    public static long and(long l, long r) {
        return l & r;
    }

    public static long xor(long l, long r) {
        return l ^ r;
    }


    public static boolean eq(long l, long r) {
        return l == r;
    }

    public static boolean neq(long l, long r) {
        return l != r;
    }

    public static boolean gt(long l, long r) {
        return l > r;
    }

    public static boolean ge(long l, long r) {
        return l >= r;
    }

    public static boolean lt(long l, long r) {
        return l < r;
    }

    public static boolean le(long l, long r) {
        return l <= r;
    }



    // float

    static float neg(float l) {
        return -l;
    }

    static float add(float l, float r) {
        return l + r;
    }

    static float sub(float l, float r) {
        return l - r;
    }

    static float mul(float l, float r) {
        return l * r;
    }

    static float div(float l, float r) {
        return l / r;
    }

    static float mod(float l, float r) {
        return l % r;
    }

    public static boolean eq(float l, float r) {
        return l == r;
    }

    public static boolean neq(float l, float r) {
        return l != r;
    }

    public static boolean gt(float l, float r) {
        return l > r;
    }

    public static boolean ge(float l, float r) {
        return l >= r;
    }

    public static boolean lt(float l, float r) {
        return l < r;
    }

    public static boolean le(float l, float r) {
        return l <= r;
    }



    // double

    static double neg(double l) {
        return -l;
    }

    static double add(double l, double r) {
        return l + r;
    }

    static double sub(double l, double r) {
        return l - r;
    }

    static double mul(double l, double r) {
        return l * r;
    }

    static double div(double l, double r) {
        return l / r;
    }

    static double mod(double l, double r) {
        return l % r;
    }



    // boolean

    static boolean eq(boolean l, boolean r) {
        return l == r;
    }

    static boolean neq(boolean l, boolean r) {
        return l != r;
    }

    static boolean and(boolean l, boolean r) {
        return l & r;
    }

    static boolean or(boolean l, boolean r) {
        return l | r;
    }

    static boolean xor(boolean l, boolean r) {
        return l ^ r;
    }


    // Primitive conversions

    // double conversion
    static double conv_double(double i) {
        return i;
    }
    static float conv_float(double i) {
        return (float) i;
    }
    static long conv_long(double i) {
        return (long) i;
    }
    static int conv_int(double i) {
        return (int) i;
    }
    static short conv_short(double i) {
        return (short) i;
    }
    static char conv_char(double i) {
        return (char) i;
    }
    static byte conv_byte(double i) {
        return (byte) i;
    }

    // float conversion
    static double conv_double(float i) {
        return i;
    }
    static float conv_float(float i) {
        return i;
    }
    static long conv_long(float i) {
        return (long) i;
    }
    static int conv_int(float i) {
        return (int) i;
    }
    static short conv_short(float i) {
        return (short) i;
    }
    static char conv_char(float i) {
        return (char) i;
    }
    static byte conv_byte(float i) {
        return (byte) i;
    }

    // long conversion
    static double conv_double(long i) {
        return (double) i;
    }
    static float conv_float(long i) {
        return (float) i;
    }
    static long conv_long(long i) {
        return i;
    }
    static int conv_int(long i) {
        return (int) i;
    }
    static short conv_short(long i) {
        return (short) i;
    }
    static char conv_char(long i) {
        return (char) i;
    }
    static byte conv_byte(long i) {
        return (byte) i;
    }

    // int conversion
    static double conv_double(int i) {
        return (double) i;
    }
    static float conv_float(int i) {
        return (float) i;
    }
    static long conv_long(int i) {
        return i;
    }
    static int conv_int(int i) {
        return i;
    }
    static short conv_short(int i) {
        return (short) i;
    }
    static char conv_char(int i) {
        return (char) i;
    }
    static byte conv_byte(int i) {
        return (byte) i;
    }

    // short conversion
    static double conv_double(short i) {
        return i;
    }
    static float conv_float(short i) {
        return i;
    }
    static long conv_long(short i) {
        return i;
    }
    static int conv_int(short i) {
        return i;
    }
    static short conv_short(short i) {
        return i;
    }
    static char conv_char(short i) {
        return (char) i;
    }
    static byte conv_byte(short i) {
        return (byte) i;
    }

    // char conversion
    static double conv_double(char i) {
        return i;
    }
    static float conv_float(char i) {
        return i;
    }
    static long conv_long(char i) {
        return i;
    }
    static int conv_int(char i) {
        return i;
    }
    static short conv_short(char i) {
        return (short) i;
    }
    static char conv_char(char i) {
        return i;
    }
    static byte conv_byte(char i) {
        return (byte) i;
    }

    // byte conversion
    static double conv_double(byte i) {
        return i;
    }
    static float conv_float(byte i) {
        return i;
    }
    static long conv_long(byte i) {
        return i;
    }
    static int conv_int(byte i) {
        return i;
    }
    static short conv_short(byte i) {
        return i;
    }
    static char conv_char(byte i) {
        return (char) i;
    }
    static byte conv_byte(byte i) {
        return i;
    }
}
