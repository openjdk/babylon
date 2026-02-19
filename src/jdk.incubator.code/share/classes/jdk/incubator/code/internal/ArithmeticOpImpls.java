package jdk.incubator.code.internal;

import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

public final class ArithmeticOpImpls {

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

    public static int compl(int l) {
        return ~l;
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

    public static int lshl(int l, int r) {
        return l << r;
    }

    public static int ashr(int l, int r) {
        return l >> r;
    }

    public static int lshr(int l, int r) {
        return l >>> r;
    }

    public static int lshl(int l, long r) {
        return l << r;
    }

    public static int ashr(int l, long r) {
        return l >> r;
    }

    public static int lshr(int l, long r) {
        return l >>> r;
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

    // byte

    public static byte neg(byte l) {
        return (byte) -l;
    }

    public static byte compl(byte l) {
        return (byte) ~l;
    }

    public static byte add(byte l, byte r) {
        return (byte) (l + r);
    }

    public static byte sub(byte l, byte r) {
        return (byte) (l - r);
    }

    public static byte mul(byte l, byte r) {
        return (byte) (l * r);
    }

    public static byte div(byte l, byte r) {
        return (byte) (l / r);
    }

    public static byte mod(byte l, byte r) {
        return (byte) (l % r);
    }

    public static byte or(byte l, byte r) {
        return (byte) (l | r);
    }

    public static byte and(byte l, byte r) {
        return (byte) (l & r);
    }

    public static byte xor(byte l, byte r) {
        return (byte) (l ^ r);
    }

    public static byte ashr(byte l, long r) {
        return (byte) (l >> r);
    }

    public static byte lshr(byte l, long r) {
        return (byte) (l >>> r);
    }

    public static byte lshl(byte l, int r) {
        return (byte) (l << r);
    }

    public static byte ashr(byte l, int r) {
        return (byte) (l >> r);
    }

    public static byte lshr(byte l, int r) {
        return (byte) (l >>> r);
    }

    public static boolean eq(byte l, byte r) {
        return l == r;
    }

    public static boolean neq(byte l, byte r) {
        return l != r;
    }

    public static boolean gt(byte l, byte r) {
        return l > r;
    }

    public static boolean ge(byte l, byte r) {
        return l >= r;
    }

    public static boolean lt(byte l, byte r) {
        return l < r;
    }

    public static boolean le(byte l, byte r) {
        return l <= r;
    }

    // short

    public static short neg(short l) {
        return (short) -l;
    }

    public static short compl(short l) {
        return (short) ~l;
    }

    public static short add(short l, short r) {
        return (short) (l + r);
    }

    public static short sub(short l, short r) {
        return (short) (l - r);
    }

    public static short mul(short l, short r) {
        return (short) (l * r);
    }

    public static short div(short l, short r) {
        return (short) (l / r);
    }

    public static short mod(short l, short r) {
        return (short) (l % r);
    }

    public static short or(short l, short r) {
        return (short) (l | r);
    }

    public static short and(short l, short r) {
        return (short) (l & r);
    }

    public static short xor(short l, short r) {
        return (short) (l ^ r);
    }

    public static short ashr(short l, long r) {
        return (short) (l >> r);
    }

    public static short lshr(short l, long r) {
        return (short) (l >>> r);
    }

    public static short lshl(short l, int r) {
        return (short) (l << r);
    }

    public static short ashr(short l, int r) {
        return (short) (l >> r);
    }

    public static short lshr(short l, int r) {
        return (short) (l >>> r);
    }

    public static boolean eq(short l, short r) {
        return l == r;
    }

    public static boolean neq(short l, short r) {
        return l != r;
    }

    public static boolean gt(short l, short r) {
        return l > r;
    }

    public static boolean ge(short l, short r) {
        return l >= r;
    }

    public static boolean lt(short l, short r) {
        return l < r;
    }

    public static boolean le(short l, short r) {
        return l <= r;
    }

    // char

    public static char neg(char l) {
        return (char) -l;
    }

    public static char compl(char l) {
        return (char) ~l;
    }

    public static char add(char l, char r) {
        return (char) (l + r);
    }

    public static char sub(char l, char r) {
        return (char) (l - r);
    }

    public static char mul(char l, char r) {
        return (char) (l * r);
    }

    public static char div(char l, char r) {
        return (char) (l / r);
    }

    public static char mod(char l, char r) {
        return (char) (l % r);
    }

    public static char or(char l, char r) {
        return (char) (l | r);
    }

    public static char and(char l, char r) {
        return (char) (l & r);
    }

    public static char xor(char l, char r) {
        return (char) (l ^ r);
    }

    public static char ashr(char l, long r) {
        return (char) (l >> r);
    }

    public static char lshr(char l, long r) {
        return (char) (l >>> r);
    }

    public static char lshl(char l, int r) {
        return (char) (l << r);
    }

    public static char ashr(char l, int r) {
        return (char) (l >> r);
    }

    public static char lshr(char l, int r) {
        return (char) (l >>> r);
    }

    public static boolean eq(char l, char r) {
        return l == r;
    }

    public static boolean neq(char l, char r) {
        return l != r;
    }

    public static boolean gt(char l, char r) {
        return l > r;
    }

    public static boolean ge(char l, char r) {
        return l >= r;
    }

    public static boolean lt(char l, char r) {
        return l < r;
    }

    public static boolean le(char l, char r) {
        return l <= r;
    }
    // long

    public static long neg(long l) {
        return -l;
    }

    public static long compl(long l) {
        return ~l;
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

    public static long lshl(long l, long r) {
        return l << r;
    }

    public static long ashr(long l, long r) {
        return l >> r;
    }

    public static long lshr(long l, long r) {
        return l >>> r;
    }

    public static long lshl(long l, int r) {
        return l << r;
    }

    public static long ashr(long l, int r) {
        return l >> r;
    }

    public static long lshr(long l, int r) {
        return l >>> r;
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

    public static float neg(float l) {
        return -l;
    }

    public static float add(float l, float r) {
        return l + r;
    }

    public static float sub(float l, float r) {
        return l - r;
    }

    public static float mul(float l, float r) {
        return l * r;
    }

    public static float div(float l, float r) {
        return l / r;
    }

    public static float mod(float l, float r) {
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

    public static double neg(double l) {
        return -l;
    }

    public static double add(double l, double r) {
        return l + r;
    }

    public static double sub(double l, double r) {
        return l - r;
    }

    public static double mul(double l, double r) {
        return l * r;
    }

    public static double div(double l, double r) {
        return l / r;
    }

    public static double mod(double l, double r) {
        return l % r;
    }

    public static boolean eq(double l, double r) {
        return l == r;
    }

    public static boolean neq(double l, double r) {
        return l != r;
    }

    public static boolean gt(double l, double r) {
        return l > r;
    }

    public static boolean ge(double l, double r) {
        return l >= r;
    }

    public static boolean lt(double l, double r) {
        return l < r;
    }

    public static boolean le(double l, double r) {
        return l <= r;
    }


    // boolean

    public static boolean eq(boolean l, boolean r) {
        return l == r;
    }

    public static boolean neq(boolean l, boolean r) {
        return l != r;
    }

    public static boolean and(boolean l, boolean r) {
        return l & r;
    }

    public static boolean or(boolean l, boolean r) {
        return l | r;
    }

    public static boolean xor(boolean l, boolean r) {
        return l ^ r;
    }


    // Primitive conversions

    // double conversion
    public static double conv_double(double i) {
        return i;
    }
    public static float conv_float(double i) {
        return (float) i;
    }
    public static long conv_long(double i) {
        return (long) i;
    }
    public static int conv_int(double i) {
        return (int) i;
    }
    public static short conv_short(double i) {
        return (short) i;
    }
    public static char conv_char(double i) {
        return (char) i;
    }
    public static byte conv_byte(double i) {
        return (byte) i;
    }
    public static boolean conv_boolean(double i) {
        return ((int)i & 1) == 1;
    }

    // float conversion
    public static double conv_double(float i) {
        return i;
    }
    public static float conv_float(float i) {
        return i;
    }
    public static long conv_long(float i) {
        return (long) i;
    }
    public static int conv_int(float i) {
        return (int) i;
    }
    public static short conv_short(float i) {
        return (short) i;
    }
    public static char conv_char(float i) {
        return (char) i;
    }
    public static byte conv_byte(float i) {
        return (byte) i;
    }
    public static boolean conv_boolean(float i) {
        return ((int)i & 1) == 1;
    }

    // long conversion
    public static double conv_double(long i) {
        return (double) i;
    }
    public static float conv_float(long i) {
        return (float) i;
    }
    public static long conv_long(long i) {
        return i;
    }
    public static int conv_int(long i) {
        return (int) i;
    }
    public static short conv_short(long i) {
        return (short) i;
    }
    public static char conv_char(long i) {
        return (char) i;
    }
    public static byte conv_byte(long i) {
        return (byte) i;
    }
    public static boolean conv_boolean(long i) {
        return (i & 1) == 1;
    }

    // int conversion
    public static double conv_double(int i) {
        return (double) i;
    }
    public static float conv_float(int i) {
        return (float) i;
    }
    public static long conv_long(int i) {
        return i;
    }
    public static int conv_int(int i) {
        return i;
    }
    public static short conv_short(int i) {
        return (short) i;
    }
    public static char conv_char(int i) {
        return (char) i;
    }
    public static byte conv_byte(int i) {
        return (byte) i;
    }
    public static boolean conv_boolean(int i) {
        return (i & 1) == 1;
    }

    // short conversion
    public static double conv_double(short i) {
        return i;
    }
    public static float conv_float(short i) {
        return i;
    }
    public static long conv_long(short i) {
        return i;
    }
    public static int conv_int(short i) {
        return i;
    }
    public static short conv_short(short i) {
        return i;
    }
    public static char conv_char(short i) {
        return (char) i;
    }
    public static byte conv_byte(short i) {
        return (byte) i;
    }
    public static boolean conv_boolean(short i) {
        return (i & 1) == 1;
    }

    // char conversion
    public static double conv_double(char i) {
        return i;
    }
    public static float conv_float(char i) {
        return i;
    }
    public static long conv_long(char i) {
        return i;
    }
    public static int conv_int(char i) {
        return i;
    }
    public static short conv_short(char i) {
        return (short) i;
    }
    public static char conv_char(char i) {
        return i;
    }
    public static byte conv_byte(char i) {
        return (byte) i;
    }
    public static boolean conv_boolean(char i) {
        return (i & 1) == 1;
    }

    // byte conversion
    public static double conv_double(byte i) {
        return i;
    }
    public static float conv_float(byte i) {
        return i;
    }
    public static long conv_long(byte i) {
        return i;
    }
    public static int conv_int(byte i) {
        return i;
    }
    public static short conv_short(byte i) {
        return i;
    }
    public static char conv_char(byte i) {
        return (char) i;
    }
    public static byte conv_byte(byte i) {
        return i;
    }
    public static boolean conv_boolean(byte i) {
        return (i & 1) == 1;
    }

    // boolean conversion
    public static double conv_double(boolean i) {
        return i ? 1d : 0d;
    }
    public static float conv_float(boolean i) {
        return i ? 1f : 0f;
    }
    public static long conv_long(boolean i) {
        return i ? 1l : 0l;
    }
    public static int conv_int(boolean i) {
        return i ? 1 : 0;
    }
    public static short conv_short(boolean i) {
        return i ? (short)1 : 0;
    }
    public static char conv_char(boolean i) {
        return i ? (char)1 : 0;
    }
    public static byte conv_byte(boolean i) {
        return i ? (byte)1 : 0;
    }
    public static boolean conv_boolean(boolean i) {
        return i;
    }

    // resolution
    public static MethodType resolveToMethodType(MethodHandles.Lookup l, FunctionType ft) {
        try {
            return MethodRef.toNominalDescriptor(ft).resolveConstantDesc(l);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static MethodHandle opHandle(MethodHandles.Lookup l, String opName, FunctionType ft) {
        MethodType mt = resolveToMethodType(l, ft).erase();
        try {
            return MethodHandles.lookup().findStatic(ArithmeticOpImpls.class, opName, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
