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

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.components.ClassPrinter;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng/othervm -Djdk.invoke.MethodHandle.dumpClassFiles=true TestBytecode
 */

public class TestBytecode {

    @CodeReflection
    static int intNumOps(int i, int j, int k) {
        k++;
        i = (i + j) / k - i % j;
        i--;
        return i;
    }

    @CodeReflection
    static byte byteNumOps(byte i, byte j, byte k) {
        k++;
        i = (byte) ((i + j) / k - i % j);
        i--;
        return i;
    }

    @CodeReflection
    static short shortNumOps(short i, short j, short k) {
        k++;
        i = (short) ((i + j) / k - i % j);
        i--;
        return i;
    }

    @CodeReflection
    static char charNumOps(char i, char j, char k) {
        k++;
        i = (char) ((i + j) / k - i % j);
        i--;
        return i;
    }

    @CodeReflection
    static long longNumOps(long i, long j, long k) {
        k++;
        i = (i + j) / k - i % j;
        i--;
        return i;
    }

    @CodeReflection
    static float floatNumOps(float i, float j, float k) {
        k++;
        i = (i + j) / k - i % j;
        i--;
        return i;
    }

    @CodeReflection
    static double doubleNumOps(double i, double j, double k) {
        k++;
        i = (i + j) / k - i % j;
        i--;
        return i;
    }

    @CodeReflection
    static int intBitOps(int i, int j, int k) {
        return i & j | k ^ j;
    }

    @CodeReflection
    static byte byteBitOps(byte i, byte j, byte k) {
        return (byte) (i & j | k ^ j);
    }

    @CodeReflection
    static short shortBitOps(short i, short j, short k) {
        return (short) (i & j | k ^ j);
    }

    @CodeReflection
    static char charBitOps(char i, char j, char k) {
        return (char) (i & j | k ^ j);
    }

    @CodeReflection
    static long longBitOps(long i, long j, long k) {
        return i & j | k ^ j;
    }

    @CodeReflection
    static boolean boolBitOps(boolean i, boolean j, boolean k) {
        return i & j | k ^ j;
    }

    @CodeReflection
    static int intShiftOps(int i, int j, int k) {
        return ((-1 >> i) << (j << k)) >>> (k - j);
    }

    @CodeReflection
    static byte byteShiftOps(byte i, byte j, byte k) {
        return (byte) (((-1 >> i) << (j << k)) >>> (k - j));
    }

    @CodeReflection
    static short shortShiftOps(short i, short j, short k) {
        return (short) (((-1 >> i) << (j << k)) >>> (k - j));
    }

    @CodeReflection
    static char charShiftOps(char i, char j, char k) {
        return (char) (((-1 >> i) << (j << k)) >>> (k - j));
    }

    @CodeReflection
    static long longShiftOps(long i, long j, long k) {
        return ((-1 >> i) << (j << k)) >>> (k - j);
    }

    @CodeReflection
    static Object[] boxingAndUnboxing(int i, byte b, short s, char c, Integer ii, Byte bb, Short ss, Character cc) {
        ii += i; ii += b; ii += s; ii += c;
        i += ii; i += bb; i += ss; i += cc;
        b += ii; b += bb; b += ss; b += cc;
        s += ii; s += bb; s += ss; s += cc;
        c += ii; c += bb; c += ss; c += cc;
        return new Object[]{i, b, s, c};
    }

    @CodeReflection
    static String constructor(String s, int i, int j) {
        return new String(s.getBytes(), i, j);
    }

    @CodeReflection
    static Class<?> classArray(int i, int j) {
        Class<?>[] ifaces = new Class[1 + i + j];
        ifaces[0] = Function.class;
        return ifaces[0];
    }

    @CodeReflection
    static String[] stringArray(int i, int j) {
        return new String[i];
    }

    @CodeReflection
    static String[][] stringArray2(int i, int j) {
        return new String[i][];
    }

    @CodeReflection
    static String[][] stringArrayMulti(int i, int j) {
        return new String[i][j];
    }

    @CodeReflection
    static int[][] initializedIntArray(int i, int j) {
        return new int[][]{{i, j}, {i + j}};
    }

    @CodeReflection
    static int ifElseCompare(int i, int j) {
        if (i < 3) {
            i += 1;
        } else {
            j += 2;
        }
        return i + j;
    }

    @CodeReflection
    static int ifElseEquality(int i, int j) {
        if (j != 0) {
            if (i != 0) {
                i += 1;
            } else {
                i += 2;
            }
        } else {
            if (j != 0) {
                i += 3;
            } else {
                i += 4;
            }
        }
        return i;
    }

    @CodeReflection
    static int conditionalExpr(int i, int j) {
        return ((i - 1 >= 0) ? i - 1 : j - 1);
    }

    @CodeReflection
    static int nestedConditionalExpr(int i, int j) {
        return (i < 2) ? (j < 3) ? i : j : i + j;
    }

    @CodeReflection
    static int tryFinally(int i, int j) {
        try {
            i = i + j;
        } finally {
            i = i + j;
        }
        return i;
    }

    public record A(String s) {}

    @CodeReflection
    static A newWithArgs(int i, int j) {
        return new A("hello world".substring(i, i + j));
    }

    @CodeReflection
    static int loop(int n, int j) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum = sum + j;
        }
        return sum;
    }


    @CodeReflection
    static int ifElseNested(int a, int b) {
        int c = a + b;
        int d = 10 - a + b;
        if (b < 3) {
            if (a < 3) {
                a += 1;
            } else {
                b += 2;
            }
            c += 3;
        } else {
            if (a > 2) {
                a += 4;
            } else {
                b += 5;
            }
            d += 6;
        }
        return a + b + c + d;
    }

    @CodeReflection
    static int nestedLoop(int m, int n) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum = sum + i + j;
            }
        }
        return sum;
    }

    @CodeReflection
    static int methodCall(int a, int b) {
        int i = Math.max(a, b);
        return Math.negateExact(i);
    }

    @CodeReflection
    static int[] primitiveArray(int i, int j) {
        int[] ia = new int[i + 1];
        ia[0] = j;
        return ia;
    }

    @CodeReflection
    static boolean not(boolean b) {
        return !b;
    }

    @CodeReflection
    static boolean notCompare(int i, int j) {
        boolean b = i < j;
        return !b;
    }

    @CodeReflection
    static int mod(int i, int j) {
        return i % (j + 1);
    }

    @CodeReflection
    static int xor(int i, int j) {
        return i ^ j;
    }

    @CodeReflection
    static int whileLoop(int i, int n) { int
        counter = 0;
        while (i < n && counter < 3) {
            counter++;
            if (counter == 4) {
                break;
            }
            i++;
        }
        return counter;
    }

    public interface Func {
        int apply(int a);
    }

    public interface QuotableFunc extends Quotable {
        int apply(int a);
    }

    static int consume(int i, Func f) {
        return f.apply(i + 1);
    }

    static int consumeQuotable(int i, QuotableFunc f) {
        Assert.assertNotNull(f.quoted());
        Assert.assertNotNull(f.quoted().op());
        Assert.assertTrue(f.quoted().op() instanceof CoreOp.LambdaOp);
        return f.apply(i + 1);
    }

    @CodeReflection
    static int lambda(int i) {
        return consume(i, a -> -a);
    }

    @CodeReflection
    static int quotableLambda(int i) {
        return consumeQuotable(i, a -> -a);
    }

    @CodeReflection
    static int lambdaWithCapture(int i, String s) {
        return consume(i, a -> a + s.length());
    }

    @CodeReflection
    static int quotableLambdaWithCapture(int i, String s) {
        return consumeQuotable(i, a -> a + s.length());
    }

    @CodeReflection
    static int nestedLambdasWithCaptures(int i, int j, String s) {
        return consume(i, a -> consume(a, b -> a + b + j - s.length()) + s.length());
    }

    @CodeReflection
    static int nestedQuotableLambdasWithCaptures(int i, int j, String s) {
        return consumeQuotable(i, a -> consumeQuotable(a, b -> a + b + j - s.length()) + s.length());
    }

    @CodeReflection
    static int methodHandle(int i) {
        return consume(i, Math::negateExact);
    }

    @CodeReflection
    static boolean compareLong(long i, long j) {
        return i > j;
    }

    @CodeReflection
    static boolean compareFloat(float i, float j) {
        return i > j;
    }

    @CodeReflection
    static boolean compareDouble(double i, double j) {
        return i > j;
    }

    @CodeReflection
    static int lookupSwitch(int i) {
        return switch (1000 * i) {
            case 1000 -> 1;
            case 2000 -> 2;
            case 3000 -> 3;
            default -> 0;
        };
    }

    @CodeReflection
    static int tableSwitch(int i) {
        return switch (i) {
            case 1 -> 1;
            case 2 -> 2;
            case 3 -> 3;
            default -> 0;
        };
    }

    int instanceField = -1;

    @CodeReflection
    int instanceFieldAccess(int i) {
        int ret = instanceField;
        instanceField = i;
        return ret;
    }

    record TestData(Method testMethod) {
        @Override
        public String toString() {
            String s = testMethod.getName() + Arrays.stream(testMethod.getParameterTypes())
                    .map(Class::getSimpleName).collect(Collectors.joining(",", "(", ")"));
            if (s.length() > 30) s = s.substring(0, 27) + "...";
            return s;
        }
    }

    @DataProvider(name = "testMethods")
    public static TestData[]testMethods() {
        return Stream.of(TestBytecode.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(CodeReflection.class))
                .map(TestData::new).toArray(TestData[]::new);
    }

    private static byte[] CLASS_DATA;
    private static ClassModel CLASS_MODEL;

    @BeforeClass
    public static void setup() throws Exception {
        CLASS_DATA = TestBytecode.class.getResourceAsStream("TestBytecode.class").readAllBytes();
        CLASS_MODEL = ClassFile.of().parse(CLASS_DATA);
    }

    private static MethodTypeDesc toMethodTypeDesc(Method m) {
        return MethodTypeDesc.of(
                m.getReturnType().describeConstable().orElseThrow(),
                Arrays.stream(m.getParameterTypes())
                        .map(cls -> cls.describeConstable().orElseThrow()).toList());
    }


    private static final Map<Class<?>, Object[]> TEST_ARGS = new IdentityHashMap<>();
    private static Object[] values(Object... values) {
        return values;
    }
    private static void initTestArgs(Object[] values, Class<?>... argTypes) {
        for (var argType : argTypes) TEST_ARGS.put(argType, values);
    }
    static {
        initTestArgs(values(1, 2, 4), int.class, Integer.class);
        initTestArgs(values((byte)1, (byte)3, (byte)4), byte.class, Byte.class);
        initTestArgs(values((short)1, (short)2, (short)3), short.class, Short.class);
        initTestArgs(values((char)2, (char)3, (char)4), char.class, Character.class);
        initTestArgs(values(false, true), boolean.class, Boolean.class);
        initTestArgs(values("Hello World"), String.class);
        initTestArgs(values(1l, 2l, 4l), long.class, Long.class);
        initTestArgs(values(1f, 3f, 4f), float.class, Float.class);
        initTestArgs(values(1d, 2d, 3d), double.class, Double.class);
    }

    interface Executor {
        void execute(Object[] args) throws Throwable;
    }

    private static void permutateAllArgs(Class<?>[] argTypes, Executor executor) throws Throwable {
        final int argn = argTypes.length;
        Object[][] argValues = new Object[argn][];
        for (int i = 0; i < argn; i++) {
            argValues[i] = TEST_ARGS.get(argTypes[i]);
        }
        int[] argIndexes = new int[argn];
        Object[] args = new Object[argn];
        while (true) {
            for (int i = 0; i < argn; i++) {
                args[i] = argValues[i][argIndexes[i]];
            }
            executor.execute(args);
            int i = argn - 1;
            while (i >= 0 && argIndexes[i] == argValues[i].length - 1) i--;
            if (i < 0) return;
            argIndexes[i++]++;
            while (i < argn) argIndexes[i++] = 0;
        }
    }

    @Test(dataProvider = "testMethods")
    public void testLift(TestData d) throws Throwable {
        CoreOp.FuncOp flift;
        try {
            flift = BytecodeLift.lift(CLASS_DATA, d.testMethod.getName(), toMethodTypeDesc(d.testMethod));
        } catch (Throwable e) {
            ClassPrinter.toYaml(ClassFile.of().parse(TestBytecode.class.getResourceAsStream("TestBytecode.class").readAllBytes())
                    .methods().stream().filter(m -> m.methodName().equalsString(d.testMethod().getName())).findAny().get(),
                    ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES, System.out::print);
            System.out.println("Lift failed, compiled model:");
            d.testMethod.getCodeModel().ifPresent(f -> f.writeTo(System.out));
            throw e;
        }
        try {
            Object receiver1, receiver2;
            if (d.testMethod.accessFlags().contains(AccessFlag.STATIC)) {
                receiver1 = null;
                receiver2 = null;
            } else {
                receiver1 = new TestBytecode();
                receiver2 = new TestBytecode();
            }
            permutateAllArgs(d.testMethod.getParameterTypes(), args ->
                Assert.assertEquals(invokeAndConvert(flift, receiver1, args), d.testMethod.invoke(receiver2, args)));
        } catch (Throwable e) {
            System.out.println("Compiled model:");
            d.testMethod.getCodeModel().ifPresent(f -> f.writeTo(System.out));
            System.out.println("Lifted model:");
            flift.writeTo(System.out);
            throw e;
        }
    }

    private static Object invokeAndConvert(CoreOp.FuncOp func, Object receiver, Object... args) {
        List argl = new ArrayList(args.length + 1);
        if (receiver != null) argl.add(receiver);
        argl.addAll(Arrays.asList(args));
        Object ret = Interpreter.invoke(MethodHandles.lookup(), func, argl);
        if (ret instanceof Integer i) {
            TypeElement rt = func.invokableType().returnType();
            if (rt.equals(JavaType.BOOLEAN)) {
                return i != 0;
            } else if (rt.equals(JavaType.BYTE)) {
                return i.byteValue();
            } else if (rt.equals(JavaType.CHAR)) {
                return (short)i.intValue();
            } else if (rt.equals(JavaType.SHORT)) {
                return i.shortValue();
            }
        }
        return ret;
    }

    @Test(dataProvider = "testMethods")
    public void testGenerate(TestData d) throws Throwable {
        CoreOp.FuncOp func = d.testMethod.getCodeModel().get();

        CoreOp.FuncOp lfunc;
        try {
            lfunc = func.transform(CopyContext.create(), OpTransformer.LOWERING_TRANSFORMER);
        } catch (UnsupportedOperationException uoe) {
            throw new SkipException("lowering caused:", uoe);
        }

        try {
            MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), lfunc);
            Object receiver1, receiver2;
            if (d.testMethod.accessFlags().contains(AccessFlag.STATIC)) {
                receiver1 = null;
                receiver2 = null;
            } else {
                receiver1 = new TestBytecode();
                receiver2 = new TestBytecode();
            }
            permutateAllArgs(d.testMethod.getParameterTypes(), args -> {
                    List argl = new ArrayList(args.length + 1);
                    if (receiver1 != null) argl.add(receiver1);
                    argl.addAll(Arrays.asList(args));
                    Assert.assertEquals(mh.invokeWithArguments(argl), d.testMethod.invoke(receiver2, args));
            });
        } catch (Throwable e) {
            func.writeTo(System.out);
            lfunc.writeTo(System.out);
            String methodName = d.testMethod().getName();
            for (var mm : CLASS_MODEL.methods()) {
                if (mm.methodName().equalsString(methodName)
                        || mm.methodName().stringValue().startsWith("lambda$" + methodName + "$")) {
                    ClassPrinter.toYaml(mm,
                                        ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES,
                                        System.out::print);
                }
            }
            Files.list(Path.of("DUMP_CLASS_FILES")).forEach(p -> {
                if (p.getFileName().toString().matches(methodName + "\\..+\\.class")) try {
                    ClassPrinter.toYaml(ClassFile.of().parse(p),
                                        ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES,
                                        System.out::print);
                } catch (IOException ignore) {}
            });
            throw e;
        }
    }
}
