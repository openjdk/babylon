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

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.classfile.ClassFile;
import java.lang.classfile.components.ClassPrinter;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestBytecode
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
    static long longBitOps(long i, long j, long k) {
        return i & j | k ^ j;
    }

    @CodeReflection
    static boolean boolBitOps(boolean i, boolean j, boolean k) {
        return i & j | k ^ j;
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
    static boolean not(int i, int j) {
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
        return f.apply(i + 1);
    }

    @CodeReflection
    @SkipLift
    static int lambda(int i) {
        return consume(i, a -> -a);
    }

    @CodeReflection
    @SkipLift
    static int quotableLambda(int i) {
        return consumeQuotable(i, a -> -a);
    }

    @CodeReflection
    @SkipLift
    static int lambdaWithCapture(int i, String s) {
        return consume(i, a -> a + s.length());
    }

    @CodeReflection
    @SkipLift
    static int quotableLambdaWithCapture(int i, String s) {
        return consumeQuotable(i, a -> a + s.length());
    }

    @CodeReflection
    @SkipLift
    static int nestedLambdasWithCaptures(int i, int j, String s) {
        return consume(i, a -> consume(a, b -> a + b + j) + s.length());
    }

    @CodeReflection
    @SkipLift
    static int nestedQuotableLambdasWithCaptures(int i, int j, String s) {
        return consumeQuotable(i, a -> consumeQuotable(a, b -> a + b + j) + s.length());
    }

    @CodeReflection
    @SkipLift
    static int methodHandle(int i) {
        return consume(i, Math::negateExact);
    }

    @Retention(RetentionPolicy.RUNTIME)
    @interface SkipLift {}

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

    @BeforeClass
    public static void setup() throws Exception {
        CLASS_DATA = TestBytecode.class.getResourceAsStream("TestBytecode.class").readAllBytes();
//        ClassPrinter.toYaml(ClassFile.of().parse(CLASS_DATA), ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
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
        initTestArgs(values(1, 2, 3, 4), int.class, Integer.class, byte.class,
                Byte.class, short.class, Short.class, char.class, Character.class);
        initTestArgs(values(false, true), boolean.class, Boolean.class);
        initTestArgs(values("Hello World"), String.class);
        initTestArgs(values(1l, 2l, 3l, 4l), long.class, Long.class);
        initTestArgs(values(1f, 2f, 3f, 4f), float.class, Float.class);
        initTestArgs(values(1d, 2d, 3d, 4d), double.class, Double.class);
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
        if (d.testMethod.getAnnotation(SkipLift.class) != null) {
            throw new SkipException("skipped");
        }
        CoreOps.FuncOp flift;
        try {
            flift = BytecodeLift.lift(CLASS_DATA, d.testMethod.getName(), toMethodTypeDesc(d.testMethod));
        } catch (Throwable e) {
            System.out.println("Lift failed, expected:");
            d.testMethod.getCodeModel().ifPresent(f -> f.writeTo(System.out));
            throw e;
        }
        try {
            permutateAllArgs(d.testMethod.getParameterTypes(), args ->
                Assert.assertEquals(invokeAndConvert(flift, args), d.testMethod.invoke(null, args)));
        } catch (Throwable e) {
            flift.writeTo(System.out);
            throw e;
        }
    }

    private static Object invokeAndConvert(CoreOps.FuncOp func, Object[] args) {
        Object ret = Interpreter.invoke(func, args);
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
        CoreOps.FuncOp func = d.testMethod.getCodeModel().get();

        CoreOps.FuncOp lfunc = func.transform(CopyContext.create(), (block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        try {
            MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), lfunc);
            permutateAllArgs(d.testMethod.getParameterTypes(), args ->
                    Assert.assertEquals(mh.invokeWithArguments(args), d.testMethod.invoke(null, args)));
        } catch (Throwable e) {
            func.writeTo(System.out);
            lfunc.writeTo(System.out);
            throw e;
        }
    }
}
