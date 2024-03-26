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

import java.lang.classfile.ClassFile;
import java.lang.classfile.components.ClassPrinter;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.runtime.CodeReflection;
import java.util.function.Function;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestBytecode
 */

public class TestBytecode {

    @CodeReflection
    static int sum(int i, int j) {
        i = i + j;
        return i;
    }

    @CodeReflection
    static int logicalOps(int i, int j) {
        return 13 & i | j ^ 13;
    }

    @CodeReflection
    static String constructor(int i, int j) {
        return new String("hello world".getBytes(), i, j);
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
    static int conditionalExpression(int i, int j) {
        return ((i - 1 >= 0) ? i - 1 : j - 1);
    }

    @CodeReflection
    static int nestedConditionalExpression(int i, int j) {
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
    static A newWithArguments(int i, int j) {
        return new A("hello world".substring(i, i + j));
    }


    @DataProvider(name = "testMethods")
    public static Object[] testMethods() {
        return Stream.of(TestBytecode.class.getDeclaredMethods()).filter(m -> m.isAnnotationPresent(CodeReflection.class)).map(Method::getName).toArray();
    }

    private static byte[] CLASS_DATA;

    @BeforeClass
    public void setup() throws Exception {
        CLASS_DATA = TestBytecode.class.getResourceAsStream("TestBytecode.class").readAllBytes();
//        ClassPrinter.toYaml(ClassFile.of().parse(CLASS_DATA), ClassPrinter.Verbosity.TRACE_ALL, System.out::print);
    }

    @Test(dataProvider = "testMethods")
    public void testLift(String methodName) throws Throwable {
        Method testMethod = TestBytecode.class.getDeclaredMethod(methodName, int.class, int.class);
        CoreOps.FuncOp flift = null;
        try {
            flift = BytecodeLift.lift(CLASS_DATA, methodName);
        } catch (Throwable e) {
            System.out.println("Lift failed, expected:");
            testMethod.getCodeModel().ifPresent(f -> f.writeTo(System.out));
            throw e;
        }
        try {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    Assert.assertEquals(Interpreter.invoke(flift, i, j), testMethod.invoke(null, i, j));
                }
            }
        } catch (Throwable e) {
            flift.writeTo(System.out);
            throw e;
        }
    }

    @Test(dataProvider = "testMethods")
    public void testGenerate(String methodName) throws Throwable {
        Method testMethod = TestBytecode.class.getDeclaredMethod(methodName, int.class, int.class);

        CoreOps.FuncOp func = testMethod.getCodeModel().get();

        @SuppressWarnings("unchecked")
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
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    Assert.assertEquals(mh.invoke(i, j), testMethod.invoke(null, i, j));
                }
            }
        } catch (Throwable e) {
            func.writeTo(System.out);
            lfunc.writeTo(System.out);
            throw e;
        }
    }
}
