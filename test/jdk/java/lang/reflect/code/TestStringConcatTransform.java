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
 * @run testng TestStringConcatTransform
 */

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.NoInjection;
import org.testng.annotations.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.StringConcatTransformer;

import java.lang.reflect.Method;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestStringConcatTransform {

    static final String TESTSTR = "TESTING STRING";

    static final Map<Class<?>, Object> valMap;

    static {
        valMap = new HashMap<>();
        valMap.put(byte.class, (byte) 42);
        valMap.put(short.class, (short) 42);
        valMap.put(int.class, 42);
        valMap.put(long.class, (long) 42);
        valMap.put(float.class, 42f);
        valMap.put(double.class, 42d);
        valMap.put(char.class, 'z');
        valMap.put(boolean.class, false);

        valMap.put(Byte.class, (byte) 42);
        valMap.put(Short.class, (short) 42);
        valMap.put(Integer.class, 42);
        valMap.put(Long.class, (long) 42);
        valMap.put(Float.class, 42f);
        valMap.put(Double.class, 42d);
        valMap.put(Character.class, 'z');
        valMap.put(Boolean.class, false);

        valMap.put(Object.class, new Object() {
            @Override
            public String toString() {
                return "I'm a test string.";
            }
        });
        valMap.put(TestObject.class, new TestObject());
        valMap.put(String.class, TESTSTR);
    }

    public static final class TestObject {
        TestObject() {
        }

        @Override
        public String toString() {
            return "TestObject String";
        }
    }

    @Test(dataProvider = "getClassMethods")
    public void testModelTransform(@NoInjection Method method) {
        CoreOp.FuncOp model = method.getCodeModel().orElseThrow();
        CoreOp.FuncOp f_transformed = model.transform(new StringConcatTransformer());
        Object[] args = prepArgs(method);

        model.writeTo(System.out);
        f_transformed.writeTo(System.out);

        var interpreted = Interpreter.invoke(model, args);
        var transformed_interpreted = Interpreter.invoke(f_transformed, args);

        Assert.assertEquals(interpreted, transformed_interpreted);

    }

    @Test(dataProvider = "getClassMethods")
    public void testSSAModelTransform(@NoInjection Method method) {
        CoreOp.FuncOp model = method.getCodeModel().orElseThrow();
        CoreOp.FuncOp transformed_model = model.transform(new StringConcatTransformer());
        CoreOp.FuncOp ssa_model = generateSSA(model);
        CoreOp.FuncOp ssa_transformed_model = ssa_model.transform(new StringConcatTransformer());
        Object[] args = prepArgs(method);

        model.writeTo(System.out);
        ssa_model.writeTo(System.out);
        ssa_transformed_model.writeTo(System.out);

        var model_interpreted = Interpreter.invoke(model, args);
        var transformed_model_interpreted = Interpreter.invoke(transformed_model, args);
        var ssa_interpreted = Interpreter.invoke(ssa_model, args);
        var ssa_transformed_interpreted = Interpreter.invoke(ssa_transformed_model, args);
        Object jvm_interpreted;
        try {
            jvm_interpreted = method.invoke(null, args);
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
        Assert.assertEquals(model_interpreted, transformed_model_interpreted);
        Assert.assertEquals(transformed_model_interpreted, ssa_interpreted);
        Assert.assertEquals(ssa_interpreted, ssa_transformed_interpreted);
        Assert.assertEquals(ssa_transformed_interpreted, jvm_interpreted);
    }

    public static Object[] prepArgs(Method m) {
        Class<?>[] argTypes = m.getParameterTypes();
        Object[] args = new Object[argTypes.length];
        for (int i = 0; i < argTypes.length; i++) {
            args[i] = valMap.get(argTypes[i]);
        }
        return args;
    }

    @DataProvider(name = "getClassMethods")
    public static Object[][] getClassMethods() {
        return getTestMethods(TestStringConcatTransform.class);
    }

    public static Object[][] getTestMethods(Class<?> clazz) {
        Object[][] res = Arrays.stream(clazz.getMethods())
                .filter((method) -> method.isAnnotationPresent(CodeReflection.class))
                .map(m -> new Object[]{m})
                .toArray(Object[][]::new);
        return res;
    }

    static CoreOp.FuncOp generateSSA(CoreOp.FuncOp f) {
        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf = SSA.transform(lf);
        lf.writeTo(System.out);
        return lf;
    }

    @CodeReflection
    public static String intConcat(int i, String s) {
        return i + s + "hello" + 52;
    }


    @CodeReflection
    public static String intConcatAssignment(int i, String s) {
        String s1 = i + s;
        return s1 + "hello" + 52;
    }

    @CodeReflection
    public static String intConcatExprAssignment(int i, String s) {
        String r;
        String inter = i + (r = s + "hello") + 52;
        return r + inter;
    }

    @CodeReflection
    public static String intConcatWideExpr(int i, String s) {
        String s1 = i + s;
        return s1 + "hello" + 52 + "world" + 26 + "!";
    }

    @CodeReflection
    public static String intConcatDoubVar(int i, String s) {
        String r;
        String inter = i + (r = s + "hello") + 52;
        String inter2 = i + (r = s + "hello") + 52 + inter;
        return r + inter2;
    }

    @CodeReflection
    public static String intConcatNestedSplit(int i, String s) {
        String q, r;
        String inter = i + (q = r = s + "hello") + 52;
        return q + r + inter;
    }

    @CodeReflection
    public static String degenerateTree(String a, String b, String c, String d) {
        String s = (a + b) + (c + d);
        return s;
    }

    //This String Builder is getting tweaked, check for side effects
    @CodeReflection
    public static String degenerateTree2(String a, String d) {
        StringBuilder sb = new StringBuilder("test");
        String s = sb + a;
        String t = s + d;
        System.out.println(sb);
        return t;
    }

    @CodeReflection
    public static String degenerateTree4(String a, String b, String c, String d) {
        String s = ((a + b) + c) + d;
        return s;
    }

    @CodeReflection
    public static String degenerateTree5(String a, String b, String c, String d) {
        String s = (a + (b + (c + d)));
        return s;
    }

    @CodeReflection
    public static String widenPrimitives(short a, byte b, int c, int d) {
        String s = (a + (b + (c + d + "hi")));
        return s;
    }
}

