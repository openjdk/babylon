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
import java.lang.reflect.code.transformations.LinearConcatTransform;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
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
        CoreOp.FuncOp f_transformed = model.transform(new LinearConcatTransform.ConcatTransform());
        Object[] args = prepArgs(method);

        System.out.println("Final Transformed");
        f_transformed.writeTo(System.out);

        var interpreted = Interpreter.invoke(model, args);
        var transformed_interpreted = Interpreter.invoke(f_transformed, args);

        Assert.assertEquals(interpreted, transformed_interpreted);

    }

    @Test(dataProvider = "getClassMethods")
    public void testSSAModelTransform(@NoInjection Method method) {
        CoreOp.FuncOp model = method.getCodeModel().orElseThrow();
        System.out.println("Basic model");
        model.writeTo(System.out);
        CoreOp.FuncOp ssa_model = generateSSA(model);
        System.out.println("Final Transformed");
        CoreOp.FuncOp f_transformed = ssa_model.transform(new LinearConcatTransform.ConcatTransform());
        f_transformed.writeTo(System.out);
        Object[] args = prepArgs(method);

        var interpreted = Interpreter.invoke(model, args);
        var transformed_interpreted = Interpreter.invoke(f_transformed, args);
        var ssa_interpreted = Interpreter.invoke(ssa_model, args);
        try {
            var jvm_interpreted = method.invoke(null, args);
            Assert.assertEquals(transformed_interpreted, jvm_interpreted);
            //Assert.assertEquals(interpreted, transformed_interpreted);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }


    }


    //@Test(dataProvider = "getClassMethods")
    public void testEqualResults(@NoInjection Method method) {
        CoreOp.FuncOp f = method.getCodeModel().orElseThrow();
        CoreOp.FuncOp f_transformed = f.transform(new LinearConcatTransform.ConcatTransform());
        CoreOp.FuncOp f_ssa = generateSSA(f).transform(new LinearConcatTransform.ConcatTransform());
        MethodHandle m_bytecode = generateBytecode(f_transformed);


        Object[] args = prepArgs(method);

        var interpreted = Interpreter.invoke(f, args);
        var transformed_interpreted = Interpreter.invoke(f_transformed, args);
        var transformed_ssa_interpreted = Interpreter.invoke(f_ssa, args);
        try {
            var jvm_executed = method.invoke(null, args);
            var code_model_bytecode_executed = m_bytecode.invokeWithArguments(args);

            //Assert.assertEquals(interpreted, transformed_interpreted);
            //Assert.assertEquals(transformed_interpreted, jvm_executed);
            //Assert.assertEquals(jvm_executed, transformed_ssa_interpreted);
            //Assert.assertEquals(transformed_ssa_interpreted, code_model_bytecode_executed);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

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
        List<Object[]> ars = Arrays.stream(clazz.getMethods())
                .filter((method) -> method.isAnnotationPresent(CodeReflection.class))
                .map(m -> new Object[]{m})
                .toList();
        Object[][] res = new Object[ars.size()][];
        for (int i = 0; i < ars.size(); i++) {
            res[i] = ars.get(i);
        }

        return res;
    }

    static CoreOp.FuncOp generateSSA(CoreOp.FuncOp f) {
        CoreOp.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf = SSA.transform(lf);

        System.out.println("Post SSA");
        lf.writeTo(System.out);

        return lf;
    }


    static MethodHandle generateBytecode(CoreOp.FuncOp f) {
        CoreOp.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf = SSA.transform(lf);
        return BytecodeGenerator.generate(MethodHandles.lookup(), lf);
    }

    static CoreOp.FuncOp lower(CoreOp.FuncOp f) {
        CoreOp.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

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
    public static String intConcatNestedSplit(int i, String s){
        String q, r;
        String inter = i + (q = r = s + "hello") + 52;
        return q + r + inter;
    }

}