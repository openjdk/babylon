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
 * @modules jdk.incubator.code
 * @run junit TestModuleOp
 * @run junit/othervm -Dbabylon.ssa=cytron TestModuleOp
 */

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

public class TestModuleOp {

    @Reflect
    public static void lambda(Consumer<Integer> consumer) {
        consumer.accept(4);
    }

    @Reflect
    public static void a() {
    }

    @Reflect
    public static int b(int i) {
        return i;
    }

    @Reflect
    public static int c(int i) {
        return (i * 2) % 5;
    }

    @Reflect
    public static int d(int i) {
        return (i / 4) + 9;
    }

    @Reflect
    public static int e(int i) {
        return f(i) + c(i + 5);
    }

    @Reflect
    public static int f(int i) {
        return c(i);
    }

    @Reflect
    public static int g(int i, int j) {
        return i + j;
    }

    @Test
    public void testEmptyLambda() {
        Runnable runnable = (@Reflect Runnable) () -> {};
        JavaOp.LambdaOp lambdaOp = SSA.transform((JavaOp.LambdaOp) Op.ofQuotable(runnable).get().op());
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().containsKey("rootLambda"));
        Assertions.assertTrue(moduleOp.functionTable().get("rootLambda").elements().allMatch(e -> e instanceof ReturnOp || e instanceof FuncOp || !(e instanceof Op)));
    }

    @Test
    public void testSingleInvoke() {
        Runnable runnable = (@Reflect Runnable) () -> a();
        JavaOp.LambdaOp lambdaOp = SSA.transform((JavaOp.LambdaOp) Op.ofQuotable(runnable).get().op());
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().containsKey("a"));
        Assertions.assertFalse(moduleOp.functionTable().containsKey("rootLambda"));
    }

    @Test
    public void testSingleInvokeWithLambdaParam() {
        IntUnaryOperator iuo = (@Reflect IntUnaryOperator) (i) -> b(i);
        JavaOp.LambdaOp lambdaOp = SSA.transform((JavaOp.LambdaOp) Op.ofQuotable(iuo).get().op());
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().containsKey("b"));
        Assertions.assertFalse(moduleOp.functionTable().containsKey("rootLambda"));
    }

    @Test
    public void testWithCapture() throws NoSuchMethodException {
        MethodRef bRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("b", int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT))
                .body(block -> {
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                lblock.op(invoke(bRef, block.parameters().get(0)));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });
        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().containsKey("b"));
        Assertions.assertFalse(moduleOp.functionTable().containsKey("rootLambda"));
    }

    // @Reflect
    // public static void testFunction4(int i) {
    //     lambda((_) -> b(c(i * 2) + e(d(i))));
    // }

    @Test
    public void testNested() throws NoSuchMethodException {
        MethodRef bRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("b", int.class));
        MethodRef cRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("c", int.class));
        MethodRef dRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("d", int.class));
        MethodRef eRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("e", int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT))
                .body(block -> {
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Block.Parameter li = block.parameters().get(0);
                                Op.Result var = lblock.op(var(li));
                                Op.Result constant = lblock.op(constant(INT, 2));
                                Op.Result varLoad = lblock.op(varLoad(var));
                                Op.Result mul = lblock.op(JavaOp.mul(varLoad, constant));
                                Op.Result qRes = lblock.op(invoke(cRef, mul));
                                Op.Result varLoad2 = lblock.op(varLoad(var));
                                Op.Result pRes = lblock.op(invoke(dRef, varLoad2));
                                Op.Result oRes = lblock.op(invoke(eRef, pRes));
                                Op.Result sum = lblock.op(add(qRes, oRes));
                                lblock.op(invoke(bRef, sum));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });

        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "b", "c", "d", "e")));
    }

    // @Reflect
    // public static void testFunction5(int i) {
    //     lambda((_) -> c(b(b(i))));
    // }

    @Test
    public void testNested2() throws NoSuchMethodException {
        MethodRef bRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("b", int.class));
        MethodRef cRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("c", int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Block.Parameter li = block.parameters().get(0);
                                Op.Result var = lblock.op(var(li));
                                Op.Result varLoad = lblock.op(varLoad(var));
                                Op.Result mRes = lblock.op(invoke(bRef, varLoad));
                                Op.Result mRes2 = lblock.op(invoke(bRef, mRes));
                                lblock.op(invoke(cRef, mRes2));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });

        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        System.out.println(moduleOp.toText());
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "b", "c")));
    }

    // @Reflect
    // public static void testFunction6(int i, int[] array) {
    //     lambda((j) -> c(b(i) + array[j]));
    // }

    @Test
    public void testArray() throws NoSuchMethodException {
        MethodRef bRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("b", int.class));
        MethodRef cRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("c", int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT, INT_ARRAY))
                .body(block -> {
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Block.Parameter li = block.parameters().get(0);
                                Block.Parameter larray = block.parameters().get(1);
                                Block.Parameter lj = lblock.parameters().get(0);
                                Op.Result vari = lblock.op(var(li));
                                Op.Result varArray = lblock.op(var(larray));
                                Op.Result varJ = lblock.op(var(lj));
                                Op.Result varLoad = lblock.op(varLoad(vari));
                                Op.Result mRes = lblock.op(invoke(bRef, varLoad));
                                Op.Result varArrayLoad = lblock.op(varLoad(varArray));
                                Op.Result varJLoad = lblock.op(varLoad(varJ));
                                Op.Result arrLoad = lblock.op(arrayLoadOp(varArrayLoad, varJLoad));
                                Op.Result sum = lblock.op(add(mRes, arrLoad));
                                lblock.op(invoke(cRef, sum));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });

        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        System.out.println(moduleOp.toText());
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "b", "c")));
    }

    // @Reflect
    // public static void testFunction7(int i, int j) {
    //     lambda((k) -> c(b(i) + b(j) + b(k)));
    // }

    @Test
    public void testNestedSum() throws NoSuchMethodException {
        MethodRef bRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("b", int.class));
        MethodRef cRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("c", int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Block.Parameter li = block.parameters().get(0);
                                Block.Parameter lj = block.parameters().get(1);
                                Block.Parameter lk = lblock.parameters().get(0);
                                Op.Result vari = lblock.op(var(li));
                                Op.Result varj = lblock.op(var(lj));
                                Op.Result vark = lblock.op(var(lk));
                                Op.Result variLoad = lblock.op(varLoad(vari));
                                Op.Result iRes = lblock.op(invoke(bRef, variLoad));
                                Op.Result varjLoad = lblock.op(varLoad(varj));
                                Op.Result jRes = lblock.op(invoke(bRef, varjLoad));

                                Op.Result sum = lblock.op(add(iRes, jRes));
                                Op.Result varkLoad = lblock.op(varLoad(vark));
                                Op.Result kRes = lblock.op(invoke(bRef, varkLoad));
                                Op.Result ssum = lblock.op(add(sum, kRes));

                                lblock.op(invoke(cRef, ssum));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });

        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "b", "c")));
    }

    // @Reflect
    // public static void testFunction8(int i, int j) {
    //     lambda((k) -> c(1));
    // }

    @Test
    public void testConstant() throws NoSuchMethodException {
        MethodRef cRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("c", int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Op.Result constant = lblock.op(constant(INT, 1));
                                lblock.op(invoke(cRef, constant));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });

        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        System.out.println(moduleOp.toText());
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "c")));
    }

    // @Reflect
    // public static void testFunction9(int i, int j) {
    //     lambda((k) -> c(g(i, k)));
    // }

    @Test
    public void testSomeUsed() throws NoSuchMethodException {
        MethodRef cRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("c", int.class));
        MethodRef gRef = MethodRef.method(TestModuleOp.class.getDeclaredMethod("g", int.class, int.class));
        CoreOp.FuncOp f = func("f", CoreType.functionType(VOID, INT, INT))
                .body(block -> {
                    JavaOp.LambdaOp lambda = JavaOp.lambda(block.parentBody(),
                                    CoreType.functionType(VOID, INT), type(IntUnaryOperator.class))
                            .body(lblock -> {
                                Op.Result vari = lblock.op(var(block.parameters().get(0)));
                                Op.Result vark = lblock.op(var(lblock.parameters().get(0)));
                                Op.Result variLoad = lblock.op(varLoad(vari));
                                Op.Result varkLoad = lblock.op(varLoad(vark));
                                Op.Result uRes = lblock.op(invoke(gRef, variLoad, varkLoad));
                                lblock.op(invoke(cRef, uRes));
                                lblock.op(CoreOp.return_());
                            });
                    block.op(lambda);
                    block.op(CoreOp.return_());
                });

        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "c", "g")));
    }

    @Reflect
    public static void testFunction10(int i, int j) {
        lambda((k) -> c(i > k ? b(i) : b(j)));
    }

    @Test
    public void testTernary() throws NoSuchMethodException {
        Method m = TestModuleOp.class.getDeclaredMethod("testFunction10", int.class, int.class);
        CoreOp.FuncOp f = Op.ofMethod(m).get();
        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) f.elements().filter(e -> e instanceof JavaOp.LambdaOp).findFirst().get();
        CoreOp.ModuleOp moduleOp = CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().containsAll(List.of("rootLambda", "b", "c")));
    }

    // @Reflect
    // public static void testFunction11(int i, int j) {
    //     lambda((k) -> rootLambda());
    // }

    @Test
    public void testRepeatLambdaName() {
        Runnable runnable = (@Reflect Runnable) () -> {};
        JavaOp.LambdaOp lambdaOp = SSA.transform((JavaOp.LambdaOp) Op.ofQuotable(runnable).get().op());
        Assertions.assertThrows(IllegalArgumentException.class, () -> CoreOp.ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "b"));
    }
}
