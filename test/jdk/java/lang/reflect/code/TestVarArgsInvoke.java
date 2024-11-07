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
 * @run testng TestVarArgsInvoke
 */

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestVarArgsInvoke {

    String m1(String... args) {
        StringBuilder sb = new StringBuilder("m1");
        for (String arg : args) {
            sb.append(arg);
        }
        return sb.toString();
    }

    static String sm1(String... args) {
        StringBuilder sb = new StringBuilder("sm1");
        for (String arg : args) {
            sb.append(arg);
        }
        return sb.toString();
    }

    String m2(String one, String... args) {
        StringBuilder sb = new StringBuilder("m2");
        sb.append(one);
        for (String arg : args) {
            sb.append(arg);
        }
        return sb.toString();
    }

    static String sm2(String one, String... args) {
        StringBuilder sb = new StringBuilder("sm2");
        sb.append(one);
        for (String arg : args) {
            sb.append(arg);
        }
        return sb.toString();
    }

    enum MethodKind {
        M1, SM1, M2, SM2;
    }

    @CodeReflection
    String fArray(String[] array, MethodKind m) {
        return switch (m) {
            case M1 -> m1(array);
            case SM1 -> sm1(array);
            case M2 -> m2("first", array);
            case SM2 -> sm2("first", array);
        };
    }

    @Test
    public void testArray() {
        CoreOp.FuncOp f = getFuncOp("fArray");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        invokes(f).forEach(iop -> {
            Assert.assertFalse(iop.isVarArgs());
            Assert.assertNull(iop.varArgOperands());
        });

        String[] array = new String[]{"second", "third"};
        for (MethodKind mk : MethodKind.values()) {
            Assert.assertEquals(
                    Interpreter.invoke(MethodHandles.lookup(), f, this, array, mk),
                    fArray(array, mk));
        }
    }

    @CodeReflection
    String fEmpty(MethodKind m) {
        return switch (m) {
            case M1 -> m1();
            case SM1 -> sm1();
            case M2 -> m2("first");
            case SM2 -> sm2("first");
        };
    }

    @Test
    public void testEmpty() {
        CoreOp.FuncOp f = getFuncOp("fEmpty");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        invokes(f).forEach(iop -> {
            Assert.assertTrue(iop.isVarArgs());
            Assert.assertTrue(iop.varArgOperands().isEmpty());
        });

        String[] array = new String[]{"second", "third"};
        for (MethodKind mk : MethodKind.values()) {
            Assert.assertEquals(
                    Interpreter.invoke(MethodHandles.lookup(), f, this, mk),
                    fEmpty(mk));
        }
    }

    @CodeReflection
    String fOne(String one, MethodKind m) {
        return switch (m) {
            case M1 -> m1(one);
            case SM1 -> sm1(one);
            case M2 -> m2("first", one);
            case SM2 -> sm2("first", one);
        };
    }

    @Test
    public void testOne() {
        CoreOp.FuncOp f = getFuncOp("fOne");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        invokes(f).forEach(iop -> {
            Assert.assertTrue(iop.isVarArgs());
            Assert.assertEquals(iop.varArgOperands().size(), 1);
        });

        for (MethodKind mk : MethodKind.values()) {
            Assert.assertEquals(
                    Interpreter.invoke(MethodHandles.lookup(), f, this, "one", mk),
                    fOne("one", mk));
        }
    }

    @CodeReflection
    String fMany(String one, String two, MethodKind m) {
        return switch (m) {
            case M1 -> m1(one, two);
            case SM1 -> sm1(one, two);
            case M2 -> m2("first", one, two);
            case SM2 -> sm2("first", one, two);
        };
    }

    @Test
    public void testMany() {
        CoreOp.FuncOp f = getFuncOp("fMany");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        invokes(f).forEach(iop -> {
            Assert.assertTrue(iop.isVarArgs());
            Assert.assertEquals(iop.varArgOperands().size(), 2);
        });

        for (MethodKind mk : MethodKind.values()) {
            Assert.assertEquals(
                    Interpreter.invoke(MethodHandles.lookup(), f, this, "one", "two", mk),
                    fMany("one", "two", mk));
        }
    }

    static Stream<CoreOp.InvokeOp> invokes(CoreOp.FuncOp f) {
        return f.elements().mapMulti((ce, c) -> {
            if (ce instanceof CoreOp.InvokeOp iop &&
                iop.invokeDescriptor().refType().equals(JavaType.type(TestVarArgsInvoke.class))) {
                c.accept(iop);
            }
        });
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestVarArgsInvoke.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }

}
