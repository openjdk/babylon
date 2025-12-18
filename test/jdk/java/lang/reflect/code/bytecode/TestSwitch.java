/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
import java.lang.classfile.MethodModel;
import java.lang.classfile.instruction.LookupSwitchInstruction;
import java.lang.classfile.instruction.TableSwitchInstruction;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.bytecode.impl
 * @library ../
 * @run junit TestSwitch
 * @run main Unreflect TestSwitch
 * @run junit TestSwitch
 */
public class TestSwitch {

    @Reflect
    static String lookupSwitchExpression(int i) {
        return switch (i) {
            case 7,8 -> "magic number";
            case 42 -> "Answer to the Ultimate Question of Life, the Universe, and Everything";
            case 101 -> "introduction to a subject";
            default -> "not important";
        };
    }

    @Reflect
    static String tableSwitchExpression(int i) {
        return switch (i) {
            case -1 -> "?";
            case 0 -> "none";
            case 1 -> "one";
            case 2 -> "two";
            case 3 -> "three";
            default -> "many";
        };
    }

    @Reflect
    static String lookupSwitchStatement(int i) {
        String ret = null;
        switch (i) {
            case 7 : ret = "magic number"; break;
            case 42 : return "Answer to the Ultimate Question of Life, the Universe, and Everything";
            case 101 : ret = "introduction to a subject"; break;
            default : return "not important";
        }
        return ret;
    }

    @Reflect
    static String tableSwitchStatement(int i) {
        String ret = null;
        switch (i) {
            case -1 : ret = "?"; break;
            case 0 : return "none";
            case 1 : ret = "one"; break;
            case 2 : return "two";
            case 3 : ret ="three"; break;
            default : return"many";
        }
        return ret;
    }

    @Reflect
    static String outOfOrderFallThrought(int i) {
        String ret = "";
        switch (i) {
            default : ret += "? ";
            case 4 : ret += "four ";
            case 2 : ret += "two ";
            case 3 : ret += "three ";
            case 1 : ret += "one";
        }
        return ret;
    }

    @Reflect
    static String nestedExpressions(int i) {
        return switch (i) {
            case -1 -> "?";
            case 0 -> "none";
            case 1 -> "one";
            case 2 -> "two";
            case 3 -> "three";
            default -> switch (i) {
                case 7,8 -> "magic number";
                case 42 -> "Answer to the Ultimate Question of Life, the Universe, and Everything";
                case 101 -> "introduction to a subject";
                default -> "not important";
            };
        };
    }

    @ParameterizedTest
    @ValueSource(strings = {
        "lookupSwitchExpression",
        "tableSwitchExpression",
        "lookupSwitchStatement",
        "tableSwitchStatement",
        "outOfOrderFallThrought",
        "nestedExpressions"
    })

    public void testSwitch(String methodName) throws Throwable {
        Method m = getMethod(methodName);
        CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();

        Assertions.assertTrue(getModel(f).code().get().elementStream()
                .anyMatch(i -> i instanceof TableSwitchInstruction || i instanceof LookupSwitchInstruction));

        MethodHandle mh = generate(f);
        for (int i = -1; i < 110; i++) {
            Assertions.assertEquals((String)m.invoke(null, i), (String)mh.invokeExact(i));
        }
    }

    static MethodHandle generate(CoreOp.FuncOp f) {
        System.out.println(f.toText());

        return BytecodeGenerator.generate(MethodHandles.lookup(), f);
    }

    static MethodModel getModel(CoreOp.FuncOp f) {
        var clm = ClassFile.of().parse(BytecodeGenerator.generateClassData(MethodHandles.lookup(), f));
        var mm = clm.methods().getFirst();
        System.out.println(mm.toDebugString());
        return mm;
    }

    static Method getMethod(String name) {
        Optional<Method> om = Stream.of(TestSwitch.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        return om.get();
    }
}
