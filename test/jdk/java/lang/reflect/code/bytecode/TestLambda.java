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
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestLambda
 */

public class TestLambda {

    public interface Func {
        int apply(int a);
    }

    static int consume(int i, Func f) {
        return f.apply(i);
    }

    @CodeReflection
    static int lambda(int i) {
        return consume(i, a -> -a);
    }

    @Test
    public void testLambda() throws Throwable {
//        ClassPrinter.toYaml(ClassFile.of().parse(TestLambda.class.getResourceAsStream("TestLambda.class").readAllBytes()), ClassPrinter.Verbosity.TRACE_ALL, System.out::print);

        CoreOps.FuncOp f = getFuncOp("lambda");
//        f.writeTo(System.out);
        MethodHandle mh = BytecodeGenerator.generate(MethodHandles.lookup(), f);
        Assert.assertEquals(mh.invoke(1), -1);

    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestLambda.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();
        return om.get().getCodeModel().get();
    }
}
