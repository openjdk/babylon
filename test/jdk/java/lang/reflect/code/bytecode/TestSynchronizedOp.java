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
 * @enablePreview
 * @run testng TestSynchronizedOp
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeModel;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.MonitorInstruction;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.java.lang.reflect.code.OpTransformer;
import jdk.incubator.code.java.lang.reflect.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TestSynchronizedOp {

    @CodeReflection
    static int f(Object o, int i, int[] a) {
        synchronized (o) {
            if (i < 0) {
                throw new RuntimeException();
            }
            a[0] = ++i;
        }
        return i;
    }

    @Test
    public void testInstructions() {
        CoreOp.FuncOp f = getFuncOp("f");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        byte[] classdata = BytecodeGenerator.generateClassData(MethodHandles.lookup(), f);
        CodeModel cmf = ClassFile.of().parse(classdata).methods().stream()
                .filter(mm -> mm.methodName().equalsString("f"))
                .findFirst().flatMap(MethodModel::code).orElseThrow();
        Map<Opcode, Long> monitorCount = cmf.elementStream().<Opcode>mapMulti((i, c) -> {
            if (i instanceof MonitorInstruction mi) {
                c.accept(mi.opcode());
            }
        }).collect(Collectors.groupingBy(oc -> oc, Collectors.counting()));
        Assert.assertEquals(monitorCount, Map.of(
                Opcode.MONITORENTER, 1L,
                Opcode.MONITOREXIT, 2L));
    }

    @Test
    public void testExecution() throws Throwable {
        CoreOp.FuncOp f = getFuncOp("f");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        MethodHandle mf = BytecodeGenerator.generate(MethodHandles.lookup(), f);

        Object monitor = new Object();
        int[] a = new int[1];
        Assert.assertEquals((int) mf.invoke(monitor, 0, a), 1);
        a[0] = 0;
        Assert.assertThrows(RuntimeException.class, () -> {
            int i = (int) mf.invoke(monitor, -1, a);
        });
        Assert.assertEquals(a[0], 0);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestSynchronizedOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
