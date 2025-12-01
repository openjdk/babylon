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

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import jdk.incubator.code.Block;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.impl.ConstantLabelSwitchOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaType;

/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.bytecode.impl
 * @run junit TestSwitch
 */
public class TestSwitch {

    @Reflect
    static String switchExpression(int i) {
        return switch (i) {
            case 7 -> "magic number";
            case 42 -> "Answer to the Ultimate Question of Life, the Universe, and Everything";
            case 101 -> "introduction to a subject";
            default -> "not important";
        };
    }

    @Test
    public void switchExpression() throws Throwable {
//        CoreOp.FuncOp f = getFuncOp("switchExpression");

        // @@@ manually constructed:
        // func @"switchExpression" (%0 : java.type:"int")java.type:"java.lang.String" -> {
        //    %1 : Var<java.type:"int"> = var %0 @"i";
        //    %2 : java.type:"int" = var.load %1;
        //    ConstantLabelSwitchOp %2 @"[7,42,101]" ^block_1 ^block_2 ^block_3 ^block_4;
        //
        //  ^block_1:
        //    %3 : java.type:"java.lang.String" = constant @"magic number";
        //    branch ^block_5(%3);
        //
        //  ^block_2:
        //    %4 : java.type:"java.lang.String" = constant @"Answer to the Ultimate Question of Life, the Universe, and Everything";
        //    branch ^block_5(%4);
        //
        //  ^block_3:
        //    %5 : java.type:"java.lang.String" = constant @"introduction to a subject";
        //    branch ^block_5(%5);
        //
        //  ^block_4:
        //    %6 : java.type:"java.lang.String" = constant @"not important";
        //    branch ^block_5(%6);
        //
        //  ^block_5(%7 : java.type:"java.lang.String"):
        //    return %7;
        //};
        JavaType stringType = JavaType.type(String.class);
        CoreOp.FuncOp f = CoreOp.FuncOp.func("switchExpression",
                CoreType.functionType(stringType, JavaType.INT)).body(bb -> {
                    Block.Builder b0 = bb.entryBlock(),
                                  b1 = bb.block(),
                                  b2 = bb.block(),
                                  b3 = bb.block(),
                                  b4 = bb.block(),
                                  b5 = bb.block(stringType);
                    Value v1 = b0.op(CoreOp.var("i", JavaType.INT, b0.parameters().getFirst()));
                    Value v2 = b0.op(CoreOp.varLoad(v1));
                    b0.op(new ConstantLabelSwitchOp(v2,
                            List.of(7, 42, 101),
                            List.of(b1.successor(), b2.successor(), b3.successor(), b4.successor())));
                    b1.op(CoreOp.branch(b5.successor(b1.op(CoreOp.constant(stringType, "magic number")))));
                    b2.op(CoreOp.branch(b5.successor(b2.op(CoreOp.constant(stringType, "Answer to the Ultimate Question of Life, the Universe, and Everything")))));
                    b3.op(CoreOp.branch(b5.successor(b3.op(CoreOp.constant(stringType, "introduction to a subject")))));
                    b4.op(CoreOp.branch(b5.successor(b4.op(CoreOp.constant(stringType, "not important")))));
                    b5.op(CoreOp.return_(b5.parameters().getFirst()));
                });

        MethodHandle mh = generate(f);

        for (int i = 0; i < 110; i++) {
            Assertions.assertEquals(switchExpression(i), (String)mh.invokeExact(i));
        }
    }

    static MethodHandle generate(CoreOp.FuncOp f) {
        System.out.println(f.toText());

        return BytecodeGenerator.generate(MethodHandles.lookup(), f);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestTry.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
