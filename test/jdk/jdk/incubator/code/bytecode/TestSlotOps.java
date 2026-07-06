/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Block;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.SlotOp;
import jdk.incubator.code.bytecode.SlotSSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Test;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestSlots
 */
public class TestSlotOps {

    static void m(int i) {

    }

    @Test
    public void test() {
        CoreOp.FuncOp f = build();
        System.out.println(f.toText());

        CoreOp.FuncOp fssa = SlotSSA.transform(f);
        System.out.println(fssa.toText());
    }

    static CoreOp.FuncOp build() {
        return CoreOp.func("f", CoreType.functionType(JavaType.J_L_STRING)).body(b -> {
            Block.Builder trueBlock = b.block();
            Block.Builder falseBlock = b.block();
            Block.Builder exitBlock = b.block();

            // Entry block
            {
                Value nullConstant = b.add(CoreOp.constant(JavaType.J_L_OBJECT, null));
                b.add(SlotOp.store(0, nullConstant));

                b.add(CoreOp.conditionalBranch(b.add(CoreOp.constant(JavaType.BOOLEAN, true)),
                        trueBlock.reference(), falseBlock.reference()));
            }

            // True block
            {
                Value oneConstant = trueBlock.add(CoreOp.constant(JavaType.INT, 1));
                trueBlock.add(SlotOp.store(0, oneConstant));

                Value loadValue = trueBlock.add(SlotOp.load(0, JavaType.INT));
                trueBlock.add(JavaOp.invoke(MethodRef.method(TestSlots.class, "m", void.class, int.class), loadValue));

                Value stringConstant = trueBlock.add(CoreOp.constant(JavaType.J_L_STRING, "TRUE"));
                trueBlock.add(SlotOp.store(0, stringConstant));

                trueBlock.add(CoreOp.branch(exitBlock.reference()));
            }

            // False block
            {
                Value stringConstant = falseBlock.add(CoreOp.constant(JavaType.J_L_STRING, "FALSE"));
                falseBlock.add(SlotOp.store(0, stringConstant));

                falseBlock.add(CoreOp.branch(exitBlock.reference()));
            }

            // Exit block
            {
                Value loadValue = exitBlock.add(SlotOp.load(0, JavaType.J_L_STRING));
                exitBlock.add(CoreOp.return_(loadValue));
            }
        });
    }
}
