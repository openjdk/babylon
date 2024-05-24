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
import org.testng.annotations.Test;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.bytecode.SlotOp;
import java.lang.reflect.code.bytecode.SlotSSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;

/*
 * @test
 * @run testng TestSlots
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
        return CoreOp.func("f", FunctionType.functionType(JavaType.J_L_STRING)).body(b -> {
            Block.Builder trueBlock = b.block();
            Block.Builder falseBlock = b.block();
            Block.Builder exitBlock = b.block();

            // Entry block
            {
                Value nullConstant = b.op(CoreOp.constant(JavaType.J_L_OBJECT, null));
                b.op(SlotOp.store(0, nullConstant));

                b.op(CoreOp.conditionalBranch(b.op(CoreOp.constant(JavaType.BOOLEAN, true)),
                        trueBlock.successor(), falseBlock.successor()));
            }

            // True block
            {
                Value oneConstant = trueBlock.op(CoreOp.constant(JavaType.INT, 1));
                trueBlock.op(SlotOp.store(0, oneConstant));

                Value loadValue = trueBlock.op(SlotOp.load(0, JavaType.INT));
                trueBlock.op(CoreOp.invoke(MethodRef.method(TestSlots.class, "m", void.class, int.class), loadValue));

                Value stringConstant = trueBlock.op(CoreOp.constant(JavaType.J_L_STRING, "TRUE"));
                trueBlock.op(SlotOp.store(0, stringConstant));

                trueBlock.op(CoreOp.branch(exitBlock.successor()));
            }

            // False block
            {
                Value stringConstant = falseBlock.op(CoreOp.constant(JavaType.J_L_STRING, "FALSE"));
                falseBlock.op(SlotOp.store(0, stringConstant));

                falseBlock.op(CoreOp.branch(exitBlock.successor()));
            }

            // Exit block
            {
                Value loadValue = exitBlock.op(SlotOp.load(0, JavaType.J_L_STRING));
                exitBlock.op(CoreOp._return(loadValue));
            }
        });
    }
}
