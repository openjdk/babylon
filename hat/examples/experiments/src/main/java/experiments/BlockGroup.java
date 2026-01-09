/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package experiments;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.codebuilders.JavaCodeBuilder;
import optkl.util.OpCodeBuilder;


import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;

public class BlockGroup {

    @Reflect
    static int m(int a, int b) {
        a += 2;
        b += 2;

        // Group these
        System.out.println(a);
        System.out.println(b);

        return a + b;
    }


    static SequencedSet<Op> findGroup(Block b) {
        SequencedSet<Op> ops = new TreeSet<>(Comparator.comparingInt(op -> op.parent().ops().indexOf(op)));
        for (Op op : b.ops()) {
            // Assuming System.out.println(t) :)
            if (op instanceof JavaOp.InvokeOp _) {
                Value out = op.operands().get(0);
                Value value = op.operands().get(1);

                ops.add(OpHelper.asOpFromResultOrNull(out));
                ops.add(OpHelper.asOpFromResultOrNull(value));
                ops.add(op);
            }
        }
        return ops;
    }

    static CoreOp.FuncOp group(CoreOp.FuncOp f) {
        SequencedSet<Op> group = findGroup(f.body().entryBlock());

        return f.transform((block, op) -> {
            if (group.contains(op)) {
                // Drop op until we reach the last one
                if (group.getLast() == op) {
                    // Create a new body builder connected as a child
                    // Use a child of the code context so values can be shared
                    Body.Builder groupBodyBuilder = Body.Builder.of(
                            block.parentBody(), CoreType.FUNCTION_TYPE_VOID,
                            CodeContext.create(block.context()));

                    // Add ops to the entry block
                    Block.Builder groupBlockBuilder = groupBodyBuilder.entryBlock();
                    group.forEach(groupBlockBuilder::op);
                    groupBlockBuilder.op(CoreOp.core_yield());

                    // Replace all those added ops with the block op
                    block.op(JavaOp.block(groupBodyBuilder));
                }
            } else {
                block.op(op);
            }
            return block;
        });
    }



    public static void main(String[] args) throws Exception {
        var lookup = MethodHandles.lookup();
        Method m = BlockGroup.class.getDeclaredMethod("m", int.class, int.class);
        CoreOp.FuncOp mModel = Op.ofMethod(m).orElseThrow();
       // System.out.println("From Code Model             -\n"+ OpCodeBuilder.toText(mModel));
        System.out.println("From Approx Jave Source ------\n"+JavaCodeBuilder.toText(lookup,mModel));
        CoreOp.FuncOp mGroupModel = group(mModel);
      //  System.out.println("To Code Model               -\n"+ OpCodeBuilder.toText(mGroupModel));
        System.out.println("To Approx Java Source    ------\n"+JavaCodeBuilder.toText(lookup,mGroupModel));

    }
}
