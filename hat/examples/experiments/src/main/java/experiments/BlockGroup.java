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
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Trxfmr;
import optkl.codebuilders.JavaCodeBuilder;
import optkl.util.Regex;


import java.io.PrintStream;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;

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


    public static void main(String[] args) throws Throwable {
        var lookup = MethodHandles.lookup();
        Method m = BlockGroup.class.getDeclaredMethod("m", int.class, int.class);
        CoreOp.FuncOp mModel = Op.ofMethod(m).orElseThrow();

       // System.out.println("From Code Model             -\n"+ OpCodeBuilder.toText(mModel));
        System.out.println("From Approx Jave Source ------\n"+JavaCodeBuilder.toText(lookup,mModel));

        BytecodeGenerator.generate(lookup, mModel).invoke(1,2);
        SequencedSet<Op> opsToGroup = new TreeSet<>(Comparator.comparingInt(op -> op.parent().ops().indexOf(op)));


        Invoke.stream(lookup,mModel.body().entryBlock()) // So this yields a stream of Invoke helpers
                .filter(invoke->// it's easier to check if is in fact println
                        invoke.refIs(PrintStream.class)
                                && invoke.named(Regex.of("print(ln|)"))
                                && invoke.returns(void.class))
                .forEach(invoke -> opsToGroup.addAll(List.of(
                                invoke.opFromOperandNOrThrow(0),// instead of op.operands().get(0).result().op()
                                invoke.opFromOperandNOrThrow(1),
                                invoke.op())
                        )
                );

         var mGroupModel=  Trxfmr.of(lookup,mModel).transform(opsToGroup::contains, c->{ // Here we use a HAT style transformer
            if (opsToGroup.getLast() == c.op()) {
                // Create a new body builder connected as a child
                // Use a child of the code context so values can be shared ???? what does this mean
                Body.Builder groupBodyBuilder = Body.Builder.of(
                        c.builder().parentBody(), CoreType.FUNCTION_TYPE_VOID,
                        CodeContext.create(c.builder().context()));

                // Add ops to the entry block
                Block.Builder groupBlockBuilder = groupBodyBuilder.entryBlock();
                opsToGroup.forEach(groupBlockBuilder::op); // transfers all to this builder?
                groupBlockBuilder.op(CoreOp.core_yield());

                c.replace(JavaOp.block(groupBodyBuilder)); // Replace all those added ops with the block op
            }else{
                c.remove(); // Unlike regular trasnformers we must actively remove
            }
            // But we don't' have to deal with anything we don't care about
            // no do we return the builder
        }).funcOp();

      //  System.out.println("To Code Model               -\n"+ OpCodeBuilder.toText(mGroupModel));
        System.out.println("To Approx Java Source    ------\n"+JavaCodeBuilder.toText(lookup,mGroupModel));

        // make sure we didnt break anything
       BytecodeGenerator.generate(lookup, mGroupModel).invoke(1,2);
    }
}
