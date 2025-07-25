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

import hat.codebuilders.C99HATComputeBuilder;
import hat.optools.FuncOpWrapper;
import hat.optools.OpWrapper;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;

import static jdk.incubator.code.dialect.core.CoreOp.return_;
import static jdk.incubator.code.dialect.core.CoreOp.closureCall;
import static jdk.incubator.code.dialect.core.CoreOp.constant;
import static jdk.incubator.code.dialect.core.CoreOp.func;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaOp.add;
import static jdk.incubator.code.dialect.java.JavaType.INT;
/*
https://github.com/openjdk/babylon/tree/code-reflection/test/jdk/java/lang/reflect/code
*/

public class QuotedTest {
    public static void quotedTest() {
        Quoted quoted = () -> {
        }; //See TestClosureOps:132
        Op qop = quoted.op();
        Op top = qop.ancestorOp().ancestorOp();


        CoreOp.FuncOp fop = (CoreOp.FuncOp) top;
    }

    public static void main(String[] args) {
        quotedTest();
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(block -> {
                    //  OpWrapper.BodyWrapper.onlyBlock(block, l->{});
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    //   captures i
                    CoreOp.ClosureOp closure = CoreOp.closure(block.parentBody(), functionType(INT, INT))
                            .body(cblock -> {
                                Block.Parameter ci = cblock.parameters().get(0);
                                cblock.op(return_(cblock.op(add(i, ci))));
                            });
                    Op.Result c = block.op(closure);
                    Op.Result fortyTwo = block.op(constant(INT, 42));
                    Op.Result or = block.op(closureCall(c, fortyTwo));
                    block.op(return_(or));
                });

        System.out.println(f.toText());
        MethodHandles.Lookup lookup =  MethodHandles.lookup();
        C99HATComputeBuilder codeBuilder = new C99HATComputeBuilder();
        FuncOpWrapper wf = OpWrapper.wrap(lookup,f);
        codeBuilder.compute(wf);
        System.out.println(codeBuilder);

        // target type of a lambda must be an interface

    }

}
