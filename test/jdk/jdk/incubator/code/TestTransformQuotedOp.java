/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.function.IntUnaryOperator;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreOp.return_;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.INT;
import static jdk.incubator.code.dialect.java.JavaType.type;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestTransformQuotedOp
 */
public class TestTransformQuotedOp {
    @Test
    void test() {
        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(CoreOp.QuotedOp.QUOTED_OP_TYPE, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    // op type = ()Quoted<LambdaOp>
                    CoreOp.QuotedOp qop = quoted(block.parentBody(), qblock -> {
                        return JavaOp.lambda(qblock.parentBody(),
                                        functionType(INT, INT), type(IntUnaryOperator.class))
                                .body(lblock -> {
                                    Block.Parameter li = lblock.parameters().get(0);

                                    lblock.op(return_(
                                            // capture i from function's body
                                            lblock.op(JavaOp.add(i, li))
                                    ));
                                });
                    });
                    Op.Result lquoted = block.op(qop);

                    block.op(return_(lquoted));
                });

        System.out.println(f.toText());

        FuncOp tf = f.transform(CodeTransformer.COPYING_TRANSFORMER);

        QuotedOp qop = (QuotedOp) tf.body().entryBlock().firstOp();

        Assertions.assertEquals(qop.bodies().getFirst().entryBlock().firstOp(), qop.quotedOp());
    }
}