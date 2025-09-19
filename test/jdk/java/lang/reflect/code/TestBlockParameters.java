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
 * @modules jdk.incubator.code
 * @run junit TestBlockParameters
 */

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.INT;

public class TestBlockParameters {
    static FuncOp m() {
        return func("f", functionType(INT, INT, INT))
                .body(fe -> {
                    JavaOp.LambdaOp lop = JavaOp.lambda(fe.parentBody(), functionType(INT, INT), JavaType.type(FunctionType.class))
                            .body(le -> {
                                le.op(return_(le.parameters().get(0)));
                            });
                    fe.op(lop);
                    Block.Builder b = fe.block(INT, INT);
                    fe.op(branch(b.successor(fe.parameters())));

                    b.op(return_(b.parameters().get(0)));
                });
    }

    @Test
    public void t() {
        FuncOp m = m();
        m.traverse(null, CodeElement.blockVisitor((_, b) -> {
            for (Block.Parameter p : b.parameters()) {
                testBlockParameter(p);
            }

            return null;
        }));
    }

    void testBlockParameter(Block.Parameter p) {
        Assertions.assertEquals(p.declaringBlock().parameters().indexOf(p), p.index());

        if (p.invokableOperation() instanceof Op.Invokable iop) {
            Assertions.assertTrue(p.declaringBlock().isEntryBlock());
            Assertions.assertEquals(iop.parameters().indexOf(p), p.index());
        } else {
            // There are no non-invokable operations with bodies in the model
            Assertions.assertFalse(p.declaringBlock().isEntryBlock());
        }
    }
}
