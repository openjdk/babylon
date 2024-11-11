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
 * @run testng TestClosureOps
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.Block;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.type.JavaType;
import java.util.ArrayList;
import java.util.List;

import static jdk.incubator.code.op.CoreOp._return;
import static jdk.incubator.code.op.CoreOp.add;
import static jdk.incubator.code.op.CoreOp.closure;
import static jdk.incubator.code.op.CoreOp.closureCall;
import static jdk.incubator.code.op.CoreOp.constant;
import static jdk.incubator.code.op.CoreOp.func;
import static jdk.incubator.code.op.CoreOp.quoted;
import static jdk.incubator.code.type.FunctionType.functionType;
import static jdk.incubator.code.type.JavaType.INT;
import static jdk.incubator.code.type.JavaType.type;

public class TestClosureOps {

    static class Builder {
        static final MethodRef ACCEPT_METHOD = MethodRef.method(type(TestClosureOps.Builder.class), "accept",
                INT, CoreOp.QuotedOp.QUOTED_TYPE);

        static int accept(Quoted c) {
            Assert.assertEquals(1, c.capturedValues().size());
            Assert.assertEquals(1, c.capturedValues().values().iterator().next());

            List<Object> arguments = new ArrayList<>();
            arguments.add(42);
            arguments.addAll(c.capturedValues().values());
            int r = (int) Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) c.op(),
                    arguments);
            return r;
        }
    }

    @Test
    public void testQuotedWithCapture() {
        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    // op descriptor = ()Quoted<ClosureOp>
                    CoreOp.QuotedOp qop = quoted(block.parentBody(), qblock -> {
                        return closure(qblock.parentBody(), functionType(INT, INT))
                                .body(cblock -> {
                                    Block.Parameter ci = cblock.parameters().get(0);

                                    cblock.op(_return(
                                            // capture i from function's body
                                            cblock.op(add(i, ci))
                                    ));
                                });
                    });
                    Op.Result cquoted = block.op(qop);

                    Op.Result or = block.op(CoreOp.invoke(TestClosureOps.Builder.ACCEPT_METHOD, cquoted));
                    block.op(_return(or));
                });

        f.writeTo(System.out);

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testWithCapture() {
        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(block -> {
                    Block.Parameter i = block.parameters().get(0);

                    // functional type = (int)int
                    //   captures i
                    CoreOp.ClosureOp closure = CoreOp.closure(block.parentBody(),
                                    functionType(INT, INT))
                            .body(cblock -> {
                                Block.Parameter ci = cblock.parameters().get(0);

                                cblock.op(_return(
                                        cblock.op(add(i, ci))));
                            });
                    Op.Result c = block.op(closure);

                    Op.Result fortyTwo = block.op(constant(INT, 42));
                    Op.Result or = block.op(closureCall(c, fortyTwo));
                    block.op(_return(or));
                });

        f.writeTo(System.out);

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testQuotableModel() {
        Quoted quoted = () -> {};
        Op qop = quoted.op();
        Op top = qop.ancestorBody().parentOp().ancestorBody().parentOp();
        Assert.assertTrue(top instanceof CoreOp.FuncOp);

        CoreOp.FuncOp fop = (CoreOp.FuncOp) top;
        Assert.assertEquals(JavaType.type(Quoted.class), fop.invokableType().returnType());
    }
}
