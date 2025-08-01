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

import jdk.incubator.code.analysis.Inliner;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.List;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.INT;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestInline
 */

public class TestInline {

    @Test
    public void testInline() {
        Quoted q = (int a, int b) -> a + b;
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();

        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    var cb = Inliner.inline(fblock, cop, List.of(i, fortyTwo), Inliner.INLINE_RETURN);
                    Assert.assertEquals(fblock, cb);
                });

        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testInlineVar() {
        Quoted q = (int a, int b) -> a + b;
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();

        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    Op.Result v = fblock.op(var(fblock.op(constant(INT, 0))));

                    var cb = Inliner.inline(fblock, cop, List.of(i, fortyTwo), (b, value) -> {
                        b.op(varStore(v, value));
                    });
                    Assert.assertEquals(fblock, cb);

                    fblock.op(return_(fblock.op(varLoad(v))));
                });

        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }


    @Test
    public void testInlineLowerMultipleReturn() {
        Quoted q = (int a, int b) ->  {
            if (a < 10) {
                return a + b;
            }
            return a - b;
        };
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();
        System.out.println(cop.toText());
        CoreOp.ClosureOp lcop = cop.transform(CopyContext.create(), OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(lcop.toText());

        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    var cb = Inliner.inline(fblock, lcop, List.of(i, fortyTwo), Inliner.INLINE_RETURN);
                    Assert.assertNotEquals(fblock, cb);
                });
        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testInlineLowerMultipleReturnVar() {
        Quoted q = (int a, int b) ->  {
            if (a < 10) {
                return a + b;
            }
            return a - b;
        };
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();
        System.out.println(cop.toText());
        CoreOp.ClosureOp lcop = cop.transform(CopyContext.create(), OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(lcop.toText());

        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    Op.Result v = fblock.op(var(fblock.op(constant(INT, 0))));

                    var cb = Inliner.inline(fblock, lcop, List.of(i, fortyTwo), (b, value) -> {
                        b.op(varStore(v, value));
                    });
                    Assert.assertNotEquals(fblock, cb);

                    cb.op(return_(cb.op(varLoad(v))));
                });
        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testInlineMultipleReturnLower() {
        Quoted q = (int a, int b) ->  {
            if (a < 10) {
                return a + b;
            }
            return a - b;
        };
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();
        System.out.println(cop.toText());

        CoreOp.FuncOp f = func("f", functionType(INT, INT))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    var cb = Inliner.inline(fblock, cop, List.of(i, fortyTwo), Inliner.INLINE_RETURN);
                    Assert.assertEquals(fblock, cb);
                });
        System.out.println(f.toText());

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(f.toText());

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testInlineVoid() {
        Quoted q = (int[] a) -> {
            a[0] = 42;
            return;
        };
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();

        // functional type = (int)int
        CoreOp.FuncOp f = func("f", functionType(JavaType.VOID, JavaType.type(int[].class)))
                .body(fblock -> {
                    Block.Parameter a = fblock.parameters().get(0);

                    var cb = Inliner.inline(fblock, cop, List.of(a), Inliner.INLINE_RETURN);
                    Assert.assertEquals(fblock, cb);
                });

        System.out.println(f.toText());

        int[] a = new int[1];
        Interpreter.invoke(MethodHandles.lookup(), f, a);
        Assert.assertEquals(a[0], 42);
    }

}
