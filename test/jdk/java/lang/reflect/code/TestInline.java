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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.util.List;

import static java.lang.reflect.code.op.CoreOps.*;
import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;
import static java.lang.reflect.code.descriptor.TypeDesc.INT;

/*
 * @test
 * @run testng TestInline
 */

public class TestInline {

    @Test
    public void testInline() {
        Quoted q = (int a, int b) -> a + b;
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();

        // functional descriptor = (int)int
        CoreOps.FuncOp f = func("f", methodType(int.class, int.class))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    var cb = fblock.inline(cop, List.of(i, fortyTwo), Block.Builder.INLINE_RETURN);
                    Assert.assertEquals(fblock, cb);
                });

        f.writeTo(System.out);

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testInlineVar() {
        Quoted q = (int a, int b) -> a + b;
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();

        // functional descriptor = (int)int
        CoreOps.FuncOp f = func("f", methodType(int.class, int.class))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    Op.Result v = fblock.op(var(fblock.op(constant(INT, 0))));

                    var cb = fblock.inline(cop, List.of(i, fortyTwo), (b, value) -> {
                        b.op(varStore(v, value));
                    });
                    Assert.assertEquals(fblock, cb);

                    fblock.op(_return(fblock.op(varLoad(v))));
                });

        f.writeTo(System.out);

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
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();
        cop.writeTo(System.out);
        CoreOps.ClosureOp lcop = cop.transform(CopyContext.create(), (block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        lcop.writeTo(System.out);

        // functional descriptor = (int)int
        CoreOps.FuncOp f = func("f", methodType(int.class, int.class))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    var cb = fblock.inline(lcop, List.of(i, fortyTwo), Block.Builder.INLINE_RETURN);
                    Assert.assertNotEquals(fblock, cb);
                });
        f.writeTo(System.out);

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
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();
        cop.writeTo(System.out);
        CoreOps.ClosureOp lcop = cop.transform(CopyContext.create(), (block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        lcop.writeTo(System.out);

        // functional descriptor = (int)int
        CoreOps.FuncOp f = func("f", methodType(int.class, int.class))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    Op.Result v = fblock.op(var(fblock.op(constant(INT, 0))));

                    var cb = fblock.inline(lcop, List.of(i, fortyTwo), (b, value) -> {
                        b.op(varStore(v, value));
                    });
                    Assert.assertNotEquals(fblock, cb);

                    cb.op(_return(cb.op(varLoad(v))));
                });
        f.writeTo(System.out);

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
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();
        cop.writeTo(System.out);

        CoreOps.FuncOp f = func("f", methodType(int.class, int.class))
                .body(fblock -> {
                    Block.Parameter i = fblock.parameters().get(0);

                    Op.Result fortyTwo = fblock.op(constant(INT, 42));

                    var cb = fblock.inline(cop, List.of(i, fortyTwo), Block.Builder.INLINE_RETURN);
                    Assert.assertEquals(fblock, cb);
                });
        f.writeTo(System.out);

        f = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        f.writeTo(System.out);

        int ir = (int) Interpreter.invoke(MethodHandles.lookup(), f, 1);
        Assert.assertEquals(ir, 43);
    }

    @Test
    public void testInlineVoid() {
        Quoted q = (int[] a) -> {
            a[0] = 42;
            return;
        };
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();

        // functional descriptor = (int)int
        CoreOps.FuncOp f = func("f", methodType(void.class, int[].class))
                .body(fblock -> {
                    Block.Parameter a = fblock.parameters().get(0);

                    var cb = fblock.inline(cop, List.of(a), Block.Builder.INLINE_RETURN);
                    Assert.assertEquals(fblock, cb);
                });

        f.writeTo(System.out);

        int[] a = new int[1];
        Interpreter.invoke(MethodHandles.lookup(), f, a);
        Assert.assertEquals(a[0], 42);
    }

}
