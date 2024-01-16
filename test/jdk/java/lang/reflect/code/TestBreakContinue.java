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
 * @run testng TestBreakContinue
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.BitSet;
import java.util.Optional;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

public class TestBreakContinue {

    @CodeReflection
    public static BitSet forLoopBreakContinue(IntUnaryOperator f) {
        BitSet b = new BitSet();
        for (int i = 0; i < 8; i++) {
            b.set(i);
            int r = f.applyAsInt(i);
            if (r == 0) {
                continue;
            } else if (r == 1) {
                break;
            }
            b.set(i * 2);
        }
        return b;
    }

    @Test
    public void testForLoopBreakContinue() {
        CoreOps.FuncOp f = getFuncOp("forLoopBreakContinue");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        IntUnaryOperator o = i -> {
            if (i <= 3) return -1;
            if (i <= 5) return 0;
            return 1;
        };
        Assert.assertEquals(Interpreter.invoke(lf, o), forLoopBreakContinue(o));
    }

    @CodeReflection
    public static BitSet nestedForLoopBreakContinue(IntUnaryOperator f) {
        BitSet b = new BitSet();
        for (int j = 0; j < 8; j++) {
            b.set(j);
            int r = f.applyAsInt(j);
            if (r == 0) {
                continue;
            } else if (r == 1) {
                break;
            }
            for (int i = 8; i < 16; i++) {
                b.set(i);
                r = f.applyAsInt(i);
                if (r == 2) {
                    continue;
                } else if (r == 3) {
                    break;
                }
                b.set(i * 2);
            }
            b.set(j * 2);
        }
        return b;
    }

    @Test
    public void testNestedForLoopBreakContinue() {
        CoreOps.FuncOp f = getFuncOp("nestedForLoopBreakContinue");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        for (int r = -1; r < 4; r++) {
            int fr = r;
            IntUnaryOperator o = i -> fr;
            Assert.assertEquals(Interpreter.invoke(lf, o), nestedForLoopBreakContinue(o));
        }
    }


    @CodeReflection
    public static BitSet forLoopLabeledBreakContinue(IntUnaryOperator f) {
        BitSet b = new BitSet();
        outer: for (int j = 0; j < 8; j++) {
            b.set(j);
            int r = f.applyAsInt(j);
            if (r == 0) {
                continue outer;
            } else if (r == 1) {
                break outer;
            }
            inner: for (int i = 8; i < 16; i++) {
                b.set(i);
                r = f.applyAsInt(i);
                if (r == 2) {
                    continue inner;
                } else if (r == 3) {
                    break inner;
                } else if (r == 4) {
                    continue outer;
                } else if (r == 5) {
                    break outer;
                }
                b.set(i * 2);
            }
            b.set(j * 2);
        }
        return b;
    }

    @Test
    public void testForLoopLabeledBreakContinue() {
        CoreOps.FuncOp f = getFuncOp("forLoopLabeledBreakContinue");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        for (int r = -1; r < 6; r++) {
            int fr = r;
            IntUnaryOperator o = i -> fr;
            Assert.assertEquals(Interpreter.invoke(lf, o), forLoopLabeledBreakContinue(o));
        }
    }

    @CodeReflection
    public static BitSet blockBreak(IntUnaryOperator f) {
        BitSet b = new BitSet();
        a: b: {
            b.set(1);
            if (f.applyAsInt(1) != 0) {
                break a;
            }
            b.set(2);
            if (f.applyAsInt(2) != 0) {
                break b;
            }
            b.set(3);
            c: {
                b.set(4);
                if (f.applyAsInt(4) != 0) {
                    break a;
                }
                b.set(5);
                if (f.applyAsInt(5) != 0) {
                    break b;
                }
                b.set(6);
                if (f.applyAsInt(6) != 0) {
                    break c;
                }
                b.set(7);
            }
            b.set(8);
        }
        return b;
    }

    @Test
    public void testBlockBreak() {
        CoreOps.FuncOp f = getFuncOp("blockBreak");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        for (int i = 0; i < 7; i++) {
            int fi = i;
            IntUnaryOperator o = v -> v == fi ? 1 : 0;
            Assert.assertEquals(Interpreter.invoke(lf, o), blockBreak(o));
        }
    }


    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestBreakContinue.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
