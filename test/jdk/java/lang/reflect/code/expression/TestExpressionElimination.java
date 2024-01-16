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

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.interpreter.Interpreter;

/*
 * @test
 * @run testng TestExpressionElimination
 */

public class TestExpressionElimination {

    @Test
    public void testAddZero() {
        CoreOps.ClosureOp lf = generate((double a) -> a + 0.0);

        Assert.assertEquals((double) Interpreter.invoke(lf, 1.0d), 1.0d);
    }

    @Test
    public void testF() {
        CoreOps.ClosureOp lf = generate((double a, double b) -> -a + b);

        Assert.assertEquals((double) Interpreter.invoke(lf, 1.0d, 1.0d), 0.0d);
    }

    static CoreOps.ClosureOp generate(Quoted q) {
        return generateF((CoreOps.ClosureOp) q.op());
    }

    static <T extends Op & Op.Invokable> T generateF(T f) {
        f.writeTo(System.out);

        @SuppressWarnings("unchecked")
        T lf = (T) f.transform(CopyContext.create(), (block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        lf.writeTo(System.out);

        lf = SSA.transform(lf);
        lf.writeTo(System.out);

        lf = ExpressionElimination.eliminate(lf);
        lf.writeTo(System.out);
        return lf;
    }
}
