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

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.interpreter.Interpreter;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestExpressionElimination
 */

public class TestExpressionElimination {

    @Test
    public void testAddZero() {
        CoreOp.ClosureOp lf = generate((double a) -> a + 0.0);

        Assert.assertEquals((double) Interpreter.invoke(MethodHandles.lookup(), lf, 1.0d), 1.0d);
    }

    @Test
    public void testF() {
        CoreOp.ClosureOp lf = generate((double a, double b) -> -a + b);

        Assert.assertEquals((double) Interpreter.invoke(MethodHandles.lookup(), lf, 1.0d, 1.0d), 0.0d);
    }

    static CoreOp.ClosureOp generate(Quoted q) {
        return generateF((CoreOp.ClosureOp) q.op());
    }

    static <T extends Op & Op.Invokable> T generateF(T f) {
        System.out.println(f.toText());

        @SuppressWarnings("unchecked")
        T lf = (T) f.transform(CopyContext.create(), OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(lf.toText());

        lf = SSA.transform(lf);
        System.out.println(lf.toText());

        lf = ExpressionElimination.eliminate(lf);
        System.out.println(lf.toText());
        return lf;
    }
}
