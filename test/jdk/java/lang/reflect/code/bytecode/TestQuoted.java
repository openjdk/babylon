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

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestQuoted
 */

public class TestQuoted {

    @Test
    public void testQuoted() throws Throwable {
        Quoted q = (int i, int j) -> {
            i = i + j;
            return i;
        };
        CoreOp.ClosureOp cop = (CoreOp.ClosureOp) q.op();

        MethodHandle mh = generate(cop);

        Assert.assertEquals(3, (int) mh.invoke(1, 2));
    }

    static <O extends Op & Op.Invokable> MethodHandle generate(O f) {
        System.out.println(f.toText());

        @SuppressWarnings("unchecked")
        O lf = (O) f.transform(CopyContext.create(), OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(lf.toText());

        return BytecodeGenerator.generate(MethodHandles.lookup(), lf);
    }
}
