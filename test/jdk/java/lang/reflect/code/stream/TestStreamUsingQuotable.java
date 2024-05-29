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
 * @run testng TestStreamUsingQuotable
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class TestStreamUsingQuotable {

    @Test
    public void testMapFilterForEach() {
        CoreOp.FuncOp f = StreamFuserUsingQuotable.fromList(Integer.class)
                .map(Object::toString)
                .filter(s -> s.length() < 10)
                .map(s -> s.concat("_XXX"))
                .filter(s -> s.length() < 10)
                // Cannot use method reference since it captures the result of the expression "System.out"
                .forEach(s -> System.out.println(s));

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        lf.writeTo(System.out);

        Interpreter.invoke(lf, List.of(List.of(1, 2, 3, 4, 5, 100_000_000, 10_000, 100_000, 20)));
    }

    @Test
    public void testMapFlatMapFilterCollect() {
        CoreOp.FuncOp f = StreamFuserUsingQuotable.fromList(Integer.class)
                .map(Object::toString)
                .flatMap(s -> List.of(s, s))
                .filter(s -> s.length() < 10)
                .map(s -> s.concat("_XXX"))
                .filter(s -> s.length() < 10)
                .collect(ArrayList::new, ArrayList::add);

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        lf.writeTo(System.out);

        List<Integer> source = List.of(1, 2, 3, 4, 5, 100_000_000, 10_000, 20);

        List<String> expected = source.stream()
                .map(Object::toString)
                .flatMap(s -> Stream.of(s, s))
                .filter(s -> s.length() < 10)
                .map(s -> s.concat("_XXX"))
                .filter(s -> s.length() < 10)
                .toList();

        @SuppressWarnings("unchecked")
        List<String> actual = (List<String>) Interpreter.invoke(lf, List.of(source));

        Assert.assertEquals(expected, actual);
    }
}
