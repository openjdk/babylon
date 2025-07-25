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
 * @run testng TestStream
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.java.JavaType.type;

public class TestStream {

    @Test
    public void testMapFilterForEach() {
        CoreOp.FuncOp f = StreamFuser.fromList(type(Integer.class))
                .map((Integer i) -> i.toString())
                .filter((String s) -> s.length() < 10)
                .map((String s) -> s.concat("_XXX"))
                .filter((String s) -> s.length() < 10)
                .forEach((String s) -> System.out.println(s));

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Interpreter.invoke(MethodHandles.lookup(), lf,
                List.of(List.of(1, 2, 3, 4, 5, 100_000_000, 10_000, 100_000, 20)));
    }

    @Test
    public void testMapFlatMapFilterCollect() {
        CoreOp.FuncOp f = StreamFuser.fromList(type(Integer.class))
                .map((Integer i) -> i.toString())
                .flatMap((String s) -> List.of(s, s))
                .filter((String s) -> s.length() < 10)
                .map((String s) -> s.concat("_XXX"))
                .filter((String s) -> s.length() < 10)
                .collect(() -> new ArrayList<String>(), (List<String> l, String e) -> l.add(e));

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        List<Integer> source = List.of(1, 2, 3, 4, 5, 100_000_000, 10_000, 20);

        List<String> expected = source.stream()
                .map(Object::toString)
                .flatMap(s -> Stream.of(s, s))
                .filter(s -> s.length() < 10)
                .map(s -> s.concat("_XXX"))
                .filter(s -> s.length() < 10)
                .toList();

        @SuppressWarnings("unchecked")
        List<String> actual = (List<String>) Interpreter.invoke(MethodHandles.lookup(), lf,
                List.of(source));

        Assert.assertEquals(expected, actual);
    }
}
