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
 * @run testng TestUsesDependsOn
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.*;
import java.lang.reflect.code.parser.OpParser;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.Function;

public class TestUsesDependsOn {

    static final String OP = """
            func @"f" (%0 : int, %1 : int)int -> {
                %2 : int = add %0 %1;
                %3 : boolean = lt %0 %1;
                %4 : void = cbranch %3 ^b1(%2, %2) ^b2(%0, %1);

              ^b1(%5 : int, %6 : int):
                %7 : void = return %5;

              ^b2(%8 : int, %9 : int):
                %10 : void = return %8;
            };
            """;

    @Test
    public void testDependsOn() {
        Op f = OpParser.fromStringOfFuncOp(OP);

        Map<String, List<String>> dependsUpon = computeValueMap(f, Value::dependsOn);

        var expected = Map.ofEntries(
                Map.entry("0", List.of()),
                Map.entry("1", List.of()),
                Map.entry("2", List.of("0", "1")),
                Map.entry("3", List.of("0", "1")),
                Map.entry("4", List.of("3", "2", "0", "1")),
                Map.entry("5", List.of()),
                Map.entry("6", List.of()),
                Map.entry("7", List.of("5")),
                Map.entry("8", List.of()),
                Map.entry("9", List.of()),
                Map.entry("10", List.of("8"))
        );

        Assert.assertEquals(dependsUpon, expected);
    }


    @Test
    public void testUses() {
        Op f = OpParser.fromStringOfFuncOp(OP);
        f.writeTo(System.out);

        Map<String, List<String>> uses = computeValueMap(f, Value::uses);

        var expected = Map.ofEntries(
                Map.entry("0", List.of("2", "3", "4")),
                Map.entry("1", List.of("2", "3", "4")),
                Map.entry("2", List.of("4")),
                Map.entry("3", List.of("4")),
                Map.entry("4", List.of()),
                Map.entry("5", List.of("7")),
                Map.entry("6", List.of()),
                Map.entry("7", List.of()),
                Map.entry("8", List.of("10")),
                Map.entry("9", List.of()),
                Map.entry("10", List.of())
        );
        System.out.println(uses.toString());
        System.out.println(expected);

        Assert.assertEquals(uses, expected);
    }

    static Map<String, List<String>> computeValueMap(Op op, Function<Value, Set<? extends Value>> f) {
        AtomicInteger ai = new AtomicInteger();

        Map<Value, String> valueNameMap = computeValues(op, new HashMap<>(), (v, m) -> {
            String name = Integer.toString(ai.getAndIncrement());
            m.put(v, name);
        });

        return computeValues(op, new HashMap<>(), (v, m) -> {
            m.put(valueNameMap.get(v), f.apply(v).stream().map(valueNameMap::get).toList());
        });
    }

    static <T> T computeValues(Op op, T t, BiConsumer<Value, T> c) {
        return op.traverse(t, (m, codeElement) -> {
            return switch (codeElement) {
                case Block b -> {
                    for (var a : b.parameters()) {
                        c.accept(a, m);
                    }

                    yield m;
                }
                case Op o -> {
                    if (o.result() != null) {
                        c.accept(o.result(), m);
                    }

                    yield m;
                }
                default -> m;
            };
        });
    }
}
