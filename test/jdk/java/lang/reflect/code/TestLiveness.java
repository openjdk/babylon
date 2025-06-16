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
 * @run testng TestLiveness
 */

import jdk.incubator.code.dialect.java.JavaOp;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.Liveness;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.parser.OpParser;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class TestLiveness {

    static final String F = """
            func @"f" (%0 : java.type:"int", %1 : java.type:"int")java.type:"int" -> {
                %2 : java.type:"int" = add %0 %1;
                return %2;
            };
            """;

    @Test
    public void testF() {
        Op op = OpParser.fromString(JavaOp.FACTORY, F).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of()));
        Assert.assertEquals(actual, expected);
    }

    static final String IF_ELSE = """
            func @"ifelse" (%0 : java.type:"int", %1 : java.type:"int", %2 : java.type:"int")java.type:"int" -> {
                %3 : java.type:"int" = constant @10;
                %4 : java.type:"boolean" = lt %2 %3;
                cbranch %4 ^block_0 ^block_1;

              ^block_0:
                %5 : java.type:"int" = constant @1;
                %6 : java.type:"int" = add %0 %5;
                branch ^block_2(%6, %1);

              ^block_1:
                %7 : java.type:"int" = constant @2;
                %8 : java.type:"int" = add %1 %7;
                branch ^block_2(%0, %8);

              ^block_2(%9 : java.type:"int", %10 : java.type:"int"):
                %11 : java.type:"int" = add %9 %10;
                return %11;
            };
            """;

    @Test
    public void testIfElse() {
        Op op = OpParser.fromString(JavaOp.FACTORY, IF_ELSE).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of(0, 1)),
                1, List.of(Set.of(0, 1), Set.of()),
                2, List.of(Set.of(0, 1), Set.of()),
                3, List.of(Set.of(), Set.of())
        );
        Assert.assertEquals(actual, expected);
    }

    static final String LOOP = """
            func @"loop" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = constant @0;
                %2 : java.type:"int" = constant @0;
                branch ^block_0(%1, %2);

              ^block_0(%3 : java.type:"int", %4 : java.type:"int"):
                %5 : java.type:"boolean" = lt %4 %0;
                cbranch %5 ^block_1 ^block_2;

              ^block_1:
                %6 : java.type:"int" = add %3 %4;
                branch ^block_3;

              ^block_3:
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"int" = add %4 %7;
                branch ^block_0(%6, %8);

              ^block_2:
                return %3;
            };
            """;

    @Test
    public void testLoop() {
        Op op = OpParser.fromString(JavaOp.FACTORY, LOOP).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of(0)),
                1, List.of(Set.of(0), Set.of(0, 3, 4)),
                2, List.of(Set.of(0, 3, 4), Set.of(0, 4, 6)),
                3, List.of(Set.of(3), Set.of()),
                4, List.of(Set.of(0, 4, 6), Set.of(0))
        );
        Assert.assertEquals(actual, expected);
    }

    static final String IF_ELSE_NESTED = """
            func @"ifelseNested" (%0 : java.type:"int", %1 : java.type:"int", %2 : java.type:"int", %3 : java.type:"int", %4 : java.type:"int")java.type:"int" -> {
                %5 : java.type:"int" = constant @20;
                %6 : java.type:"boolean" = lt %4 %5;
                cbranch %6 ^block_0 ^block_1;

              ^block_0:
                %7 : java.type:"int" = constant @10;
                %8 : java.type:"boolean" = lt %4 %7;
                cbranch %8 ^block_2 ^block_3;

              ^block_2:
                %9 : java.type:"int" = constant @1;
                %10 : java.type:"int" = add %0 %9;
                branch ^block_4(%10, %1);

              ^block_3:
                %11 : java.type:"int" = constant @2;
                %12 : java.type:"int" = add %1 %11;
                branch ^block_4(%0, %12);

              ^block_4(%13 : java.type:"int", %14 : java.type:"int"):
                %15 : java.type:"int" = constant @3;
                %16 : java.type:"int" = add %2 %15;
                branch ^block_5(%13, %14, %16, %3);

              ^block_1:
                %17 : java.type:"int" = constant @20;
                %18 : java.type:"boolean" = gt %4 %17;
                cbranch %18 ^block_6 ^block_7;

              ^block_6:
                %19 : java.type:"int" = constant @4;
                %20 : java.type:"int" = add %0 %19;
                branch ^block_8(%20, %1);

              ^block_7:
                %21 : java.type:"int" = constant @5;
                %22 : java.type:"int" = add %1 %21;
                branch ^block_8(%0, %22);

              ^block_8(%23 : java.type:"int", %24 : java.type:"int"):
                %25 : java.type:"int" = constant @6;
                %26 : java.type:"int" = add %3 %25;
                branch ^block_5(%23, %24, %2, %26);

              ^block_5(%27 : java.type:"int", %28 : java.type:"int", %29 : java.type:"int", %30 : java.type:"int"):
                %31 : java.type:"int" = add %27 %28;
                %32 : java.type:"int" = add %31 %29;
                %33 : java.type:"int" = add %32 %30;
                return %33;
            };
            """;

    @Test
    public void testIfElseNested() {
        Op op = OpParser.fromString(JavaOp.FACTORY, IF_ELSE_NESTED).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of(0, 1, 2, 3, 4)),
                1, List.of(Set.of(0, 1, 2, 3, 4), Set.of(0, 1, 2, 3)),
                2, List.of(Set.of(0, 1, 2, 3, 4), Set.of(0, 1, 2, 3)),
                3, List.of(Set.of(0, 1, 2, 3), Set.of(2, 3)),
                4, List.of(Set.of(0, 1, 2, 3), Set.of(2, 3)),
                5, List.of(Set.of(2, 3), Set.of()),
                6, List.of(Set.of(), Set.of()),
                7, List.of(Set.of(0, 1, 2, 3), Set.of(2, 3)),
                8, List.of(Set.of(0, 1, 2, 3), Set.of(2, 3)),
                9, List.of(Set.of(2, 3), Set.of())
        );
        Assert.assertEquals(actual, expected);
    }

    static final String LOOP_NESTED = """
            func @"loopNested" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"int" = constant @0;
                %2 : java.type:"int" = constant @0;
                branch ^block_0(%1, %2);

              ^block_0(%3 : java.type:"int", %4 : java.type:"int"):
                %5 : java.type:"boolean" = lt %4 %0;
                cbranch %5 ^block_1 ^block_2;

              ^block_1:
                %6 : java.type:"int" = constant @0;
                branch ^block_3(%3, %6);

              ^block_3(%7 : java.type:"int", %8 : java.type:"int"):
                %9 : java.type:"boolean" = lt %8 %0;
                cbranch %9 ^block_4 ^block_5;

              ^block_4:
                %10 : java.type:"int" = add %7 %4;
                %11 : java.type:"int" = add %10 %8;
                branch ^block_6;

              ^block_6:
                %12 : java.type:"int" = constant @1;
                %13 : java.type:"int" = add %8 %12;
                branch ^block_3(%11, %13);

              ^block_5:
                branch ^block_7;

              ^block_7:
                %14 : java.type:"int" = constant @1;
                %15 : java.type:"int" = add %4 %14;
                branch ^block_0(%7, %15);

              ^block_2:
                return %3;
            };
            """;

    @Test
    public void testLoopNested() {
        Op op = OpParser.fromString(JavaOp.FACTORY, LOOP_NESTED).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of(0)),
                1, List.of(Set.of(0), Set.of(0, 3, 4)),
                2, List.of(Set.of(0, 3, 4), Set.of(0, 4)),
                3, List.of(Set.of(3), Set.of()),
                4, List.of(Set.of(0, 4), Set.of(0, 4, 7, 8)),
                5, List.of(Set.of(0, 4, 7, 8), Set.of(0, 4, 8, 11)),
                6, List.of(Set.of(0, 4, 7), Set.of(0, 4, 7)),
                7, List.of(Set.of(0, 4, 8, 11), Set.of(0, 4)),
                8, List.of(Set.of(0, 4, 7), Set.of(0))
        );
        Assert.assertEquals(actual, expected);
    }

    static Map<Integer, List<Set<Integer>>> liveness(Op op) {
        Liveness l = new Liveness(op);
        System.out.println(l);

        Map<Value, Integer> valueMap = valueNameMapping(op);
        Map<Block, Integer> blockMap = blockNameMapping(op);

        return op.traverse(new HashMap<>(),
                CodeElement.blockVisitor((m, b) -> {
                    if (b.parentBody().parentOp() == op) {
                        Liveness.BlockInfo lbi = l.getLiveness(b);
                        m.put(blockMap.get(b),
                                List.of(
                                        lbi.liveIn().stream().map(valueMap::get).collect(Collectors.toSet()),
                                        lbi.liveOut().stream().map(valueMap::get).collect(Collectors.toSet())
                                ));
                    }
                    return m;
                }));
    }

    static Map<Block, Integer> blockNameMapping(Op top) {
        AtomicInteger i = new AtomicInteger();
        return top.traverse(new HashMap<>(), CodeElement.blockVisitor((m, b) -> {
            if (b.parentBody().parentOp() != top) {
                return m;
            }

            m.computeIfAbsent(b, _ -> i.getAndIncrement());
            for (Block.Reference s : b.successors()) {
                m.computeIfAbsent(s.targetBlock(), _ -> i.getAndIncrement());
            }

            return m;
        }));
    }

    static Map<Value, Integer> valueNameMapping(Op top) {
        AtomicInteger i = new AtomicInteger();
        return top.traverse(new HashMap<>(), (m, e) -> {
            switch (e) {
                case Block b -> {
                    for (Block.Parameter p : b.parameters()) {
                        m.put(p, i.getAndIncrement());
                    }
                }
                case Op op -> {
                    Op.Result r = op.result();
                    if (r != null && !r.type().equals(JavaType.VOID)) {
                        m.put(r, i.getAndIncrement());
                    }
                }
                default -> {
                }
            }
            return m;
        });
    }
}