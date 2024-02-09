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
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.type.JavaType;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.lang.reflect.code.op.CoreOps._return;
import static java.lang.reflect.code.op.CoreOps.branch;
import static java.lang.reflect.code.op.CoreOps.conditionalBranch;
import static java.lang.reflect.code.op.CoreOps.constant;
import static java.lang.reflect.code.op.CoreOps.func;

/*
 * @test
 * @run testng TestDominate
 */

public class TestDominate {

    @Test
    public void testIfElse() {
        CoreOps.FuncOp f = func("f", MethodTypeDesc.VOID).body(entry -> {
            Block.Builder ifBlock = entry.block();
            Block.Builder elseBlock = entry.block();
            Block.Builder end = entry.block();

            Op.Result p = entry.op(constant(JavaType.BOOLEAN, true));
            entry.op(conditionalBranch(p, ifBlock.successor(), elseBlock.successor()));

            ifBlock.op(branch(end.successor()));

            elseBlock.op(branch(end.successor()));

            end.op(_return());
        });

        boolean[][] bvs = new boolean[][]{
                {true, false, false, false},
                {true, true, false, false},
                {true, false, true, false},
                {true, false, false, true}
        };

        test(f, bvs);
    }

    @Test
    public void testForwardSuccessors() {
        CoreOps.FuncOp f = func("f", MethodTypeDesc.VOID).body(entry -> {
            Block.Builder b1 = entry.block();
            Block.Builder b2 = entry.block();
            Block.Builder b3 = entry.block();
            Block.Builder b4 = entry.block();
            Block.Builder b5 = entry.block();

            Op.Result p = entry.op(constant(JavaType.BOOLEAN, true));
            entry.op(conditionalBranch(p, b4.successor(), b2.successor()));

            b4.op(conditionalBranch(p, b5.successor(), b3.successor()));

            b2.op(conditionalBranch(p, b5.successor(), b1.successor()));

            b5.op(_return());

            b3.op(branch(b1.successor()));

            b1.op(_return());
        });

        f.writeTo(System.out);
        boolean[][] bvs = new boolean[][]{
                {true, false, false, false, false, false},
                {true, true, false, false, false, false},
                {true, true, true, false, false, false},
                {true, false, false, true, false, false},
                {true, false, false, false, true, false},
                {true, false, false, false, false, true},
        };

        test(f, bvs);
    }

    @Test
    public void testBackbranch() {
        CoreOps.FuncOp f = func("f", MethodTypeDesc.VOID).body(entry -> {
            Block.Builder cond = entry.block();
            Block.Builder body = entry.block();
            Block.Builder update = entry.block();
            Block.Builder end = entry.block();

            Op.Result p = entry.op(constant(JavaType.BOOLEAN, true));
            entry.op(branch(cond.successor()));

            cond.op(conditionalBranch(p, body.successor(), end.successor()));

            body.op(branch(update.successor()));

            update.op(branch(cond.successor()));

            end.op(_return());

        });

        boolean[][] bvs = new boolean[][]{
                {true, false, false, false, false},
                {true, true, false, false, false},
                {true, true, true, false, false},
                {true, true, true, true, false},
                {true, true, false, false, true},
        };
        test(f, bvs);
    }

    static void test(CoreOps.FuncOp f, boolean[][] bvs) {
        Block[] bs = f.body().blocks().toArray(Block[]::new);
        for (int i = 0; i < bs.length; i++) {
            for (int j = 0; j < bs.length; j++) {
                Block x = bs[i];
                Block y = bs[j];
                Assert.assertEquals(y.isDominatedBy(x), bvs[j][i]);
            }
        }
    }


    @Test
    public void testImmediateDominators() {
        CoreOps.FuncOp f = func("f", MethodTypeDesc.VOID).body(entry -> {
            Block.Builder b6 = entry.block();
            Block.Builder b5 = entry.block();
            Block.Builder b4 = entry.block();
            Block.Builder b3 = entry.block();
            Block.Builder b2 = entry.block();
            Block.Builder b1 = entry.block();

            Op.Result p = entry.op(constant(JavaType.BOOLEAN, true));
            entry.op(branch(b6.successor()));

            b6.op(conditionalBranch(p, b5.successor(), b4.successor()));

            b5.op(branch(b1.successor()));

            b4.op(conditionalBranch(p, b2.successor(), b3.successor()));

            b1.op(branch(b2.successor()));

            b2.op(conditionalBranch(p, b1.successor(), b3.successor()));

            b3.op(branch(b2.successor()));
        });
        f.writeTo(System.out);
        Map<Block, Block> idoms = f.body().immediateDominators();
        System.out.println(idoms);

        Block entry = f.body().entryBlock();
        Block b6 = entry.successors().get(0).targetBlock();

        for (Block b : f.body().blocks()) {
            if (b == entry || b == b6) {
                Assert.assertEquals(idoms.get(b), entry);
            } else {
                Assert.assertEquals(idoms.get(b), b6);
            }
        }
    }

    @Test
    public void testCytronExample() {
        CoreOps.FuncOp f = func("f", MethodTypeDesc.VOID).body(entry -> {
            Block.Builder exit = entry.block();
            Block.Builder b12 = entry.block();
            Block.Builder b11 = entry.block();
            Block.Builder b10 = entry.block();
            Block.Builder b9 = entry.block();
            Block.Builder b8 = entry.block();
            Block.Builder b7 = entry.block();
            Block.Builder b6 = entry.block();
            Block.Builder b5 = entry.block();
            Block.Builder b4 = entry.block();
            Block.Builder b3 = entry.block();
            Block.Builder b2 = entry.block();
            Block.Builder b1 = entry.block();

            Op.Result p = entry.op(constant(JavaType.BOOLEAN, true));

            entry.op(conditionalBranch(p, exit.successor(), b1.successor()));

            b1.op(branch(b2.successor()));

            b2.op(conditionalBranch(p, b3.successor(), b7.successor()));

            b3.op(conditionalBranch(p, b4.successor(), b5.successor()));

            b4.op(branch(b6.successor()));

            b5.op(branch(b6.successor()));

            b6.op(branch(b8.successor()));

            b7.op(branch(b8.successor()));

            b8.op(branch(b9.successor()));

            b9.op(conditionalBranch(p, b10.successor(), b11.successor()));

            b10.op(branch(b11.successor()));

            b11.op(conditionalBranch(p, b12.successor(), b9.successor()));

            b12.op(conditionalBranch(p, exit.successor(), b2.successor()));

            exit.op(_return());
        });

        f.writeTo(System.out);

        Map<Block, Block> idoms = f.body().immediateDominators();
        Node<String> domTree = buildDomTree(f.body().entryBlock(), idoms).transform(b -> Integer.toString(b.index()));
        Node<String> domTreeExpected =
                node("0",
                        node("1",
                                node("2",
                                        node("7"),
                                        node("8",
                                                node("9",
                                                        node("10"),
                                                        node("11",
                                                                node("12")))),
                                        node("3",
                                                node("4"), node("5"), node("6")))),
                        node("13"));
        Assert.assertEquals(domTree, domTreeExpected);


        Map<String, Set<String>> df = f.body().dominanceFrontier().entrySet().stream()
                .map(e -> Map.entry(Integer.toString(e.getKey().index()),
                        e.getValue().stream().map(b -> Integer.toString(b.index())).collect(Collectors.toSet())))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        Map<String, Set<String>> dfExpected = Map.ofEntries(
                Map.entry("1", Set.of("13")),
                Map.entry("2", Set.of("13", "2")),
                Map.entry("3", Set.of("8")),
                Map.entry("4", Set.of("6")),
                Map.entry("5", Set.of("6")),
                Map.entry("6", Set.of("8")),
                Map.entry("7", Set.of("8")),
                Map.entry("8", Set.of("13", "2")),
                Map.entry("9", Set.of("13", "2", "9")),
                Map.entry("10", Set.of("11")),
                Map.entry("11", Set.of("13", "2", "9")),
                Map.entry("12", Set.of("13", "2"))
        );
        Assert.assertEquals(df, dfExpected);
    }

    static Node<Block> buildDomTree(Block entryBlock, Map<Block, Block> idoms) {
        Map<Block, Node<Block>> m = new HashMap<>();
        for (Map.Entry<Block, Block> e : idoms.entrySet()) {
            Block id = e.getValue();
            Block b = e.getKey();
            if (b == entryBlock) {
                continue;
            }

            Node<Block> parent = m.computeIfAbsent(id, _k -> new Node<>(_k, new HashSet<>()));
            Node<Block> child = m.computeIfAbsent(b, _k -> new Node<>(_k, new HashSet<>()));
            parent.children.add(child);
        }
        return m.get(entryBlock);
    }

    @SafeVarargs
    static <T> Node<T> node(T t, Node<T>... children) {
        return new Node<>(t, Set.of(children));
    }

    static <T> Node<T> node(T t, Set<Node<T>> children) {
        return new Node<>(t, children);
    }

    record Node<T>(T t, Set<Node<T>> children) {
        <U> Node<U> transform(Function<T, U> f) {
            Set<Node<U>> mchildren = new HashSet<>();
            for (Node<T> nc : children) {
                mchildren.add(nc.transform(f));
            }
            return node(f.apply(t), mchildren);
        }
    }
}
