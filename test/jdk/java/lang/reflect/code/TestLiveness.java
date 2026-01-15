/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
 * @run junit TestLiveness
 */

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.OpParser;
import jdk.incubator.code.extern.OpWriter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.StringWriter;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TestLiveness {

    /**
     * Provides liveness information for values declared in the bodies of an operation.
     */
    public static class Liveness {

        /**
         * Liveness information associated with a block.
         * Each block has two sets of values, live-in values and live-out values.
         */
        public static final class BlockInfo {
            final Block block;
            final Deque<Value> inValues;
            final Deque<Value> outValues;

            BlockInfo(Block block) {
                this.block = block;
                this.inValues = new ArrayDeque<>();
                this.outValues = new ArrayDeque<>();
            }

            /**
             * {@return the block associated with the liveness information}
             */
            public Block getBlock() {
                return block;
            }

            /**
             * Returns true if a value is live-in for the associated block.
             * <p>
             * A value is live-in for a block if it is not declared in the block
             * and is used in the block or (transitively) by some successor.
             *
             * @param value the value
             * @return true if the value is live-in
             */
            public boolean isLiveIn(Value value) {
                return inValues.contains(value);
            }

            /**
             * {@return the set of live-in values}
             */
            public Set<Value> liveIn() {
                return new HashSet<>(inValues);
            }

            /**
             * Returns true if a value is live-out for the associated block.
             * <p>
             * A value is live-out for a block if it is used (transitively) by some successor.
             *
             * @param value the value
             * @return true if the value is live-out
             */
            public boolean isLiveOut(Value value) {
                return outValues.contains(value);
            }

            /**
             * {@return the set of live-out values}
             */
            public Set<Value> liveOut() {
                return new HashSet<>(outValues);
            }

            /**
             * Returns the first operation associated with a value and the associated block.
             * <p>
             * If the value is live-in or a block argument then the blocks first operation
             * is returned. Otherwise, the value is an operation result and its operation
             * is returned.
             *
             * @param value the value
             * @return first operation associated with a value and the associated block.
             */
            public Op getStartOperation(Value value) {
                if (isLiveIn(value) || value instanceof Block.Parameter) {
                    // @@@ Check value is from this block
                    return block.firstOp();
                } else {
                    // @@@ Check value is from block
                    Op.Result or = (Op.Result) value;
                    return or.op();
                }
            }

            /**
             * Returns the end operation associated with a value and the associated block.
             * <p>
             * If the value is live-out then the blocks last (and terminating) operation
             * is returned. Otherwise, the value is dying in this block and the last
             * operation to use this value is returned.
             *
             * @param value the value
             * @return first operation associated with a value and the associated block.
             */
            public Op getEndOperation(Value value, Op startOp) {
                // Value is used by some other operation
                if (isLiveOut(value)) {
                    return block.terminatingOp();
                }

                // Value may be last used in this block, if so find it
                // @@@ Check startOp is of this block
                Op endOp = startOp;
                for (Op.Result useOpr : value.uses()) {
                    Op useOp = useOpr.op();
                    // Find the operation in the current block
                    useOp = findChildAncestor(block, useOp);
                    // Update if after
                    if (useOp != null && isBeforeInBlock(endOp, useOp)) {
                        endOp = useOp;
                    }
                }
                return endOp;
            }
        }

        final Op op;
        final Map<Block, Liveness.BlockInfo> livenessMapping;

        /**
         * Constructs liveness information for values declared in the bodies
         * of an operation.
         *
         * @param op the operation.
         */
        @SuppressWarnings("this-escape")
        public Liveness(Op op) {
            this.op = op;
            this.livenessMapping = new HashMap<>();
            for (Body cfg : op.bodies()) {
                Compute_LiveSets_SSA_ByVar(cfg);
            }
        }

    /*
    The algorithm to compute liveness information is derived from
    Domaine, & Brandner, Florian & Boissinot, Benoit & Darte, Alain & Dinechin, Benoit & Rastello, Fabrice.
    (2011). Computing Liveness Sets for SSA-Form Programs.
    https://inria.hal.science/inria-00558509v2/document
    Specifically Algorithm 6 & 7, adapted to work with block arguments and
    block parameters instead of phi operations.
    This is a simple algorithm that is easy to understand. We may need to review
    its usage within exception regions.
    We also may revisit this later with a more performant implementation
    perhaps based on the well known algorithm that uses fixpoint iteration.
     */

        void Compute_LiveSets_SSA_ByVar(Body CFG) {
            for (Block b : CFG.blocks()) {
                livenessMapping.put(b, new Liveness.BlockInfo(b));
            }
            for (Block b : CFG.blocks()) {
                for (Block.Parameter p : b.parameters()) {
                    Compute_LiveSets_SSA_ByVar(CFG, p);
                }

                for (Op op : b.ops()) {
                    Compute_LiveSets_SSA_ByVar(CFG, op.result());
                }
            }
        }

        void Compute_LiveSets_SSA_ByVar(Body CFG, Value v) {
            for (Op.Result use : v.uses()) {
                Block B = findChildAncestor(CFG, use.declaringBlock());
                Up_and_Mark_Stack(B, v);
            }
        }

        void Up_and_Mark_Stack(Block B, Value v) {
            if (v.declaringBlock() == B) {
                return;
            }
            var lbi = livenessMapping.get(B);
            if (lbi.inValues.peek() == v) {
                return;
            }
            lbi.inValues.push(v);
            for (Block P : B.predecessors()) {
                lbi = livenessMapping.get(P);
                if (lbi.outValues.peek() != v) {
                    lbi.outValues.push(v);
                }
                Up_and_Mark_Stack(P, v);
            }
        }

        /**
         * {@return the liveness information as a string}
         */
        public String toString() {
            StringWriter w = new StringWriter();
            writeTo(w);
            return w.toString();
        }

        /**
         * Writes the liveness information to the given writer.
         *
         * @param w the writer to write to.
         */
        public void writeTo(Writer w) {
            OpWriter ow = new OpWriter(w);
            ow.writeOp(op);
            try {
                w.write("\n");
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            Function<CodeItem, String> namer = ow.namer();
            op.elements().forEach(e -> {
                if (!(e instanceof Block b)) {
                    return;
                }
                Liveness.BlockInfo liveness = getLiveness(b);
                try {
                    w.write("^" + namer.apply(b));
                    w.write("\n");
                    w.write("  Live-in values: ");
                    w.write(liveness.inValues.stream()
                            .map(v -> "%" + namer.apply(v))
                            .collect(Collectors.joining(",")));
                    w.write("\n");
                    w.write("  Live-out values: ");
                    w.write(liveness.outValues.stream()
                            .map(v -> "%" + namer.apply(v))
                            .collect(Collectors.joining(",")));
                    w.write("\n");
                } catch (IOException ex) {
                    throw new UncheckedIOException(ex);
                }
            });
        }

        /**
         * Returns true if a value is last used by an operation.
         * <p>
         * The liveness information for the operation's parent block
         * is obtained. If the value is live-out then the value escapes
         * the block and is therefore not the last use, and this method
         * returns false.
         * If the operation is the last to use the value, this method
         * returns true. If the operation does not use the value and
         * the {@link Liveness.BlockInfo#getEndOperation end operation}
         * occurs before the operation, this method returns true.
         * Otherwise, this method returns false.
         *
         * @param value the value
         * @param op    the operation
         * @return true if a value is last used by an operation
         */
        public boolean isLastUse(Value value, Op op) {
            Block block = op.ancestorBlock();
            Liveness.BlockInfo liveness = getLiveness(block);

            // Value is used by some successor
            if (liveness.isLiveOut(value))
                return false;

            Op endOp = liveness.getEndOperation(value, op);
            // Last use or operation is after last use
            return endOp == op || isBeforeInBlock(endOp, op);
        }

        /**
         * {@return the liveness information associated with a block}
         *
         * @param block the block
         * @throws IllegalArgumentException if the block has no liveness information
         */
        public Liveness.BlockInfo getLiveness(Block block) {
            Liveness.BlockInfo lbi = livenessMapping.get(block);
            if (lbi == null) {
                throw new IllegalArgumentException("Block has no liveness information");
            }
            return lbi;
        }

        private static boolean isBeforeInBlock(Op thisOp, Op thatOp) {
            if (thisOp.result() == null || thatOp.result() == null) {
                throw new IllegalArgumentException("This or the given operation is not assigned to a block");
            }

            if (thisOp.ancestorBlock() != thatOp.ancestorBlock()) {
                throw new IllegalArgumentException("This and that operation are not assigned to the same blocks");
            }

            List<Op> ops = thisOp.ancestorBlock().ops();
            return ops.indexOf(thisOp) < ops.indexOf(thatOp);
        }

        /**
         * Finds the child of the parent element that is an ancestor of the given descendant element,
         * otherwise returns the descendant element if a child of this element, otherwise
         * returns {@code null} if there is no such child.
         *
         * @param parent the parent element
         * @param descendant the descendant element
         * @return the child that is an ancestor of the given descendant element, otherwise the descendant
         * element if a child of this element, otherwise {@code null}.
         * @throws IllegalStateException if an operation with unbuilt parent block is encountered.
         */
        private static <C extends CodeElement<C, ?>> C findChildAncestor(CodeElement<?, C> parent, CodeElement<?, ?> descendant) {
            Objects.requireNonNull(descendant);

            CodeElement<?, ?> e = descendant;
            while (e != null && e.parent() != parent) {
                e = e.parent();
            }

            @SuppressWarnings("unchecked")
            C child = (C) e;
            return child;
        }
    }



    static final String F = """
            func @"f" (%0 : java.type:"int", %1 : java.type:"int")java.type:"int" -> {
                %2 : java.type:"int" = add %0 %1;
                return %2;
            };
            """;

    @Test
    public void testF() {
        Op op = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, F).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of()));
        Assertions.assertEquals(expected, actual);
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
        Op op = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, IF_ELSE).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of(0, 1)),
                1, List.of(Set.of(0, 1), Set.of()),
                2, List.of(Set.of(0, 1), Set.of()),
                3, List.of(Set.of(), Set.of())
        );
        Assertions.assertEquals(expected, actual);
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
        Op op = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, LOOP).getFirst();

        var actual = liveness(op);
        var expected = Map.of(
                0, List.of(Set.of(), Set.of(0)),
                1, List.of(Set.of(0), Set.of(0, 3, 4)),
                2, List.of(Set.of(0, 3, 4), Set.of(0, 4, 6)),
                3, List.of(Set.of(3), Set.of()),
                4, List.of(Set.of(0, 4, 6), Set.of(0))
        );
        Assertions.assertEquals(expected, actual);
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
        Op op = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, IF_ELSE_NESTED).getFirst();

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
        Assertions.assertEquals(expected, actual);
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
        Op op = OpParser.fromString(JavaOp.JAVA_DIALECT_FACTORY, LOOP_NESTED).getFirst();

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
        Assertions.assertEquals(expected, actual);
    }

    static Map<Integer, List<Set<Integer>>> liveness(Op op) {
        Liveness l = new Liveness(op);
        System.out.println(l);

        Map<Value, Integer> valueMap = valueNameMapping(op);
        Map<Block, Integer> blockMap = blockNameMapping(op);

        Map<Integer, List<Set<Integer>>> m = new HashMap<>();
        op.elements().forEach(e -> {
            if (e instanceof Block b) {
                if (b.ancestorOp() == op) {
                    Liveness.BlockInfo lbi = l.getLiveness(b);
                    m.put(blockMap.get(b),
                            List.of(
                                    lbi.liveIn().stream().map(valueMap::get).collect(Collectors.toSet()),
                                    lbi.liveOut().stream().map(valueMap::get).collect(Collectors.toSet())
                            ));
                }
            }
        });
        return m;
    }

    static Map<Block, Integer> blockNameMapping(Op top) {
        AtomicInteger i = new AtomicInteger();
        Map<Block, Integer> m = new HashMap<>();
        top.elements().forEach(e -> {
            if (e instanceof Block b) {
                if (b.ancestorOp() != top) {
                    return;
                }

                m.computeIfAbsent(b, _ -> i.getAndIncrement());
                for (Block.Reference s : b.successors()) {
                    m.computeIfAbsent(s.targetBlock(), _ -> i.getAndIncrement());
                }
            }
        });
        return m;
    }

    static Map<Value, Integer> valueNameMapping(Op top) {
        AtomicInteger i = new AtomicInteger();
        Map<Value, Integer> m = new HashMap<>();
        top.elements().forEach(e -> {
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
        });
        return m;
    }
}