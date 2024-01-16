/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package java.lang.reflect.code.analysis;

import java.io.StringWriter;
import java.io.Writer;
import java.lang.reflect.code.*;
import java.lang.reflect.code.impl.GlobalValueBlockNaming;
import java.lang.reflect.code.impl.OpWriter;
import java.util.*;

/**
 * Provides liveness information for values declared in the bodies of an operation.
 */
public class Liveness {

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
                useOp = block.findAncestorOpInBlock(useOp);
                // Update if after
                if (useOp != null && isBeforeInBlock(endOp, useOp)) {
                    endOp = useOp;
                }
            }
            return endOp;
        }
    }

    final Op op;
    final Map<Block, BlockInfo> livenessMapping;

    /**
     * Constructs liveness information for values declared in the bodies
     * of an operation.
     *
     * @param op the operation.
     */
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
            livenessMapping.put(b, new BlockInfo(b));
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
            Block B = CFG.findAncestorBlockInBody(use.declaringBlock());
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
        GlobalValueBlockNaming gn = new GlobalValueBlockNaming();

        OpWriter ow = new OpWriter(w, gn);
        ow.writeOp(op);
        ow.write("\n");

        op.traverse(null, CodeElement.blockVisitor((_, b) -> {
            BlockInfo liveness = getLiveness(b);
            ow.write("^");
            ow.write(gn.getBlockName(b));
            ow.write("\n");
            ow.write("  Live-in values: ");
            ow.writeCommaSeparatedList(liveness.inValues, v -> {
                ow.write("%");
                ow.write(gn.getValueName(v));
            });
            ow.write("\n");
            ow.write("  Live-out values: ");
            ow.writeCommaSeparatedList(liveness.outValues, v -> {
                ow.write("%");
                ow.write(gn.getValueName(v));
            });
            ow.write("\n");

            return null;
        }));
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
     * the {@link BlockInfo#getEndOperation end operation}
     * occurs before the operation, this method returns true.
     * Otherwise, this method returns false.
     *
     * @param value the value
     * @param op    the operation
     * @return true if a value is last used by an operation
     */
    public boolean isLastUse(Value value, Op op) {
        Block block = op.parentBlock();
        BlockInfo liveness = getLiveness(block);

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
    public BlockInfo getLiveness(Block block) {
        BlockInfo lbi = livenessMapping.get(block);
        if (lbi == null) {
            throw new IllegalArgumentException("Block has no liveness information");
        }
        return lbi;
    }

    private static boolean isBeforeInBlock(Op thisOp, Op thatOp) {
        if (thisOp.result() == null || thatOp.result() == null) {
            throw new IllegalArgumentException("This or the given operation is not assigned to a block");
        }

        if (thisOp.parentBlock() != thatOp.parentBlock()) {
            throw new IllegalArgumentException("This and that operation are not assigned to the same blocks");
        }

        List<Op> ops = thisOp.parentBlock().ops();
        return ops.indexOf(thisOp) < ops.indexOf(thatOp);
    }

}
