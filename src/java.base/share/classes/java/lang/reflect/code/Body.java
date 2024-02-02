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

package java.lang.reflect.code;

import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;

import java.util.*;
import java.util.function.BiFunction;

/**
 * A body containing a sequence of (basic) blocks.
 * <p>
 * The sequence of blocks form a graph. The last operation in a block, a terminating operation,
 * may refer to other blocks in the sequence as successors, thus forming the graph. Otherwise, the last
 * operation defines how the body passes control flow back to the parent operation, and in doing so may optionally
 * yield a value.
 * <p>
 * The first block in the sequence is the entry block, and no other blocks refer to it as a successor.
 * <p>
 * A body has a descriptor whose return type is the body's return type and whose parameter types are the entry block's
 * parameters types, in order.
 * The descriptor describes the sequence of input parameters types for arguments that are passed to the
 * body when control flow is passed it, and describes the return type of values that are returned when body passes
 * control back to the operation.
 */
public final class Body implements CodeElement<Body, Block> {
    // Parent operation
    // Non-null when body is built, and therefore bound to operation
    Op parentOp;

    // The ancestor body, when null the body is isolated and cannot refer to values defined outside
    // When non-null and body is built, ancestorBody == parentOp.result.block.parentBody
    final Body ancestorBody;

    final TypeDesc yieldType;

    // Sorted in reverse postorder
    final List<Block> blocks;

    // Map of a block to its immediate dominator
    // Computed lazily, null if not computed
    Map<Block, Block> idoms;

    /**
     * Constructs a body, whose ancestor is the given ancestor body.
     *
     */
    Body(Body ancestorBody, TypeDesc yieldType) {
        this.ancestorBody = ancestorBody;
        this.yieldType = yieldType;
        this.blocks = new ArrayList<>();
    }

    /**
     * {@return the yield type of this body}
     */
    public TypeDesc yieldType() {
        return yieldType;
    }

    /**
     * Returns the descriptor of this body.
     * <p>The descriptor is composed of the body's entry block parameter types and
     * the body's yield type.
     *
     * @return the descriptor of this body.
     */
    public MethodTypeDesc descriptor() {
        Block entryBlock = entryBlock();
        return MethodTypeDesc.methodType(yieldType, entryBlock.parameterTypes());
    }

    /**
     * Returns this body's parent operation.
     *
     * @return the body's parent operation.
     */
    public Op parentOp() {
        return parentOp;
    }

    /**
     * Finds the block in this body that is the ancestor of the given block.
     *
     * @param b the given block.
     * @return the block in this body that is the ancestor of the given block,
     * otherwise {@code null}
     */
    public Block findAncestorBlockInBody(Block b) {
        Objects.requireNonNull(b);

        while (b != null && b.parentBody() != this) {
            b = b.parentBody().parentOp().parentBlock();
        }

        return b;
    }

    /**
     * Returns body's blocks in reverse-postorder as an unmodifiable list.
     *
     * @return the body's blocks in reverse-postorder.
     */
    public List<Block> blocks() {
        return Collections.unmodifiableList(blocks);
    }

    @Override
    public List<Block> children() {
        return blocks();
    }

    /**
     * Returns this body's entry block.
     * <p>
     * The entry block is the first block in the sequence. No other blocks refer to it as a successor.
     *
     * @return the body's entry block
     */
    public Block entryBlock() {
        return blocks.getFirst();
    }

    /**
     * Returns a map of block to its immediate dominator.
     *
     * @return a map of block to its immediate dominator
     */
    public Map<Block, Block> immediateDominators() {
        /*
         * Compute dominators of blocks in a body.
         * <p>
         * https://www.cs.rice.edu/~keith/EMBED/dom.pdf
         * A Simple, Fast Dominance Algorithm
         * Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
         */

        if (idoms != null) {
            return idoms;
        }

        Map<Block, Block> doms = idoms = new HashMap<>();
        doms.put(entryBlock(), entryBlock());

        // Blocks are sorted in reverse postorder
        boolean changed;
        do {
            changed = false;
            // Iterate through blocks in reverse postorder, except for entry block
            for (int i = 1; i < blocks.size(); i++) {
                Block b = blocks.get(i);

                // Find first processed predecessor of b
                Block newIdom = null;
                for (Block p : b.predecessors()) {
                    if (doms.containsKey(p)) {
                        newIdom = p;
                        break;
                    }
                }
                assert b.predecessors().isEmpty() || newIdom != null : b;

                // For all other predecessors, p, of b
                for (Block p : b.predecessors()) {
                    if (p == newIdom) {
                        continue;
                    }

                    if (doms.containsKey(p)) {
                        // If already calculated
                        newIdom = intersect(doms, p, newIdom);
                    }
                }

                if (doms.get(b) != newIdom) {
                    doms.put(b, newIdom);
                    changed = true;
                }
            }
        } while (changed);

        return doms;
    }

    static Block intersect(Map<Block, Block> doms, Block b1, Block b2) {
        while (b1 != b2) {
            while (b1.index > b2.index) {
                b1 = doms.get(b1);
            }

            while (b2.index > b1.index) {
                b2 = doms.get(b2);
            }
        }

        return b1;
    }

    /**
     * Returns the dominance frontier of each block in the body.
     * <p>
     * The dominance frontier of block, {@code B} say, is the set of all blocks, {@code C} say,
     * such that {@code B} dominates a predecessor of {@code C} but does not strictly dominate
     * {@code C}.
     *
     * @return the dominance frontier of each block in the body
     */
    public Map<Block, Set<Block>> dominanceFrontier() {
        // @@@ cache result?
        Map<Block, Block> idoms = immediateDominators();
        Map<Block, Set<Block>> df = new HashMap<>();

        for (Block b : blocks) {
            Set<Block> preds = b.predecessors();

            if (preds.size() > 1) {
                for (Block p : preds) {
                    Block runner = p;
                    while (runner != idoms.get(b)) {
                        df.computeIfAbsent(runner, k -> new LinkedHashSet<>()).add(b);
                        runner = idoms.get(runner);
                    }
                }
            }
        }

        return df;
    }

    /**
     * Returns {@code true} if this body is dominated by the given body {@code dom}.
     * <p>
     * A body, {@code b} say, is dominated by {@code dom} if {@code b} is the same as {@code dom} or a descendant of
     * {@code dom}. Specifically, if {@code b} and {@code dom} are not equal then {@code b} becomes the nearest ancestor
     * body, the result of {@code b.parentOp().parentBlock().parentBody()}, and so on until either:
     * {@code b == dom}, therefore {@code b} is dominated by {@code dom} and this method returns {@code true};
     * or {@code b.parentOp().parentBlock() == null}, therefore {@code b} is <b>not</b> dominated
     * by {@code dom} and this method returns {@code false}.
     *
     * @param dom the dominating body
     * @return {@code true} if this body is dominated by the given body {@code dom}.
     */
    public boolean isDominatedBy(Body dom) {
        return isDominatedBy(this, dom);
    }

    static boolean isDominatedBy(Body r, Body dom) {
        while (r != dom) {
            Block eb = r.parentOp().parentBlock();
            if (eb == null) {
                return false;
            }

            r = eb.parentBody();
        }

        return true;
    }

    /**
     * A builder of a body.
     * <p>
     * When the body builder is built any associated block builders are also considered built.
     */
    public final class Builder {
        /**
         * Creates a body build with a new context, and a copying transformer.
         *
         * @param ancestorBody the nearest ancestor body builder
         * @param desc         the body descriptor
         * @return the body builder
         * @throws IllegalStateException if the ancestor body builder is built
         * @see #of(Builder, MethodTypeDesc, CopyContext, OpTransformer)
         */
        public static Builder of(Builder ancestorBody, MethodTypeDesc desc) {
            // @@@ Creation of CopyContext
            return of(ancestorBody, desc, CopyContext.create(), OpTransformer.COPYING_TRANSFORMER);
        }

        /**
         * Creates a body build with a copying transformer.
         *
         * @param ancestorBody the nearest ancestor body builder
         * @param desc         the body descriptor
         * @param cc           the context
         * @return the body builder
         * @throws IllegalStateException if the ancestor body builder is built
         * @see #of(Builder, MethodTypeDesc, CopyContext, OpTransformer)
         */
        public static Builder of(Builder ancestorBody, MethodTypeDesc desc, CopyContext cc) {
            return of(ancestorBody, desc, cc, OpTransformer.COPYING_TRANSFORMER);
        }

        /**
         * Creates a body builder.
         * <p>
         * Structurally, the created body builder must be built before its ancestor body builder (if non-null) is built,
         * otherwise an {@code IllegalStateException} will occur.
         * <p>
         * The body descriptor defines the body's yield type and the initial sequence of entry block parameters.
         * The body's yield is the descriptors return type.
         * An entry block builder is created with appended block parameters corresponding, in order, to
         * the body descriptor parameter types.
         * <p>
         * If the ancestor body is null then the created body builder is isolated and descendant operations may only
         * refer to values declared within the created body builder. Otherwise, operations
         * may refer to values declared in the ancestor body builders (outside the created body builder).
         *
         * @param ancestorBody the nearest ancestor body builder, may be null if isolated
         * @param desc         the body descriptor
         * @param cc           the context
         * @param ot           the transformer
         * @return the body builder
         * @throws IllegalStateException if the ancestor body builder is built
         * @see #of(Builder, MethodTypeDesc, CopyContext, OpTransformer)
         */
        public static Builder of(Builder ancestorBody, MethodTypeDesc desc,
                                 CopyContext cc, OpTransformer ot) {
            Body body = new Body(ancestorBody != null ? ancestorBody.target() : null, desc.returnType());
            return body.new Builder(ancestorBody, desc, cc, ot);
        }

        // The ancestor body, may be null
        final Builder ancestorBody;

        // The entry block of this body, whose parameters are given by the body's descriptor
        final Block.Builder entryBlock;

        // When non-null contains one or more great-grandchildren
        List<Builder> greatgrandchildren;

        // True when built
        boolean closed;

        Builder(Builder ancestorBody, MethodTypeDesc descriptor,
                CopyContext cc, OpTransformer ot) {
            // Structural check
            // The ancestor body should not be built before this body is created
            if (ancestorBody != null) {
                ancestorBody.check();
                ancestorBody.addGreatgrandchild(this);
            }

            this.ancestorBody = ancestorBody;
            // Create entry block from descriptor
            Block eb = Body.this.createBlock(descriptor.parameters());
            this.entryBlock = eb.new Builder(this, cc, ot);
        }

        void addGreatgrandchild(Builder greatgrandchild) {
            var l = greatgrandchildren == null
                    ? (greatgrandchildren = new ArrayList<>()) : greatgrandchildren;
            l.add(greatgrandchild);
        }

        /**
         * Builds the body and its blocks, associating the body with a parent operation.
         * <p>
         * Structurally, any descendant body builders must be built before this body builder is built,
         * otherwise an {@code IllegalStateException} will occur.
         * <p>
         * Blocks are sorted in reserve postorder.
         * <p>
         * Any unreferenced empty blocks are removed. An unreferenced block is a non-entry block with no predecessors.
         *
         * @param op the parent operation
         * @return the build body
         * @throws IllegalStateException if this body builder is built
         * @throws IllegalStateException if any descendant body builders are not built
         * @throws IllegalStateException if a block has no terminal operation, unless unreferenced and empty
         */
        // @@@ Validation
        // e.g., every operand dominates the operation result (potentially expensive)
        public Body build(Op op) {
            // Structural check
            // This body should not be closed
            check();
            closed = true;

            // Structural check
            // All great-grandchildren bodies should be built
            if (greatgrandchildren != null) {
                for (Builder greatgrandchild : greatgrandchildren) {
                    if (!greatgrandchild.closed) {
                        throw new IllegalStateException("Descendant body builder is not built");
                    }
                }
            }

            Iterator<Block> i = blocks.iterator();
            while (i.hasNext()) {
                Block block = i.next();

                // Structural check
                // All referenced blocks should have a terminating operation as the last operation
                if (block.ops.isEmpty()) {
                    if (block.isEntryBlock() || !block.predecessors.isEmpty()) {
                        throw noTerminatingOperation();
                    }

                    // Remove unreferenced empty block
                    assert !block.isEntryBlock() && block.predecessors.isEmpty();
                    i.remove();
                } else if (!(block.ops.getLast() instanceof Op.Terminating)) {
                    throw noTerminatingOperation();
                }
            }

            sortReversePostorder();

            Body.this.parentOp = op;
            return Body.this;
        }

        static IllegalStateException noTerminatingOperation() {
            return new IllegalStateException("Block has no terminating operation as the last operation");
        }

        /**
         * Returns the body's descriptor.
         * <p>
         * The descriptor is composed of the body's yield type, as the descriptor's return type, and the currently built
         * entry block's parameter types, in order, as the descriptor's parameter types.
         * @return the body's descriptor
         */
        public MethodTypeDesc descriptor() {
            TypeDesc returnType = Body.this.yieldType();
            Block eb = Body.this.entryBlock();
            return MethodTypeDesc.methodType(returnType, eb.parameterTypes());
        }

        /**
         * {@return the body builder's nearest ancestor body builder}
         */
        public Builder ancestorBody() {
            return ancestorBody;
        }

        /**
         * {@return the body's entry block builder}
         */
        public Block.Builder entryBlock() {
            return entryBlock;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            return o instanceof Builder that && Body.this == that.target();
        }

        @Override
        public int hashCode() {
            return Body.this.hashCode();
        }

        void check() {
            if (closed) {
                throw new IllegalStateException("Builder is closed");
            }
        }

        Body target() {
            return Body.this;
        }

        // Build new block in body
        Block.Builder block(List<TypeDesc> params, CopyContext cc, OpTransformer ot) {
            check();
            Block block = Body.this.createBlock(params);

            return block.new Builder(this, cc, ot);
        }
    }

    /**
     * Copies the contents of this body.
     *
     * @param cc the copy context
     * @return the builder of a body containing the copied body
     * @see #transform(CopyContext, OpTransformer)
     */
    public Builder copy(CopyContext cc) {
        return transform(cc, OpTransformer.COPYING_TRANSFORMER);
    }

    /**
     * Transforms this body.
     * <p>
     * A new body builder is created with the same descriptor as this body.
     * Then, this body is {@link Block.Builder#transformBody(Body, java.util.List, CopyContext, OpTransformer) transformed}
     * into the body builder's entry block builder with the given copy context, operation transformer, and values
     * that are the entry block's parameters.
     *
     * @param cc the copy context
     * @param ot the operation transformer
     * @return a body builder containing the transformed body
     */
    public Builder transform(CopyContext cc, OpTransformer ot) {
        Block.Builder ancestorBlockBuilder = ancestorBody != null
                ? cc.getBlock(ancestorBody.entryBlock()) : null;
        Builder ancestorBodyBuilder = ancestorBlockBuilder != null
                ? ancestorBlockBuilder.parentBody() : null;
        Builder body = Builder.of(ancestorBodyBuilder,
                // Create descriptor with just the return type and add parameters afterward
                MethodTypeDesc.methodType(yieldType),
                cc, ot);

        for (Block.Parameter p : entryBlock().parameters()) {
            body.entryBlock.parameter(p.type());
        }

        body.entryBlock.transformBody(this, body.entryBlock.parameters(), cc, ot);
        return body;
    }

    // Sort blocks in reverse post order
    // After sorting the following holds for a block
    //   block.parentBody().blocks().indexOf(block) == block.index()
    private void sortReversePostorder() {
        if (blocks.size() < 2) {
            for (int i = 0; i < blocks.size(); i++) {
                blocks.get(i).index = i;
            }
            return;
        }

        // Reset block indexes
        // Also ensuring blocks with no predecessors occur last
        for (Block b : blocks) {
            b.index = Integer.MAX_VALUE;
        }

        Deque<Block> stack = new ArrayDeque<>();
        stack.push(blocks.get(0));

        // Postorder iteration
        int index = blocks.size();
        while (!stack.isEmpty()) {
            Block n = stack.peek();
            if (n.index == Integer.MIN_VALUE) {
                // If n's successor has been processed then add n
                stack.pop();
                n.index = --index;
            } else if (n.index < Integer.MAX_VALUE) {
                // If n has already been processed then ignore
                stack.pop();
            } else {
                // Mark before processing successors, a successor may refer back to n
                n.index = Integer.MIN_VALUE;
                for (Block.Reference s : n.successors()) {
                    if (s.target.index < Integer.MAX_VALUE) {
                        continue;
                    }

                    stack.push(s.target);
                }
            }
        }

        blocks.sort(Comparator.comparingInt(b -> b.index));
        if (blocks.get(0).index > 0) {
            // There are blocks with no predecessors
            // Reassign indexes to their natural indexes, sort order is preserved
            for (int i = 0; i < blocks.size(); i++) {
                blocks.get(i).index = i;
            }
        }
    }

    // Modifying methods

    // Create block
    private Block createBlock(List<TypeDesc> params) {
        Block b = new Block(this, params);
        blocks.add(b);
        return b;
    }
}
