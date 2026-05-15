/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code;

import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import java.util.*;

/**
 * A body containing a sequence of (basic) blocks.
 * <p>
 * The sequence of blocks form a graph topologically sorted in reverse postorder.
 * The first block in the sequence is the entry block, and no other blocks refer to it as a successor.
 * The last operation in a block, a terminating operation, may refer to other blocks in the sequence as successors,
 * thus forming the graph. Otherwise, the last operation defines how the body passes control back to the parent
 * operation, and in doing so may optionally yield a value.
 * <p>
 * A body has a signature, a function type, whose return type is the body's yield type and whose parameter types are the
 * entry block's parameters types, in order.
 * The signature describes the sequence of input parameters types for arguments that are passed to the
 * body when control is passed to it, and describes the return type of values that are yielded when the body passes
 * control back to its parent operation.
 * <p>
 * A body is either open or isolated. An open body may {@link #capturedValues() capture} values, depending on how the
 * body's descendant operations use values. An {@link #isIsolated() isolated} body is guaranteed to never capture
 * values.
 * <p>
 * A body is built using a {@link Body.Builder}, which specifies the
 * <a href="Body.Builder.html#body-building-process">building process</a>. An open body is built by a
 * <a href="Body.Builder.html#connected-builder">connected</a> body builder. An isolated body is built by an
 * <a href="Body.Builder.html#isolated-builder">isolated</a> body builder.
 */
public final class Body implements CodeElement<Body, Block> {
    // @Stable?
    // Parent operation
    // Non-null when body is built, and therefore child of an operation
    Op parentOp;

    // The connected ancestor body
    // When non-null the body is open and when built/observable, connectedAncestorBody == this.ancestorBody()
    // When null the body is isolated, and cannot refer to values defined outside
    final Body connectedAncestorBody;

    final CodeType yieldType;

    // Sorted in reverse postorder
    final List<Block> blocks;

    // Lazily computed map of a block to its immediate dominator
    // Computed after body is built
    // @@@ when dominance checks are implemented, may be computed and used in build method
    LazyConstant<Map<Block, Block>> idoms = LazyConstant.of(this::computeImmediateDominators);

    /**
     * Constructs a body, whose connected ancestor body is the given ancestor body.
     */
    Body(Body connectedAncestorBody, CodeType yieldType) {
        this.connectedAncestorBody = connectedAncestorBody;
        this.yieldType = yieldType;
        this.blocks = new ArrayList<>();
    }

    @Override
    public String toString() {
        return "body@" + Integer.toHexString(hashCode());
    }

    /**
     * {@return the body's parent operation.}
     */
    @Override
    public Op parent() {
        return parentOp;
    }

    @Override
    public List<Block> children() {
        return blocks();
    }

    /**
     * Returns body's blocks in reverse-postorder as an unmodifiable list.
     *
     * @return the body's blocks in reverse-postorder.
     */
    public List<Block> blocks() {
        return Collections.unmodifiableList(blocks);
    }

    /**
     * {@return the yield type of this body}
     */
    public CodeType yieldType() {
        return yieldType;
    }

    /**
     * Returns the body's signature, represented as a function type.
     * <p>
     * The signature's return type is the body's yield type and its parameter types are the
     * body's entry block parameter types, in order.
     *
     * @return the body's signature.
     */
    public FunctionType bodySignature() {
        Block entryBlock = entryBlock();
        return CoreType.functionType(yieldType, entryBlock.parameterTypes());
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
     * @return a map of block to its immediate dominator, as an unmodifiable map
     */
    public Map<Block, Block> immediateDominators() {
        return idoms.get();
    }

    // Called by LazyConstant field
    private Map<Block, Block> computeImmediateDominators() {
        Map<Block, Block> idoms = new HashMap<>();

        // Repeatedly compute for each graph
        int start = 0;
        while (start < blocks.size()) {
            start = computeImmediateDominators(idoms, start);
        }

        return Collections.unmodifiableMap(idoms);
    }

    private int computeImmediateDominators(Map<Block, Block> doms, int start) {
        /*
         * Compute dominators of blocks in a body.
         * <p>
         * https://www.cs.rice.edu/~keith/EMBED/dom.pdf
         * A Simple, Fast Dominance Algorithm
         * Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
         */

        Block root = blocks.get(start);
        int component = root.component;
        doms.put(root, root);

        int end = blocks.size();
        boolean changed;
        do {
            changed = false;
            // Iterate through blocks in reverse postorder, except for root block
            for (int i = start + 1; i < end; i++) {
                Block b = blocks.get(i);
                if (b.component != component) {
                    end = i;
                    break;
                }

                // Find first processed predecessor of b
                Block newIdom = null;
                for (Block p : b.predecessors()) {
                    if (p.component != b.component) {
                        continue;
                    }

                    if (doms.containsKey(p)) {
                        newIdom = p;
                        break;
                    }
                }
                assert newIdom != null : b;

                // For all other predecessors, p, of b
                for (Block p : b.predecessors()) {
                    if (p.component != b.component || p == newIdom) {
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

        return end;
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
     * @return the dominance frontier of each block in the body, as a modifiable map
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
                        df.computeIfAbsent(runner, _ -> new LinkedHashSet<>()).add(b);
                        runner = idoms.get(runner);
                    }
                }
            }
        }

        return df;
    }

    /**
     * A synthetic exit block used when computing immediate post dominators.
     * It represents the post dominator of all blocks when two or more blocks
     * in the body have no successors.
     * <p>
     * Computing the immediate post dominators requires a single exit point,
     * one block with no successors. When a body has two or more blocks
     * with no successors then this block acts as the single exit point.
     */
    public static final Block IPDOM_EXIT;
    static {
        IPDOM_EXIT = new Block(null);
        IPDOM_EXIT.index = Integer.MAX_VALUE;
    }

    /**
     * Returns a map of block to its immediate post dominator.
     * <p>
     * If there are two or more blocks with no successors then
     * a single exit point is synthesized using the {@link #IPDOM_EXIT}
     * block, which represents the immediate post dominator of those blocks.
     *
     * @return a map of block to its immediate post dominator, as an unmodifiable map
     */
    public Map<Block, Block> immediatePostDominators() {
        Map<Block, Block> pdoms = new HashMap<>();

        // If there are multiple exit blocks (those with zero successors)
        // then use the block IPDOM_EXIT that is the synthetic successor of
        // the exit blocks
        boolean nSuccessors = blocks.stream().filter(b -> b.successors().isEmpty()).count() > 1;

        if (nSuccessors) {
            pdoms.put(IPDOM_EXIT, IPDOM_EXIT);
        } else {
            Block exit = blocks.getLast();
            assert blocks.stream().filter(b -> b.successors().isEmpty()).findFirst().orElseThrow() == exit;
            pdoms.put(exit, exit);
        }

        // Blocks are sorted in reverse postorder
        boolean changed;
        do {
            changed = false;
            // Iterate in reverse through blocks in reverse postorder, except for exit block
            for (int i = blocks.size() - (nSuccessors ? 1 : 2); i >= 0; i--) {
                Block b = blocks.get(i);

                // Find first processed successor of b
                Block newIpdom = null;
                Collection<Block> targets = b.successorTargets();
                for (Block s : nSuccessors && targets.isEmpty() ? List.of(IPDOM_EXIT) : targets) {
                    if (pdoms.containsKey(s)) {
                        newIpdom = s;
                        break;
                    }
                }

                if (newIpdom == null) {
                    // newIpdom can be null if all successors reference
                    // prior blocks (back branch) yet to be encountered
                    // in the dominator treee
                    continue;
                }

                // For all other successors, s, of b
                for (Block s : b.successorTargets()) {
                    if (s == newIpdom) {
                        continue;
                    }

                    if (pdoms.containsKey(s)) {
                        // If already calculated
                        newIpdom = postIntersect(pdoms, s, newIpdom, blocks.size());
                    }
                }

                if (pdoms.get(b) != newIpdom) {
                    pdoms.put(b, newIpdom);
                    changed = true;
                }
            }
        } while (changed);

        return Collections.unmodifiableMap(pdoms);
    }

    static Block postIntersect(Map<Block, Block> doms, Block b1, Block b2, int exitIndex) {
        while (b1 != b2) {
            while (b1.index() < b2.index()) {
                b1 = doms.get(b1);
            }

            while (b2.index() < b1.index()) {
                b2 = doms.get(b2);
            }
        }

        return b1;
    }

    /**
     * Returns the post dominance frontier of each block in the body.
     * <p>
     * The post dominance frontier of block, {@code B} say, is the set of all blocks, {@code C} say,
     * such that {@code B} post dominates a successor of {@code C} but does not strictly post dominate
     * {@code C}.
     *
     * @return the post dominance frontier of each block in the body, as a modifiable map
     */
    public Map<Block, Set<Block>> postDominanceFrontier() {
        // @@@ cache result?
        Map<Block, Block> idoms = immediatePostDominators();
        Map<Block, Set<Block>> df = new HashMap<>();

        for (Block b : blocks) {
            Set<Block> succs = b.successorTargets();

            if (succs.size() > 1) {
                for (Block s : succs) {
                    Block runner = s;
                    while (runner != idoms.get(b)) {
                        df.computeIfAbsent(runner, _ -> new LinkedHashSet<>()).add(b);
                        runner = idoms.get(runner);
                    }
                }
            }
        }

        return df;
    }

    /**
     * {@return true if this body is isolated}
     * <p>
     * An isolated body, built by an <a href="Body.Builder.html#isolated-builder">isolated</a> body builder, is
     * guaranteed to never {@link #capturedValues() capture} values. Conversely, an open body, built by a
     * <a href="Body.Builder.html#connected-builder">connected</a> body builder, may or may not capture values,
     * depending on how the body's descendant operations use values.
     *
     * @see #capturedValues()
     */
    public boolean isIsolated() {
        return connectedAncestorBody == null;
    }

    /**
     * Computes values captured by this body. A captured value is a value that is used
     * but not declared by any descendant block or operation of this body.
     * <p>
     * The order of the captured values is first use encountered in depth
     * first search of this body's descendant operations.
     *
     * @return the list of captured values, modifiable
     */
    public List<Value> capturedValues() {
        Set<Value> cvs = new LinkedHashSet<>();

        capturedValues(cvs, new ArrayDeque<>(), this);
        return new ArrayList<>(cvs);
    }

    static void capturedValues(Set<Value> capturedValues, Deque<Body> bodyStack, Body body) {
        bodyStack.push(body);

        for (Block b : body.blocks()) {
            for (Op op : b.ops()) {
                for (Body childBody : op.bodies()) {
                    capturedValues(capturedValues, bodyStack, childBody);
                }

                for (Value a : op.operands()) {
                    if (!bodyStack.contains(a.declaringBlock().ancestorBody())) {
                        capturedValues.add(a);
                    }
                }

                for (Block.Reference s : op.successors()) {
                    for (Value a : s.arguments()) {
                        if (!bodyStack.contains(a.declaringBlock().ancestorBody())) {
                            capturedValues.add(a);
                        }
                    }
                }
            }
        }

        bodyStack.pop();
    }

    /**
     * A builder for a body.
     * <p>
     * <a id="body-building-process"></a>
     * The process of building a body starts with the {@link Builder#of(Builder, FunctionType, CodeContext, CodeTransformer) creation}
     * of a body builder, which {@link Builder#entryBlock exposes} a {@link Block.Builder block builder} for the body's
     * entry block.
     * <p>
     * Building then progresses with the building of the body's structure, where:
     * <ul>
     * <li>
     * the entry block builder is used to {@link Block.Builder#block(List) create} block builders for sibling blocks,
     * and likewise those block builders can also be used to create block builders for additional sibling blocks and so
     * on;
     * <li>
     * a block builder is used to {@link Block.Builder#op(Op) append} operations to the block,
     * {@link Block.Builder#parameter(CodeType) append} parameters to the block's parameters, and
     * {@link Block.Builder#reference(List) create} references to the block, which can be used as successors of a
     * terminating operation that is the last operation that is appended to the block or a sibling block; and
     * <li>
     * <a id="body-building-observability"></a>
     * the body and its child blocks are not observable; attempts to observe them through appended operations, their
     * operation results, block parameters, or block references, throw an {@link IllegalStateException}.
     * </ul>
     * <p>
     * Building finishes by invoking {@link #build(Op)}, with a given operation that becomes the body's
     * parent.
     * <p>
     * <a id="body-building-finishing"></a>
     * After building finishes, the body and its child blocks become observable, and the body builder and its block
     * builders all become inoperable, regardless of whether building succeeds or fails with an exception.
     * Further attempts to operate on the builders throw an exception.
     * <p>
     * A body builder may be connected to its {@link #connectedAncestorBody() nearest ancestor} body builder. This
     * connection constrains the order in which the connected builders can finish building, ancestors cannot finish
     * before their descendants, and determines the <a href="Block.Builder.html#reachable-value">reachability</a> of
     * values used by appended operations.
     * <p>
     * Body builders are not thread-safe. Block builders associated with a body builder are also not thread-safe.
     */
    public final class Builder {
        /**
         * Creates a body builder, with an entry block {@link #entryBlock builder} that has a
         * {@link CodeContext#create() new} code context and a {@link CodeTransformer#COPYING_TRANSFORMER copying}
         * code transformer.
         *
         * @param connectedAncestorBody  the nearest ancestor body builder if the created body builder is connected, or
         * {@code null} if the created body builder is isolated
         * @param bodySignature the initial body signature
         * @return the body builder
         * @throws IllegalStateException if the ancestor body builder is finished
         * @see #of(Builder, FunctionType, CodeContext, CodeTransformer)
         */
        public static Builder of(Builder connectedAncestorBody, FunctionType bodySignature) {
            // @@@ Creation of CodeContext
            return of(connectedAncestorBody, bodySignature, CodeContext.create(), CodeTransformer.COPYING_TRANSFORMER);
        }

        /**
         * Creates a body builder, with an entry block {@link #entryBlock builder} that has the given code context
         * and a {@link CodeTransformer#COPYING_TRANSFORMER copying} code transformer.
         *
         * @param connectedAncestorBody  the nearest ancestor body builder if the created body builder is connected, or
         * {@code null} if the created body builder is isolated
         * @param bodySignature the initial body signature
         * @param cc            the code context
         * @return the body builder
         * @throws IllegalStateException if the ancestor body builder is finished
         * @see #of(Builder, FunctionType, CodeContext, CodeTransformer)
         */
        public static Builder of(Builder connectedAncestorBody, FunctionType bodySignature, CodeContext cc) {
            return of(connectedAncestorBody, bodySignature, cc, CodeTransformer.COPYING_TRANSFORMER);
        }

        /**
         * Creates a body builder whose entry block {@link #entryBlock builder} uses the given code context and code
         * transformer.
         * <p>
         * If {@code connectedAncestorBody} is non-{@code null}, the created body builder is
         * <a id="connected-builder"><i>connected</i></a> to {@code connectedAncestorBody} as the
         * {@link #connectedAncestorBody() nearest ancestor} body builder, builds an <i>open</i> body, and the following
         * apply:
         * <ul>
         * <li>
         * the created body builder must finish before the nearest ancestor body builder finishes, which implies the
         * ancestor body builder cannot finish until all body builders connected to it finish; and
         * <li>
         * the body built by the created body builder must have, as its nearest {@link Body#ancestorBody() ancestor body},
         * the body built by the nearest ancestor body builder.
         * </ul>
         * If {@code connectedAncestorBody} is {@code null}, the created body builder is
         * <a id="isolated-builder"><i>isolated</i></a>, it has no nearest ancestor body builder, builds an
         * <i>isolated</i> body, and the following applies:
         * <ul>
         * <li>
         * the scope of <a href="Block.Builder.html#reachable-value">reachable</a> values used by operations is
         * reduced to that up to and including the created body builder.
         * </ul>
         * <p>
         * One or more body builders can be connected to the created body builder, as their nearest ancestor body
         * builder, whether the created body builder be connected or isolated, which implies the created body builder
         * cannot finish until all of its connected body builders finish.
         * <p>
         * The initial body signature's return type defines the body's yield type, and its parameter types are used,
         * in order, to create the initial parameters of the entry block builder.
         *
         * @param connectedAncestorBody  the nearest ancestor body builder if the created body builder is connected, or
         * {@code null} if the created body builder is isolated
         * @param bodySignature the initial body signature
         * @param cc            the code context for the entry block builder
         * @param ct            the code transformer for the entry block builder
         * @return the body builder
         * @throws IllegalStateException if the connected ancestor body builder is finished
         */
        public static Builder of(Builder connectedAncestorBody, FunctionType bodySignature,
                                 CodeContext cc, CodeTransformer ct) {
            Body body = new Body(connectedAncestorBody != null ? connectedAncestorBody.target() : null,
                    bodySignature.returnType());
            return body.new Builder(connectedAncestorBody, bodySignature, cc, ct);
        }

        // The connected nearest ancestor body, may be null
        final Builder connectedAncestorBody;

        // The entry block of this body, whose parameters are given by the body's function type
        final Block.Builder entryBlock;

        // When non-null contains one or more great-grandchildren
        List<Builder> greatgrandchildren;

        // True when finished
        boolean finished;

        Builder(Builder connectedAncestorBody, FunctionType bodySignature,
                CodeContext cc, CodeTransformer ct) {
            // Structural check
            // The connected ancestor body should not be built before this body is built
            if (connectedAncestorBody != null) {
                connectedAncestorBody.check();
                connectedAncestorBody.addGreatgrandchild(this);
            }

            this.connectedAncestorBody = connectedAncestorBody;
            // Create entry block from the body's function type
            Block eb = Body.this.createBlock(bodySignature.parameterTypes());
            this.entryBlock = eb.new Builder(this, cc, ct);
        }

        void addGreatgrandchild(Builder greatgrandchild) {
            var l = greatgrandchildren == null
                    ? (greatgrandchildren = new ArrayList<>()) : greatgrandchildren;
            l.add(greatgrandchild);
        }

        /**
         * Finishes building the body and its child blocks, associating the body with a parent operation.
         * <p>
         * The parent operation must report the built body as one of its child bodies.
         * <p>
         * After building finishes, the body builder and its block builders all become inoperable, regardless of whether
         * building succeeds or fails with an exception. Further attempts to operate on the builders throw an exception.
         * <p>
         * Body builders connected to this body builder must finish building before this body builder finishes.
         * <p>
         * Any unreferenced empty blocks are ignored and do not become children of the body. An unreferenced block is
         * a non-entry block with no predecessors.
         *
         * @apiNote
         * This method is commonly called from the parent operation's constructor, which holds a reference to the built
         * body so it can report it as one of its child bodies.
         *
         * @param op the parent operation
         * @return the built body
         * @throws IllegalStateException if this body builder has finished
         * @throws IllegalStateException if any connected body builder is not finished
         * @throws IllegalStateException if a block has no terminating operation, unless unreferenced and empty
         */
        public Body build(Op op) {
            Objects.requireNonNull(op);

            // Structural check
            // This body builder should not be finished
            check();
            finished = true;

            // Structural check
            // All great-grandchildren bodies should be built
            if (greatgrandchildren != null) {
                for (Builder greatgrandchild : greatgrandchildren) {
                    if (!greatgrandchild.finished) {
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

            // Validate each use of a value declared in the body.
            // The use's declaring block must be dominated by the value's declaring block
            if (blocks.size() > 1) {
                // Only need to check when there is more than one block, since for one block
                // the use's declaring block will be the same as or a descendant of the
                // value's declaring block
                for (Block block : blocks) {
                    for (Block.Parameter p : block.parameters()) {
                        for (Op.Result use : p.uses()) {
                            if (!use.declaringBlock().isDominatedBy(block)) {
                                throw new IllegalStateException("Use of value is not dominated by value");
                            }
                        }
                    }

                    for (Op o : block.ops()) {
                        Op.Result r = o.result();
                        for (Op.Result use : r.uses()) {
                            if (!use.declaringBlock().isDominatedBy(block)) {
                                throw new IllegalStateException("Use of value is not dominated by value");
                            }
                        }
                    }
                }
            }

            Body.this.parentOp = op;
            return Body.this;
        }

        static IllegalStateException noTerminatingOperation() {
            return new IllegalStateException("Block has no terminating operation as the last operation");
        }

        /**
         * Returns this body builder's signature, represented as a function type.
         * <p>
         * The signature's return type is the body builder's yield type and parameter types are
         * the currently built entry block's parameter types, in order.
         *
         * @return the body builder's signature
         */
        public FunctionType bodySignature() {
            CodeType returnType = Body.this.yieldType();
            Block eb = Body.this.entryBlock();
            return CoreType.functionType(returnType, eb.parameterTypes());
        }

        /**
         * {@return this body builder's connected ancestor body builder if this body builder is
         * <a href="#connected-builder">connected</a>, otherwise {@code null} if this body builder is isolated}
         */
        public Builder connectedAncestorBody() {
            return connectedAncestorBody;
        }

        /**
         * {@return this body builder's entry block builder}
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
            if (finished) {
                throw new IllegalStateException("Builder is finished");
            }
        }

        Body target() {
            return Body.this;
        }

        // Build new block in body
        Block.Builder block(List<CodeType> params, CodeContext cc, CodeTransformer ct) {
            check();
            Block block = Body.this.createBlock(params);

            return block.new Builder(this, cc, ct);
        }
    }

    /**
     * Transforms this body, returning an output body builder containing the transformed body.
     * <p>
     * This method creates an output body builder for this input body's {@link #bodySignature() signature}, with a
     * {@link CodeContext#create(CodeContext) child} of the given parent code context, and the given code transformer.
     * <p>
     * The output body builder is <a href="Body.Builder.html#connected-builder">connected</a> to a body builder, as its
     * nearest ancestor body builder, if that builder can be determined from this input body and the given parent code
     * context. Otherwise, the output body builder is <a href="Body.Builder.html#isolated-builder">isolated</a>.
     * <p>
     * This method then transforms this input body by invoking
     * {@link CodeTransformer#acceptBody(Block.Builder, Body, List)} with the created output body builder's
     * {@link Body.Builder#entryBlock() entry} block builder, this input body, and that entry block builder's
     * parameters.
     *
     * @apiNote
     * To copy a body use the {@link CodeTransformer#COPYING_TRANSFORMER copying transformer}.
     * <p>
     * The body builder connected to the output body builder can be explicitly determined when this
     * input body's {@link Body#ancestorBody() nearest ancestor} body is present and observable, and the given parent
     * code context can be used to {@link CodeContext#queryBody(Body) query} the present body builder for that ancestor
     * body. For example, in such cases:
     * {@snippet lang = "java":
     * Body nearestAncestorBody = this.ancestorBody(); // @link substring="ancestorBody" target="jdk.incubator.code.CodeElement#ancestorBody"
     * Body.Builder connectedBodyBuilder = cc.queryBody(nearestAncestorBody).orElseThrow(); // @link substring="queryBody" target="jdk.incubator.code.CodeContext#queryBody"
     * }
     *
     * @param cc the parent code context
     * @param ct the code transformer
     * @return a body builder containing the transformed body
     */
    public Builder transform(CodeContext cc, CodeTransformer ct) {
        Builder connectedAncestorBodyBuilder = connectedAncestorBody != null
                ? cc.queryBody(connectedAncestorBody).orElse(null)
                : null;
        Builder bodyBuilder = Builder.of(connectedAncestorBodyBuilder,
                bodySignature(),
                // Create child context for mapped code items contained in this body
                // thereby not polluting the given context
                CodeContext.create(cc), ct);

        // Transform body starting from the entry block builder
        ct.acceptBody(bodyBuilder.entryBlock, this, bodyBuilder.entryBlock.parameters());
        return bodyBuilder;
    }

    private static final int UNASSIGNED_INDEX = Integer.MIN_VALUE;
    private static final int UNSORTED_INDEX = Integer.MAX_VALUE;

    private void sortReversePostorder() {
        if (blocks.size() == 1) {
            Block b = blocks.getFirst();
            b.index = 0;
            b.component = 0;
            return;
        }

        // Set block indexes and components to indicate unsorted state
        for (Block b : blocks) {
            b.index = UNSORTED_INDEX;
            b.component = -1;
        }

        // Repeatedly sort for each graph
        int start = 0;
        while (start < blocks().size()) {
            start = sortReversePostorder(start);
        }
    }

    // Sort blocks in reverse post order
    // After sorting the following holds for a block in the range of [start, end)
    //   block.parentBody().blocks().indexOf(block) == block.index()
    private int sortReversePostorder(int start) {
        assert assertUnsorted(blocks, start);

        Deque<Block> stack = new ArrayDeque<>();
        stack.push(blocks.get(start));

        // Postorder iteration
        int index = blocks.size();
        while (!stack.isEmpty()) {
            Block n = stack.peek();
            if (n.index == UNASSIGNED_INDEX) {
                // If n's successor has been processed then add n
                stack.pop();
                n.index = --index;
            } else if (n.index != UNSORTED_INDEX) {
                // If n has already been processed then ignore
                stack.pop();
            } else {
                // Mark before processing successors, a successor may refer back to n
                n.index = UNASSIGNED_INDEX;
                for (Block.Reference s : n.successors()) {
                    Block target = s.target;
                    if (target.index != UNSORTED_INDEX) {
                        continue;
                    }

                    stack.push(target);
                }
            }
        }

        // The number of blocks in the graph
        int nBlocks = blocks.size() - blocks.get(start).index;
        int end = start + nBlocks;

        // Sort by indexes
        List<Block> listToSort = (start == 0) ? blocks :  blocks.subList(start, blocks.size());
        listToSort.sort(Comparator.comparingInt(b -> b.index));

        // Reassign block indexes to their natural indexes, sort order is preserved
        for (int i = start; i < end; i++) {
            Block b = blocks.get(i);
            b.index = i;
            b.component = start;
        }
        return end;
    }

    private static boolean assertUnsorted(List<Block> blocks, int start) {
        for (int i = start; i < blocks.size(); i++) {
            if (blocks.get(i).index != UNSORTED_INDEX) {
                return false;
            }
        }
        return true;
    }

    // Modifying methods

    // Create block
    private Block createBlock(List<CodeType> params) {
        Block b = new Block(this, params);
        blocks.add(b);
        return b;
    }
}
