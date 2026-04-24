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

import java.util.*;
import java.util.stream.Collectors;

/**
 * A (basic) block containing an ordered sequence of operations, where the last operation is
 * a {@link Op.Terminating terminating} operation.
 * <p>
 * The terminating operation, according to its specification, may branch to other blocks contained in the
 * same parent body, by way of its {@link Op#successors() successors}, or exit the parent body and optionally
 * yield a result.
 * <p>
 * Blocks declare zero or more block parameters.
 * <p>
 * A block is built using a {@link Block.Builder block builder} that is used to append operations to the block
 * being built.
 */
public final class Block implements CodeElement<Block, Op> {

    /**
     * A value that is a block parameter
     */
    public static final class Parameter extends Value {
        Parameter(Block block, CodeType type) {
            super(block, type);
        }

        @Override
        public String toString() {
            return "%param@" + Integer.toHexString(hashCode());
        }

        @Override
        public SequencedSet<Value> dependsOn() {
            return Collections.emptyNavigableSet();
        }

        /**
         * Returns the invokable operation associated with this block parameter.
         * <p>
         * If this block parameter is declared in an entry block and that
         * block's ancestor operation (the parent of the entry block's parent body)
         * is an instance of {@link Op.Invokable}, then that instance is returned,
         * otherwise {@code null} is returned.
         * <p>
         * A non-{@code null} result implies this parameter is an invokable parameter.
         *
         * @apiNote
         * This method may be used to pattern match on the returned result:
         * {@snippet lang = "java":
         *     if (p.invokableOperation() instanceof CoreOp.FuncOp f) {
         *         assert f.parameters().indexOf(p) == p.index(); // @link substring="parameters()" target="Op.Invokable#parameters()"
         *     }
         *}
         *
         * @return the invokable operation, otherwise {@code null} if the operation
         * is not an instance of {@link Op.Invokable}.
         * @see Op.Invokable#parameters()
         */
        public Op.Invokable invokableOperation() {
            if (declaringBlock().isEntryBlock() &&
                    declaringBlock().ancestorOp() instanceof Op.Invokable o) {
                return o;
            } else {
                return null;
            }
        }

        /**
         * {@return the index of this block parameter in the parameters of its declaring block.}
         * @see Value#declaringBlock()
         * @see Block#parameters()
         */
        public int index() {
            return declaringBlock().parameters().indexOf(this);
        }
    }

    /**
     * A block reference that refers to a block with arguments.
     * <p>
     * A terminating operation may refer, via a block reference, to one or more blocks as its successors.
     * When control is passed from a block to a successor block the values of the block reference's arguments are
     * assigned, in order, to the successor block's parameters.
     */
    public static final class Reference implements CodeItem {
        final Block target;
        final List<Value> arguments;

        /**
         * Constructs a block reference for a given target block and arguments.
         *
         * @param target    the target block.
         * @param arguments the target block arguments, a copy will be made as needed.
         */
        Reference(Block target, List<? extends Value> arguments) {
            this.target = target;
            this.arguments = List.copyOf(arguments);
        }

        /**
         * {@return the target block.}
         * @throws IllegalStateException if the target block is being built and is not observable.
         */
        public Block targetBlock() {
            if (!isBuilt()) {
                throw new IllegalStateException("Target block is being built and is not observable");
            }

            return target;
        }

        /**
         * {@return the block arguments.}
         */
        public List<Value> arguments() {
            return arguments;
        }

        boolean isBuilt() {
            return target.isBuilt();
        }
    }

    final Body parentBody;

    final List<Parameter> parameters;

    final List<Op> ops;

    // @@@ In topological order
    // @@@ Create lazily
    //     Can the representation be more efficient e.g. an array?
    final SequencedSet<Block> predecessors;

    // Reverse postorder index
    // Set when block's body has sorted its blocks and therefore set when built
    // Block is inoperable when < 0 i.e., when not built
    int index = -1;

    Block(Body parentBody) {
        this(parentBody, List.of());
    }

    Block(Body parentBody, List<CodeType> parameterTypes) {
        this.parentBody = parentBody;
        this.parameters = new ArrayList<>();
        for (CodeType param : parameterTypes) {
            parameters.add(new Parameter(this, param));
        }
        this.ops = new ArrayList<>();
        this.predecessors = new LinkedHashSet<>();
    }


    @Override
    public String toString() {
        return "^block_" + index + "@" + Integer.toHexString(hashCode());
    }

    /**
     * Returns this block's parent body.
     *
     * @return this block's parent body.
     */
    @Override
    public Body parent() {
        return parentBody;
    }

    @Override
    public List<Op> children() {
        return ops();
    }

    /**
     * Returns the sequence of operations contained in this block.
     *
     * @return returns the sequence operations, as an unmodifiable list.
     */
    public List<Op> ops() {
        return Collections.unmodifiableList(ops);
    }

    /**
     * Returns this block's index within the parent body's blocks.
     * <p>
     * The following identity holds true:
     * {@snippet lang = "java" :
     *     this.parentBody().blocks().indexOf(this) == this.index();
     * }
     *
     * @apiNote
     * The block's index may be used to efficiently track blocks using
     * bits sets or boolean arrays.
     *
     * @return the block index.
     */
    public int index() {
        return index;
    }

    /**
     * Returns the block parameters.
     *
     * @return the block parameters, as an unmodifiable list.
     */
    public List<Parameter> parameters() {
        return Collections.unmodifiableList(parameters);
    }

    /**
     * Returns the block parameter types.
     *
     * @return the block parameter types, as am unmodifiable list.
     */
    public List<CodeType> parameterTypes() {
        return parameters.stream().map(Value::type).toList();
    }

    /**
     * Returns the first operation in this block.
     *
     * @return the first operation in this block.
     */
    public Op firstOp() {
        return ops.getFirst();
    }

    /**
     * Returns the last, terminating, operation in this block.
     * <p>
     * The terminating operation implements {@link Op.Terminating}.
     *
     * @return the last, terminating, operation in this block.
     */
    public Op terminatingOp() {
        Op lop = ops.getLast();
        assert lop instanceof Op.Terminating;
        return lop;
    }

    /**
     * Returns the next operation after the given operation, otherwise {@code null}
     * if this operation is the last operation.
     *
     * @param op the operation
     * @return the next operation after the given operation.
     * @throws IllegalArgumentException if the operation is not a child of this block
     */
    public Op nextOp(Op op) {
        int i = ops.indexOf(op);
        if (i == -1) {
            throw new IllegalArgumentException();
        }
        return i < ops().size() - 1 ? ops.get(i + 1) : null;
    }

    /**
     * Returns the set of predecessors, the set containing each block in the parent
     * body that refers to this block as a successor.
     *
     * @return the set of predecessors, as an unmodifiable sequenced set. The encounter order is unspecified
     * and determined by the order in which operations are built.
     * @apiNote A block may refer to itself as a successor and therefore also be its predecessor.
     */
    public SequencedSet<Block> predecessors() {
        return Collections.unmodifiableSequencedSet(predecessors);
    }

    /**
     * Returns the list of predecessor references to this block.
     * <p>
     * This method behaves is if it returns the result of the following expression:
     * {@snippet lang = java:
     * predecessors.stream().flatMap(p -> successors().stream())
     *    .filter(r -> r.targetBlock() == this)
     *    .toList();
     *}
     *
     * @return the list of predecessor references to this block, as an unmodifiable list.
     * @apiNote A predecessor block may reference it successor block one or more times.
     */
    public List<Block.Reference> predecessorReferences() {
        return predecessors.stream().flatMap(p -> p.successors().stream())
                .filter(r -> r.targetBlock() == this)
                .toList();
    }

    /**
     * Returns the list of successors referring to other blocks.
     * <p>
     * The successors are declared by the terminating operation contained in this block.
     *
     * @return the list of successors, as an unmodifiable list.
     * @apiNote given a block, A say, whose successor targets a block, B say, we can
     * state that B is a successor block of A and A is a predecessor block of B.
     */
    public List<Reference> successors() {
        return ops.getLast().successors();
    }

    /**
     * Returns the set of target blocks referred to by the successors of this block.
     * <p>
     * This method behaves is if it returns the result of the following expression:
     * {@snippet lang = java:
     * successors().stream()
     *     .map(Block.Reference::targetBlock)
     *     .collect(Collectors.toCollection(LinkedHashSet::new));
     *}
     *
     * @return the set of target blocks, as an unmodifiable set.
     */
    public SequencedSet<Block> successorTargets() {
        LinkedHashSet<Block> targets = successors().stream().map(Reference::targetBlock)
                .collect(Collectors.toCollection(LinkedHashSet::new));
        return Collections.unmodifiableSequencedSet(targets);
    }

    /**
     * Returns true if this block is an entry block, the first block occurring
     * in the parent body's list of blocks.
     *
     * @return true if this block is an entry block.
     */
    public boolean isEntryBlock() {
        return parentBody.entryBlock() == this;
    }

    /**
     * Returns {@code true} if this block is dominated by the given block {@code dom}.
     * <p>
     * A block {@code b} is dominated by {@code dom} if every path from the entry block of {@code dom}'s
     * parent body to {@code b} passes through {@code dom}.
     * <p>
     * If this block and {@code dom} have different parent bodies, this method first
     * repeatedly replaces this block with its {@link #ancestorBlock() nearest ancestor} block until:
     * <ul>
     * <li>{@code null} is reached, in which case this method returns {@code false}; or</li>
     * <li>both blocks are in the same parent body, in which case
     * <a href="https://en.wikipedia.org/wiki/Dominator_(graph_theory)">dominance</a> is tested within that body.</li>
     * </ul>
     *
     * @apiNote
     * The method {@link Body#immediateDominators()} can be used to test for dominance, by repeatedly querying a block's
     * immediately dominating block until {@code null} or {@code dom} is reached.
     *
     * @param dom the dominating block
     * @return {@code true} if this block is dominated by the given block.
     * @see Body#immediateDominators()
     * @see Value#isDominatedBy
     */
    public boolean isDominatedBy(Block dom) {
        Block b = findBlockForDomBody(this, dom.ancestorBody());
        if (b == null) {
            return false;
        }

        // A block non-strictly dominates itself
        if (b == dom) {
            return true;
        }

        // The entry block in b's body dominates all other blocks in the body
        Block entry = b.ancestorBody().entryBlock();
        if (dom == entry) {
            return true;
        }

        // Traverse the immediate dominators until dom is reached or the entry block
        Map<Block, Block> idoms = b.ancestorBody().immediateDominators();
        Block idom = idoms.get(b);
        while (idom != entry) {
            if (idom == dom) {
                return true;
            }

            idom = idoms.get(idom);
        }

        return false;
    }

    /**
     * Returns the immediate dominator of this block, otherwise {@code null} if this block is the entry block.
     * Both this block and the immediate dominator (if defined) have the same parent body.
     * <p>
     * The immediate dominator is the unique block that strictly dominates this block, but does not strictly dominate
     * any other block that strictly dominates this block.
     *
     * @return the immediate dominator of this block, otherwise {@code null} if this block is the entry block.
     */
    public Block immediateDominator() {
        if (this == ancestorBody().entryBlock()) {
            return null;
        }

        Map<Block, Block> idoms = ancestorBody().immediateDominators();
        return idoms.get(this);
    }

    /**
     * Returns the immediate post dominator of this block.
     * <p>
     * If this block has no successors then this method returns the synthetic block
     * {@link Body#IPDOM_EXIT} representing the synthetic exit used to compute
     * the immediate post dominators.
     * <p>
     * Both this block and the immediate post dominator (if defined) have the same parent body,
     * except for the synthetic block {@link Body#IPDOM_EXIT}.
     * <p>
     * The immediate post dominator is the unique block that strictly post dominates this block,
     * but does not strictly post dominate any other block that strictly post dominates this block.
     *
     * @return the immediate post dominator of this block, otherwise {@code Body#IPDOM_EXIT}.
     */
    public Block immediatePostDominator() {
        Map<Block, Block> ipdoms = ancestorBody().immediatePostDominators();
        Block ipdom = ipdoms.get(this);
        return ipdom == this ? Body.IPDOM_EXIT : ipdom;
    }

    // @@@ isPostDominatedBy and immediatePostDominator

    private static Block findBlockForDomBody(Block b, final Body domr) {
        Body rb = b.ancestorBody();
        while (domr != rb) {
            // @@@ What if body is isolated

            b = rb.ancestorBlock();
            // null when op is top-level (and its body is isolated), or not yet assigned to block
            if (b == null) {
                return null;
            }
            rb = b.ancestorBody();
        }
        return b;
    }

    /**
     * A builder for a block.
     * <p>
     * A block builder defines the structure of the block that is being built. It is used to {@link #op(Op) append}
     * operations to its block. It can also be used to {@link Block.Builder#block(List) create} block builders for
     * sibling blocks.
     * <p>
     * A block builder has a {@link Block.Builder#parentBody() parent body builder}.
     * <p>
     * A block builder is operable while its parent body builder is building the parent body.
     * After the parent body is {@link Body.Builder#build(Op) built}, further attempts to operate on the block builder
     * throw an exception.
     * <p>
     * The block being built is not observable while building is in progress. Attempts to observe it through its
     * parameters, appended operations, their operation results, or block references, throw an exception.
     * <p>
     * A block builder always has a code {@link #context() context} and code {@link #transformer() transformer}. These
     * are used when {@link #op appending} an attached or root operation. Any sibling block builder
     * {@link #block(List) created} from a block builder will have the same code context and code transformer.
     * <p>
     * A block builder may be obtained with a different code context and code transformer by calling
     * {@link #withContextAndTransformer(CodeContext, CodeTransformer)}. Such a block builder can be used to apply
     * alternative transformations to attached or root operations that are appended.
     * <p>
     * During {@link CodeTransformer code transformation}, a block builder may also serve as the current output block
     * builder.
     */
    public final class Builder {
        final Body.Builder parentBody;
        final CodeContext cc;
        final CodeTransformer ct;

        Builder(Body.Builder parentBody, CodeContext cc, CodeTransformer ct) {
            this.parentBody = parentBody;
            this.cc = cc;
            this.ct = ct;
        }

        void check() {
            parentBody.check();
        }

        Block target() {
            return Block.this;
        }

        /**
         * {@return this block builder's code transformer}
         */
        public CodeTransformer transformer() {
            return ct;
        }

        /**
         * {@return this block builder's code context}
         */
        public CodeContext context() {
            return cc;
        }

        /**
         * {@return this block builder's parent body builder}
         */
        public Body.Builder parentBody() {
            return parentBody;
        }

        /**
         * Returns the entry block builder of this builder's parent body.
         * <p>
         * The returned block builder has this block builder's code context and code transformer.
         *
         * @return the entry block builder of this builder's parent body
         */
        public Block.Builder entryBlock() {
            return parentBody.entryBlock.withContextAndTransformer(cc, ct);
        }

        /**
         * {@return true if this block builder builds the entry block of its parent body}
         */
        public boolean isEntryBlock() {
            return Block.this == parentBody.target().entryBlock();
        }

        /**
         * Returns a block builder for the same block with the given code context and code transformer.
         * <p>
         * Both this block builder and the returned block builder may be operated on to build the same block. Both are
         * equal to each other, and both become inoperable when the parent body is built.
         *
         * @param cc the code context
         * @param ct the code transformer
         * @return the block builder with the given code context and code transformer
         */
        public Block.Builder withContextAndTransformer(CodeContext cc, CodeTransformer ct) {
            return this.cc == cc && this.ct == ct
                    ? this
                    : this.target().new Builder(parentBody(), cc, ct);
        }

        /**
         * Creates a builder for a new sibling block in this builder's parent body.
         * <p>
         * The returned builder has the same code context and code transformer as this
         * block builder.
         *
         * @param params the parameter types of the new block
         * @return the new block builder
         */
        public Block.Builder block(CodeType... params) {
            return block(List.of(params));
        }

        /**
         * Creates a builder for a new sibling block in this builder's parent body.
         * <p>
         * The returned builder has the same code context and code transformer as this
         * block builder.
         *
         * @param params the parameter types of the new block
         * @return the new block builder
         */
        public Block.Builder block(List<CodeType> params) {
            return parentBody.block(params, cc, ct);
        }

        /**
         * Returns an unmodifiable list of this block's parameters.
         *
         * @return the unmodifiable list of this block's parameters
         */
        public List<Parameter> parameters() {
            return Collections.unmodifiableList(parameters);
        }

        /**
         * Appends a parameter of the given type to this block.
         *
         * @param p the parameter type
         * @return the appended block parameter
         */
        public Parameter parameter(CodeType p) {
            check();
            return appendBlockParameter(p);
        }

        /**
         * Creates a reference to this block that can be used as a successor of a terminating operation.
         * <p>
         * A reference can only be created with arguments whose declaring block is being built.
         *
         * @param args the block arguments
         * @return a reference to this block
         * @throws IllegalStateException if this block builder builds the entry block.
         * @throws IllegalArgumentException if any argument's declaring block is built.
         */
        public Reference reference(Value... args) {
            return reference(List.of(args));
        }

        /**
         * Creates a reference to this block that can be used as a successor of a terminating operation.
         * <p>
         * A reference can only be created with arguments whose declaring block is being built.
         *
         * @param args the block arguments
         * @return a reference to this block
         * @throws IllegalStateException if this block builder builds the entry block.
         * @throws IllegalArgumentException if any argument's declaring block is built.
         */
        public Reference reference(List<? extends Value> args) {
            if (isEntryBlock()) {
                throw new IllegalStateException("Entry block cannot be referenced and used as a successor");
            }
            for (Value operand : args) {
                if (operand.isBuilt()) {
                    throw new IllegalArgumentException("Argument's declaring block is built: " + operand);
                }
            }

            return new Reference(Block.this, List.copyOf(args));
        }

        /**
         * Transforms a body using this block builder as the current output block builder, with a
         * {@link CodeContext#create(CodeContext) child} of this block builder's code context and the given code
         * transformer.
         *
         * @param body the body to transform
         * @param values the output values to map to the input parameters of the body's entry block
         * @param ct the code transformer
         * @see #body(Body, List, CodeContext, CodeTransformer)
         */
        public void body(Body body, List<? extends Value> values,
                         CodeTransformer ct) {
            check();

            body(body, values, CodeContext.create(cc), ct);
        }

        /**
         * Transforms a body using this block builder as the current output block builder, with the given code context
         * and code transformer.
         * <p>
         * This method first obtains a block builder with the given code context and code transformer by calling
         * {@link #withContextAndTransformer(CodeContext, CodeTransformer)}, and then transforms the body using the code
         * transformer by {@link CodeTransformer#acceptBody(Builder, Body, List) accepting} the obtained block builder,
         * the body, and the values.
         *
         * @apiNote
         * Supplying an explicit code context can ensure block and value mappings produced by the transformation do not
         * affect this builder's code context.
         *
         * @param body the body to transform
         * @param values the output values to map to the input parameters of the body's entry block
         * @param cc the code context
         * @param ct the code transformer
         * @see #withContextAndTransformer(CodeContext, CodeTransformer)
         * @see CodeTransformer#acceptBody(Builder, Body, List)
         */
        public void body(Body body, List<? extends Value> values,
                         CodeContext cc, CodeTransformer ct) {
            check();

            ct.acceptBody(withContextAndTransformer(cc, ct), body, values);
        }

        /**
         * Appends an operation to this block.
         * <p>
         * If the operation is unattached, it is appended directly to this block.
         * <p>
         * If the operation is attached to a block or is a root operation, this method performs
         * <a id="transform-on-append"><i>transform-on-append</i></a>: the operation is first
         * {@link Op#transform(CodeContext, CodeTransformer) transformed} using this block builder's code context and
         * code transformer; and then the resulting unattached operation is appended to this block.
         * If the operation being appended has a result, it is {@link CodeContext#mapValueIfAbsent mapped},
         * if no such mapping already exists, to the result of the appended operation in this block builder's code
         * context.
         * <p>
         * The appended operation must be structurally valid for this block, requiring:
         * <ul>
         * <li>the body builder for each child body is isolated, or has this block builder's parent body builder as its
         * nearest ancestor body builder.
         * <li>each operand is <a href="Body.Builder.html#reachable-value">reachable</a> from the operation;
         * <li>each successor argument is <a href="Body.Builder.html#reachable-value">reachable</a> from the operation;
         * <li>each successor target is a sibling of this block; and
         * <li>this block does not already end with a terminating operation.
         * </ul>
         *
         * @apiNote
         * Copying is a special case of transform-on-append when this block builder's code transformer is, or
         * behaves as a copying transformer, such as {@link CodeTransformer#COPYING_TRANSFORMER}.
         *
         * @param op the operation to append
         * @return the result of the appended operation
         * @throws IllegalStateException if the operation is structurally invalid
         */
        public Op.Result op(Op op) {
            check();

            // Perform transform-on-append for an attached or root operation
            Op outputOp = op.isAttached() || op.isRoot()
                    ? op.transform(cc, ct)
                    : op;
            assert outputOp.result == null;

            Op.Result outputResult = insertOp(outputOp);

            Op.Result inputResult = op.result();
            if (inputResult != null) {
                // Map the result of the first transformation
                // @@@ If the same operation is transformed more than once then subsequent
                //  transformed ops will not get implicitly mapped
                //  Should this be an error?
                cc.mapValueIfAbsent(inputResult, outputResult);
            }

            return outputResult;
        }

        /**
         * Returns true if this block builder is equal to the other object.
         * <p>This block builder is equal if the other object is an instance of a block builder, and they build
         * the same block (but maybe bound to different code contexts and code transformers).
         *
         * @param o the other object
         * @return true if this block builder is equal to the other object.
         */
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            return o instanceof Builder that && Block.this == that.target();
        }

        @Override
        public int hashCode() {
            return Block.this.hashCode();
        }
    }

    // Modifying methods

    // Create block parameter associated with this block
    private Parameter appendBlockParameter(CodeType type) {
        Parameter blockParameter = new Parameter(this, type);
        parameters.add(blockParameter);

        return blockParameter;
    }

    // Create an operation, adding to the end of the list of existing operations
    private Op.Result insertOp(Op op) {
        Op.Result opResult = new Op.Result(this, op);
        bindOp(opResult, op);

        ops.add(op);
        return opResult;
    }

    private void bindOp(Op.Result opr, Op op) {
        // Structural checks
        if (!ops.isEmpty() && ops.getLast() instanceof Op.Terminating) {
            throw new IllegalStateException("Operation cannot be appended, the block has a terminating operation");
        }

        for (Body b : op.bodies()) {
            if (b.ancestorBody != null && b.ancestorBody != this.parentBody) {
                throw new IllegalStateException("Body of operation is connected to a different ancestor body: ");
            }
        }

        for (Value v : op.operands()) {
            if (!isReachable(v)) {
                throw new IllegalStateException(
                        String.format("Operand of operation %s is not defined in tree: %s", op, v));
            }
            assert !v.isBuilt();
        }

        for (Reference s : op.successors()) {
            if (s.target.parentBody != this.parentBody) {
                throw new IllegalStateException("Target of block reference is not a sibling of this block");
            }

            for (Value v : s.arguments()) {
                if (!isReachable(v)) {
                    throw new IllegalStateException(
                            String.format("Argument of block reference %s of terminating operation %s is not defined in tree: %s", s, op, v));
                }
                assert !v.isBuilt();
            }
        }

        // State updates after structural checks
        // @@@ The alternative is to close the body builder on failure, rendering it inoperable,
        // so checks and updates can be merged
        for (Value v : op.operands()) {
            v.uses.add(opr);
        }

        for (Reference s : op.successors()) {
            for (Value v : s.arguments()) {
                v.uses.add(opr);
            }

            s.target.predecessors.add(Block.this);
        }

        op.result = opr;
    }

    // Determine if the parent body of value's block is an ancestor of this block
    private boolean isReachable(Value v) {
        Body b = parentBody;
        while (b != null && b != v.block.parentBody) {
            b = b.ancestorBody;
        }
        return b != null;
    }

    //

    boolean isBuilt() {
        return index >= 0;
    }
}
