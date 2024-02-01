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

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.descriptor.TypeDesc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

import static java.lang.reflect.code.op.CoreOps._return;
import static java.lang.reflect.code.op.CoreOps.branch;

/**
 * A (basic) block containing an ordered sequence of operations, where the last operation is
 * a {@link Op.Terminating terminating} operation.
 * <p>
 * The terminating operation, according to its specification, may branch to other blocks contained in the
 * same parent body, by way of its {@link Op#successors() successors}, or exit the parent body and optionally
 * yield a result.
 * <p>
 * Blocks may declare one or more block parameters.
 */
public final class Block implements CodeElement<Block, Op> {

    /**
     * A value that is a block parameter
     */
    public static final class Parameter extends Value {
        Parameter(Block block, TypeDesc type) {
            super(block, type);
        }

        @Override
        public Set<Value> dependsOn() {
            return Set.of();
        }
    }

    /**
     * A block reference that refers to a block with arguments.
     * <p>
     * A terminating operation may refer, via a block reference, to one or more blocks as its successors.
     * When control is passed from a block to a successor block the values of the block reference's arguments are
     * assigned, in order, to the successor block's parameters.
     */
    public static final class Reference {
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
         * @throws IllegalStateException if the target block is partially built
         */
        public Block targetBlock() {
            if (!isBound()) {
                throw new IllegalStateException("Target block is partially built");
            }

            return target;
        }

        /**
         * {@return the block arguments.}
         */
        public List<Value> arguments() {
            return arguments;
        }

        boolean isBound() {
            return target.isBound();
        }
    }

    final Body parentBody;

    final List<Parameter> parameters;

    final List<Op> ops;

    // @@@ In topological order
    // @@@ Create lazily
    //     Can the representation be more efficient e.g. an array?
    final Set<Block> predecessors;

    // Reverse postorder index
    // Set when block's body has sorted its blocks and therefore set when built
    // Block is inoperable when < 0 i.e., when partially built
    int index = -1;

    Block(Body parentBody) {
        this(parentBody, List.of());
    }

    Block(Body parentBody, List<TypeDesc> parameterTypes) {
        this.parentBody = parentBody;
        this.parameters = new ArrayList<>();
        for (TypeDesc param : parameterTypes) {
            parameters.add(new Parameter(this, param));
        }
        this.ops = new ArrayList<>();
        this.predecessors = new LinkedHashSet<>();
    }

    /**
     * Returns this block's parent body.
     *
     * @return this block's parent body.
     */
    public Body parentBody() {
        return parentBody;
    }

    /**
     * Returns this block's index within the parent body's blocks.
     * <p>
     * The following identity holds true:
     * {@snippet lang = "java"
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
    public List<TypeDesc> parameterTypes() {
        return parameters.stream().map(Value::type).toList();
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
     * Finds the operation in this block that is the ancestor of the given operation.
     *
     * @param op the given operation.
     * @return the operation in this block that is the ancestor of the given operation,
     * otherwise {@code null}
     */
    public Op findAncestorOpInBlock(Op op) {
        Objects.requireNonNull(op);

        while (op != null && op.parentBlock() != this) {
            Body encBody = op.ancestorBody();
            if (encBody == null) {
                return null;
            }

            op = encBody.parentOp();
        }

        return op;
    }

    @Override
    public List<Op> children() {
        return ops();
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
     */
    public Op nextOp(Op op) {
        int i = ops.indexOf(op);
        if (i == -1) {
            throw new IllegalArgumentException();
        }
        return i < ops().size() - 1 ? ops.get(i + 1) : null;
    }

    /**
     * Returns the list of predecessors, namely each block in the parent body that refers
     * to this block as a successor.
     *
     * @return the set of predecessors, as an unmodifiable list.
     * @apiNote A block may refer to itself as a successor and therefore also be its predecessor.
     */
    public Set<Block> predecessors() {
        return Collections.unmodifiableSet(predecessors);
    }

    /**
     * Returns the list of successors referring to other blocks in the parent body.
     * <p>
     * The successors are declared by the terminating operation contained in this block.
     *
     * @return the list of successors, as an unmodifiable list.
     */
    public List<Reference> successors() {
        Op lopr = ops.get(ops.size() - 1);
        return lopr.successors();
    }

    /**
     * Returns true if this block is an entry block.
     *
     * @return true if this block is an entry block.
     */
    public boolean isEntryBlock() {
        return parentBody.entryBlock() == this;
    }

    /**
     * Returns {@code true} if this block is
     * <a href="https://en.wikipedia.org/wiki/Dominator_(graph_theory)">dominated by</a> the given block {@code dom}.
     * This block is dominated by {@code dom}, if every path from the root entry block to this block passes through
     * {@code dom}.
     * <p>
     * If this block, {@code b} say, and {@code dom} are not in the same parent body,
     * then {@code b} becomes the nearest ancestor block, result of {@code b.parentBody().parentOp().parentBlock()},
     * and so on until either:
     * {@code b} is {@code null}, therefore {@code b} is <b>not</b> dominated by {@code dom} and this method
     * returns {@code false}; or
     * {@code b.parentBody() == dom.parentBody()}, therefore this method returns the result
     * of {@code b.isDominatedBy(dom)}.
     * <p>
     * If this method returns {@code true} then {@code dom.isDominatedBy(this)}
     * will return {@code false}. However, if this method returns {@code false} then it
     * does not imply {@code dom.isDominatedBy(this)} returns {@code true}, as neither
     * block may dominate the other.
     *
     * @param dom the dominating block
     * @return {@code true} if this block is dominated by the given block.
     */
    // @@@ Should this be reversed and named dominates(Block b)
    public boolean isDominatedBy(Block dom) {
        Block b = findBlockForDomBody(this, dom.parentBody());
        if (b == null) {
            return false;
        }

        // A block non-strictly dominates itself
        if (b == dom) {
            return true;
        }

        // The entry block in b's body dominates all other blocks in the body
        Block entry = b.parentBody().entryBlock();
        if (dom == entry) {
            return true;
        }

        // Traverse the immediate dominators until dom is reached or the entry block
        Map<Block, Block> idoms = b.parentBody().immediateDominators();
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
        if (this == parentBody().entryBlock()) {
            return null;
        }

        Map<Block, Block> idoms = parentBody().immediateDominators();
        return idoms.get(this);
    }

    // @@@ isPostDominatedBy and immediatePostDominator

    private static Block findBlockForDomBody(Block b, final Body domr) {
        Body rb = b.parentBody();
        while (domr != rb) {
            // @@@ What if body is isolated

            b = rb.parentOp().parentBlock();
            // null when op is top-level (and its body is isolated), or not yet assigned to block
            if (b == null) {
                return null;
            }
            rb = b.parentBody();
        }
        return b;
    }

    /**
     * A builder of a block.
     * <p>
     * When the parent body builder is built this block builder is also built. If a built builder
     * is operated on to append a block parameter, append an operation, or add a block, then
     * an {@code IllegalStateException} is thrown.
     */
    public final class Builder implements Function<Op, Op.Result> {
        final Body.Builder parentBody;
        final CopyContext cc;
        final OpTransformer ot;

        Builder(Body.Builder parentBody, CopyContext cc, OpTransformer ot) {
            this.parentBody = parentBody;
            this.cc = cc;
            this.ot = ot;
        }

        void check() {
            parentBody.check();
        }

        Block target() {
            return Block.this;
        }

        /**
         * {@return the block builder's operation transformer}
         */
        public OpTransformer transformer() {
            return ot;
        }

        /**
         * {@return the block builder's context}
         */
        public CopyContext context() {
            return cc;
        }

        /**
         * {@return the parent body builder}
         */
        public Body.Builder parentBody() {
            return parentBody;
        }

        /**
         * Returns the entry block builder for parent body.
         *
         * <p>The returned block is rebound if necessary to this block builder's
         * context and transformer.
         *
         * @return the entry block builder for parent body builder
         */
        public Block.Builder entryBlock() {
            return parentBody.entryBlock.rebind(cc, ot);
        }

        /**
         * {@return true if this block builder is a builder of the entry block}
         */
        public boolean isEntryBlock() {
            return Block.this == parentBody.target().entryBlock();
        }

        /**
         * Rebinds this block builder with the given context and operation transformer.
         *
         * <p>Either this block builder and the returned block builder may be operated on to build
         * the same block.
         * Both are equal to each other, and both are closed when the parent body builder is closed.
         *
         * @param cc the context
         * @param ot the operation transformer
         * @return the rebound block builder
         */
        public Block.Builder rebind(CopyContext cc, OpTransformer ot) {
            return this.cc == cc && this.ot == ot
                    ? this
                    : this.target().new Builder(parentBody(), cc, ot);
        }

        /**
         * Adds a new block to the parent body.
         *
         * @param params the block's parameter types
         * @return the new block builder
         */
        public Block.Builder block(TypeDesc... params) {
            return block(List.of(params));
        }

        /**
         * Adds a new block to the parent body.
         *
         * @param params the block's parameter types
         * @return the new block builder
         */
        public Block.Builder block(List<TypeDesc> params) {
            return parentBody.block(params, cc, ot);
        }

        /**
         * Returns an unmodifiable list of the block's parameters.
         *
         * @return the unmodifiable list of the block's parameters
         */
        public List<Parameter> parameters() {
            return Collections.unmodifiableList(parameters);
        }

        /**
         * Appends a block parameter to the block's parameters.
         *
         * @param p the parameter type
         * @return the appended block parameter
         */
        public Parameter parameter(TypeDesc p) {
            check();
            return appendBlockParameter(p);
        }

        /**
         * Creates a reference to this block that can be used as a successor of a terminating operation.
         *
         * @param args the block arguments
         * @return the reference to this block
         * @throws IllegalStateException if this block builder is associated with the entry block.
         */
        public Reference successor(Value... args) {
            return successor(List.of(args));
        }

        /**
         * Creates a reference to this block that can be used as a successor of a terminating operation.
         *
         * @param args the block arguments
         * @return the reference to this block
         * @throws IllegalStateException if this block builder is associated with the entry block.
         */
        public Reference successor(List<? extends Value> args) {
            if (isEntryBlock()) {
                throw new IllegalStateException("Entry block cannot be referred to as a successor");
            }

            return new Reference(Block.this, List.copyOf(args));
        }

        /**
         * An inline consumer that inserts a return operation with a value, if non-null.
         */
        public static final BiConsumer<Block.Builder, Value> INLINE_RETURN = (block, value) -> {
            block.op(value != null ? _return(value) : _return());
        };

        /**
         * Inlines the invokable operation into this block and returns the block builder from which to
         * continue building.
         * <p>
         * This method {@link #transformBody(Body, List, CopyContext, OpTransformer) transforms} the body of the
         * invokable operation with the given arguments, a new context, and an operation transformer that
         * replaces return operations by applying the given consumer to a block builder and a return value.
         * <p>
         * The operation transformer copies all operations except return operations whose nearest invokable operation
         * ancestor is the given the invokable operation. When such a return operation is encountered, then on
         * first encounter of its grandparent body a return block builder is computed and used for this return operation
         * and encounters of subsequent return operations with the same grandparent body.
         * <p>
         * If the grandparent body has only one block then operation transformer's block builder is the return
         * block builder. Otherwise, if the grandparent body has one or more blocks then the return block builder is
         * created from the operation transformer's block builder. The created return block builder will have a block
         * parameter whose type corresponds to the return type, or will have no parameter for void return.
         * The computation finishes by applying the return block builder and a return value to the inlining consumer.
         * If the grandparent body has only one block then the return value is the value mapped from the return
         * operation's operand, or is null for void return. Otherwise, if the grandparent body has one or more blocks
         * then the value is the block parameter of the created return block builder, or is null for void return.
         * <p>
         * For every encounter of a return operation the associated return block builder is compared against the
         * operation transformer's block builder. If they are not equal then a branch operation is added to the
         * operation transformer's block builder whose successor is the return block builder with a block argument
         * that is the value mapped from the return operation's operand, or with no block argument for void return.
         * @apiNote
         * It is easier to inline an invokable op if its body is in lowered form (there are no operations in the blocks
         * of the body that are lowerable). This ensures a single exit point can be created (paired with the single
         * entry point). If there are one or more nested return operations, then there is unlikely to be a single exit.
         * Transforming the model to create a single exit point while preserving nested structure is in general
         * non-trivial and outside the scope of this method. In such cases the invokable operation can be transformed
         * with a lowering transformation after which it can then be inlined.
         *
         * @param invokableOp the invokable operation
         * @param args the arguments to map to the invokable operation's parameters
         * @param inlineConsumer the consumer applied to process the return from the invokable operation.
         *                       This is called once for each grandparent body of a return operation, with a block to
         *                       build replacement operations and the return value, or null for void return.
         * @return the block builder to continue building from
         * @param <O> The invokable type
         */
        public <O extends Op & Op.Invokable> Block.Builder inline(O invokableOp, List<? extends Value> args,
                                                                  BiConsumer<Block.Builder, Value> inlineConsumer) {
            Map<Body, Block.Builder> returnBlocks = new HashMap<>();
            // Create new context, ensuring inlining is isolated
            transformBody(invokableOp.body(), args, CopyContext.create(), (block, op) -> {
                // If the return operation is associated with the invokable operation
                if (op instanceof CoreOps.ReturnOp rop && getNearestInvokeableAncestorOp(op) == invokableOp) {
                    // Compute the return block
                    Block.Builder returnBlock = returnBlocks.computeIfAbsent(rop.ancestorBody(), _body -> {
                        Block.Builder rb;
                        // If the body has one block we know there is just one return op declared, otherwise there may
                        // one or more. If so, create a new block that joins all the returns.
                        // Note: we could count all return op in a body to avoid creating a new block for a body
                        // with two or more blocks with only one returnOp is declared.
                        Value r;
                        if (rop.ancestorBody().blocks().size() != 1) {
                            List<TypeDesc> param = rop.returnValue() != null
                                    ? List.of(invokableOp.funcDescriptor().returnType())
                                    : List.of();
                            rb = block.block(param);
                            r = !param.isEmpty()
                                    ? rb.parameters().get(0)
                                    : null;
                        } else {
                            r = rop.returnValue() != null
                                    ? block.context().getValue(rop.returnValue())
                                    : null;
                            rb = block;
                        }

                        // Inline the return
                        inlineConsumer.accept(rb, r);

                        return rb;
                    });

                    // Replace the return op with a branch to the return block, if needed
                    if (!returnBlock.equals(block)) {
                        // Replace return op with branch to return block, with given return value
                        List<Value> arg = rop.returnValue() != null
                                ? List.of(block.context().getValue(rop.returnValue()))
                                : List.of();
                        block.apply(branch(returnBlock.successor(arg)));
                    }

                    return block;
                }

                block.apply(op);
                return block;
            });


            Builder builder = returnBlocks.get(invokableOp.body());
            return builder != null ? builder : this;
        }

        private static Op getNearestInvokeableAncestorOp(Op op) {
            do {
                op = op.ancestorBody().parentOp();
            } while (!(op instanceof Op.Invokable));
            return op;
        }

        /**
         * Transforms a body into this block, with this block builder's context.
         *
         * @param bodyToTransform the body to transform
         * @param args        the list of values to map to the parameters of the body's entry block
         * @param ot          the operation transformer
         * @see #transformBody(Body, List, CopyContext, OpTransformer)
         */
        public void transformBody(Body bodyToTransform, List<? extends Value> args,
                                  OpTransformer ot) {
            transformBody(bodyToTransform, args, cc, ot);
        }

        /**
         * Transforms a body into this block.
         * <p>
         * First, a new context is created from the given context and that new context is used to map values and
         * blocks.
         * <p>
         * Second, the entry block is mapped to this block builder rebound with the given operation transformer and
         * copy context, the input block parameters of the body's entry block are mapped to the given arguments.
         * <p>
         * Third, for each input block in the body (except the entry block) an output block builder is created with
         * equivalent parameters as the input block and with the given operation transformer and copy context.
         * The input block parameters are mapped to the output block parameters, and the input block is mapped to the
         * output block builder.
         * <p>
         * Fourth, for each input block in the body (in order) the input block's operations are transformed
         * by applying the output block builder and input block to the given operation transformer.
         * <p>
         * When the parent body is built any empty non-entry blocks that have no successors will be removed.
         *
         * @param bodyToTransform the body to transform
         * @param args            the list of values to map to the parameters of the body's entry block
         * @param cc              the copy context, for values captured in the body
         * @param ot              the operation transformer
         */
        public void transformBody(Body bodyToTransform, List<? extends Value> args,
                                  CopyContext cc, OpTransformer ot) {
            check();

            // @@@ This might be a new context e.g., when transforming a body
            cc = CopyContext.create(cc);

            Block entryBlockToTransform  = bodyToTransform.entryBlock();
            List<Block> blocksToTransform = bodyToTransform.blocks();

            // Map entry block
            // Rebind this block builder to the created context and transformer
            Block.Builder startingBlock = rebind(cc, ot);
            cc.mapBlock(entryBlockToTransform, startingBlock);
            cc.mapValues(entryBlockToTransform.parameters(), args);

            // Map subsequent blocks up front, for forward referencing successors
            for (int i = 1; i < blocksToTransform.size(); i++) {
                Block blockToTransform = blocksToTransform.get(i);
                if (cc.getBlock(blockToTransform) != null) {
                    throw new IllegalStateException("Block is already transformed");
                }

                // Create block and map block
                Block.Builder transformedBlock = startingBlock.block(List.of());
                for (Block.Parameter ba : blockToTransform.parameters()) {
                    transformedBlock.parameter(ba.type());
                }
                cc.mapBlock(blockToTransform, transformedBlock);
                cc.mapValues(blockToTransform.parameters(), transformedBlock.parameters());
            }

            for (Block blockToTransform : blocksToTransform) {
                ot.apply(cc.getBlock(blockToTransform), blockToTransform);
            }
        }

        /**
         * Appends operations into the block builder in the scope of the builder as an argument
         * to the given consumer.
         *
         * @param c the consumer.
         */
        // @@@ Is this needed?
        public void ops(Consumer<Builder> c) {
            c.accept(this);
        }

        /**
         * Appends an operation to this block, with no operation result name, and this builder's transformer.
         *
         * @param op the operation to append
         * @return the operation result of the appended operation
         * @throws IllegalStateException if the operation is structurally invalid
         * @see #op(Op, OpTransformer)
         */
        @Override
        public Op.Result apply(Op op) {
            return op(op, ot);
        }

        /**
         * Appends an operation to this block, with no operation result name, and this builder's transformer.
         *
         * @param op the operation to append
         * @return the operation result of the appended operation
         * @throws IllegalStateException if the operation is structurally invalid
         * @see #op(Op, OpTransformer)
         */
        public Op.Result op(Op op) {
            return op(op, ot);
        }

        /**
         * Appends an operation to this block.
         * <p>
         * If the operation is not bound to a block, then the operation is appended and bound to this block.
         * Otherwise, if the operation is bound, the operation is first
         * {@link Op#transform(CopyContext, OpTransformer) transformed} with this builder's context and the given
         * operation transformer, the unbound transformed operation is appended, and the operation's result is mapped
         * to the transformed operation's result (using the builder's context).
         * <p>
         * If the unbound operation (transformed, or otherwise) is structurally invalid then an
         * {@code IllegalStateException} is thrown. An unbound operation is structurally invalid if:
         * <ul>
         * <li>any of its bodies does not have the same ancestor body as this block's parent body.
         * <li>any of its operands (values) is not reachable from this block.
         * <li>any of its successors is not a sibling of this block.
         * <li>any of its successors arguments (values) is not reachable from this block.
         * </ul>
         * A value is reachable from this block if there is a path from this block's parent body,
         * via its ancestor bodies, to the value's block's parent body. (Note this structural check
         * ensures values are only used from the same tree being built, but it is weaker than a
         * dominance check that may be performed when the parent body is built.)
         *
         * @param op the operation to append
         * @param transformer the transformer to use when appending a bound operation
         * @return the operation result of the appended operation
         * @throws IllegalStateException if the operation is structurally invalid
         */
        public Op.Result op(Op op, OpTransformer transformer) {
            check();
            final Op.Result oprToTransform = op.result();

            Op transformedOp = op;
            if (oprToTransform != null) {
                // If operation is assigned to block, then copy it and transform its contents
                transformedOp = op.transform(cc, transformer);
                assert transformedOp.result == null;
            }

            Op.Result transformedOpr = insertOp(transformedOp);

            if (oprToTransform != null) {
                // Map the result of the first transformation
                // @@@ If the same operation is transformed more than once then subsequent
                //  transformed ops will not get implicitly mapped
                //  Should this be an error?
                cc.mapValueIfAbsent(oprToTransform, transformedOpr);
            }

            return transformedOpr;
        }

        /**
         * Returns true if this block builder is equal to the other object.
         * <p>This block builder is equal if the other object is an instance of a block builder, and they are
         * associated with the same block (but perhaps bound to different contexts and transformers).
         *
         * @param o the other object
         * @return true if this builder is equal to the other object.
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
    private Parameter appendBlockParameter(TypeDesc type) {
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
            throw new IllegalStateException("Operation cannot be appended, the block has a terminal operation");
        }

        for (Body b : op.bodies()) {
            if (b.ancestorBody != null && b.ancestorBody != this.parentBody) {
                throw new IllegalStateException("Body of operation is bound to a different ancestor body: ");
            }
        }

        for (Value v : op.operands()) {
            if (!isReachable(v)) {
                throw new IllegalStateException(
                        String.format("Operand of operation %s is not defined in tree: %s", op, v));
            }
            assert !v.isBound();
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
                assert !v.isBound();
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

    boolean isBound() {
        return index >= 0;
    }
}
