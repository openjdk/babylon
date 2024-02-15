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

import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.StringWriter;
import java.io.Writer;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.writer.OpWriter;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.BiFunction;

/**
 * An operation modelling a unit of functionality.
 * <p>
 * An operation might model the addition of two 32-bit integers, or a Java method call.
 * Alternatively an operation may model something more complex like method bodies, lambda bodies, or
 * try/catch/finally statements. In this case such an operation will contain one or more bodies modelling
 * the nested structure.
 */
public non-sealed abstract class Op implements CodeElement<Op, Body> {

    /**
     * An operation characteristic indicating the operation is pure and has no side effects.
     */
    public interface Pure {
    }

    /**
     * An operation characteristic indicating the operation has one or more bodies.
     */
    public interface Nested {
        List<Body> bodies();
    }

    /**
     * An operation characteristic indicating the operation represents a loop
     */
    public interface Loop extends Nested {
        Body loopBody();
    }

    /**
     * An operation characteristic indicating the operation has one or more bodies,
     * all of which are isolated.
     */
    public interface Isolated extends Nested {
    }

    /**
     * An operation characteristic indicating the operation is invokable, so the operation may be interpreted
     * or compiled.
     */
    public interface Invokable extends Nested {
        /**
         * @return the body of the invokable operation.
         */
        Body body();

        /**
         * @return the function type describing the invokable operation's parameter types and return type.
         */
        FunctionType invokableType();
    }

    /**
     * An operation characteristic indicating the operation can replace itself with a lowered form,
     * consisting only of operations in the core dialect.
     */
    public interface Lowerable {
        default Block.Builder lower(Block.Builder b) {
            return lower(b, OpTransformer.NOOP_TRANSFORMER);
        }

        Block.Builder lower(Block.Builder b, OpTransformer opT);
    }

    /**
     * An operation characteristic indicating the operation is a terminating operation
     * occurring as the last operation in a block.
     * <p>
     * A terminating operation passes control to either another block within the same parent body
     * or to that parent body.
     */
    public interface Terminating {
    }

    /**
     * An operation characteristic indicating the operation is a body terminating operation
     * occurring as the last operation in a block.
     * <p>
     * A body terminating operation passes control back to its nearest ancestor body.
     */
    public interface BodyTerminating extends Terminating {
    }

    /**
     * An operation characteristic indicating the operation is a block terminating operation
     * occurring as the last operation in a block.
     * <p>
     * The operation has one or more successors to other blocks within the same parent body, and passes
     * control to one of those blocks.
     */
    public interface BlockTerminating extends Terminating {
        List<Block.Reference> successors();
    }

    /**
     * A value that is the result of an operation.
     */
    public static final class Result extends Value {
        final Op op;

        Result(Block block, Op op) {
            super(block, op.resultType());

            this.op = op;
        }

        @Override
        public Set<Value> dependsOn() {
            Set<Value> depends = new LinkedHashSet<>(op.operands());
            if (op instanceof Terminating) {
                op.successors().stream().flatMap(h -> h.arguments().stream()).forEach(depends::add);
            }

            return Collections.unmodifiableSet(depends);
        }

        /**
         * Returns the result's operation.
         *
         * @return the result's operation.
         */
        public Op op() {
            return op;
        }
    }

    // Set when op is bound to block, otherwise null when unbound
    Result result;

    final String name;

    final List<Value> operands;

    /**
     * Constructs an operation by copying given operation.
     *
     * @param that the operation to copy.
     * @param cc   the copy context.
     * @implSpec The default implementation calls the constructor with the operation's name, result type, and a list
     * values computed, in order, by mapping the operation's operands using the copy context.
     */
    protected Op(Op that, CopyContext cc) {
        this(that.name, cc.getValues(that.operands));
    }

    /**
     * Copies this operation and its bodies, if any.
     * <p>
     * The returned operation is structurally identical to this operation and is otherwise independent
     * of the values declared and used.
     *
     * @return the copied operation.
     */
    public Op copy() {
        return transform(CopyContext.create(), OpTransformer.COPYING_TRANSFORMER);
    }

    /**
     * Copies this operation and its bodies, if any.
     * <p>
     * The returned operation is structurally identical to this operation and is otherwise independent
     * of the values declared and used.
     *
     * @param cc the copy context.
     * @return the copied operation.
     */
    public Op copy(CopyContext cc) {
        return transform(cc, OpTransformer.COPYING_TRANSFORMER);
    }

    /**
     * Copies this operation and transforms its bodies, if any.
     * <p>
     * Bodies are {@link Body#transform(CopyContext, OpTransformer) transformed} with the given copy context and
     * operation transformer.
     *
     * @param cc the copy context.
     * @param ot the operation transformer.
     * @return the transformed operation.
     */
    public abstract Op transform(CopyContext cc, OpTransformer ot);

    /**
     * Constructs an operation with a name and list of operands.
     *
     * @param name       the operation name.
     * @param operands   the list of operands, a copy of the list is performed if required.
     */
    protected Op(String name, List<? extends Value> operands) {
        this.name = name;
        this.operands = List.copyOf(operands);
    }

    /**
     * Returns the operation's result, otherwise {@code null} if the operation is not assigned to a block.
     *
     * @return the operation's result, or {@code null} if not assigned to a block.
     */
    public final Result result() {
        return result;
    }

    /**
     * Returns this operation's parent block, otherwise {@code null} if the operation is not assigned to a block.
     *
     * @return operation's parent block, or {@code null} if the operation is not assigned to a block.
     */
    public final Block parentBlock() {
        if (result == null) {
            return null;
        }

        if (!result.block.isBound()) {
            throw new IllegalStateException("Parent block is partially constructed");
        }

        return result.block;
    }

    /**
     * Returns this operation's nearest ancestor body (the parent body of this operation's parent block),
     * otherwise {@code null} if the operation is not assigned to a block.
     *
     * @return operation's nearest ancestor body, or {@code null} if the operation is not assigned to a block.
     */
    public final Body ancestorBody() {
        if (result == null) {
            return null;
        }

        if (!result.block.isBound()) {
            throw new IllegalStateException("Parent body is partially constructed");
        }

        return result.block.parentBody;
    }

    /**
     * {@return the operation name}
     */
    public String opName() {
        return name;
    }

    /**
     * {@return the operation's operands, as an unmodifiable list}
     */
    public List<Value> operands() {
        return operands;
    }

    /**
     * {@return the operation's successors, as an unmodifiable list}
     */
    public List<Block.Reference> successors() {
        return List.of();
    }

    /**
     * The attribute value that represents null.
     */
    public static final Object NULL_ATTRIBUTE_VALUE = new Object();

    /**
     * Returns the operation's attributes.
     *
     * <p>A null attribute value is represented by the constant value {@link #NULL_ATTRIBUTE_VALUE}.
     *
     * @return the operation's attributes, as an unmodifiable map
     */
    public Map<String, Object> attributes() {
        return Map.of();
    }

    /**
     * {@return the operation's result type}
     */
    public abstract TypeElement resultType();

    /**
     * Returns the operation's function type.
     * <p>
     * The function type's result type is the operation's result type and the function type's parameter types are the
     * operation's operand types, in order.
     *
     * @return the function type
     */
    public FunctionType opType() {
        List<TypeElement> operandTypes = operands.stream().map(Value::type).toList();
        return FunctionType.functionType(resultType(), operandTypes);
    }

    /**
     * {@return the operation's bodies, as an unmodifiable list}
     * @implSpec this implementation returns an unmodifiable empty list.
     */
    public List<Body> bodies() {
        return List.of();
    }

    @Override
    public final List<Body> children() {
        return bodies();
    }

    /**
     * Traverse the operands of this operation that are the results of prior operations, recursively.
     * <p>
     * Traversal is performed in pre-order, reporting the operation of each operand to the visitor.
     *
     * @param t   the traversing accumulator
     * @param v   the visitor
     * @param <T> accumulator type
     * @return the traversing accumulator
     * @apiNote A visitor that implements the abstract method of {@code OpVisitor} and does not override any
     * other default method will only visit operations. As such a lambda expression or method reference
     * may be used to visit operations.
     */
    public final <T> T traverseOperands(T t, BiFunction<T, Op, T> v) {
        for (Value arg : operands()) {
            if (arg instanceof Result or) {
                t = v.apply(t, or.op);
                t = or.op.traverseOperands(t, v);
            }
        }

        return t;
    }

    /**
     * Writes the textual form of this operation to the given output stream, using the UTF-8 character set.
     *
     * @param out the stream to write to.
     */
    public void writeTo(OutputStream out) {
        writeTo(new OutputStreamWriter(out, StandardCharsets.UTF_8));
    }

    /**
     * Writes the textual form of this operation to the given writer.
     *
     * @param w the writer to write to.
     */
    public void writeTo(Writer w) {
        OpWriter.writeTo(w, this);
    }

    /**
     * Returns the textual form of this operation.
     *
     * @return the textual form of this operation.
     */
    public String toText() {
        StringWriter w = new StringWriter();
        writeTo(w);
        return w.toString();
    }
}
