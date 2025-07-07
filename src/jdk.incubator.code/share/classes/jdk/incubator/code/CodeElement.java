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

package jdk.incubator.code;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.stream.Gatherer;
import java.util.stream.Stream;

/**
 * A code element, one of {@link Body body}, {@link Block block}, or {@link Op operation}.
 * <p>
 * A code may have a parent code element. An unbound code element is an operation, an unbound operation, that has no
 * parent block. An unbound operation may also be considered a root operation if never bound. A code element and all its
 * ancestors can be traversed, up to and including the unbound or root operation.
 * <p>
 * A code element may have child code elements, and so on. An unbound or root operation and all its descendants can be
 * traversed, down to and including operations with no children. Bodies and blocks have at least one child element.
 *
 * @param <E> the code element type
 * @param <C> the child code element type.
 * @sealedGraph
 */
// @@@ E may not be needed
public sealed interface CodeElement<
        E extends CodeElement<E, C>,
        C extends CodeElement<C, ?>>
        extends CodeItem
        permits Body, Block, Op {

    /**
     * {@return a stream of code elements sorted topologically in pre-order traversal.}
     */
    default Stream<CodeElement<?, ?>> elements() {
        return Stream.of(Void.class).gather(() -> (_, _, downstream) -> traversePreOrder(downstream));
    }

    private boolean traversePreOrder(Gatherer.Downstream<? super CodeElement<?, ?>> v) {
        if (!v.push(this)) {
            return false;
        }
        for (C c : children()) {
            if (!((CodeElement<?, ?>) c).traversePreOrder(v)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Traverses this code element and any descendant code elements.
     * <p>
     * Traversal is performed in pre-order, reporting each code element to the visitor.
     *
     * @param t   the traversing accumulator
     * @param v   the code element visitor
     * @param <T> accumulator type
     * @return the traversing accumulator
     */
    default <T> T traverse(T t, BiFunction<T, CodeElement<?, ?>, T> v) {
        t = v.apply(t, this);
        for (C r : children()) {
            t = r.traverse(t, v);
        }

        return t;
    }

    /**
     * Creates a visiting function for bodies.
     *
     * @param v   the body visitor
     * @param <T> accumulator type
     * @return the visiting function for bodies
     */
    static <T> BiFunction<T, CodeElement<?, ?>, T> bodyVisitor(BiFunction<T, Body, T> v) {
        return (t, e) -> e instanceof Body f
                ? v.apply(t, f)
                : t;
    }

    /**
     * Creates a visiting function for blocks.
     *
     * @param v   the block visitor
     * @param <T> accumulator type
     * @return the visiting function for blocks
     */
    static <T> BiFunction<T, CodeElement<?, ?>, T> blockVisitor(BiFunction<T, Block, T> v) {
        return (t, e) -> e instanceof Block f
                ? v.apply(t, f)
                : t;
    }

    /**
     * Creates a visiting function for operations.
     *
     * @param v   the operation visitor
     * @param <T> accumulator type
     * @return the visiting function for operations
     */
    static <T> BiFunction<T, CodeElement<?, ?>, T> opVisitor(BiFunction<T, Op, T> v) {
        return (t, e) -> e instanceof Op f
                ? v.apply(t, f)
                : t;
    }

    /**
     * Returns the parent element, otherwise {@code null}
     * if there is no parent.
     *
     * @return the parent code element.
     * @throws IllegalStateException if this element is an operation whose parent block is unbuilt.
     */
    CodeElement<?, E> parent();

    // Nearest ancestors

    /**
     * Finds the nearest ancestor operation, otherwise {@code null}
     * if there is no nearest ancestor.
     *
     * @return the nearest ancestor operation.
     * @throws IllegalStateException if an operation with unbuilt parent block is encountered.
     */
    default Op ancestorOp() {
        return switch (this) {
            // block -> body -> op~
            case Block block -> block.parent().parent();
            // body -> op~
            case Body body -> body.parent();
            // op -> block? -> body -> op~
            case Op op -> {
                // Throws ISE if op is not bound
                Block parent = op.parent();
                yield parent == null ? null : parent.parent().parent();
            }
        };
    }

    /**
     * Finds the nearest ancestor body, otherwise {@code null}
     * if there is no nearest ancestor.
     *
     * @return the nearest ancestor body.
     * @throws IllegalStateException if an operation with unbuilt parent block is encountered.
     */
    default Body ancestorBody() {
        return switch (this) {
            // block -> body
            case Block block -> block.parent();
            // body -> op~ -> block? -> body
            case Body body -> {
                // Throws ISE if block is partially constructed
                Block ancestor = body.parent().parent();
                yield ancestor == null ? null : ancestor.parent();
            }
            // op~ -> block? -> body
            case Op op -> {
                // Throws ISE if op is not bound
                Block parent = op.parent();
                yield parent == null ? null : parent.parent();
            }
        };
    }

    /**
     * Finds the nearest ancestor block, otherwise {@code null}
     * if there is no nearest ancestor.
     *
     * @return the nearest ancestor block.
     * @throws IllegalStateException if an operation with unbuilt parent block is encountered.
     */
    default Block ancestorBlock() {
        return switch (this) {
            // block -> body -> op~ -> block?
            // Throws ISE if op is not bound
            case Block block -> block.parent().parent().parent();
            // body -> op~ -> block?
            // Throws ISE if op is not bound
            case Body body -> body.parent().parent();
            // op~ -> block?
            // Throws ISE if op is not bound
            case Op op -> op.parent();
        };
    }

    /**
     * Returns true if this element is an ancestor of the descendant element.
     *
     * @param descendant the descendant element.
     * @return true if this element is an ancestor of the descendant element.
     */
    default boolean isAncestorOf(CodeElement<?, ?> descendant) {
        Objects.requireNonNull(descendant);

        CodeElement<?, ?> e = descendant.parent();
        while (e != null && e != this) {
            e = e.parent();
        }
        return e != null;
    }

    /**
     * Finds the child of this element that is an ancestor of the given descendant element,
     * otherwise returns the descendant element if a child of this element, otherwise
     * returns {@code null} if there is no such child.
     *
     * @param descendant the descendant element
     * @return the child that is an ancestor of the given descendant element, otherwise the descendant
     * element if a child of this element, otherwise {@code null}.
     * @throws IllegalStateException if an operation with unbuilt parent block is encountered.
     */
    default C findChildAncestor(CodeElement<?, ?> descendant) {
        Objects.requireNonNull(descendant);

        CodeElement<?, ?> e = descendant;
        while (e != null && e.parent() != this) {
            e = e.parent();
        }

        @SuppressWarnings("unchecked")
        C child = (C) e;
        return child;
    }

    /**
     * Returns the child code elements, as an unmodifiable list.
     *
     * @return the child code elements
     */
    List<C> children();

    // Siblings
    // Left, right

    // Des
}
