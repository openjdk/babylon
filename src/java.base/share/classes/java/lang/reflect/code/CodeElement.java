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

import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * A code element, one of {@link Body body}, {@link Block block}, or {@link Op operation}.
 * <p>
 * A code element may have child code elements, and so on, to form a tree. A (root) code element and all its descendants
 * can be traversed.
 *
 * @param <E> the code element type
 * @param <C> the child code element type.
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
    // Code copied into the compiler cannot depend on new gatherer API
    default Stream<CodeElement<?, ?>> elements() {
/*__throw new UnsupportedOperationException();__*/        return Stream.of(Void.class).gather(() -> (_, _, downstream) -> traversePreOrder(downstream::push));
    }

//    private boolean traversePreOrder(Gatherer.Downstream<? super CodeElement<?, ?>> v) {
    private boolean traversePreOrder(Predicate<? super CodeElement<?, ?>> v) {
        if (!v.test(this)) {
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
     * Returns the parent code element.
     * <p>
     * If this element is an instance of {@code Op} then the parent may be {@code null}
     * if operation is not assigned to a block.
     *
     * @return the parent code element
     */
    CodeElement<?, E> parent();

    /**
     * Returns the child code elements, as an unmodifiable list.
     *
     * @return the child code elements
     */
    List<C> children();
}
