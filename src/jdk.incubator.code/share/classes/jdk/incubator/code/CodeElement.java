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
import java.util.stream.Gatherer;
import java.util.stream.Stream;

/**
 * A code element, one of {@link Body body}, {@link Block block}, or {@link Op operation}, is an element in a code
 * model.
 * <p>
 * Code elements form a tree. A code element may have a parent code element. A root code element is an operation,
 * a root operation, that has no parent element (a block). A code element and all its ancestors can be traversed,
 * up to and including the root operation.
 * <p>
 * A code element may have child code elements. A root code element and all its descendants can be
 * traversed, down to and including elements with no children. Bodies and blocks have at least one child element.
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
        return Stream.of(this).gather(() -> (_, e, downstream) -> traversePreOrder(e, downstream));
    }

    private static boolean traversePreOrder(CodeElement<?, ?> e, Gatherer.Downstream<? super CodeElement<?, ?>> d) {
        if (!d.push(e)) {
            return false;
        }
        for (CodeElement<?, ?> c : e.children()) {
            if (!traversePreOrder(c, d)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns the parent element, otherwise {@code null} if this element is an operation that has no parent block.
     *
     * @return the parent code element.
     * @throws IllegalStateException if this element is an operation whose parent block is unbuilt.
     */
    CodeElement<?, E> parent();

    // Nearest ancestors

    /**
     * Finds the nearest ancestor operation, otherwise {@code null} if there is no nearest ancestor.
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
     * Finds the nearest ancestor body, otherwise {@code null} if there is no nearest ancestor.
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
     * Finds the nearest ancestor block, otherwise {@code null} if there is no nearest ancestor.
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
     * @throws IllegalStateException if an operation with unbuilt parent block is encountered.
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
     * Returns the child code elements, as an unmodifiable list.
     *
     * @return the child code elements
     */
    List<C> children();

    /**
     * Compares two code elements by comparing their pre-order traversal positions in the code model.
     * <p>
     * The pre-order traversal position of a code element, {@code e} say, is equivalent to result of
     * the expression {@code root.elements().toList().indexOf(e)}, where {@code root} is the root
     * code element of the code model containing {@code e}.
     *
     * @param a the first code element to compare
     * @param b the second code element to compare
     * @return the value {@code 0} if {@code a == b}; {@code -1} if {@code a}'s pre-order traversal position
     * is less than {@code b}'s position; and {@code 1} if {@code a}'s pre-order traversal position
     * is greater than {@code b}'s position
     * @throws IllegalArgumentException if {@code a} and {@code b} are not present in the same code model
     * @throws IllegalStateException if an operation with partially built block is encountered.
     */
    static int compare(CodeElement<?, ?> a, CodeElement<?, ?> b) {
        if (a == b) {
            return 0;
        }

        // Find the common ancestor of a and b and the respective children
        int depthA = getDepth(a);
        int depthB = getDepth(b);

        CodeElement<?, ?> childA = a;
        CodeElement<?, ?> childB = b;
        while (depthA > depthB) {
            childA = a;
            a = a.parent();
            depthA--;
        }

        while (depthB > depthA) {
            childB = b;
            b = b.parent();
            depthB--;
        }

        while (a != b) {
            childA = a;
            a = a.parent();
            childB = b;
            b = b.parent();
        }

        if (a == null) {
            // No common ancestor, a and b are not in the same code model
            throw new IllegalArgumentException("Comparing code elements in different code models");
        } else if (a == childA) {
            // a is an ancestor of b
            return -1;
        } else if (a == childB) {
            // b is an ancestor of a
            return 1;
        } else {
            // a and b share a common ancestor
            List<?> children = a.children();
            return Integer.compare(children.indexOf(childA), children.indexOf(childB));
        }
    }

    private static int getDepth(CodeElement<?, ?> a) {
        int depth = 0;
        while (a.parent() != null) {
            a = a.parent();
            depth++;
        }
        return depth;
    }
}
