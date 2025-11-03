/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code.internal;

import java.lang.annotation.*;

/**
 * Annotation for embedding a code reflection model alongside a Java method.
 */
@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface CodeModel {

    /**
     * The root operation associated with the method.
     */
    Op funcOp();

    /**
     * The collection of bodies referenced by operations and blocks in this model. The indexing of these entries is
     * local to this array.
     */
    Body[] bodies();

    /**
     * Describes an operation in the code reflection model. An operation may reference operands, attributes, successors,
     * and types.
     *
     * @see jdk.incubator.code.Op
     */
    @interface Op {

        /**
         * Operation name or opcode identifier.
         *
         * @see jdk.incubator.code.Op#externalizeOpName()
         */
        String name();

        /**
         * Operand references, expressed as indices local to the current context (previously produced
         * results or block parameters).
         *
         * @see jdk.incubator.code.Op#operands()
         */
        int[] operands() default {};

        /**
         * Control-flow successors for this operation, if any (e.g., branch targets). Each successor references a block
         * and optional argument indices.
         *
         * @see jdk.incubator.code.Op#successors()
         */
        BlockReference[] successors() default {};

        /**
         * Result type for this operation, if applicable.
         *
         * @see jdk.incubator.code.Op#resultType()
         */
        String resultType() default "";

        /**
         * Optional default attribute value for this operation.
         *
         * @see jdk.incubator.code.Op#externalize()
         */
        String defaultAttribute() default "";

        /**
         * Optional reference to source file information.
         *
         * @see jdk.incubator.code.Op#location()
         * @see jdk.incubator.code.Location#sourceRef()
         */
        String sourceRef() default "";

        /**
         * Optional source location coordinates (e.g., line/column pairs).
         *
         * @see jdk.incubator.code.Op#location()
         * @see jdk.incubator.code.Location#line()
         * @see jdk.incubator.code.Location#column()
         */
        int[] location() default {};

        /**
         * Additional operation attributes as string pairs (key, value).
         *
         * @see jdk.incubator.code.Op#externalize()
         */
        String[] attributes() default {};

        /**
         * References to body definitions that belong to this operation. Indices are relative to the
         * {@link CodeModel#bodies()} array.
         *
         * @see jdk.incubator.code.Op#bodies()
         */
        int[] bodyDefinitions() default {};
    }

    /**
     * A body groups one or more blocks and may declares a yield type.
     *
     * @see jdk.incubator.code.Body
     */
    @interface Body {

        /**
         * Type produced by this body.
         *
         * @see jdk.incubator.code.Body#yieldType()
         */
        String yieldType();

        /**
         * Blocks contained within this body. Block indices are local to this array.
         *
         * @see jdk.incubator.code.Body#blocks()
         */
        Block[] blocks();
    }

    /**
     * A basic block consisting of optional parameters and a sequence of operations.
     *
     * @see jdk.incubator.code.Block
     */
    @interface Block {

        /**
         * Types of block parameters, if any.
         *
         * @see jdk.incubator.code.Block#parameterTypes()
         */
        String[] paramTypes() default {};

        /**
         * Operations contained in this block, evaluated in order.
         *
         * @see jdk.incubator.code.Block#ops()
         */
        Op[] ops();
    }

    /**
     * A reference to a block and the argument indices provided to it.
     *
     * @see jdk.incubator.code.Block.Reference
     */
    @interface BlockReference {

        /**
         * Index of the target block within the enclosing {@link Body#blocks()} array.
         *
         * @see jdk.incubator.code.Block.Reference#targetBlock()
         */
        int targetBlock();

        /**
         * Indices of the arguments passed to the target block’s parameters. Argument indices are local to the
         * referencing context.
         *
         * @see jdk.incubator.code.Block.Reference#arguments()
         */
        int[] arguments() default {};
    }
}
