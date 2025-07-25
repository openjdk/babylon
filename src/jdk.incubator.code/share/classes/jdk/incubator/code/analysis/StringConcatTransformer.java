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

package jdk.incubator.code.analysis;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.*;

import java.util.*;

/**
 * StringConcatTransformer is an {@link jdk.incubator.code.OpTransformer} that removes concatenation operations
 * from blocks and replaces them with equivalent {@link java.lang.StringBuilder} operations. This provides a pathway
 * to remove {@link jdk.incubator.code.dialect.java.JavaOp.ConcatOp} for easier lowering to Bytecode.
 */
public final class StringConcatTransformer implements OpTransformer {

    private static final JavaType J_L_STRING_BUILDER = JavaType.type(StringBuilder.class);
    private static final MethodRef SB_TO_STRING_REF = MethodRef.method(
            J_L_STRING_BUILDER, "toString", JavaType.J_L_STRING);

    public StringConcatTransformer() {}

    @Override
    public Block.Builder apply(Block.Builder block, Op op) {
        switch (op) {
            case JavaOp.ConcatOp cz when isRootConcat(cz) -> {
                // Create a string builder and build by traversing tree of operands
                Op.Result builder = block.op(JavaOp.new_(ConstructorRef.constructor(J_L_STRING_BUILDER)));
                buildFromTree(block, builder, cz);
                // Convert to string
                Value s = block.op(JavaOp.invoke(SB_TO_STRING_REF, builder));
                block.context().mapValue(cz.result(), s);
            }
            case JavaOp.ConcatOp _ -> {
                // Process later when building from root concat
            }
            default -> block.op(op);
        }
        return block;
    }

    static boolean isRootConcat(JavaOp.ConcatOp cz) {
        // Root of concat tree, zero uses, two or more uses,
        // or one use that is not a subsequent concat op
        Set<Op.Result> uses = cz.result().uses();
        return uses.size() != 1 || !(uses.iterator().next().op() instanceof JavaOp.ConcatOp);
    }

    static void buildFromTree(Block.Builder block, Op.Result builder, JavaOp.ConcatOp cz) {
        // Process concat op's operands from left to right
        buildFromTree(block, builder, cz.operands().get(0));
        buildFromTree(block, builder, cz.operands().get(1));
    }

    static void buildFromTree(Block.Builder block, Op.Result builder, Value v) {
        if (v instanceof Op.Result r &&
                r.op() instanceof JavaOp.ConcatOp cz &&
                r.uses().size() == 1) {
            // Node of tree, recursively traverse the operands
            buildFromTree(block, builder, cz);
        } else {
            // Leaf of tree, append value to builder
            // Note leaf can be the result of a ConcatOp with multiple uses
            block.op(append(block, builder, block.context().getValue(v)));
        }
    }

    private static Op append(Block.Builder block, Value builder, Value arg) {
        // Check if we need to widen unsupported integer types in the StringBuilder API
        // Strings are fed in as-is, everything else given as an Object.
        TypeElement type = arg.type();
        if (type instanceof PrimitiveType) {
            //Widen Short and Byte to Int.
            if (type.equals(JavaType.BYTE) || type.equals(JavaType.SHORT)) {
                arg = block.op(JavaOp.conv(JavaType.INT, arg));
                type = JavaType.INT;
            }
        } else if (!type.equals(JavaType.J_L_STRING)){
            type = JavaType.J_L_OBJECT;
        }

        MethodRef methodDesc = MethodRef.method(J_L_STRING_BUILDER, "append", J_L_STRING_BUILDER, type);
        return JavaOp.invoke(methodDesc, builder, arg);
    }


}