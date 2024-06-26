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

package java.lang.reflect.code.analysis;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.*;

/**
 * StringConcatTransformer is an {@link java.lang.reflect.code.OpTransformer} that removes concatenation operations
 * from blocks and replaces them with equivalent {@link java.lang.StringBuilder} operations. This provides a pathway
 * to remove {@link java.lang.reflect.code.op.CoreOp.ConcatOp} for easier lowering to Bytecode.
 */
public class StringConcatTransformer implements OpTransformer {

    private static final JavaType J_L_OBJECT = JavaType.type(Object.class);
    private static final JavaType J_L_STRING_BUILDER = JavaType.type(StringBuilder.class);

    private static final MethodRef SB_TO_STRING_REF = MethodRef.method(
            J_L_STRING_BUILDER, "toString", JavaType.J_L_STRING);

    public StringConcatTransformer() {}

    @Override
    public Block.Builder apply(Block.Builder block, Op op) {
        switch (op) {
            case CoreOp.ConcatOp cz when isRootConcat(cz) -> {
                // Create a string builder and build by traversing tree of operands
                Op.Result builder = block.apply(CoreOp._new(FunctionType.functionType(J_L_STRING_BUILDER)));
                buildFromTree(block, builder, cz);
                // Convert to string
                Value s = block.op(CoreOp.invoke(SB_TO_STRING_REF, builder));
                block.context().mapValue(cz.result(), s);
            }
            case CoreOp.ConcatOp cz -> {
                // Process later when building from root concat
            }
            default -> block.op(op);
        }
        return block;
    }

    static boolean isRootConcat(CoreOp.ConcatOp cz) {
        // Root of concat tree, zero uses, two or more uses,
        // or one use that is not a subsequent concat op
        Set<Op.Result> uses = cz.result().uses();
        return uses.size() != 1 || !(uses.iterator().next().op() instanceof CoreOp.ConcatOp);
    }

    static void buildFromTree(Block.Builder block, Op.Result builder, CoreOp.ConcatOp cz) {
        // Process concat op's operands from left to right
        buildFromTree(block, builder, cz.operands().get(0));
        buildFromTree(block, builder, cz.operands().get(1));
    }

    static void buildFromTree(Block.Builder block, Op.Result builder, Value v) {
        if (v instanceof Op.Result r &&
                r.op() instanceof CoreOp.ConcatOp cz &&
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
        return append(block, builder, arg, arg.type());
    }

    private static Op append(Block.Builder block, Value builder, Value arg, TypeElement type) {
        // Check if we need to widen unsupported integer types in the StringBuilder API
        // Strings are fed in as-is, everything else given as an Object.
        if (type instanceof PrimitiveType) {
            if (List.of(JavaType.BYTE, JavaType.SHORT).contains(type)) {
                Value widened = block.op(CoreOp.conv(JavaType.INT, arg));
                MethodRef methodDesc = MethodRef.method(J_L_STRING_BUILDER, "append", J_L_STRING_BUILDER, JavaType.INT);
                return CoreOp.invoke(methodDesc, builder, widened);
            } else {
                MethodRef methodDesc = MethodRef.method(J_L_STRING_BUILDER, "append", J_L_STRING_BUILDER, type);
                return CoreOp.invoke(methodDesc, builder, arg);
            }
        } else if (type.equals(JavaType.J_L_STRING)) {
            MethodRef methodDesc = MethodRef.method(J_L_STRING_BUILDER, "append", J_L_STRING_BUILDER, type);
            return CoreOp.invoke(methodDesc, builder, arg);
        } else {
            MethodRef methodDesc = MethodRef.method(J_L_STRING_BUILDER, "append", J_L_STRING_BUILDER, J_L_OBJECT);
            return CoreOp.invoke(methodDesc, builder, arg);
        }
    }

}