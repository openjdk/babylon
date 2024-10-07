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
package java.lang.reflect.code.bytecode;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.util.List;

final class NullTypeResolver implements OpTransformer {

    static final TypeElement NULL_TYPE = new TypeElement() {

        private static final TypeElement.ExternalizedTypeElement EXT_TYPE = new TypeElement.ExternalizedTypeElement("NULL", List.of());

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return EXT_TYPE;
        }
    };

    @Override
    public Block.Builder apply(Block.Builder block, Op op) {
        // @@@ null type may appear also in block parameters
        if (op instanceof CoreOp.ConstantOp && op.resultType() == NULL_TYPE) {
           block.context().mapValue(op.result(), block.op(CoreOp.constant(pullReferenceTypeFromUses(op.result()), null)));
        } else {
            block.op(op);
        }
        return block;
    }

    private static TypeElement pullReferenceTypeFromUses(Op.Result r) {
        for (Op.Result u : r.uses()) {
            // Pull block parameter type when used as block argument
            for (Block.Reference sr : u.op().successors()) {
                int i = sr.arguments().indexOf(r);
                if (i >= 0) {
                    TypeElement bpt = sr.targetBlock().parameters().get(i).type();
                    if (bpt != NULL_TYPE) {
                        return bpt;
                    }
                }
            }
            // @@@ Pull type from specific ops when used as operand
        }
        return JavaType.J_L_OBJECT;
    }
}
