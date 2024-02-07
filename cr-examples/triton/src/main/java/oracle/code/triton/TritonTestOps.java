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

package oracle.code.triton;

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.lang.reflect.code.op.OpDeclaration;
import java.lang.reflect.code.op.OpDefinition;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.op.OpWithDefinition;
import java.util.List;

public class TritonTestOps {

    @OpDeclaration(ConsumeOp.NAME)
    public static class ConsumeOp extends OpWithDefinition {
        public static final String NAME = "tt.consume";

        public ConsumeOp(OpDefinition def) {
            super(def);
        }

        ConsumeOp(ConsumeOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ConsumeOp transform(CopyContext cc, OpTransformer ot) {
            return new ConsumeOp(this, cc);
        }

        ConsumeOp(List<Value> values) {
            super(NAME, values);
        }

        @Override
        public TypeDesc resultType() {
            return TypeDesc.VOID;
        }
    }


    public static final OpFactory FACTORY = OpFactory.OP_FACTORY.get(TritonTestOps.class);

    public static ConsumeOp consume(Value... operands) {
        return consume(List.of(operands));
    }

    public static ConsumeOp consume(List<Value> operands) {
        return new ConsumeOp(operands);
    }
}
