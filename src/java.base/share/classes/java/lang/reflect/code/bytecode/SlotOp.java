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

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.JavaType;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class SlotOp extends ExternalizableOp {
    public static final String ATTRIBUTE_SLOT = "slot";

    final int slot;

    protected SlotOp(SlotOp that, CopyContext cc) {
        super(that, cc);
        this.slot = that.slot;
    }

    protected SlotOp(String name, List<? extends Value> operands, int slot) {
        super(name, operands);
        this.slot = slot;
    }

    protected SlotOp(ExternalizedOp def) {
        super(def);

        this.slot = def.extractAttributeValue(ATTRIBUTE_SLOT, true,
                v -> switch (v) {
                    case String s -> Integer.parseInt(s);
                    case Integer i -> i;
                    default -> throw new UnsupportedOperationException("Unsupported slot value:" + v);
                });
    }

    public int slot() {
        return slot;
    }

    @Override
    public Map<String, Object> attributes() {
        HashMap<String, Object> m = new HashMap<>(super.attributes());
        m.put("", slot);
        return Collections.unmodifiableMap(m);
    }

    @OpFactory.OpDeclaration(SlotLoadOp.NAME)
    public static final class SlotLoadOp extends SlotOp {
        public static final String NAME = "slot.load";

        final TypeElement resultType;

        public SlotLoadOp(ExternalizedOp opdef) {
            super(opdef);

            if (opdef.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand");
            }

            this.resultType = opdef.resultType();
        }

        SlotLoadOp(SlotLoadOp that, CopyContext cc) {
            super(that, cc);
            this.resultType = that.resultType;
        }

        @Override
        public SlotLoadOp transform(CopyContext cc, OpTransformer ot) {
            return new SlotLoadOp(this, cc);
        }

        SlotLoadOp(int slot, TypeElement resultType) {
            super(NAME, List.of(), slot);
            this.resultType = resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    @OpFactory.OpDeclaration(SlotStoreOp.NAME)
    public static final class SlotStoreOp extends SlotOp {
        public static final String NAME = "slot.store";

        public SlotStoreOp(ExternalizedOp opdef) {
            super(opdef);

            if (opdef.operands().size() != 2) {
                throw new IllegalArgumentException("Operation must have two operands");
            }
        }

        SlotStoreOp(SlotStoreOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SlotStoreOp transform(CopyContext cc, OpTransformer ot) {
            return new SlotStoreOp(this, cc);
        }

        SlotStoreOp(int slot, Value v) {
            super(NAME, List.of(v), slot);
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    public static SlotLoadOp load(int slot) {
        return load(slot, JavaType.J_L_OBJECT);
    }

    public static SlotLoadOp load(int slot, TypeElement resultType) {
        return new SlotLoadOp(slot, resultType);
    }

    public static SlotStoreOp store(int slot, Value v) {
        return new SlotStoreOp(slot, v);
    }
}
