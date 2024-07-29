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
package hat.backend;

public class PTXRegister {
    private String name;
    private final Type type;

    public enum Type {
        S8 (8, BasicType.SIGNED, "s8", "%s"),
        S16 (16, BasicType.SIGNED, "s16", "%s"),
        S32 (32, BasicType.SIGNED, "s32", "%s"),
        S64 (64, BasicType.SIGNED, "s64", "%sd"),
        U8 (8, BasicType.UNSIGNED, "u8", "%r"),
        U16 (16, BasicType.UNSIGNED, "u16", "%r"),
        U32 (32, BasicType.UNSIGNED, "u32", "%r"),
        U64 (64, BasicType.UNSIGNED, "u64", "%rd"),
        F16 (16, BasicType.FLOATING, "f16", "%f"),
        F16X2 (16, BasicType.FLOATING, "f16", "%f"),
        F32 (32, BasicType.FLOATING, "f32", "%f"),
        F64 (64, BasicType.FLOATING, "f64", "%fd"),
        B8 (8, BasicType.BIT, "b8", "%b"),
        B16 (16, BasicType.BIT, "b16", "%b"),
        B32 (32, BasicType.BIT, "b32", "%b"),
        B64 (64, BasicType.BIT, "b64", "%bd"),
        B128 (128, BasicType.BIT, "b128", "%b"),
        PREDICATE (1, BasicType.PREDICATE, "pred", "%p");

        public enum BasicType {
            SIGNED,
            UNSIGNED,
            FLOATING,
            BIT,
            PREDICATE
        }

        private final int size;
        private final BasicType basicType;
        private final String name;
        private final String regPrefix;

        Type(int size, BasicType type, String name, String regPrefix) {
            this.size = size;
            this.basicType = type;
            this.name = name;
            this.regPrefix = regPrefix;
        }

        public int getSize() {
            return this.size;
        }

        public BasicType getBasicType() {
            return this.basicType;
        }

        public String getName() {
            return this.name;
        }

        public String getRegPrefix() {
            return this.regPrefix;
        }
    }

    public PTXRegister(int num, Type type) {
        this.type = type;
        this.name = type.regPrefix + num;
    }

    public String name() {
        return this.name;
    }

    public void name(String name) {
        this.name = name;
    }

    public Type type() {
        return this.type;
    }
}
