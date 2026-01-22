/*
 * Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.dialect;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.ops.Precedence.LoadOrConv;

import java.util.List;

public abstract sealed class HATThreadOp extends HATOp implements Dim, LoadOrConv {

    protected HATThreadOp(List<Value> operands) {
        super(operands);
    }

    protected HATThreadOp(HATThreadOp that, CodeContext cc) {
        super(that, cc);
    }

    @Override
    public final TypeElement resultType() {
        return JavaType.INT;
    }

    public static HATThreadOp create(String name) {
        return switch (name) {
            case "gix" -> new HATThreadOp.HAT_GI.HAT_GIX();
            case "giy" -> new HATThreadOp.HAT_GI.HAT_GIY();
            case "giz" -> new HATThreadOp.HAT_GI.HAT_GIZ();
            case "gsx" -> new HATThreadOp.HAT_GS.HAT_GSX();
            case "gsy" -> new HATThreadOp.HAT_GS.HAT_GSY();
            case "gsz" -> new HATThreadOp.HAT_GS.HAT_GSZ();
            case "lix" -> new HATThreadOp.HAT_LI.HAT_LIX();
            case "liy" -> new HATThreadOp.HAT_LI.HAT_LIY();
            case "liz" -> new HATThreadOp.HAT_LI.HAT_LIZ();
            case "lsx" -> new HATThreadOp.HAT_LS.HAT_LSX();
            case "lsy" -> new HATThreadOp.HAT_LS.HAT_LSY();
            case "lsz" -> new HATThreadOp.HAT_LS.HAT_LSZ();
            case "bix" -> new HATThreadOp.HAT_BI.HAT_BIX();
            case "biy" -> new HATThreadOp.HAT_BI.HAT_BIY();
            case "biz" -> new HATThreadOp.HAT_BI.HAT_BIZ();
            case "bsx" -> new HATThreadOp.HAT_BS.HAT_BSX();
            case "bsy" -> new HATThreadOp.HAT_BS.HAT_BSY();
            case "bsz" -> new HATThreadOp.HAT_BS.HAT_BSZ();
            default -> throw new RuntimeException("[ERROR] Illegal/unsupported parallel construct: " + name);
        };
    }

    public abstract static sealed class HAT_LI extends HATThreadOp {

        protected HAT_LI() {
            super(List.of());
        }

        protected HAT_LI(HAT_LI op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HAT_LIX extends HAT_LI implements Dim.X {
            public HAT_LIX(HAT_LIX op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_LIX() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_LIX(this, copyContext);
            }
        }

        public static final class HAT_LIY extends HAT_LI implements Dim.Y {
            public HAT_LIY(HAT_LIY op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_LIY() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_LIY(this, copyContext);
            }
        }

        public static final class HAT_LIZ extends HAT_LI implements Dim.Z {
            public HAT_LIZ(HAT_LI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_LIZ() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_LIZ(this, copyContext);
            }
        }
    }

    public abstract static sealed class HAT_BI extends HATThreadOp {
        protected HAT_BI() {
            super(List.of());
        }

        protected HAT_BI(HAT_BI op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HAT_BIX extends HAT_BI implements Dim.X {
            public HAT_BIX(HAT_BI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_BIX() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_BIX(this, copyContext);
            }
        }

        public static final class HAT_BIY extends HAT_BI implements Dim.Y {
            public HAT_BIY(HAT_BI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_BIY() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_BIY(this, copyContext);
            }
        }

        public static final class HAT_BIZ extends HAT_BI implements Dim.Z {
            public HAT_BIZ(HAT_BI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_BIZ() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_BIZ(this, copyContext);
            }
        }
    }

    public abstract static sealed class HAT_BS extends HATThreadOp {

        protected HAT_BS() {
            super(List.of());
        }

        protected HAT_BS(HAT_BS op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HAT_BSX extends HAT_BS implements Dim.X {

            public HAT_BSX(HAT_BS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_BSX() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_BSX(this, copyContext);
            }
        }

        public static final class HAT_BSY extends HAT_BS implements Dim.Y {

            public HAT_BSY(HAT_BS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_BSY() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_BSY(this, copyContext);
            }

        }

        public static final class HAT_BSZ extends HAT_BS implements Dim.Z {

            public HAT_BSZ(HAT_BS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_BSZ() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_BSZ(this, copyContext);
            }
        }
    }

    public abstract static sealed class HAT_LS extends HATThreadOp {

        protected HAT_LS() {
            super(List.of());
        }

        protected HAT_LS(HAT_LS op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HAT_LSX extends HAT_LS implements Dim.X {
            public HAT_LSX(HAT_LS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_LSX() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_LSX(this, copyContext);
            }
        }

        public static final class HAT_LSY extends HAT_LS implements Dim.Y {
            public HAT_LSY(HAT_LS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_LSY() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_LSY(this, copyContext);
            }
        }

        public static final class HAT_LSZ extends HAT_LS implements Dim.Z {
            public HAT_LSZ(HAT_LS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_LSZ() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_LSZ(this, copyContext);
            }
        }
    }

    public abstract static sealed class HAT_GI extends HATThreadOp {

        protected HAT_GI() {
            super(List.of());
        }

        protected HAT_GI(HAT_GI op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HAT_GIX extends HAT_GI implements Dim.X {
            public HAT_GIX(HAT_GI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_GIX() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_GIX(this, copyContext);
            }
        }

        public static final class HAT_GIY extends HAT_GI implements Dim.Y {
            public HAT_GIY(HAT_GI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_GIY() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_GIY(this, copyContext);
            }
        }

        public static final class HAT_GIZ extends HAT_GI implements Dim.Z {
            public HAT_GIZ(HAT_GI op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_GIZ() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_GIZ(this, copyContext);
            }
        }
    }

    public abstract static sealed class HAT_GS extends HATThreadOp {

        protected HAT_GS() {
            super(List.of());
        }

        protected HAT_GS(HAT_GS op, CodeContext copyContext) {
            super(op, copyContext);
        }

        public static final class HAT_GSX extends HAT_GS implements Dim.X {
            public HAT_GSX(HAT_GS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_GSX() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_GSX(this, copyContext);
            }
        }

        public static final class HAT_GSY extends HAT_GS implements Dim.Y {
            public HAT_GSY(HAT_GS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_GSY() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_GSY(this, copyContext);
            }
        }

        public static final class HAT_GSZ extends HAT_GS implements Dim.Z {
            public HAT_GSZ(HAT_GS op, CodeContext copyContext) {
                super(op, copyContext);
            }

            public HAT_GSZ() {
                super();
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HAT_GSZ(this, copyContext);
            }
        }
    }
}