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

import jdk.incubator.code.*;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.OpFactory;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ArithMathOps {

    static abstract class ArithMathOp extends Op {
        final TypeElement resultType;

        public ArithMathOp(ExternalizedOp def) {
            super(def.name(), def.operands());;

            this.resultType = def.resultType();
        }

        ArithMathOp(ArithMathOp that, CopyContext cc) {
            super(that, cc);

            this.resultType = that.resultType;
        }

        ArithMathOp(String name, TypeElement resultType, List<? extends Value> operands) {
            super(name, operands);

            this.resultType = resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    @OpFactoryHelper.OpDeclaration(ConstantOp.NAME)
    public static class ConstantOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.constant";
        public static final String ATTRIBUTE_CONSTANT_VALUE = "value";

        final Object value;

        public static ConstantOp create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have zero operands");
            }

            Object value = def.extractAttributeValue(ATTRIBUTE_CONSTANT_VALUE,true,
                    v -> processConstantValue(def.resultType(), v));
            return new ConstantOp(def, value);
        }

        static Object processConstantValue(TypeElement t, Object value) {
            if (t.equals(JavaType.BOOLEAN) && value instanceof Boolean) {
                return value;
            } else if (t.equals(JavaType.BYTE) && value instanceof Number n) {
                return n.byteValue();
            } else if (t.equals(JavaType.SHORT) && value instanceof Number n) {
                return n.shortValue();
            } else if (t.equals(JavaType.CHAR) && value instanceof Character) {
                return value;
            } else if (t.equals(JavaType.INT) && value instanceof Number n) {
                return n.intValue();
            } else if (t.equals(JavaType.LONG) && value instanceof Number n) {
                return n.longValue();
            } else if (t.equals(JavaType.FLOAT) && value instanceof Number n) {
                return n.floatValue();
            } else if (t.equals(Float16.FLOAT_16_TYPE) && value instanceof Number n) {
                // represent as a float for now
                return n.floatValue();
            } else if (t.equals(JavaType.DOUBLE) && value instanceof Number n) {
                return n.doubleValue();
            } else if (t instanceof TensorType tt) {
                return processConstantValue(tt.eType(), value);
            }

            throw new UnsupportedOperationException("Unsupported constant type and value: " + t + " " + value);
        }

        ConstantOp(ExternalizedOp def, Object value) {
            super(def);

            this.value = value;
        }

        ConstantOp(ConstantOp that, CopyContext cc) {
            super(that, cc);

            this.value = that.value;
        }

        @Override
        public ConstantOp transform(CopyContext cc, OpTransformer ot) {
            return new ConstantOp(this, cc);
        }

        ConstantOp(TypeElement type, Object value) {
            super(NAME, type, List.of());

            this.value = value;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of(ATTRIBUTE_CONSTANT_VALUE, value);
        }

        public Object value() {
            return value;
        }
    }

    @OpFactoryHelper.OpDeclaration(AddOp.NAME)
    public static class AddOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.add";

        public AddOp(ExternalizedOp def) {
            super(def);
        }

        AddOp(AddOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AddOp transform(CopyContext cc, OpTransformer ot) {
            return new AddOp(this, cc);
        }

        AddOp(Value a, Value b) {
            super(NAME + nameSuffixFromType(a.type(), false), a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(SubOp.NAME)
    public static class SubOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.sub";

        public SubOp(ExternalizedOp def) {
            super(def);
        }

        SubOp(SubOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SubOp transform(CopyContext cc, OpTransformer ot) {
            return new SubOp(this, cc);
        }

        SubOp(Value a, Value b) {
            super(NAME + nameSuffixFromType(a.type(), false), a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(MulOp.NAME)
    public static class MulOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.mul";

        public MulOp(ExternalizedOp def) {
            super(def);
        }

        MulOp(MulOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public MulOp transform(CopyContext cc, OpTransformer ot) {
            return new MulOp(this, cc);
        }

        MulOp(Value a, Value b) {
            super(NAME + nameSuffixFromType(a.type(), false), a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(DivOp.NAME)
    public static class DivOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.div";

        public DivOp(ExternalizedOp def) {
            super(def);
        }

        DivOp(DivOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public DivOp transform(CopyContext cc, OpTransformer ot) {
            return new DivOp(this, cc);
        }

        DivOp(Value a, Value b) {
            super(NAME + nameSuffixFromType(a.type(), true), a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(RemOp.NAME)
    public static class RemOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.rem";

        public RemOp(ExternalizedOp def) {
            super(def);
        }

        RemOp(RemOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public RemOp transform(CopyContext cc, OpTransformer ot) {
            return new RemOp(this, cc);
        }

        RemOp(Value a, Value b) {
            super(NAME + nameSuffixFromType(a.type(), true), a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(AndOp.NAME)
    public static class AndOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.andi";

        public AndOp(ExternalizedOp def) {
            super(def);
        }

        AndOp(AndOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AndOp transform(CopyContext cc, OpTransformer ot) {
            return new AndOp(this, cc);
        }

        AndOp(Value a, Value b) {
            super(NAME, a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(MaxOp.NAME)
    public static class MaxOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.max";

        public MaxOp(ExternalizedOp def) {
            super(def);
        }

        MaxOp(MaxOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public MaxOp transform(CopyContext cc, OpTransformer ot) {
            return new MaxOp(this, cc);
        }

        MaxOp(Value a, Value b) {
            super(NAME + maxMinSuffixFromType(a.type()) + nameSuffixFromType(a.type(), true),
                    a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(MinOp.NAME)
    public static class MinOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.min";

        public MinOp(ExternalizedOp def) {
            super(def);
        }

        MinOp(MinOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public MinOp transform(CopyContext cc, OpTransformer ot) {
            return new MinOp(this, cc);
        }

        MinOp(Value a, Value b) {
            super(NAME + maxMinSuffixFromType(a.type()) + nameSuffixFromType(a.type(), true),
                    a.type(), List.of(a, b));
        }
    }

    @OpFactoryHelper.OpDeclaration(ExpOp.NAME)
    public static class TruncOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.trunc";

        public TruncOp(ExternalizedOp def) {
            super(def);
        }

        TruncOp(TruncOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public TruncOp transform(CopyContext cc, OpTransformer ot) {
            return new TruncOp(this, cc);
        }

        TruncOp(TypeElement t, Value a) {
            super(NAME + nameSuffixFromType(a.type(), false),
                    t, List.of(a));
        }
    }

    @OpFactoryHelper.OpDeclaration(ExpOp.NAME)
    public static class ExpOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "math.exp";

        public ExpOp(ExternalizedOp def) {
            super(def);
        }

        ExpOp(ExpOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ExpOp transform(CopyContext cc, OpTransformer ot) {
            return new ExpOp(this, cc);
        }

        ExpOp(Value a) {
            super(NAME, a.type(), List.of(a));
        }
    }

    @OpFactoryHelper.OpDeclaration(CompareOp.NAME)
    public static class CompareOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.cmp";
        public static final String ATTRIBUTE_PREDICATE = "predicate";

        // https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpipredicate
        // The ordinal values correspond to the MLIR symbol's values
        // Need to refine when considering comparisons of floating point numbers which is in a different namespace
        public enum CompareKind {
            eq,
            ne,
            slt,
            sle,
            sgt,
            sge,
            ult,
            ule,
            ugt,
            uge
        }

        final CompareKind ck;

        public static CompareOp create(ExternalizedOp def) {
            CompareKind ck = def.extractAttributeValue(ATTRIBUTE_PREDICATE, true,
                    v -> switch (v) {
                        case String s -> CompareKind.valueOf(s);
                        case CompareKind k -> k;
                        case null, default -> throw new UnsupportedOperationException("Unsupported start value:" + v);
                    });
            return new CompareOp(def, ck);
        }

        CompareOp(ExternalizedOp def, CompareKind ck) {
            super(def);

            this.ck = ck;
        }

        CompareOp(CompareOp that, CopyContext cc) {
            super(that, cc);

            this.ck = that.ck;
        }

        @Override
        public CompareOp transform(CopyContext cc, OpTransformer ot) {
            return new CompareOp(this, cc);
        }

        CompareOp(CompareKind ck, Value a, Value b) {
            TypeElement t;
            if (a.type() instanceof TensorType ot) {
                t = new TensorType(JavaType.BOOLEAN, ot.shape());
            }
            else {
                t = JavaType.BOOLEAN;
            }
            super(NAME + nameSuffixFromType(a.type(), false), t, List.of(a, b));

            this.ck = ck;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of(ATTRIBUTE_PREDICATE, Long.valueOf(ck.ordinal()));
        }

        public CompareKind kind() {
            return ck;
        }
    }

    static String maxMinSuffixFromType(TypeElement t) {
        if (t instanceof TensorType tt) {
            return maxMinSuffixFromType(tt.eType());
        } else if (t instanceof PtrType pt) {
            return maxMinSuffixFromType(pt.rType());
        } else if (t.equals(JavaType.INT)) {
            return "";
        } else if (t.equals(JavaType.FLOAT)) {
            return "imum";
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + t);
        }
    }

    static String nameSuffixFromType(TypeElement t, boolean signed) {
        if (t instanceof TensorType tt) {
            return nameSuffixFromType(tt.eType(), signed);
        } else if (t instanceof PtrType pt) {
            return nameSuffixFromType(pt.rType(), signed);
        } else if (t.equals(JavaType.INT) || t.equals(JavaType.LONG)) {
            return (signed ? "s" : "") + "i";
        } else if (t.equals(JavaType.FLOAT) || t.equals(JavaType.DOUBLE) ||
                Float16.FLOAT_16_TYPE.equals(t)) {
            return "f";
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + t);
        }
    }

    public static final OpFactory OP_FACTORY = def -> {
        return switch (def.name()) {
            case ConstantOp.NAME -> ConstantOp.create(def);
            case ExpOp.NAME -> new ExpOp(def);
            case AddOp.NAME + "i", AddOp.NAME + "f" -> new AddOp(def);
            case SubOp.NAME + "i", SubOp.NAME + "f" -> new SubOp(def);
            case MulOp.NAME + "i", MulOp.NAME + "f" -> new MulOp(def);
            case DivOp.NAME + "si", DivOp.NAME + "f" -> new DivOp(def);
            case RemOp.NAME + "si", RemOp.NAME + "f" -> new DivOp(def);
            case AndOp.NAME -> new AndOp(def);
            case MaxOp.NAME + "si", MaxOp.NAME + "imumf" -> new MaxOp(def);
            case MinOp.NAME + "si", MinOp.NAME + "imumf" -> new MinOp(def);
            case TruncOp.NAME + "i", TruncOp.NAME + "f" -> new TruncOp(def);
            case CompareOp.NAME + "i", CompareOp.NAME + "f" -> CompareOp.create(def);
            default -> null;
        };
    };

    // Arith

    public static ConstantOp constant(TypeElement type, Object value) {
        return new ConstantOp(type, value);
    }

    public static MulOp mul(Value a, Value b) {
        return new MulOp(a, b);
    }

    public static AddOp add(Value a, Value b) {
        return new AddOp(a, b);
    }

    public static SubOp sub(Value a, Value b) {
        return new SubOp(a, b);
    }

    public static DivOp div(Value a, Value b) {
        return new DivOp(a, b);
    }

    public static RemOp rem(Value a, Value b) {
        return new RemOp(a, b);
    }

    public static AndOp and(Value a, Value b) {
        return new AndOp(a, b);
    }

    public static MaxOp maximum(Value a, Value b) {
        return new MaxOp(a, b);
    }

    public static MinOp minimum(Value a, Value b) {
        return new MinOp(a, b);
    }

    public static CompareOp cmp(CompareOp.CompareKind ck, Value a, Value b) {
        return new CompareOp(ck, a, b);
    }

    public static TruncOp trunc(TypeElement type, Value a) {
        return new TruncOp(type, a);
    }

    // Math

    public static ExpOp exp(Value a) {
        return new ExpOp(a);
    }
}
