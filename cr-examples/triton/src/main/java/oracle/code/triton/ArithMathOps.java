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

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.*;
import java.lang.reflect.code.type.JavaType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ArithMathOps {

    static abstract class ArithMathOp extends OpWithDefinition {
        final TypeElement resultType;

        public ArithMathOp(OpDefinition def) {
            super(def);

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

    @OpDeclaration(ConstantOp.NAME)
    public static class ConstantOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.constant";
        public static final String ATTRIBUTE_CONSTANT_VALUE = NAME + ".value";

        final Object value;

        public static ConstantOp create(OpDefinition def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have zero operands");
            }

            Object value = def.extractAttributeValue(ATTRIBUTE_CONSTANT_VALUE,true,
                    v -> processConstantValue(def.resultType(), v));
            return new ConstantOp(def, value);
        }

        static Object processConstantValue(TypeElement t, Object value) {
            if (t.equals(JavaType.BOOLEAN)) {
                if (value instanceof String s) {
                    return Boolean.valueOf(s);
                } else if (value instanceof Boolean) {
                    return value;
                }
            } else if (t.equals(JavaType.BYTE)) {
                if (value instanceof String s) {
                    return Byte.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.byteValue();
                }
            } else if (t.equals(JavaType.SHORT)) {
                if (value instanceof String s) {
                    return Short.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.shortValue();
                }
            } else if (t.equals(JavaType.CHAR)) {
                if (value instanceof String s) {
                    return s.charAt(0);
                } else if (value instanceof Character) {
                    return value;
                }
            } else if (t.equals(JavaType.INT)) {
                if (value instanceof String s) {
                    return Integer.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.intValue();
                }
            } else if (t.equals(JavaType.LONG)) {
                if (value instanceof String s) {
                    return Long.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.longValue();
                }
            } else if (t.equals(JavaType.FLOAT)) {
                if (value instanceof String s) {
                    return Float.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.floatValue();
                }
            } else if (t.equals(JavaType.DOUBLE)) {
                if (value instanceof String s) {
                    return Double.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.doubleValue();
                }
            } else if (t instanceof TensorType tt) {
                return processConstantValue(TritonType.fromType(tt.eType()), value);
            }

            throw new UnsupportedOperationException("Unsupported constant type and value: " + t + " " + value);
        }

        ConstantOp(OpDefinition def, Object value) {
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
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", value);
            return attrs;
        }

        public Object value() {
            return value;
        }
    }

    @OpDeclaration(AddOp.NAME)
    public static class AddOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.add";

        public AddOp(OpDefinition def) {
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

    @OpDeclaration(SubOp.NAME)
    public static class SubOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.sub";

        public SubOp(OpDefinition def) {
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

    @OpDeclaration(MulOp.NAME)
    public static class MulOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.mul";

        public MulOp(OpDefinition def) {
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

    @OpDeclaration(DivOp.NAME)
    public static class DivOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.div";

        public DivOp(OpDefinition def) {
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

    @OpDeclaration(RemOp.NAME)
    public static class RemOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.rem";

        public RemOp(OpDefinition def) {
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

    @OpDeclaration(AndOp.NAME)
    public static class AndOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.andi";

        public AndOp(OpDefinition def) {
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

    @OpDeclaration(MaxOp.NAME)
    public static class MaxOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.max";

        public MaxOp(OpDefinition def) {
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

    @OpDeclaration(MinOp.NAME)
    public static class MinOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.min";

        public MinOp(OpDefinition def) {
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

    @OpDeclaration(ExpOp.NAME)
    public static class TruncOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.trunc";

        public TruncOp(OpDefinition def) {
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

    @OpDeclaration(ExpOp.NAME)
    public static class ExpOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "math.exp";

        public ExpOp(OpDefinition def) {
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

    @OpDeclaration(CompareOp.NAME)
    public static class CompareOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.cmp";
        public static final String ATTRIBUTE_CONSTANT_VALUE = NAME + ".compare";

        public enum CompareKind {
            slt
        }

        final CompareKind ck;

        public static CompareOp create(OpDefinition def) {
            CompareKind ck = def.extractAttributeValue(ATTRIBUTE_CONSTANT_VALUE, true,
                    v -> switch (v) {
                        case String s -> CompareKind.valueOf(s);
                        case CompareKind k -> k;
                        default -> throw new UnsupportedOperationException("Unsupported start value:" + v);
                    });
            return new CompareOp(def, ck);
        }

        CompareOp(OpDefinition def, CompareKind ck) {
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
            super(NAME + nameSuffixFromType(a.type(), false), a.type(), List.of(a, b));

            this.ck = ck;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", ck);
            return attrs;
        }

        public CompareKind kind() {
            return ck;
        }
    }

    static String maxMinSuffixFromType(TypeElement t) {
        if (t instanceof TensorType tt) {
            return maxMinSuffixFromType(TritonType.fromType(tt.eType()));
        } else if (t instanceof PtrType pt) {
            return maxMinSuffixFromType(TritonType.fromType(pt.rType()));
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
            return nameSuffixFromType(TritonType.fromType(tt.eType()), signed);
        } else if (t instanceof PtrType pt) {
            return nameSuffixFromType(TritonType.fromType(pt.rType()), signed);
        } else if (t.equals(JavaType.INT) || t.equals(JavaType.LONG)) {
            return (signed ? "s" : "") + "i";
        } else if (t.equals(JavaType.FLOAT) || t.equals(JavaType.DOUBLE) ||
                Float16.FLOAT_16_TYPE.equals(t)) {
            return "f";
        } else {
            throw new UnsupportedOperationException("Unsupported type: " + t);
        }
    }

    public static final OpFactory FACTORY = def -> {
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
