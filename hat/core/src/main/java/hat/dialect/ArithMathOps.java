package hat.dialect;

import hat.codetypes.ConstantType;
import hat.codetypes.TensorType;
import jdk.incubator.code.AbstractOp;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedOp;
import optkl.util.ops.Precedence;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

// Arithmetic Math Ops extended from Triton PoC and adapted to the Tile Programming Model
public class ArithMathOps {
    private ArithMathOps() {
        /* This utility class should not be instantiated */
    }

    public abstract static class ArithMathOp extends AbstractOp implements ExternalizedOp.Externalizable {
        final String opName;
        final CodeType resultType;

        protected ArithMathOp(ExternalizedOp def) {
            super(def.operands());

            this.opName = def.name();
            this.resultType = def.resultType();
        }

        ArithMathOp(ArithMathOp that, CodeContext cc) {
            super(that, cc);

            this.opName = that.opName;
            this.resultType = that.resultType;
        }

        ArithMathOp(String name, CodeType resultType, List<? extends Value> operands) {
            super(operands);

            this.opName = name;
            this.resultType = resultType;
        }

        protected StringBuilder externalizeShape() {
            StringBuilder builder = new StringBuilder();
            if (resultType instanceof ConstantType constantType
                    && constantType.value() instanceof TensorType tensorType) {
                builder.append("shape: ").append(Arrays.toString(tensorType.shape().toArray()));
            }
            return builder;
        }

        @Override
        public CodeType resultType() {
            return resultType;
        }

        @Override
        public String externalizeOpName() {
            return opName;
        }
    }

    public static class ConstantOp extends ArithMathOp implements Op.Pure, Precedence.Invoke {
        public static final String NAME = "arith.constant";
        public static final String ATTRIBUTE_CONSTANT_VALUE = "value";

        final Object value;

        public static ConstantOp create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have zero operands");
            }
            Object value = processConstantValue(def.resultType(), getDefaultAttributeValue(def, ATTRIBUTE_CONSTANT_VALUE));
            return new ConstantOp(def, value);
        }

        static Object processConstantValue(CodeType codeType, Object value) {
            if (codeType.equals(JavaType.BOOLEAN) && value instanceof Boolean) {
                return value;
            } else if (codeType.equals(JavaType.BYTE) && value instanceof Number n) {
                return n.byteValue();
            } else if (codeType.equals(JavaType.SHORT) && value instanceof Number n) {
                return n.shortValue();
            } else if (codeType.equals(JavaType.CHAR) && value instanceof Character) {
                return value;
            } else if (codeType.equals(JavaType.INT) && value instanceof Number n) {
                return n.intValue();
            } else if (codeType.equals(JavaType.LONG) && value instanceof Number n) {
                return n.longValue();
            } else if (codeType.equals(JavaType.FLOAT) && value instanceof Number n) {
                return n.floatValue();
            } else if (codeType.equals(JavaType.DOUBLE) && value instanceof Number n) {
                return n.doubleValue();
            } else if (codeType instanceof TensorType tt) {
                return processConstantValue(tt.elementType(), value);
            }
            throw new UnsupportedOperationException("Unsupported constant type and value: " + codeType + " " + value);
        }

        ConstantOp(ExternalizedOp def, Object value) {
            super(def);

            this.value = value;
        }

        ConstantOp(ConstantOp that, CodeContext cc) {
            super(that, cc);

            this.value = that.value;
        }

        @Override
        public ConstantOp transform(CodeContext cc, CodeTransformer ot) {
            return new ConstantOp(this, cc);
        }

        ConstantOp(CodeType type, Object value) {
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

    public static class AddOp extends ArithMathOp implements Op.Pure, Precedence.Additive {
        public static final String NAME = "arith.add";

        public AddOp(ExternalizedOp def) {
            super(def);
        }

        AddOp(AddOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public AddOp transform(CodeContext cc, CodeTransformer ot) {
            return new AddOp(this, cc);
        }

        AddOp(CodeType type, Value a, Value b) {
            super(NAME, type, List.of(a, b));
        }

        @Override
        public String externalizeOpName() {
            StringBuilder builder = externalizeShape();
            builder.append(" >>>>>> ").append(opName).append(" - resultType: ").append(resultType);
            return builder.toString();
        }
    }

    public static class TransposeOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.transpose";

        public TransposeOp(ExternalizedOp def) {
            super(def);
        }

        TransposeOp(TransposeOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TransposeOp transform(CodeContext cc, CodeTransformer ot) {
            return new TransposeOp(this, cc);
        }

        TransposeOp(CodeType type, Value a) {
            super(NAME, type, List.of(a));
        }

        @Override
        public String externalizeOpName() {
            StringBuilder builder = externalizeShape();
            builder.append(" >>>>>> " + opName + " - resultType: " + resultType);
            return builder.toString();
        }
    }

    public static class ReshapeOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.reshape";

        public ReshapeOp(ExternalizedOp def) {
            super(def);
        }

        ReshapeOp(ReshapeOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ReshapeOp transform(CodeContext cc, CodeTransformer ot) {
            return new ReshapeOp(this, cc);
        }

        ReshapeOp(CodeType type, Value a) {
            super(NAME, type, List.of(a));
        }

        @Override
        public String externalizeOpName() {
            StringBuilder builder = externalizeShape();
            builder.append(" >>>>>> " + opName + " - resultType: " + resultType);
            return builder.toString();
        }
    }

    public static class PermuteOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.permute";

        public PermuteOp(ExternalizedOp def) {
            super(def);
        }

        PermuteOp(PermuteOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public PermuteOp transform(CodeContext cc, CodeTransformer ot) {
            return new PermuteOp(this, cc);
        }

        PermuteOp(CodeType type, Value a) {
            super(NAME, type, List.of(a));
        }

        @Override
        public String externalizeOpName() {
            StringBuilder builder = externalizeShape();
            builder.append(" >>>>>> " + opName + " - resultType: " + resultType);
            return builder.toString();
        }
    }

    public static class MMAOp extends ArithMathOp implements Op.Pure, Precedence.Invoke {
        public static final String NAME = "arith.mma";

        public MMAOp(ExternalizedOp def) {
            super(def);
        }

        MMAOp(MMAOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public MMAOp transform(CodeContext cc, CodeTransformer ot) {
            return new MMAOp(this, cc);
        }

        MMAOp(Value tensorA, Value tensorB, Value tensorC) {
            super(NAME, tensorC.type(), List.of(tensorA, tensorB, tensorC));
        }

        @Override
        public String externalizeOpName() {
            StringBuilder builder = externalizeShape();
            builder.append(" >>>>>> " + opName + " - resultType: " + resultType);
            return builder.toString();
        }
    }

    public static class SubOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.sub";

        public SubOp(ExternalizedOp def) {
            super(def);
        }

        SubOp(SubOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public SubOp transform(CodeContext cc, CodeTransformer ot) {
            return new SubOp(this, cc);
        }

        SubOp(CodeType type, Value a, Value b) {
            super(NAME, type, List.of(a, b));
        }
    }

    public static class MulOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.mul";

        public MulOp(ExternalizedOp def) {
            super(def);
        }

        MulOp(MulOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public MulOp transform(CodeContext cc, CodeTransformer ot) {
            return new MulOp(this, cc);
        }

        MulOp(CodeType type, Value a, Value b) {
            super(NAME, type, List.of(a, b));
        }
    }

    public static class CDivOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.cdiv";

        public CDivOp(ExternalizedOp def) {
            super(def);
        }

        CDivOp(CDivOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public CDivOp transform(CodeContext cc, CodeTransformer ot) {
            return new CDivOp(this, cc);
        }

        CDivOp(CodeType type, Value a, Value b) {
            super(NAME, type, List.of(a, b));
        }
    }

    public static class TrueDivOp extends ArithMathOp implements Op.Pure {
        public static final String NAME = "arith.truediv";

        public TrueDivOp(ExternalizedOp def) {
            super(def);
        }

        TrueDivOp(TrueDivOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TrueDivOp transform(CodeContext cc, CodeTransformer ot) {
            return new TrueDivOp(this, cc);
        }

        TrueDivOp(CodeType type, Value a, Value b) {
            super(NAME, type, List.of(a, b));
        }
    }

    static Object getDefaultAttributeValue(ExternalizedOp def, String attributeName) {
        return TileOps.getDefaultAttributeValue(def, attributeName);
    }

    // Arith

    public static ConstantOp constant(CodeType type, Object value) {
        return new ConstantOp(type, value);
    }

    public static MulOp mul(CodeType type, Value a, Value b) {
        return new MulOp(type, a, b);
    }

    public static AddOp add(CodeType type, Value a, Value b) {
        return new AddOp(type, a, b);
    }

    public static SubOp sub(CodeType type, Value a, Value b) {
        return new SubOp(type, a, b);
    }

    public static CDivOp cdiv(CodeType type, Value a, Value b) {
        return new CDivOp(type, a, b);
    }

    public static TrueDivOp truediv(CodeType type, Value a, Value b) {
        return new TrueDivOp(type, a, b);
    }

    public static TransposeOp transpose(CodeType type, Value a) {
        return new TransposeOp(type, a);
    }

    public static ReshapeOp reshape(CodeType type, Value a) {
        return new ReshapeOp(type, a);
    }

    public static PermuteOp permute(CodeType type, Value a) {
        return new PermuteOp(type, a);
    }

    public static MMAOp mma(Value tensorA, Value tensorB, Value tensorC) {
        return new MMAOp(tensorA, tensorB, tensorC);
    }
}