package hat.dialect;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;

import java.util.List;
import java.util.Map;

public abstract class HATBFLOATBinaryOp extends HATBFloat16Op {

    protected final TypeElement elementType;
    protected final BinaryOpType operationType;
    protected final List<Boolean> references;
    protected final byte f32;

    public static final byte FIRST_OP = 0x01;
    public static final byte LAST_OP = 0x10;

    public enum BinaryOpType {
        ADD("+"),
        SUB("-"),
        MUL("*"),
        DIV("/");

        String symbol;

        BinaryOpType(String symbol) {
            this.symbol = symbol;
        }

        public String symbol() {
            return symbol;
        }
    }

    public HATBFLOATBinaryOp(TypeElement typeElement, BinaryOpType operationType, List<Boolean> references, byte f32, List<Value> operands) {
        super("", operands);
        this.elementType = typeElement;
        this.operationType = operationType;
        this.references = references;
        this.f32 = f32;
    }

    public HATBFLOATBinaryOp(HATBFLOATBinaryOp op, CodeContext copyContext) {
        super(op, copyContext);
        this.elementType = op.elementType;
        this.operationType = op.operationType;
        this.references = op.references;
        this.f32 = op.f32;
    }

    @Override
    public TypeElement resultType() {
        return this.elementType;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.bfloat16." + varName(), operationType.symbol());
    }

    public BinaryOpType binaryOperationType() {
        return operationType;
    }

    public List<Boolean> references() {
        return references;
    }

    public byte getF32() {
        return f32;
    }
}
