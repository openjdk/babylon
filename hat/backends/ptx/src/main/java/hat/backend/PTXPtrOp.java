package hat.backend;

import hat.ifacemapper.Schema;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExternalizableOp;
import java.util.List;

public class PTXPtrOp extends ExternalizableOp {
    public String fieldName;
    public static final String NAME = "ptxPtr";
    final TypeElement resultType;
    public Schema<?> schema;

    PTXPtrOp(TypeElement resultType, String fieldName, List<Value> operands, Schema<?> schema) {
        super(NAME, operands);
        this.resultType = resultType;
        this.fieldName = fieldName;
        this.schema = schema;
    }

    PTXPtrOp(PTXPtrOp that, CopyContext cc) {
        super(that, cc);
        this.resultType = that.resultType;
        this.fieldName = that.fieldName;
        this.schema = that.schema;
    }

    @Override
    public PTXPtrOp transform(CopyContext cc, OpTransformer ot) {
        return new PTXPtrOp(this, cc);
    }

    @Override
    public TypeElement resultType() {
        return resultType;
    }
}
