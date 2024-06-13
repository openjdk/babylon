package hat.ops;

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.util.List;

public abstract class HatOp extends Op {
    private final TypeElement type;

    HatOp(String opName, TypeElement type, List<Value> operands) {
        super(opName, operands);
        this.type = type;
    }

    HatOp(HatOp that, CopyContext cc) {
        super(that, cc);
        this.type = that.type;
    }

    @Override
    public TypeElement resultType() {
        return type;
    }
}
