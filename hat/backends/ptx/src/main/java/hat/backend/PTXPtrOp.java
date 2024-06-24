package hat.backend;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExternalizableOp;
import java.util.List;

public class PTXPtrOp extends ExternalizableOp {
    public static final String NAME = "ptx.ptr.op";
    public String fieldName;

    PTXPtrOp(PTXPtrOp that, CopyContext cc) {
        super(that, cc);
    }

    public PTXPtrOp(Value ptr, String fieldName) {
        super(NAME, List.of(ptr));
        this.fieldName = fieldName;
    }

    @Override
    public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
        return null;
    }

    @Override
    public TypeElement resultType() {
        return null;
    }
}
