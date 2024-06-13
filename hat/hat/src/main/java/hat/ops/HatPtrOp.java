package hat.ops;

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.util.List;

public class HatPtrOp extends HatOp {

    public HatPtrOp(TypeElement typeElement, List<Value> operands) {
        super("hat.ptr", typeElement, operands);
    }

    public HatPtrOp(HatOp that, CopyContext cc) {
        super(that, cc);
    }

    @Override
    public Op transform(CopyContext cc, OpTransformer ot) {
        return new HatPtrOp(this, cc);
    }
}
