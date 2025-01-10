package oracle.code.onnx.ir;

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.ExternalizableOp;

import java.util.List;
import java.util.Map;

public abstract class OnnxOp extends ExternalizableOp {
    final TypeElement resultType;

    OnnxOp(ExternalizedOp def) {
        super(def);

        this.resultType = def.resultType();
    }

    OnnxOp(OnnxOp that, CopyContext cc) {
        super(that, cc);

        this.resultType = that.resultType;
    }

    OnnxOp(String name, TypeElement resultType, List<? extends Value> operands) {
        super(name, operands);

        this.resultType = resultType;
    }

    @Override
    public TypeElement resultType() {
        return resultType;
    }

    public Map<String, Object> onnxAttributes() {
        return Map.of();
    }
}
