package oracle.code.onnx.ir;

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.ExternalizableOp;

import java.util.List;
import java.util.Map;

public abstract class OnnxOp extends ExternalizableOp {

    OnnxOp(ExternalizedOp def) {
        super(def);
    }

    OnnxOp(OnnxOp that, CopyContext cc) {
        super(that, cc);
    }

    OnnxOp(String name, List<? extends Value> operands) {
        super(name, operands);
    }

    public Map<String, Object> onnxAttributes() {
        return Map.of();
    }
}
