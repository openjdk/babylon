package oracle.code.onnx;

import oracle.code.onnx.ir.OnnxOp;

import java.util.List;

public class OnnxInterpreter {
    public static Object interpret(Class<? extends OnnxOp> opClass,
                                   List<Object> inputs,
                                   List<Object> attributes) {
        throw new UnsupportedOperationException();
    }
}
