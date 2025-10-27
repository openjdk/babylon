package oracle.code.onnx;

import java.util.Map;

public record OnnxProvider(String name, Map<String, String> options) { }
