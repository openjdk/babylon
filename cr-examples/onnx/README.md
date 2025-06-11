### MavenStyleProject using code reflection with a Java-based ONNX programming model.

Running the demo:
```
JAVA_HOME=<path to the Babylon JDK home>;mvn process-test-classes exec:java -Dexec.classpathScope=test -Dexec.mainClass=oracle.code.onnx.MNISTDemo
```

### Onnx Generation API to create and run LLM Onnx models.

Example of direct execution of existing Onnx LLM model:
```
// model-specific prompt format
static final String PROMPT_TEMPLATE = "<|...|>%s<|...|><|...|>";

public static void main(String... args) {

    // compatible `libonnxruntime` library must be present in the same folder as `libonnxruntime-genai` library
    // native library extension (.dylib, .so or .dll) is platform specific
    System.load("path/To/libonnxruntime-genai.dylib");

    // model folder must contain the Onnx model file and all configuration and external data files
    try (OnnxGenRuntimeSession session = new OnnxGenRuntimeSession(Path.of("path/To/Onnx/Model/Folder/")) {
        // each LLM model has specific prompt format
        session.prompt(PROMPT_TEMPLATE.formatted("Tell me a joke"), System.out::print);
    }
}
```

Example of a custom LLM Onnx model generation from Java sources and execution:
```
// model-specific prompt format
static final String PROMPT_TEMPLATE = "<|...|>%s<|...|><|...|>";

public static void main(String... args) {

    // compatible `libonnxruntime` library must be present in the same folder as `libonnxruntime-genai` library
    // native library extension (.dylib or .so or .dll) is platform specific
    System.load("path/To/libonnxruntime-genai.dylib");

    // instance of a custom Onnx LLM model
    MyCustomLLMModel myCustomModelInstance = ...;

    // target model folder must contain all configuration files
    // `genai_config.json` must be configured following way:
    //     - model filename to match generated model file name (below)
    //     - model inputs to match main model method argument names
    //     - model outputs to match main model result record component names
    Path targetModelFolder = ...;

    // Onnx model file and external data file are generated to the target model folder
    // and the session is created from the generated model
    try (OnnxGenRuntimeSession session = OnnxGenRuntimeSession.buildFromCodeReflection(myCustomModelInstance, "myMainModelMethod", targetModelFolder, "MyModelFileName.onnx", "MyDataFileName")) {
        // each LLM model has specific prompt format
        session.prompt(PROMPT_TEMPLATE.formatted("Tell me a joke"), System.out::print);
    }
}
```

Example of a custom LLM Onnx model Java source:
```
import oracle.code.onnx.Tensor;
import jdk.incubator.code.CodeReflection;
import static oracle.code.onnx.OnnxOperators.*;

public final class MyCustomLLMModel {

     public final Tensor<Float> myModelWeights...
     public final Tensor<Byte> otherMyModelWeights...

     public MyCustomLLMModel(...) {
         // initilize all weight tensors
         // large tensors data can be memory-mapped
         this.myModelWeights = ...
         this.otherMyModelWeights = ...
         ...
     }

     // custom record with main model method response
     public record MyModelResponse(Tensor<Float> logits, Tensor<Float> presentKey0, Tensor<Float> presentValue0, ...) {
     }

     @CodeReflection
     public MyModelResponse myMainModelMethod(Tensor<Long> inputIds, Tensor<Long> attentionMask, Tensor<Float> pastKey0, Tensor<Float> pastValue0, ...) {

         // computation of the model using oracle.code.onnx.OnnxOperators.* method calls
         ...
         Tensor<Float> logits = MatMul(...

         // composition of the return record
         return new MyModelResponse(logits, key0, value0, ...);
     }
}
```
