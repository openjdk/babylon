## MavenStyleProject using code reflection with a Java-based ONNX programming model.

### ONNX Runtime running convolution neural network from Java source

Running the MNIST demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.mnist.MNISTDemo
```

### ONNX GenAI running large language model from Java source.

Setup:
 - Download [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/releases) native library coresponding to your system/architecture, unzip and put it into `cr-examples/onnx/lib` folder.
 - Download `model.onnx.data`, `tokenizer.json` and `tokenizer_config.json` data files from [Llama-3.2-1B-Instruct-ONNX](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4) and put them into `cr-examples/onnx/src/test/resources/oracle/code/onnx/llm` folder.

Running the Llama demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.llm.LlamaDemo
```

/Users/ammbra/Documents/experiments/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h,

jextract --target-package oracle.code.onnx.foreign.coreml \
-l /Users/ammbra/Documents/experiments/onnxruntime/build/MacOS/Release/libonnxruntime.dylib \
-I /Users/ammbra/Documents/experiments/onnxruntime/include/onnxruntime/core/session \
--output src/main/java /Users/ammbra/Documents/experiments/onnxruntime/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
