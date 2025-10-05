## MavenStyleProject using code reflection with a Java-based ONNX programming model.

### ONNX Runtime running convolution neural network from Java source

Running the MNIST demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.mnist.MNISTDemo
```

### ONNX Runtime with CoreML running facial emotion recognition from Java source.

Build and install custom ONNX Runtime with CoreML enabled (for Mac users):

```
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --use_coreml --parallel
```

Now install the built library:

```
sudo cp build/MacOS/Release/lib/libonnxruntime.* /usr/local/lib/
sudo cp -r include/onnxruntime /usr/local/include/
```

Running the FER demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.fer.FERCoreMLDemo
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