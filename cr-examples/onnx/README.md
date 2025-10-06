## MavenStyleProject using code reflection with a Java-based ONNX programming model.

### ONNX Runtime running convolution neural network from Java source

Running the MNIST demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.mnist.MNISTDemo
```

### ONNX Runtime with CoreML running facial emotion recognition from Java source.

For demo purposes, we isolated the CoreML generated bindings and an FFM only `OnnxRuntime` in `oracle.code.onnx.coreml.foreign`.

Running the FER demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.fer.FERCoreMLDemo
```

Babylon JDK is based on current OpenJDK mainline.
This means that the FFM compatible parts of the `onnx` project can be ran with JDK 25 or OpenJDK 26 Early Access Builds.
You can try that by pointing your $JAVA_HOME to JDK 25 then run `run-jdk.sh` script:

```shell
JAVA_HOME=<path to JDK 25>
./run-jdk.sh
```

#### How to (Re)Generate the CoreML Java Bindings

Build and install custom ONNX Runtime with CoreML enabled (for Mac users):

```
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --use_coreml --parallel
# get the path to where current built library is available
pwd
```

Inside `cr-examples/onnx` project you will find the `setup.sh` script that takes as argument the path to your cloned `onnxruntime` and uses `jextract` to regenerate the binaries.
Prior to running it make sure that `jextract` is in your system `$PATH` :

```shell
jextract --version
```
Provide the path to your cloned `onnxruntime` and the script will regenerate the CoreML Java bindings inside the `oracle.code.onnx.coreml.foreign`:

```
sh setup.sh path/to/cloned/onnxruntime
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