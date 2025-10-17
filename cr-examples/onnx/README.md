## MavenStyleProject using code reflection with a Java-based ONNX programming model.

### ONNX Runtime running convolution neural network from Java source

Running the MNIST demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.mnist.MNISTDemo
```

### ONNX Runtime with CoreML running facial emotion recognition from Java source.

Running the FER demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.fer.FERCoreMLDemo
```

#### How to (Re)Generate the CoreML Java Bindings

The following instructions are for Mac users only as the CoreML Execution Provider (EP) requires iOS devices with iOS 13 or higher, or Mac computers with macOS 10.15 or higher.
Build and install custom ONNX Runtime with CoreML enabled:

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
Provide the path to your cloned `onnxruntime` and the script will regenerate the CoreML Java bindings inside the `oracle.code.onnx.foreign`:

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