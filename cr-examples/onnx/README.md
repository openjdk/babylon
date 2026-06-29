## MavenStyleProject using code reflection with a Java-based ONNX programming model.

### ONNX Runtime running convolution neural network from Java source

Running the MNIST demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.mnist.MNISTDemo
```

### ONNX Runtime with CoreML running facial emotion recognition from Java source

Download the `.data` file from [emotion-ferplus-8.onnx.data](https://github.com/ammbra/fer-model-weights/raw/refs/heads/main/emotion-ferplus-8.onnx.data) and place it under `cr-examples/onnx/src/test/resources/oracle/code/onnx/fer` folder.

Running the FER demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.fer.FERCoreMLDemo
```

### ONNX Runtime running an embedding model from Java source

[all-MiniLM-L6-v2](https://huggingface.co/onnx-community/all-MiniLM-L6-v2-ONNX) is a popular, lightweight, sentence embedding model.
This model converts sentences into 384 dimensional vectors, an aspect very useful for semantic search and information retrieval.

Download `model.onnx_data` and `vocab.json` data files from [all-MiniLM-L6-v2](https://huggingface.co/onnx-community/all-MiniLM-L6-v2-ONNX) and put them into `cr-examples/onnx/src/test/resources/oracle/code/onnx/bert` folder.

To run the EmbeddingDemo, execute:

```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.bert.EmbeddingDemo
```

#### How to (Re)Generate the Java Bindings

The following instructions are for Mac users only as the CoreML Execution Provider (EP) requires iOS devices with iOS 13 or higher, or Mac computers with macOS 10.15 or higher.
Download ONNX Runtime from a published release.

Inside `cr-examples/onnx/opgen` project you will find the `setup.sh` script that takes as argument the path to your cloned `onnxruntime` and uses `jextract` to regenerate the binaries.
Prior to running it make sure that `jextract` is in your system `$PATH` :

```shell
jextract --version
```

Provide the path to your cloned `onnxruntime` and the script will regenerate the Java bindings inside the `oracle.code.onnx.foreign`:

```
export ONNX_INCLUDE_DIR=/path/to/onnx/include/onnxruntime/include
export ONNX_GEN_AI_INCLUDE_DIR=/path/to/onnx/include/onnxruntime-genai/include
sh setup.sh
```

### ONNX GenAI running large language model from Java source.

Setup:
- Download [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/releases) native library coresponding to your system/architecture, unzip and put it into `cr-examples/onnx/lib` folder.
- Download `model_q4.onnx_data`, `tokenizer.json` and `tokenizer_config.json` data files from [Llama-3.2-1B-Instruct-ONNX](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/tree/main) and put them into `cr-examples/onnx/src/test/resources/oracle/code/onnx/llm` folder.

Running the Llama demo:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn process-test-classes exec:java -Dexec.mainClass=oracle.code.onnx.llm.LlamaDemo
```

### Lifting ONNX model from binary to Java source.

OnnxLift is an experimental tool for lifting ONNX binary models to ONNX code reflection model, extraction of weights, and generation of Java model source.

Running the OnnxLift:
```
JAVA_HOME=<path to the Babylon JDK home>
mvn package exec:java -Dexec.mainClass=oracle.code.onnx.lift.OnnxLift -Dexec.args="<model.onnx> <target folder> [class simple name]"
```
