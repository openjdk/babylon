/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package oracle.code.onnx.genai;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.stream.Stream;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import oracle.code.onnx.OnnxProtoBuilder;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.compiler.OnnxTransformer;

import static oracle.code.onnx.foreign.OrtGenApi.*;

/**
 * Class wrapping Onnx Generation API to create and run LLM Onnx model in a session with configured tokenizer and generator.
 * <p>
 * Example of direct execution of existing Onnx LLM model:
 * {@snippet lang="java" :
 *  // model-specific prompt format
 *  static final String PROMPT_TEMPLATE = "<|...|>%s<|...|><|...|>";
 *
 *  public static void main(String... args) {
 *
 *      // compatible `libonnxruntime` library must be present in the same folder as `libonnxruntime-genai` library
 *      // native library extension (.dylib or .so or .dll) is platform specific
 *      System.load("path/To/libonnxruntime-genai.dylib");
 *
 *      // model folder must contain the Onnx model file and all configuration and external data files
 *      try (OnnxGenRuntimeSession session = new OnnxGenRuntimeSession(Path.of("path/To/Onnx/Model/Folder/")) {
 *          // each LLM model has specific prompt format
 *          session.prompt(PROMPT_TEMPLATE.formatted("Tell me a joke"), System.out::print);
 *      }
 *   }
 * }
 * <p>
 * Example of a custom LLM Onnx model generation from Java sources and execution:
 * {@snippet lang="java" :
 *  // model-specific prompt format
 *  static final String PROMPT_TEMPLATE = "<|...|>%s<|...|><|...|>";
 *
 *  public static void main(String... args) {
 *
 *      // compatible `libonnxruntime` library must be present in the same folder as `libonnxruntime-genai` library
 *      // native library extension (.dylib or .so or .dll) is platform specific
 *      System.load("path/To/libonnxruntime-genai.dylib");
 *
 *      // instance of a custom Onnx LLM model
 *      MyCustomLLMModel myCustomModelInstance = ...;
 *
 *      // target model folder must contain all configuration files
 *      // `genai_config.json` must be configured following way:
 *      //     - model filename to match generated model file name (below)
 *      //     - model inputs to match main model method argument names
 *      //     - model outputs to match main model result record component names
 *      Path targetModelFolder = ...;
 *
 *      // Onnx model file and external data file are generated to the target model folder
 *      // and the session is created from the generated model
 *      try (OnnxGenRuntimeSession session = OnnxGenRuntimeSession.buildFromCodeReflection(myCustomModelInstance, "myMainModelMethod", targetModelFolder, "MyModelFileName.onnx", "MyDataFileName")) {
 *          // each LLM model has specific prompt format
 *          session.prompt(PROMPT_TEMPLATE.formatted("Tell me a joke"), System.out::print);
 *      }
 *   }
 * }
 * <p>
 * Example of a custom LLM Onnx model Java source:
 * {@snippet lang="java" :
 *   import oracle.code.onnx.Tensor;
 *   import jdk.incubator.code.CodeReflection;
 *   import static oracle.code.onnx.OnnxOperators.*;
 *
 *   public final class MyCustomLLMModel {
 *
 *       public final Tensor<Float> myModelWeights...
 *       public final Tensor<Byte> otherMyModelWeights...
 *
 *       public MyCustomLLMModel(...) {
 *           // initilize all weight tensors
 *           // large tensors data can be memory-mapped
 *           this.myModelWeights = ...
 *           this.otherMyModelWeights = ...
 *           ...
 *       }
 *
 *       // custom record with main model method response
 *       public record MyModelResponse(Tensor<Float> logits, Tensor<Float> presentKey0, Tensor<Float> presentValue0, ...) {
 *       }
 *
 *       @CodeReflection
 *       public MyModelResponse myMainModelMethod(Tensor<Long> inputIds, Tensor<Long> attentionMask, Tensor<Float> pastKey0, Tensor<Float> pastValue0 ...) {
 *
 *           // computation of the model using oracle.code.onnx.OnnxOperators.* method calls
 *           ...
 *           Tensor<Float> logits = MatMul(...
 *
 *           // composition of the return record
 *           return new MyModelResponse(logits, key0, value0, ...);
 *       }
 *   }
 * }
 */
public class OnnxGenRuntimeSession implements AutoCloseable {

    /**
     * Builds Onnx model from the provided Java model instance and loads it into a constructs the Onnx Generate API session.
     * @param codeReflectionModelInstance Instance of a class representing Onnx LLM model.
     * @param methodName Main model method name.
     * @param targetOnnxModelDir Target folder for generation of Onnx model and external tensor data file.
     * @param targetOnnxModelFileName Target Onnx model file name.
     * @param targetExternalDataFileName Target external tensor data file name.
     * @return a live session instance
     * @throws IOException In case of any IO problems during model generation.
     */
    public static OnnxGenRuntimeSession buildFromCodeReflection(Object codeReflectionModelInstance, String methodName, Path targetOnnxModelDir, String targetOnnxModelFileName, String targetExternalDataFileName) throws IOException {
        Method method = Stream.of(codeReflectionModelInstance.getClass().getDeclaredMethods()).filter(m -> m.getName().equals(methodName)).findFirst().orElseThrow();
        CoreOp.FuncOp javaModel = Op.ofMethod(method).orElseThrow();
        OnnxTransformer.ModuleAndInitializers onnxModel = OnnxTransformer.transform(MethodHandles.lookup(), javaModel);
        List<Object> initializers = OnnxRuntime.getInitValues(MethodHandles.lookup(), onnxModel.initializers(), List.of(codeReflectionModelInstance));
        try (OutputStream dataOutput = Files.newOutputStream(targetOnnxModelDir.resolve(targetExternalDataFileName))) {
            AtomicLong offset = new AtomicLong();
            byte[] protobufModel = OnnxProtoBuilder.buildModel("llm", onnxModel.module(), initializers, onnxModel.namesMap(), t -> {
                byte[] data = t.data().toArray(ValueLayout.JAVA_BYTE);
                try {
                    dataOutput.write(data);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                return new OnnxProtoBuilder.ExternalTensorDataInfo(targetExternalDataFileName, offset.getAndAdd(data.length), data.length);
            });
            Files.write(targetOnnxModelDir.resolve(targetOnnxModelFileName), protobufModel);
        }
        return new OnnxGenRuntimeSession(targetOnnxModelDir);
    }

    private final Arena arena;
    private final MemorySegment ret, model, tokenizer, tokenizerStream, generatorParams, generator, count;

    /**
     * Constructs Onnx Generate API session (including model, tokenizer and generator) from assets stored in the Onnx model directory.
     * @param onnxModelDir Path to the Onnx model directory with Onnx model file, data file(s) and configuration files.
     */
    public OnnxGenRuntimeSession(Path onnxModelDir) {
        this.arena = Arena.ofConfined();
        ret = arena.allocate(C_POINTER);
        model = call(OgaCreateModel(arena.allocateFrom(onnxModelDir.toString()), ret));
        tokenizer = call(OgaCreateTokenizer(model, ret));
        tokenizerStream = call(OgaCreateTokenizerStream(tokenizer, ret));
        generatorParams = call(OgaCreateGeneratorParams(model, ret));
        generator = call(OgaCreateGenerator(model, generatorParams, ret));
        count = arena.allocate(C_LONG);
    }

    private MemorySegment call(MemorySegment status) {
        try {
            if (!status.equals(MemorySegment.NULL)) {
                status = status.reinterpret(C_INT.byteSize());
                if (status.get(C_INT, 0) != 0) {
                    String errString = OgaResultGetError(status)
                            .reinterpret(Long.MAX_VALUE)
                            .getString(0L);
                    throw new RuntimeException(errString);
                }
            }
            return ret.get(C_POINTER, 0);
        } finally {
            OgaDestroyResult(status);
        }
    }

    /**
     * Runs generator with the provided prompt and feeds decoded response to the provided consumer.
     * @param prompt Text prompt to tokenize and append to the LLM model input.
     * @param outputConsumer Consumer receiving decoded model response from the model generator.
     */
    public void prompt(String prompt, Consumer<String> outputConsumer) {
        var inputTokens = call(OgaCreateSequences(ret));
        try {
            call(OgaTokenizerEncode(tokenizer, arena.allocateFrom(prompt), inputTokens));
            call(OgaGenerator_AppendTokenSequences(generator, inputTokens));
            while (!OgaGenerator_IsDone(generator)) {
                call(OgaGenerator_GenerateNextToken(generator));
                int nextToken = call(OgaGenerator_GetNextTokens(generator, ret, count)).get(C_INT, 0);
                String response = call(OgaTokenizerStreamDecode(tokenizerStream, nextToken, ret)).getString(0);
                outputConsumer.accept(response);
            }
            outputConsumer.accept("\n");
        } finally {
            OgaDestroySequences(inputTokens);
        }
    }

    /**
     * Closes the session and all its related assets (arena, generator, tokenizer and model).
     */
    @Override
    public void close() {
        arena.close();
        OgaDestroyGenerator(generator);
        OgaDestroyGeneratorParams(generatorParams);
        OgaDestroyTokenizerStream(tokenizerStream);
        OgaDestroyTokenizer(tokenizer);
        OgaDestroyModel(model);
    }
}