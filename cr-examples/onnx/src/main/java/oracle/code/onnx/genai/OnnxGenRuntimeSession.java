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
import jdk.incubator.code.op.CoreOp;
import oracle.code.onnx.OnnxProtoBuilder;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.compiler.OnnxTransformer;

import static oracle.code.onnx.foreign.OrtGenApi.*;

public class OnnxGenRuntimeSession implements AutoCloseable {

    public static OnnxGenRuntimeSession buildFromCodeReflection(Object codeReflectionModelInstance, String methodName, String promptTemplate, Path targetOnnxModelDir, String targetOnnxModelFileName, String targetExternalDataFileName) throws IOException {
        Method method = Stream.of(codeReflectionModelInstance.getClass().getDeclaredMethods()).filter(m -> m.getName().equals(methodName)).findFirst().orElseThrow();
        CoreOp.FuncOp javaModel = Op.ofMethod(method).orElseThrow();
        OnnxTransformer.ModuleAndInitializers onnxModel = OnnxTransformer.transform(MethodHandles.lookup(), javaModel, true);
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
        return new OnnxGenRuntimeSession(targetOnnxModelDir.toString(), promptTemplate);
    }

    private final Arena arena;
    private final MemorySegment ret, model, tokenizer, tokenizerStream, generatorParams, generator, count;
    private final String promptTemplate;

    public OnnxGenRuntimeSession(String onnxModelDir, String promptTemplate) {
        this.arena = Arena.ofConfined();
        ret = arena.allocate(C_POINTER);
        model = call(OgaCreateModel(arena.allocateFrom(onnxModelDir), ret));
        tokenizer = call(OgaCreateTokenizer(model, ret));
        tokenizerStream = call(OgaCreateTokenizerStream(tokenizer, ret));
        generatorParams = call(OgaCreateGeneratorParams(model, ret));
        generator = call(OgaCreateGenerator(model, generatorParams, ret));
        count = arena.allocate(C_LONG);
        this.promptTemplate = promptTemplate;
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

    public void prompt(String userPrompt, Consumer<String> outputConsumer) {
        var inputTokens = call(OgaCreateSequences(ret));
        try {
            call(OgaTokenizerEncode(tokenizer, arena.allocateFrom(promptTemplate.formatted(userPrompt)), inputTokens));
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

    @Override
    public void close() throws Exception {
        arena.close();
        OgaDestroyGenerator(generator);
        OgaDestroyGeneratorParams(generatorParams);
        OgaDestroyTokenizerStream(tokenizerStream);
        OgaDestroyTokenizer(tokenizer);
        OgaDestroyModel(model);
    }
}