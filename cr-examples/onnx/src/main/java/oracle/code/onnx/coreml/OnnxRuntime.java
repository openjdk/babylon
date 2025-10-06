/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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

package oracle.code.onnx.coreml;

import java.io.File;
import java.io.IOException;
import java.lang.foreign.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;
import oracle.code.onnx.foreign.OrtApi;
import oracle.code.onnx.foreign.OrtApiBase;

import static oracle.code.onnx.foreign.onnxruntime_c_api_h.*;

public final class OnnxRuntime {

    static final boolean DEBUG = Boolean.getBoolean("oracle.code.onnx.coreml.OnnxRuntime.DEBUG");

    static {
        String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH).startsWith("aarch64") ? "aarch64" : "x64";
        String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
        String libResource;
        if (os.contains("mac") || os.contains("darwin")) {
            libResource = "/ai/onnxruntime/native/osx-" + arch + "/libonnxruntime.dylib";
        } else if (os.contains("win")) {
            libResource = "/ai/onnxruntime/native/win-" + arch + "/libonnxruntime.dll";
        } else if (os.contains("nux")) {
            libResource = "/ai/onnxruntime/native/linux-" + arch + "/libonnxruntime.so";
        } else {
            throw new IllegalStateException("Unsupported os:" + os);
        }
        try {
            // workaround to avoid CNFE when the ReleaseEnv class is attempted to load in the shutdown hook from already closed classloader
            Class.forName("oracle.code.onnx.foreign.OrtApi$ReleaseEnv");
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
        try (var libStream = oracle.code.onnx.OnnxRuntime.class.getResourceAsStream(libResource)) {
            var libFile = File.createTempFile("libonnxruntime", "");
            Path libFilePath = libFile.toPath();
            Files.copy(libStream, libFilePath, StandardCopyOption.REPLACE_EXISTING);
            System.load(libFilePath.toAbsolutePath().toString());
            libFile.deleteOnExit();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static OnnxRuntime getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new OnnxRuntime();
        }
        return INSTANCE;
    }

    private static final String LOG_ID = "onnx-ffm-java";
    private static OnnxRuntime INSTANCE;

    private final MemorySegment runtimeAddress, ret, envAddress, defaultAllocatorAddress;

    private OnnxRuntime() {
        var arena = Arena.ofAuto();
        ret = arena.allocate(C_POINTER);
        //  const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t)apiVersion);
        var apiBase = OrtApiBase.reinterpret(OrtGetApiBase(), arena, null);
        runtimeAddress = OrtApi.reinterpret(OrtApiBase.GetApi(apiBase, ORT_API_VERSION()), arena, null);
        envAddress = retAddr(OrtApi.CreateEnv(runtimeAddress, ORT_LOGGING_LEVEL_ERROR(), arena.allocateFrom(LOG_ID), ret));
        defaultAllocatorAddress = retAddr(OrtApi.GetAllocatorWithDefaultOptions(runtimeAddress, ret)).reinterpret(arena, null);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            OrtApi.ReleaseEnv(runtimeAddress, envAddress);
        }));
    }

    public Session createSession(Arena arena, String modelPath) {
        return createSession(arena, modelPath, createSessionOptions(arena));
    }

    public Session createSession(Arena arena, String modelPath, SessionOptions options) {
        return new Session(arena, retAddr(OrtApi.CreateSession(runtimeAddress, envAddress, arena.allocateFrom(modelPath), options.sessionOptionsAddress, ret)));
    }

    public Session createSession(Arena arena, byte[] model) {
        return createSession(arena, model, createSessionOptions(arena));
    }

    public Session createSession(Arena arena, byte[] model, SessionOptions options) {
        return new Session(arena, retAddr(OrtApi.CreateSessionFromArray(runtimeAddress, envAddress, arena.allocateFrom(ValueLayout.JAVA_BYTE, model), model.length, options.sessionOptionsAddress, ret)));
    }

    public final class Session {

        private final MemorySegment sessionAddress;

        private Session(Arena arena, MemorySegment sessionAddress) {
            this.sessionAddress = sessionAddress.reinterpret(arena,
                    session -> OrtApi.ReleaseSession(runtimeAddress, session));
        }

        public int getNumberOfInputs() {
            return retInt(OrtApi.SessionGetInputCount(runtimeAddress, sessionAddress, ret));
        }

        public String getInputName(int inputIndex) {
            return retString(OrtApi.SessionGetInputName(runtimeAddress, sessionAddress, inputIndex, defaultAllocatorAddress, ret));
        }

        public int getNumberOfOutputs() {
            return retInt(OrtApi.SessionGetOutputCount(runtimeAddress, sessionAddress, ret));
        }

        public String getOutputName(int inputIndex) {
            return retString(OrtApi.SessionGetOutputName(runtimeAddress, sessionAddress, inputIndex, defaultAllocatorAddress, ret));
        }

        // @@@ only tensors are supported yet
        public List<Tensor> run(Arena arena, List<Tensor> inputValues) {
            var runOptions = MemorySegment.NULL;
            int inputLen = getNumberOfInputs();
            int outputLen = getNumberOfOutputs();
            var inputNames = arena.allocate(C_POINTER, inputLen);
            var inputs = arena.allocate(C_POINTER, inputLen);
            long index = 0;
            for (int i = 0; i < inputLen; i++) {
                inputNames.setAtIndex(C_POINTER, index, arena.allocateFrom(getInputName(i)));
                inputs.setAtIndex(C_POINTER, index++, inputValues.get(i).tensorAddr);
            }
            var outputNames = arena.allocate(C_POINTER, outputLen);
            var outputs = arena.allocate(C_POINTER, outputLen);
            for (int i = 0; i < outputLen; i++) {
                outputNames.setAtIndex(C_POINTER, i, arena.allocateFrom(getOutputName(i)));
                outputs.setAtIndex(C_POINTER, i, MemorySegment.NULL);
            }
            checkStatus(OrtApi.Run(runtimeAddress, sessionAddress, runOptions, inputNames, inputs, (long)inputLen, outputNames, (long)outputLen, outputs));
            var retArr = new Tensor[outputLen];
            for (int i = 0; i < outputLen; i++) {
                var tensorAddr = outputs.getAtIndex(C_POINTER, i)
                        .reinterpret(arena, value -> OrtApi.ReleaseValue(runtimeAddress, value));
                retArr[i] = new Tensor(tensorData(tensorAddr).reinterpret(arena, null),
                                       tensorAddr);
            }
            return List.of(retArr);
        }
    }

    public MemorySegment createTensor(Arena arena, MemorySegment flatData, Tensor.ElementType elementType, long[] shape) {
        var allocatorInfo = retAddr(OrtApi.AllocatorGetInfo(runtimeAddress, defaultAllocatorAddress, ret));
        return retAddr(OrtApi.CreateTensorWithDataAsOrtValue(
                runtimeAddress,
                allocatorInfo,
                flatData, flatData.byteSize(),
                shape.length == 0 ? MemorySegment.NULL : autoShape(arena, shape, 8l * flatData.byteSize() / elementType.bitSize()), (long)shape.length,
                elementType.id,
                ret)).reinterpret(arena, value -> OrtApi.ReleaseValue(runtimeAddress, value));
    }

    private static MemorySegment autoShape(Arena arena, long[] shape, long elementsCount) {
        int auto = -1;
        long elCount = 1;
        for (int i = 0; i < shape.length; i++) {
            long dim = shape[i];
            if (dim == -1) {
                if (auto == -1) {
                    auto = i;
                } else {
                    throw new IllegalArgumentException("Multiple automatic dimensions in shape");
                }
            } else {
                elCount *= dim;
            }
        }
        var ms = arena.allocateFrom(C_LONG_LONG, shape);
        if (auto != -1) {
            long autoDim = elementsCount / elCount;
            ms.setAtIndex(C_LONG, auto, autoDim);
            elCount *= autoDim;
        }
        if (elCount != elementsCount) {
            throw new IllegalArgumentException("Tensor shape does not match data");
        }
        return ms;
    }

    public Tensor.ElementType tensorElementType(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        return Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType(runtimeAddress, infoAddr, ret)));
    }

    public long[] tensorShape(MemorySegment tensorAddr) {
        try (var arena = Arena.ofConfined()) {
            var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
            long dims = retLong(OrtApi.GetDimensionsCount(runtimeAddress, infoAddr, ret));
            var shape = arena.allocate(C_LONG_LONG, dims);
            checkStatus(OrtApi.GetDimensions(runtimeAddress, infoAddr, shape, dims));
            return shape.toArray(C_LONG_LONG);
        }
    }

    public MemorySegment tensorData(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        long size = retLong(OrtApi.GetTensorShapeElementCount(runtimeAddress, infoAddr, ret))
                * Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType(runtimeAddress, infoAddr, ret))).bitSize() / 8;
        return retAddr(OrtApi.GetTensorMutableData(runtimeAddress, tensorAddr, ret))
                .reinterpret(size);
    }

    public SessionOptions createSessionOptions(Arena arena) {
        return new SessionOptions(retAddr(OrtApi.CreateSessionOptions(runtimeAddress, ret))
                .reinterpret(arena, opts -> OrtApi.ReleaseSessionOptions(runtimeAddress, opts)));
    }

    public final class SessionOptions {

        private final MemorySegment sessionOptionsAddress;

        public SessionOptions(MemorySegment sessionOptionsAddress) {
            this.sessionOptionsAddress = sessionOptionsAddress;
            setInterOpNumThreads(1);
        }

        public void setInterOpNumThreads(int numThreads) {
            checkStatus(OrtApi.SetInterOpNumThreads(runtimeAddress, sessionOptionsAddress, numThreads));
        }
    }

    private MemorySegment retAddr(MemorySegment res) {
        checkStatus(res);
        return ret.get(C_POINTER, 0);
    }

    private int retInt(MemorySegment res) {
        checkStatus(res);
        return ret.get(C_INT, 0);
    }

    private long retLong(MemorySegment res) {
        checkStatus(res);
        return ret.get(C_LONG_LONG, 0);
    }

    private String retString(MemorySegment res) {
        return retAddr(res).reinterpret(Long.MAX_VALUE)
                .getString(0);
    }

    private void checkStatus(MemorySegment status) {
        try {
            if (!status.equals(MemorySegment.NULL)) {
                status = status.reinterpret(Long.MAX_VALUE);
                if (status.get(C_INT, 0) != 0) {
                    throw new RuntimeException(status.getString(C_INT.byteSize()));
                }
            }
        } finally {
            OrtApi.ReleaseStatus(runtimeAddress, status);
        }
    }
}
