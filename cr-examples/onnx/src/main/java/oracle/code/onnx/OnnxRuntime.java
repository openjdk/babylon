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

package oracle.code.onnx;

import java.io.File;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import jdk.incubator.code.*;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import oracle.code.onnx.compiler.OnnxTransformer;
import oracle.code.onnx.foreign.OrtApi;
import oracle.code.onnx.foreign.OrtApiBase;
import oracle.code.onnx.proto.OnnxModel;

import static oracle.code.onnx.foreign.onnxruntime_c_api_h.*;

public final class OnnxRuntime {

    static final boolean DEBUG = Boolean.getBoolean("oracle.code.onnx.OnnxRuntime.DEBUG");

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
        try (var libStream = OnnxRuntime.class.getResourceAsStream(libResource)) {
            var libFile = File.createTempFile("libonnxruntime", "");
            Path libFilePath = libFile.toPath();
            Files.copy(libStream, libFilePath, StandardCopyOption.REPLACE_EXISTING);
            System.load(libFilePath.toAbsolutePath().toString());
            libFile.deleteOnExit();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @FunctionalInterface
    public interface OnnxFunction<T> extends Supplier<T>, Quotable {
    }

    // @@@ temporary set public for ongoing experiments
    public static List<Object> getInitValues(MethodHandles.Lookup lookup, SequencedCollection<FieldRef> initializers, SequencedCollection<Object> possibleReceivers) {
        return initializers.stream().map(i -> {
            try {
                Field initializerField = i.resolveToMember(lookup);
                VarHandle handle = lookup.unreflectVarHandle(initializerField);
                if (initializerField.accessFlags().contains(AccessFlag.STATIC)) {
                    return handle.get();
                } else {
                    Class<?> initializerClass = initializerField.getDeclaringClass();
                    return handle.get(possibleReceivers.stream().filter(initializerClass::isInstance).findFirst().orElseThrow());
                }
            } catch (ReflectiveOperationException ex) {
                throw new RuntimeException(ex);
            }
        }).toList();
    }

    static class CachedSessionClassValue extends ClassValue<Session> {

        private MethodHandles.Lookup l;
        private Quoted q;

        Session computeIfAbsent(Class<?> lambdaClass, MethodHandles.Lookup l,  Quoted q) {
            try {
                this.l = l;
                this.q = q;
                // not very nice way to pass additional arguments to computeValue method
                return get(lambdaClass);
            } finally {
                this.l = null;
                this.q = null;
            }
        }

        @Override
        protected Session computeValue(Class<?> type) {
            OnnxTransformer.ModuleAndInitializers mi = OnnxTransformer.transform(l, q);

            String domainName = type.getSimpleName().split("\\$")[0];
            byte[] protobufModel = OnnxProtoBuilder.buildModel(domainName, mi.module(), getInitValues(l, mi.initializers(), q.capturedValues().sequencedValues()));

            if (DEBUG) {
                System.out.println(mi.module().toText());
//                System.out.println(OnnxModel.readFrom(protobufModel).toText());
                try {
                    var export = Path.of(domainName + ".onnx");
                    Files.write(export, protobufModel);
                    System.out.println("Onnx model exported to: " + export.toAbsolutePath());
                } catch (IOException _) {}
            }

            return getInstance().createSession(
                    Arena.ofAuto(), // cached session must be created under its own auto arena
                    protobufModel);

        }
    }

    private static final CachedSessionClassValue SESSION_CACHE = new CachedSessionClassValue();

    public static <T> T execute(OnnxFunction<T> codeLambda) {
        return execute(MethodHandles.lookup(), codeLambda);
    }

    public static <T> T execute(MethodHandles.Lookup l, OnnxFunction<T> codeLambda) {
        return execute(Arena.ofAuto(), l, codeLambda);
    }


    private static void expandArg(Object val, Consumer<Tensor> args) {
        switch (val) {
            case CoreOp.Var<?> v -> expandArg(v.value(), args);
            case Tensor t -> args.accept(t);
            case Record r -> {
                for (var rc : r.getClass().getRecordComponents()) try {
                    expandArg(rc.getAccessor().invoke(r), args);
                } catch (ReflectiveOperationException e) {
                    throw new IllegalStateException(e);
                }
            }
            // @@@ constant array last object must be consumed or the statically detected size and the actual size missmatch
            case Object[] os -> {
                for (var o : os) {
                    expandArg(o, args);
                }
            }
            default -> {
            }
        }
    }

    public static <T> T execute(Arena arena, MethodHandles.Lookup l, OnnxFunction<T> codeLambda) {
        var q = Op.ofQuotable(codeLambda).orElseThrow();

        var model = SESSION_CACHE.computeIfAbsent(codeLambda.getClass(), l, q);

        List<Tensor> arguments = q.capturedValues().sequencedValues().stream()
                .mapMulti(OnnxRuntime::expandArg)
                .toList();
        List<Tensor> ret = model.run(arena, arguments);

        TypeElement type = ((JavaOp.LambdaOp)q.op()).invokableType().returnType();
        if (type instanceof ArrayType) {
            return (T)ret.toArray(Tensor[]::new);
        }
        ClassType retType = ((ClassType)type).rawType();
        if (retType.equals(TENSOR_RAW_TYPE)) {
            return (T)ret.getFirst();
        } else if(retType.equals(LIST_RAW_TYPE)) {
            return (T)ret;
        } else if(getRecordClass(l, retType) instanceof Class cls) {
            try {
                return (T)cls.getConstructors()[0].newInstance(unflat(ret, cls.getRecordComponents()));
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        } else {
            throw new UnsupportedOperationException("Unsupported return type: " + q.op().resultType());
        }
    }

    static Object[] unflat(List<Tensor> values, RecordComponent[] rcs) {
        Object[] ret = new Object[rcs.length];
        for (int i = 0, j = 0; i < rcs.length; i++) {
            if (rcs[i].getType().isArray() && rcs[i].getAnnotation(ExplicitOnnxOperators.ArrayLen.class) instanceof ExplicitOnnxOperators.ArrayLen al) {
                ret[i] = values.subList(j, j + al.value()).toArray(Tensor[]::new);
                j += al.value();
            } else {
                ret[i] = values.get(j++);
            }
        }
        return ret;
    }


    static Class getRecordClass(MethodHandles.Lookup l, ClassType ct) {
        try {
            var t = ct.resolve(l);
            while (t instanceof ParameterizedType pt) t = pt.getRawType();
            if (t instanceof Class c && c.isRecord()) return c;
        } catch (ReflectiveOperationException _) {
        }
        return null;
    }

    static final JavaType TENSOR_RAW_TYPE = JavaType.type(Tensor.class);
    static final JavaType LIST_RAW_TYPE = JavaType.type(List.class);

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

    public List<Tensor> runOp(Arena arena, String opName, List<Tensor> inputValues, int numOutputs, Map<String, Object> attributes) {
        var outputNames = IntStream.range(0, numOutputs).mapToObj(o -> "o" + o).toList();
        var protoModel = OnnxProtoBuilder.buildModel(
                List.of(),
                IntStream.range(0, inputValues.size()).mapToObj(i -> OnnxProtoBuilder.tensorInfo("i" + i, inputValues.get(i).elementType().id)).toList(),
                List.of(OnnxProtoBuilder.node(
                        opName,
                        IntStream.range(0, inputValues.size()).mapToObj(i -> "i" + i).toList(),
                        outputNames,
                        attributes)),
                outputNames);
        return createSession(arena, protoModel)
                .run(arena, inputValues);
    }

    public List<Tensor> run(Arena arena, Block block, List<Tensor> inputValues, int initializers) {
        var protoModel = OnnxProtoBuilder.buildModel(block, inputValues.subList(0, initializers));
        return createSession(arena, protoModel)
                .run(arena, inputValues.subList(initializers, inputValues.size()));
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

    private Session createSession(Arena arena, byte[] model, SessionOptions options) {
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
