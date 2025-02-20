package oracle.code.onnx;

import java.io.File;
import java.io.IOException;
import java.lang.foreign.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

import jdk.incubator.code.op.CoreOp;
import oracle.code.onnx.foreign.OrtApi;
import oracle.code.onnx.foreign.OrtApiBase;
import oracle.code.onnx.ir.OnnxOp;

import static java.lang.foreign.ValueLayout.*;
import static oracle.code.onnx.foreign.onnxruntime_c_api_h.*;

public final class OnnxRuntime {

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

    private static final String LOG_ID = "onnx-ffm-java";

    public static OnnxRuntime getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new OnnxRuntime();
        }
        return INSTANCE;
    }

    private static final AddressLayout ADDR_WITH_ADDR = ADDRESS.withTargetLayout(ADDRESS);
    private static final AddressLayout ADDR_WITH_STRING = ADDRESS.withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, JAVA_BYTE));

    private static OnnxRuntime INSTANCE;

    private final Arena         arena;
    private final MemorySegment runtimeAddress, ret, envAddress, defaultAllocatorAddress;

    private OnnxRuntime() {
        arena = Arena.ofAuto();
        ret = arena.allocate(ADDR_WITH_ADDR);
        //  const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t)apiVersion);
        var apiBase = OrtApiBase.reinterpret(OrtGetApiBase(), arena, null);
        runtimeAddress = OrtApi.reinterpret(OrtApiBase.GetApi(apiBase, ORT_API_VERSION()), arena, null);
        envAddress = retAddr(OrtApi.CreateEnv(runtimeAddress, ORT_LOGGING_LEVEL_ERROR(), arena.allocateFrom(LOG_ID), ret));
        defaultAllocatorAddress = retAddr(OrtApi.GetAllocatorWithDefaultOptions(runtimeAddress, ret));
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            OrtApi.ReleaseEnv(runtimeAddress, envAddress);
        }));
    }

    private List<Optional<Tensor.ElementType>> toElementTypes(List<Optional<MemorySegment>> values) {
        return values.stream().map(ot -> ot.map(this::tensorElementType)).toList();
    }

    public List<MemorySegment> runOp(OnnxOp.OnnxSchema schema, List<Optional<MemorySegment>> inputValues, List<Object> attributes) {
        var protoModel = OnnxProtoBuilder.buildOpModel(schema, toElementTypes(inputValues), attributes);
        try (var session = createSession(protoModel)) {
            return session.run(inputValues);
        }
    }

    public List<MemorySegment> runFunc(CoreOp.FuncOp model, List<Optional<MemorySegment>> inputValues) {
        var protoModel = OnnxProtoBuilder.buildFuncModel(model);
        try (var session = createSession(protoModel)) {
            return session.run(inputValues);
        }
    }

    public Session createSession(String modelPath) {
        return createSession(modelPath, createSessionOptions());
    }

    public Session createSession(String modelPath, SessionOptions options) {
        return new Session(retAddr(OrtApi.CreateSession(runtimeAddress, envAddress, arena.allocateFrom(modelPath), options.sessionOptionsAddress, ret)));
    }

    public Session createSession(ByteBuffer model) {
        return createSession(model, createSessionOptions());
    }

    private Session createSession(ByteBuffer model, SessionOptions options) {
        return new Session(retAddr(OrtApi.CreateSessionFromArray(runtimeAddress, envAddress, MemorySegment.ofBuffer(model.rewind()), model.limit(), options.sessionOptionsAddress, ret)));
    }

    public final class Session implements AutoCloseable {

        private final MemorySegment sessionAddress;

        private Session(MemorySegment sessionAddress) {
            this.sessionAddress = sessionAddress;
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
        public List<MemorySegment> run(List<Optional<MemorySegment>> inputValues) {
            var runOptions = MemorySegment.NULL;
            int inputLen = getNumberOfInputs();
            int outputLen = getNumberOfOutputs();
            var inputNames = arena.allocate(ADDRESS, inputLen);
            var inputs = arena.allocate(ADDRESS, inputLen);
            long index = 0;
            for (int i = 0; i < inputLen; i++) {
                if (inputValues.get(i).isPresent()) {
                    inputNames.setAtIndex(ADDRESS, index, arena.allocateFrom(getInputName(i)));
                    inputs.setAtIndex(ADDRESS, index++, inputValues.get(i).get());
                }
            }
            var outputNames = arena.allocate(ADDRESS, outputLen);
            var outputs = arena.allocate(ADDRESS, outputLen);
            for (int i = 0; i < outputLen; i++) {
                outputNames.setAtIndex(ADDRESS, i, arena.allocateFrom(getOutputName(i)));
                outputs.setAtIndex(ADDRESS, i, MemorySegment.NULL);
            }
            checkStatus(OrtApi.Run(runtimeAddress, sessionAddress, runOptions, inputNames, inputs, (long)inputLen, outputNames, (long)outputLen, outputs));
            var retArr = new MemorySegment[outputLen];
            for (int i = 0; i < outputLen; i++) {
                retArr[i] = outputs.getAtIndex(ADDRESS, i);
            }
            return List.of(retArr);
        }

        @Override
        public void close() {
            OrtApi.ReleaseSession(runtimeAddress, sessionAddress);
        }
    }

    public MemorySegment createTensor(MemorySegment flatData, Tensor.ElementType elementType, long[] shape) {
        var allocatorInfo = retAddr(OrtApi.AllocatorGetInfo(runtimeAddress, defaultAllocatorAddress, ret));
        var shapeAddr = shape.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_LONG, shape);
        return retAddr(OrtApi.CreateTensorWithDataAsOrtValue(runtimeAddress, allocatorInfo, flatData, flatData.byteSize(), shapeAddr, shape.length, elementType.id, ret));
    }

    public Tensor.ElementType tensorElementType(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        return Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType(runtimeAddress, infoAddr, ret)));
    }

    public long[] tensorShape(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        long dims = retLong(OrtApi.GetDimensionsCount(runtimeAddress, infoAddr, ret));
        var shape = arena.allocate(JAVA_LONG, dims);
        checkStatus(OrtApi.GetDimensions(runtimeAddress, infoAddr, shape, dims));
        return shape.toArray(JAVA_LONG);
    }

    public ByteBuffer tensorBuffer(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        long size = retLong(OrtApi.GetTensorShapeElementCount(runtimeAddress, infoAddr, ret))
                * Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType(runtimeAddress, infoAddr, ret))).size();
        return retAddr(OrtApi.GetTensorMutableData(runtimeAddress, tensorAddr, ret))
                .reinterpret(size)
                .asByteBuffer().order(ByteOrder.nativeOrder());
    }

    public SessionOptions createSessionOptions() {
        return new SessionOptions(retAddr(OrtApi.CreateSessionOptions(runtimeAddress, ret)));
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
        return ret.get(ADDR_WITH_ADDR, 0);
    }

    private int retInt(MemorySegment res) {
        checkStatus(res);
        return ret.get(JAVA_INT, 0);
    }

    private long retLong(MemorySegment res) {
        checkStatus(res);
        return ret.get(JAVA_LONG, 0);
    }

    private String retString(MemorySegment res) {
        checkStatus(res);
        return ret.get(ADDR_WITH_STRING, 0).getString(0);
    }

    private void checkStatus(MemorySegment status) {
        if (!status.equals(MemorySegment.NULL)) {
            status = status.reinterpret(Long.MAX_VALUE);
            if (status.get(JAVA_INT, 0) != 0) {
                throw new RuntimeException(status.getString(JAVA_INT.byteSize()));
            }
        }
    }
}
