package oracle.code.onnx;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.lang.reflect.UndeclaredThrowableException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.*;

public final class OnnxRuntime {

    private static final Path LIB_PATH;
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
            LIB_PATH = libFile.toPath();
            Files.copy(libStream, LIB_PATH, StandardCopyOption.REPLACE_EXISTING);
            libFile.deleteOnExit();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static final Linker LINKER = Linker.nativeLinker();

    private static final AddressLayout ADDR_WITH_ADDR = ADDRESS.withTargetLayout(ADDRESS);
    private static final AddressLayout ADDR_WITH_STRING = ADDRESS.withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, JAVA_BYTE));

    private static final VarHandle VH_ADDRESS = ADDRESS.varHandle();

    private static Environment ENV;

    public static Environment defaultEnvironment() {
        if (ENV == null) ENV = new OnnxRuntime().createEnv();
        return ENV;
    }

    private final Arena         arena;
    private final SymbolLookup  library;
    private final MemorySegment runtimeAddress, ret;
    private final MethodHandle  allocatorGetInfo,
                                createTensorWithDataAsOrtValue,
                                castTypeInfoToMapTypeInfo,
                                castTypeInfoToSequenceTypeInfo,
                                castTypeInfoToTensorInfo,
                                createEnv,
                                createSession,
                                createSessionFromArray,
                                createSessionOptions,
                                getAllocatorWithDefaultOptions,
                                getDimensions,
                                getDimensionsCount,
                                getOnnxTypeFromTypeInfo,
                                getTensorElementType,
                                getTensorMutableData,
                                getTensorShapeElementCount,
                                getTensorTypeAndShape,
                                releaseSession,
                                run,
                                sessionGetInputCount,
                                sessionGetInputName,
                                sessionGetInputTypeInfo,
                                sessionGetOutputCount,
                                sessionGetOutputName,
                                sessionGetOutputTypeInfo,
                                setInterOpNumThreads;

    public OnnxRuntime() {
        this(14);
    }

    public OnnxRuntime(int version) {
        arena = Arena.ofAuto();
        library = SymbolLookup.libraryLookup(LIB_PATH, arena);
        ret = arena.allocate(ADDR_WITH_ADDR);
        try {
            //  const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t)apiVersion);
            var apiBase = (MemorySegment)LINKER.downcallHandle(
                    library.findOrThrow("OrtGetApiBase"),
                    FunctionDescriptor.of(ADDR_WITH_ADDR)).invokeExact();
            runtimeAddress = MemorySegment.ofAddress((long)LINKER.downcallHandle(
                            (MemorySegment)VH_ADDRESS.get(apiBase, 0),
                            FunctionDescriptor.of(JAVA_LONG, JAVA_INT)).invokeExact(version))
                    .reinterpret(285 * ADDRESS.byteSize());
        } catch (Throwable t) {
            throw wrap(t);
        }
        allocatorGetInfo               = handle( 77, ADDRESS, ADDRESS);
        createTensorWithDataAsOrtValue = handle( 49, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS);
        castTypeInfoToMapTypeInfo      = handle(103, ADDRESS, ADDRESS);
        castTypeInfoToSequenceTypeInfo = handle(104, ADDRESS, ADDRESS);
        castTypeInfoToTensorInfo       = handle( 55, ADDRESS, ADDRESS);
        createEnv                      = handle(  3, JAVA_INT, ADDRESS, ADDRESS);
        createSession                  = handle(  7, ADDRESS, ADDRESS, ADDRESS, ADDRESS);
        createSessionFromArray         = handle(  8, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS);
        createSessionOptions           = handle( 10, ADDRESS);
        getAllocatorWithDefaultOptions = handle( 78, ADDRESS);
        getDimensions                  = handle( 62, ADDRESS, ADDRESS, JAVA_LONG);
        getDimensionsCount             = handle( 61, ADDRESS, ADDRESS);
        getOnnxTypeFromTypeInfo        = handle( 56, ADDRESS, ADDRESS);
        getTensorElementType           = handle( 60, ADDRESS, ADDRESS);
        getTensorMutableData           = handle( 51, ADDRESS, ADDRESS);
        getTensorShapeElementCount     = handle( 64, ADDRESS, ADDRESS);
        getTensorTypeAndShape          = handle( 65, ADDRESS, ADDRESS);
        releaseSession                 = handle( 95, ADDRESS);
        run                            = handle(  9, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS);
        sessionGetInputCount           = handle( 30, ADDRESS, ADDRESS);
        sessionGetInputName            = handle( 36, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        sessionGetInputTypeInfo        = handle( 33, ADDRESS, JAVA_INT, ADDRESS);
        sessionGetOutputCount          = handle( 31, ADDRESS, ADDRESS);
        sessionGetOutputName           = handle( 37, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        sessionGetOutputTypeInfo       = handle( 34, ADDRESS, JAVA_INT, ADDRESS);
        setInterOpNumThreads           = handle( 25, ADDRESS, JAVA_INT);
    }

    private MethodHandle handle(int methodIndex, MemoryLayout... args) {
        var mh = LINKER.downcallHandle((MemorySegment)VH_ADDRESS.get(runtimeAddress, methodIndex * ADDRESS.byteSize()),
                                     FunctionDescriptor.of(ADDRESS, args));
        return mh.asType(mh.type().changeReturnType(Object.class));
    }

    public Environment createEnv() {
        return createEnv(LoggingLevel.VERBOSE, "onnx-ffm-java");
    }

    public Environment createEnv(LoggingLevel logLevel, String logId) {
        try {
            return new Environment(retAddr(createEnv.invokeExact(logLevel.ordinal(), arena.allocateFrom(logId), ret)),
                                   retAddr(getAllocatorWithDefaultOptions.invokeExact(ret)));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public enum LoggingLevel {
        VERBOSE, INFO, WARNING, ERROR, FATAL
    }

    public enum ONNXType {
        UNKNOWN, TENSOR, SEQUENCE, MAP, OPAQUE, SPARSETENSOR, OPTIONAL
    }

    interface ProtoBuf extends Consumer<ByteBuffer> {
    }

    private static ByteBuffer storeVarInt(ByteBuffer bb, int number) {
        long expanded = Long.expand(Integer.toUnsignedLong(number), 0x7f7f7f7f7f7f7f7fl);
        int bytesSize = Math.max(1, 8 - Long.numberOfLeadingZeros(expanded) / 8);
        for (int i = 1; i < bytesSize; i++) {
            bb.put((byte)(0x80 | expanded & 0x7f));
            expanded >>= 8;
        }
        return bb.put((byte)(expanded & 0x7f));
    }

    static ProtoBuf stringField(int fieldIndex, String value) {
        var bytes = value.getBytes(StandardCharsets.UTF_8);
        return bb -> storeVarInt(storeVarInt(bb, fieldIndex << 3 | 2), bytes.length).put(bytes);
    }

    static ProtoBuf intField(int fieldIndex, int value) {
        return bb -> storeVarInt(storeVarInt(bb, fieldIndex << 3), value);
    }

    static ProtoBuf subField(int fieldIndex, ProtoBuf... values) {
        return bb -> {
            int start = storeVarInt(bb, fieldIndex << 3 | 2).position();
            bb.put((byte)0);
            for (var v : values) v.accept(bb);
            int end = bb.position();
            storeVarInt(bb.position(start), end - start - 1); // patch size
            if (bb.position() == start + 1) {
                bb.position(end); // size < 128, all OK
            } else {
                for (var v : values) v.accept(bb); // replay shifted
            }
        };
    }

    static final int IR_VERSION = 10;
    static final int OPSET_VERSION = 14;

    static ByteBuffer unaryOp(String opName, Tensor.ElementType type) {
        var bb = ByteBuffer.allocateDirect(40 + opName.length());
        intField(1, IR_VERSION).accept(bb);
        subField(7, // Graph
                subField(1, stringField(1, "x"), stringField(2, "y"), stringField(4, opName)), // Op node
                subField(11, stringField(1, "x"), subField(2, subField(1, intField(1, type.id)))), // Input
                subField(12, stringField(1, "y"), subField(2, subField(1, intField(1, type.id))))).accept(bb); // Output
        subField(8, intField(2, OPSET_VERSION)).accept(bb); // Opset import
        int len = bb.position();
        return bb.limit(len).asReadOnlyBuffer();
    }

    static ByteBuffer binaryOp(String opName, Tensor.ElementType type) {
        var bb = ByteBuffer.allocateDirect(54 + opName.length());
        intField(1, IR_VERSION).accept(bb);
        subField(7, // Graph
                subField(1, stringField(1, "a"), stringField(1, "b"), stringField(2, "c"), stringField(4, opName)), // Op node
                subField(11, stringField(1, "a"), subField(2, subField(1, intField(1, type.id)))), // Input
                subField(11, stringField(1, "b"), subField(2, subField(1, intField(1, type.id)))), // Input
                subField(12, stringField(1, "c"), subField(2, subField(1, intField(1, type.id))))).accept(bb); // Output
        subField(8, intField(2, OPSET_VERSION)).accept(bb); // Opset import
        int len = bb.position();
        return bb.limit(len).asReadOnlyBuffer();
    }

    public final class Environment {

        private final MemorySegment envAddress,
                                    defaultAllocatorAddress;

        private Environment(MemorySegment envAddress, MemorySegment defaultAllocatorAddress) {
            this.envAddress = envAddress;
            this.defaultAllocatorAddress = defaultAllocatorAddress;
        }

        public OrtTensor runBinaryOp(String opName, Tensor.ElementType elementType, OrtTensor arg1, OrtTensor arg2) {
            // @@ sessions caching and closing
            var session = createSession(binaryOp(opName, elementType));
            return (OrtTensor)session.run(Map.of(session.getInputName(0), arg1, session.getInputName(1), arg2), session.getOutputName(0))[0];
        }

        public Session createSession(String modelPath) {
            return createSession(modelPath, createSessionOptions());
        }

        public Session createSession(String modelPath, SessionOptions options) {
            try {
                return new Session(retAddr(createSession.invokeExact(envAddress, arena.allocateFrom(modelPath), options.sessionOptionsAddress, ret)));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public Session createSession(ByteBuffer model) {
            return createSession(model, createSessionOptions());
        }

        private Session createSession(ByteBuffer model, SessionOptions options) {
            try {
                return new Session(retAddr(createSessionFromArray.invokeExact(envAddress, MemorySegment.ofBuffer(model.rewind()), (long)model.limit(), options.sessionOptionsAddress, ret)));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public final class Session implements AutoCloseable {

            private final MemorySegment sessionAddress;

            private Session(MemorySegment sessionAddress) {
                this.sessionAddress = sessionAddress;
            }

            public int getNumberOfInputs() {
                try {
                    return retInt(sessionGetInputCount.invokeExact(sessionAddress, ret));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            public String getInputName(int inputIndex) {
                try {
                    return retString(sessionGetInputName.invokeExact(sessionAddress, inputIndex, defaultAllocatorAddress, ret));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            public int getNumberOfOutputs() {
                try {
                    return retInt(sessionGetOutputCount.invokeExact(sessionAddress, ret));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            public String getOutputName(int inputIndex) {
                try {
                    return retString(sessionGetOutputName.invokeExact(sessionAddress, inputIndex, defaultAllocatorAddress, ret));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            public OrtTypeInfo getInputTypeInfo(int inputIndex) {
                try {
                    return getTypeInfo(retAddr(sessionGetInputTypeInfo.invokeExact(sessionAddress, inputIndex, ret)));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            public OrtTypeInfo getOutputTypeInfo(int outputIndex) {
                try {
                    return getTypeInfo(retAddr(sessionGetOutputTypeInfo.invokeExact(sessionAddress, outputIndex, ret)));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            private OrtTypeInfo getTypeInfo(MemorySegment typeAddress) {
                try {
                    var type = ONNXType.values()[retInt(getOnnxTypeFromTypeInfo.invokeExact(typeAddress, ret))];
                    return switch (type) {
                        case TENSOR, SPARSETENSOR ->
                            new OrtTensorTypeAndShapeInfo(retAddr(castTypeInfoToTensorInfo.invokeExact(typeAddress, ret)));
                        case SEQUENCE ->
                            new SequenceTypeInfo(retAddr(castTypeInfoToSequenceTypeInfo.invokeExact(typeAddress, ret)));
                        case MAP ->
                            new MapTypeInfo(retAddr(castTypeInfoToMapTypeInfo.invokeExact(typeAddress, ret)));
                        default ->
                            throw new IllegalArgumentException("Invalid element type found in sequence " + type);
                    };
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            public OrtValue[] run(Map<String, OrtValue> inputMap, String... outputNames) {
                var runOptions = MemorySegment.NULL;
                int inputLen = inputMap.size();
                var inputNames = arena.allocate(ADDRESS, inputLen);
                var inputs = arena.allocate(ADDRESS, inputLen);
                long index = 0;
                for (var input : inputMap.entrySet()) {
                    inputNames.setAtIndex(ADDRESS, index, arena.allocateFrom(input.getKey()));
                    inputs.setAtIndex(ADDRESS, index++, input.getValue().valueAddress());
                }
                var outputNamesArr = arena.allocate(ADDRESS, outputNames.length);
                var outputs = arena.allocate(ADDRESS, outputNames.length);
                for (int i = 0; i < outputNames.length; i++) {
                    outputNamesArr.setAtIndex(ADDRESS, i, arena.allocateFrom(outputNames[i]));
                    outputs.setAtIndex(ADDRESS, i, MemorySegment.NULL);
                }
                try {
                    checkStatus(run.invokeExact(sessionAddress, runOptions, inputNames, inputs, (long)inputLen, outputNamesArr, (long)outputNames.length, outputs));
                    var retArr = new OrtValue[outputNames.length];
                    for (int i = 0; i < outputNames.length; i++) {
                        retArr[i] = new OrtTensor(outputs.getAtIndex(ADDRESS, i));
                    }
                    return retArr;
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }

            @Override
            public void close() throws Exception {
                try {
                    checkStatus(releaseSession.invokeExact(sessionAddress));
                } catch (Throwable t) {
                    throw wrap(t);
                }
            }
        }

        public OrtTensor loadTensorFromFlatMemoryMappedDataFile(String file, TensorShape shape, Tensor.ElementType elementType) throws IOException {
            var f = new RandomAccessFile(file, "r");
            return createTensor(f.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, f.length(), arena), shape, elementType);
        }

        OrtTensor createFlatTensor(float... elements) {
            return createTensor(new TensorShape((long)elements.length), elements);
        }

        OrtTensor createTensor(TensorShape shape, float... elements) {
            return createTensor(arena.allocateFrom(JAVA_FLOAT, elements), shape, Tensor.ElementType.FLOAT);
        }

        private OrtTensor createTensor(MemorySegment flatData, TensorShape shape, Tensor.ElementType elementType) {
            try {
                var allocatorInfo = retAddr(allocatorGetInfo.invokeExact(defaultAllocatorAddress, ret));
                return new OrtTensor(retAddr(createTensorWithDataAsOrtValue.invokeExact(allocatorInfo, flatData, flatData.byteSize(), shape.dataAddress, shape.getDimensionsCount(), elementType.id, ret)));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }
    }

    public final class TensorShape {

        private final MemorySegment dataAddress;

        public TensorShape(long... dimensions) {
            this(arena.allocateFrom(JAVA_LONG, dimensions));
        }

        private TensorShape(MemorySegment dataAddress) {
            this.dataAddress = dataAddress;
        }

        public long getDimensionsCount() {
            return dataAddress.byteSize() / JAVA_LONG.byteSize();
        }

        public long getDimension(long index) {
            return dataAddress.getAtIndex(JAVA_LONG, index);
        }
    }

    public sealed interface OrtValue {

        MemorySegment valueAddress();

    }

    public final class OrtTensor implements OrtValue {

        private final MemorySegment valueAddress;

        private OrtTensor(MemorySegment valueAddress) {
            this.valueAddress = valueAddress;
        }

        @Override
        public MemorySegment valueAddress() {
            return valueAddress;
        }

        public OrtTensorTypeAndShapeInfo getTensorTypeAndShape() {
            try {
                return new OrtTensorTypeAndShapeInfo(retAddr(getTensorTypeAndShape.invokeExact(valueAddress, ret)));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public ByteBuffer asByteBuffer() {
            var type = getTensorTypeAndShape();
            long size = type.getTensorShapeElementCount() * type.getTensorElementType().size();
            try {
                return retAddr(getTensorMutableData.invokeExact(valueAddress, ret))
                        .reinterpret(size)
                        .asByteBuffer().order(ByteOrder.nativeOrder());
            } catch (Throwable t) {
                throw wrap(t);
            }
        }
    }

    public SessionOptions createSessionOptions() {
        try {
            return new SessionOptions(retAddr(createSessionOptions.invokeExact(ret)));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public final class SessionOptions {

        private final MemorySegment sessionOptionsAddress;

        public SessionOptions(MemorySegment sessionOptionsAddress) {
            this.sessionOptionsAddress = sessionOptionsAddress;
            setInterOpNumThreads(1);
        }

        public void setInterOpNumThreads(int numThreads) {
            try {
                checkStatus(setInterOpNumThreads.invokeExact(sessionOptionsAddress, numThreads));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }
    }

    public sealed interface OrtTypeInfo {
    }

    public final class OrtTensorTypeAndShapeInfo implements OrtTypeInfo {

        private final MemorySegment infoAddress;

        private OrtTensorTypeAndShapeInfo(MemorySegment infoAddress) {
            this.infoAddress = infoAddress;
        }

        public long getDimensionsCount() {
            try {
                return retLong(getDimensionsCount.invokeExact(infoAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public Tensor.ElementType getTensorElementType() {
            try {
                return Tensor.ElementType.fromOnnxId(retInt(getTensorElementType.invokeExact(infoAddress, ret)));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public long getTensorShapeElementCount() {
            try {
                return retLong(getTensorShapeElementCount.invokeExact(infoAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public TensorShape getShape() {
            long dims = getDimensionsCount();
            var shape = arena.allocate(JAVA_LONG, dims);
            try {
                checkStatus(getDimensions.invokeExact(infoAddress, shape, dims));
            } catch (Throwable t) {
                throw wrap(t);
            }
            return new TensorShape(shape);
        }
    }

    public final class SequenceTypeInfo implements OrtTypeInfo {

        private final MemorySegment infoAddress;

        private SequenceTypeInfo(MemorySegment infoAddress) {
            this.infoAddress = infoAddress;
        }
    }

    public final class MapTypeInfo implements OrtTypeInfo {

        private final MemorySegment infoAddress;

        private MapTypeInfo(MemorySegment infoAddress) {
            this.infoAddress = infoAddress;
        }
    }

    private MemorySegment retAddr(Object res) {
        checkStatus(res);
        return ret.get(ADDR_WITH_ADDR, 0);
    }

    private int retInt(Object res) {
        checkStatus(res);
        return ret.get(JAVA_INT, 0);
    }

    private long retLong(Object res) {
        checkStatus(res);
        return ret.get(JAVA_LONG, 0);
    }

    private String retString(Object res) {
        checkStatus(res);
        return ret.get(ADDR_WITH_STRING, 0).getString(0);
    }

    private void checkStatus(Object res) {
        if (!res.equals(MemorySegment.NULL) && res instanceof MemorySegment status) {
            status = status.reinterpret(Long.MAX_VALUE);
            if (status.get(JAVA_INT, 0) != 0) {
                throw new RuntimeException(status.getString(JAVA_INT.byteSize()));
            }
        }
    }

    private static RuntimeException wrap(Throwable t) {
        return t instanceof RuntimeException e ? e : new UndeclaredThrowableException(t);
    }
}
