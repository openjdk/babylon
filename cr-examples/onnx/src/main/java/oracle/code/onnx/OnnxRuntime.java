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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import jdk.incubator.code.op.CoreOp;
import oracle.code.onnx.ir.OnnxOp;

import static java.lang.foreign.ValueLayout.*;

public final class OnnxRuntime {

    private static final int ORT_VERSION = 20;
    private static final int LOG_LEVEL = 3; // 0 - verbose, 1 - info, 2 - warning, 3 - error, 4 - fatal
    private static final String LOG_ID = "onnx-ffm-java";

    public static OnnxRuntime getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new OnnxRuntime();
        }
        return INSTANCE;
    }

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

    private static OnnxRuntime INSTANCE;

    private final Arena         arena;
    private final SymbolLookup  library;
    private final MemorySegment runtimeAddress, ret, envAddress, defaultAllocatorAddress;
    private final MethodHandle  allocatorGetInfo,
                                createTensorWithDataAsOrtValue,
                                createEnv,
                                createSession,
                                createSessionFromArray,
                                createSessionOptions,
                                getAllocatorWithDefaultOptions,
                                getDimensions,
                                getDimensionsCount,
                                getTensorElementType,
                                getTensorMutableData,
                                getTensorShapeElementCount,
                                getTensorTypeAndShape,
                                releaseEnv,
                                releaseSession,
                                run,
                                sessionGetInputCount,
                                sessionGetInputName,
                                sessionGetOutputCount,
                                sessionGetOutputName,
                                setInterOpNumThreads;

    private OnnxRuntime() {
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
                            FunctionDescriptor.of(JAVA_LONG, JAVA_INT)).invokeExact(ORT_VERSION))
                    .reinterpret(285 * ADDRESS.byteSize());
        } catch (Throwable t) {
            throw wrap(t);
        }
        allocatorGetInfo               = handle( 77, ADDRESS, ADDRESS);
        createTensorWithDataAsOrtValue = handle( 49, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS);
        createEnv                      = handle(  3, JAVA_INT, ADDRESS, ADDRESS);
        createSession                  = handle(  7, ADDRESS, ADDRESS, ADDRESS, ADDRESS);
        createSessionFromArray         = handle(  8, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS);
        createSessionOptions           = handle( 10, ADDRESS);
        getAllocatorWithDefaultOptions = handle( 78, ADDRESS);
        getDimensions                  = handle( 62, ADDRESS, ADDRESS, JAVA_LONG);
        getDimensionsCount             = handle( 61, ADDRESS, ADDRESS);
        getTensorElementType           = handle( 60, ADDRESS, ADDRESS);
        getTensorMutableData           = handle( 51, ADDRESS, ADDRESS);
        getTensorShapeElementCount     = handle( 64, ADDRESS, ADDRESS);
        getTensorTypeAndShape          = handle( 65, ADDRESS, ADDRESS);
        releaseEnv                     = handle( 92, ADDRESS, ADDRESS);
        releaseSession                 = handle( 95, ADDRESS);
        run                            = handle(  9, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS);
        sessionGetInputCount           = handle( 30, ADDRESS, ADDRESS);
        sessionGetInputName            = handle( 36, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        sessionGetOutputCount          = handle( 31, ADDRESS, ADDRESS);
        sessionGetOutputName           = handle( 37, ADDRESS, JAVA_INT, ADDRESS, ADDRESS);
        setInterOpNumThreads           = handle( 25, ADDRESS, JAVA_INT);
        try {
            envAddress = retAddr(createEnv.invokeExact(LOG_LEVEL, arena.allocateFrom(LOG_ID), ret));
            defaultAllocatorAddress = retAddr(getAllocatorWithDefaultOptions.invokeExact(ret));
        } catch (Throwable t) {
            throw wrap(t);
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                checkStatus(releaseEnv.invokeExact(envAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }));
    }

    private MethodHandle handle(int methodIndex, MemoryLayout... args) {
        var mh = LINKER.downcallHandle((MemorySegment)VH_ADDRESS.get(runtimeAddress, methodIndex * ADDRESS.byteSize()),
                                     FunctionDescriptor.of(ADDRESS, args));
        return mh.asType(mh.type().changeReturnType(Object.class));
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
            try {
                checkStatus(run.invokeExact(sessionAddress, runOptions, inputNames, inputs, (long)inputLen, outputNames, (long)outputLen, outputs));
                var retArr = new MemorySegment[outputLen];
                for (int i = 0; i < outputLen; i++) {
                    retArr[i] = outputs.getAtIndex(ADDRESS, i);
                }
                return List.of(retArr);
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        @Override
        public void close() {
            try {
                checkStatus(releaseSession.invokeExact(sessionAddress));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }
    }

    public MemorySegment loadFlatTensorFromMemoryMappedDataFile(String file, Tensor.ElementType elementType) throws IOException {
        var f = new RandomAccessFile(file, "r");
        return createTensor(f.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, f.length(), arena), elementType, new long[]{f.length() / elementType.size()});
    }

    public MemorySegment createScalar(long element) {
        return createScalar(arena.allocateFrom(JAVA_LONG, element), Tensor.ElementType.INT64);
    }

    public MemorySegment createScalar(float element) {
        return createScalar(arena.allocateFrom(JAVA_FLOAT, element), Tensor.ElementType.FLOAT);
    }

    private MemorySegment createScalar(MemorySegment flatData, Tensor.ElementType elementType) {
        try {
            var allocatorInfo = retAddr(allocatorGetInfo.invokeExact(defaultAllocatorAddress, ret));
            return retAddr(createTensorWithDataAsOrtValue.invokeExact(allocatorInfo, flatData, flatData.byteSize(), MemorySegment.NULL, 0l, elementType.id, ret));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public MemorySegment createFlatTensor(long... elements) {
        return createTensor(arena.allocateFrom(JAVA_LONG, elements), Tensor.ElementType.INT64, new long[]{elements.length});
    }

    public MemorySegment createFlatTensor(float... elements) {
        return createTensor(arena.allocateFrom(JAVA_FLOAT, elements), Tensor.ElementType.FLOAT, new long[]{elements.length});
    }

    public MemorySegment createTensor(MemorySegment flatData, Tensor.ElementType elementType, long[] shape) {
        try {
            var allocatorInfo = retAddr(allocatorGetInfo.invokeExact(defaultAllocatorAddress, ret));
            var shapeAddr = arena.allocateFrom(JAVA_LONG, shape);
            return retAddr(createTensorWithDataAsOrtValue.invokeExact(allocatorInfo, flatData, flatData.byteSize(), shapeAddr, (long)shape.length, elementType.id, ret));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public Tensor.ElementType tensorElementType(MemorySegment tensorAddr) {
        try {
            var infoAddr = retAddr(getTensorTypeAndShape.invokeExact(tensorAddr, ret));
            return Tensor.ElementType.fromOnnxId(retInt(getTensorElementType.invokeExact(infoAddr, ret)));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public long[] tensorShape(MemorySegment tensorAddr) {
        try {
            var infoAddr = retAddr(getTensorTypeAndShape.invokeExact(tensorAddr, ret));
            long dims = retLong(getDimensionsCount.invokeExact(infoAddr, ret));
            var shape = arena.allocate(JAVA_LONG, dims);
            checkStatus(getDimensions.invokeExact(infoAddr, shape, dims));
            return shape.toArray(JAVA_LONG);
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public ByteBuffer tensorBuffer(MemorySegment tensorAddr) {
        try {
            var infoAddr = retAddr(getTensorTypeAndShape.invokeExact(tensorAddr, ret));
            long size = retLong(getTensorShapeElementCount.invokeExact(infoAddr, ret))
                    * Tensor.ElementType.fromOnnxId(retInt(getTensorElementType.invokeExact(infoAddr, ret))).size();
            return retAddr(getTensorMutableData.invokeExact(tensorAddr, ret))
                    .reinterpret(size)
                    .asByteBuffer().order(ByteOrder.nativeOrder());
        } catch (Throwable t) {
            throw wrap(t);
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
