package oracle.code.onnx;

import java.io.File;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.UndeclaredThrowableException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import jdk.incubator.code.op.CoreOp;

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
    private static final MethodHandle  allocatorGetInfo,
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

    static {
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
    }

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
        try {
            envAddress = retAddr(createEnv.invokeExact(runtimeAddress, LOG_LEVEL, arena.allocateFrom(LOG_ID), ret));
            defaultAllocatorAddress = retAddr(getAllocatorWithDefaultOptions.invokeExact(runtimeAddress, ret));
        } catch (Throwable t) {
            throw wrap(t);
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                checkStatus(releaseEnv.invokeExact(runtimeAddress, envAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }));
    }

    private static MethodHandle handle(int methodIndex, MemoryLayout... args) {
        // create a "virtual" downcall handle with given function descriptor (MS, ...)->R
        var mh = LINKER.downcallHandle(FunctionDescriptor.of(ADDRESS, args));
        // obtain an indexed address method handle getter - (MS, long, long)->MS
        var addressGetter = ADDRESS.arrayElementVarHandle()
                .toMethodHandle(VarHandle.AccessMode.GET);
        // inject provided method index into the address method handle getter - (MS)->MS
        addressGetter = MethodHandles.insertArguments(addressGetter, 1, 0L, methodIndex);
        // filter address argument of virtual downcall handle using the address method handle getter - (MS, ...)->R
        // The resulting method handle expects 'runtimeAddress' as first parameter, and will access it accordingly
        // to find the target address for the downcall
        mh = MethodHandles.filterArguments(mh, 0, addressGetter);
        return mh.asType(mh.type().changeReturnType(Object.class));
    }

    public List<MemorySegment> runOp(String opName, List<MemorySegment> inputValues, int numOutputs, Map<String, Object> attributes) {
        var protoModel = OnnxProtoBuilder.buildOpModel(opName, inputValues.stream().map(this::tensorElementType).toList(), numOutputs, attributes);
        try (var session = createSession(protoModel)) {
            return session.run(inputValues);
        }
    }

    public List<MemorySegment> runFunc(CoreOp.FuncOp model, List<MemorySegment> inputValues) {
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
            return new Session(retAddr(createSession.invokeExact(runtimeAddress, envAddress, arena.allocateFrom(modelPath), options.sessionOptionsAddress, ret)));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public Session createSession(ByteBuffer model) {
        return createSession(model, createSessionOptions());
    }

    private Session createSession(ByteBuffer model, SessionOptions options) {
        try {
            return new Session(retAddr(createSessionFromArray.invokeExact(runtimeAddress, envAddress, MemorySegment.ofBuffer(model.rewind()), (long)model.limit(), options.sessionOptionsAddress, ret)));
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
                return retInt(sessionGetInputCount.invokeExact(runtimeAddress, sessionAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public String getInputName(int inputIndex) {
            try {
                return retString(sessionGetInputName.invokeExact(runtimeAddress, sessionAddress, inputIndex, defaultAllocatorAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public int getNumberOfOutputs() {
            try {
                return retInt(sessionGetOutputCount.invokeExact(runtimeAddress, sessionAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        public String getOutputName(int inputIndex) {
            try {
                return retString(sessionGetOutputName.invokeExact(runtimeAddress, sessionAddress, inputIndex, defaultAllocatorAddress, ret));
            } catch (Throwable t) {
                throw wrap(t);
            }
        }

        // @@@ only tensors are supported yet
        public List<MemorySegment> run(List<MemorySegment> inputValues) {
            var runOptions = MemorySegment.NULL;
            int inputLen = getNumberOfInputs();
            int outputLen = getNumberOfOutputs();
            var inputNames = arena.allocate(ADDRESS, inputLen);
            var inputs = arena.allocate(ADDRESS, inputLen);
            long index = 0;
            for (int i = 0; i < inputLen; i++) {
                inputNames.setAtIndex(ADDRESS, index, arena.allocateFrom(getInputName(i)));
                inputs.setAtIndex(ADDRESS, index++, inputValues.get(i));
            }
            var outputNames = arena.allocate(ADDRESS, outputLen);
            var outputs = arena.allocate(ADDRESS, outputLen);
            for (int i = 0; i < outputLen; i++) {
                outputNames.setAtIndex(ADDRESS, i, arena.allocateFrom(getOutputName(i)));
                outputs.setAtIndex(ADDRESS, i, MemorySegment.NULL);
            }
            try {
                checkStatus(run.invokeExact(runtimeAddress, sessionAddress, runOptions, inputNames, inputs, (long)inputLen, outputNames, (long)outputLen, outputs));
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
                Object o = releaseSession.invokeExact(runtimeAddress, sessionAddress);
            } catch (Throwable t) {
                throw wrap(t);
            }
        }
    }

    public MemorySegment createTensor(MemorySegment flatData, Tensor.ElementType elementType, long[] shape) {
        try {
            var allocatorInfo = retAddr(allocatorGetInfo.invokeExact(runtimeAddress, defaultAllocatorAddress, ret));
            var shapeAddr = shape.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_LONG, shape);
            return retAddr(createTensorWithDataAsOrtValue.invokeExact(runtimeAddress, allocatorInfo, flatData, flatData.byteSize(), shapeAddr, (long)shape.length, elementType.id, ret));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public Tensor.ElementType tensorElementType(MemorySegment tensorAddr) {
        try {
            var infoAddr = retAddr(getTensorTypeAndShape.invokeExact(runtimeAddress, tensorAddr, ret));
            return Tensor.ElementType.fromOnnxId(retInt(getTensorElementType.invokeExact(runtimeAddress, infoAddr, ret)));
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public long[] tensorShape(MemorySegment tensorAddr) {
        try {
            var infoAddr = retAddr(getTensorTypeAndShape.invokeExact(runtimeAddress, tensorAddr, ret));
            long dims = retLong(getDimensionsCount.invokeExact(runtimeAddress, infoAddr, ret));
            var shape = arena.allocate(JAVA_LONG, dims);
            checkStatus(getDimensions.invokeExact(runtimeAddress, infoAddr, shape, dims));
            return shape.toArray(JAVA_LONG);
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public ByteBuffer tensorBuffer(MemorySegment tensorAddr) {
        try {
            var infoAddr = retAddr(getTensorTypeAndShape.invokeExact(runtimeAddress, tensorAddr, ret));
            long size = retLong(getTensorShapeElementCount.invokeExact(runtimeAddress, infoAddr, ret))
                    * Tensor.ElementType.fromOnnxId(retInt(getTensorElementType.invokeExact(runtimeAddress, infoAddr, ret))).size();
            return retAddr(getTensorMutableData.invokeExact(runtimeAddress, tensorAddr, ret))
                    .reinterpret(size)
                    .asByteBuffer().order(ByteOrder.nativeOrder());
        } catch (Throwable t) {
            throw wrap(t);
        }
    }

    public SessionOptions createSessionOptions() {
        try {
            return new SessionOptions(retAddr(createSessionOptions.invokeExact(runtimeAddress, ret)));
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
                checkStatus(setInterOpNumThreads.invokeExact(runtimeAddress, sessionOptionsAddress, numThreads));
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
