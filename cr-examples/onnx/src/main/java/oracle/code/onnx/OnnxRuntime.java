package oracle.code.onnx;

import java.io.File;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import jdk.incubator.code.*;

import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.ClassType;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.type.VarType;
import oracle.code.onnx.compiler.OnnxTransformer;
import oracle.code.onnx.foreign.OrtApi;
import oracle.code.onnx.foreign.OrtApiBase;

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

    @FunctionalInterface
    public interface OnnxFunction<T> extends Supplier<T>, Quotable {
    }

    private static final WeakHashMap<Class<?>, byte[]> MODEL_CACHE = WeakHashMap.newWeakHashMap(10);

    public static <T> Tensor<T> execute(MethodHandles.Lookup l, OnnxFunction<Tensor<T>> codeLambda) {
        var quotable = Op.ofQuotable(codeLambda).orElseThrow();

        var model = MODEL_CACHE.computeIfAbsent(codeLambda.getClass(), _ -> {

            var lambda = (CoreOp.LambdaOp) quotable.op();

            CoreOp.FuncOp onnxFunc;

            // Shortcut for lambda expressions that call just one method
            if (singleMethodInvocation(lambda) instanceof
                    SingleMethod(CoreOp.InvokeOp iop, Map<Value, Value> valueMapping)) {
                Method m;
                try {
                    m = iop.invokeDescriptor().resolveToMethod(l, iop.invokeKind());
                } catch (ReflectiveOperationException e) {
                    throw new RuntimeException(e);
                }

                CoreOp.FuncOp f = Op.ofMethod(m).orElseThrow();
                onnxFunc = OnnxTransformer.transform(l, f);

                // @@@ turn valueMapping into a static shuffle map of captured values indexes

//            arguments = iop.operands().stream().toList().stream()
//                    .map(valueMapping::get)
//                    .map(v -> quotable.capturedValues().get(v))
//                    .map(val -> val instanceof CoreOp.Var<?> v ? v.value() : val)
//                    .map(val -> (Tensor) val)
//                    .toList();
            } else {
                var capturedValues = lambda.capturedValues();
                var functionType = FunctionType.functionType(lambda.invokableType().returnType(),
                        capturedValues.stream().map(Value::type)
                                .map(t -> t instanceof VarType vt ? vt.valueType() : t).toList());
                onnxFunc = OnnxTransformer.transform(l, CoreOp.func("onnxCode", functionType)
                        .body(bb -> {
                            bb.context().mapValues(capturedValues, bb.parameters());
                            for (Op op : lambda.body().entryBlock().ops()) {
                                int i;
                                if (op instanceof CoreOp.VarAccessOp.VarLoadOp load &&
                                        (i = capturedValues.indexOf(load.varOp().result())) >= 0) {
                                    bb.context().mapValue(op.result(), bb.parameters().get(i)); // remap var load result to block param
                                } else {
                                    bb.apply(op);
                                }
                            }
                        }));

            }
            return OnnxProtoBuilder.build(onnxFunc.body().entryBlock());
        });

        // @@@ shuffle captured values according to the above created static map
        List<Tensor> arguments = quotable.capturedValues().values().stream()
                .map(val -> val instanceof CoreOp.Var<?> v ? v.value() : val)
                .map(val -> (Tensor) val)
                .toList();

        try (var session = getInstance().createSession(model)) {
            return session.run(arguments).getFirst();
        }
    }

    record SingleMethod(CoreOp.InvokeOp iop, Map<Value, Value> valueMapping) {}
    static SingleMethod singleMethodInvocation(CoreOp.LambdaOp lop) {
        // Single block
        if (lop.body().blocks().size() > 1) {
            return null;
        }

        Map<Value, Value> valueMapping = new HashMap<>();
        CoreOp.InvokeOp methodRefInvokeOp = extractMethodInvoke(valueMapping, lop.body().entryBlock().ops());
        if (methodRefInvokeOp == null) {
            return null;
        }

        return new SingleMethod(methodRefInvokeOp, valueMapping);
    }

    static CoreOp.InvokeOp extractMethodInvoke(Map<Value, Value> valueMapping, List<Op> ops) {
        CoreOp.InvokeOp methodRefInvokeOp = null;
        for (Op op : ops) {
            switch (op) {
                case CoreOp.VarOp varOp -> {
                    if (isValueUsedWithOp(varOp.result(), o -> o instanceof CoreOp.VarAccessOp.VarStoreOp)) {
                        return null;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                    Value v = varLoadOp.varOp().result();
                    valueMapping.put(varLoadOp.result(), valueMapping.getOrDefault(v, v));
                }
                case CoreOp.InvokeOp iop when isBoxOrUnboxInvocation(iop) -> {
                    Value v = iop.operands().getFirst();
                    valueMapping.put(iop.result(), valueMapping.getOrDefault(v, v));
                }
                case CoreOp.InvokeOp iop -> {
                    if (methodRefInvokeOp != null) {
                        return null;
                    }

                    for (Value o : iop.operands()) {
                        valueMapping.put(o, valueMapping.getOrDefault(o, o));
                    }
                    methodRefInvokeOp = iop;
                }
                case CoreOp.ReturnOp rop -> {
                    if (methodRefInvokeOp == null) {
                        return null;
                    }
                    Value r = rop.returnValue();
                    if (!(valueMapping.getOrDefault(r, r) instanceof Op.Result invokeResult)) {
                        return null;
                    }
                    if (invokeResult.op() != methodRefInvokeOp) {
                        return null;
                    }
                    assert methodRefInvokeOp.result().uses().size() == 1;
                }
                default -> {
                    return null;
                }
            }
        }

        return methodRefInvokeOp;
    }

    private static boolean isValueUsedWithOp(Value value, Predicate<Op> opPredicate) {
        for (Op.Result user : value.uses()) {
            if (opPredicate.test(user.op())) {
                return true;
            }
        }
        return false;
    }

    // @@@ Move to functionality on JavaType(s)
    static final Set<String> UNBOX_NAMES = Set.of(
            "byteValue",
            "shortValue",
            "charValue",
            "intValue",
            "longValue",
            "floatValue",
            "doubleValue",
            "booleanValue");

    private static boolean isBoxOrUnboxInvocation(CoreOp.InvokeOp iop) {
        MethodRef mr = iop.invokeDescriptor();
        return mr.refType() instanceof ClassType ct && ct.unbox().isPresent() &&
                (UNBOX_NAMES.contains(mr.name()) || mr.name().equals("valueOf"));
    }

    public static OnnxRuntime getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new OnnxRuntime();
        }
        return INSTANCE;
    }

    private static final String LOG_ID = "onnx-ffm-java";
    private static OnnxRuntime INSTANCE;

    private final Arena         arena;
    private final MemorySegment runtimeAddress, ret, envAddress, defaultAllocatorAddress;

    private OnnxRuntime() {
        arena = Arena.ofAuto();
        ret = arena.allocate(C_POINTER);
        //  const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t)apiVersion);
        var apiBase = OrtApiBase.reinterpret(OrtGetApiBase(), arena, null);
        runtimeAddress = OrtApi.reinterpret(OrtApiBase.GetApi(apiBase, ORT_API_VERSION()), arena, null);
        envAddress = retAddr(OrtApi.CreateEnv(runtimeAddress, ORT_LOGGING_LEVEL_ERROR(), arena.allocateFrom(LOG_ID), ret));
        defaultAllocatorAddress = retAddr(OrtApi.GetAllocatorWithDefaultOptions(runtimeAddress, ret));
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            OrtApi.ReleaseEnv(runtimeAddress, envAddress);
        }));
    }

    public List<Tensor> runOp(String opName, List<Tensor> inputValues, int numOutputs, Map<String, Object> attributes) {
        var outputNames = IntStream.range(0, numOutputs).mapToObj(o -> "o" + o).toList();
        var protoModel = OnnxProtoBuilder.build(
                IntStream.range(0, inputValues.size()).mapToObj(i -> OnnxProtoBuilder.tensorInfo("i" + i, inputValues.get(i).elementType().id)).toList(),
                List.of(OnnxProtoBuilder.node(
                        opName,
                        IntStream.range(0, inputValues.size()).mapToObj(i -> "i" + i).toList(),
                        outputNames,
                        attributes)),
                outputNames);
        try (var session = createSession(protoModel)) {
            return session.run(inputValues);
        }
    }

    public List<Tensor> run(Block block, List<Tensor> inputValues) {
        var protoModel = OnnxProtoBuilder.build(block);
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

    public Session createSession(byte[] model) {
        return createSession(model, createSessionOptions());
    }

    private Session createSession(byte[] model, SessionOptions options) {
        return new Session(retAddr(OrtApi.CreateSessionFromArray(runtimeAddress, envAddress, arena.allocateFrom(ValueLayout.JAVA_BYTE, model), model.length, options.sessionOptionsAddress, ret)));
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
        public List<Tensor> run(List<Tensor> inputValues) {
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
                retArr[i] = new Tensor(outputs.getAtIndex(C_POINTER, i)
                        .reinterpret(arena, null));
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
        var shapeAddr = shape.length == 0 ? MemorySegment.NULL : arena.allocateFrom(C_LONG_LONG, shape);
        return retAddr(OrtApi.CreateTensorWithDataAsOrtValue(runtimeAddress, allocatorInfo, flatData, flatData.byteSize(), shapeAddr, shape.length, elementType.id, ret));
    }

    public Tensor.ElementType tensorElementType(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        return Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType(runtimeAddress, infoAddr, ret)));
    }

    public long[] tensorShape(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        long dims = retLong(OrtApi.GetDimensionsCount(runtimeAddress, infoAddr, ret));
        var shape = arena.allocate(C_LONG_LONG, dims);
        checkStatus(OrtApi.GetDimensions(runtimeAddress, infoAddr, shape, dims));
        return shape.toArray(C_LONG_LONG);
    }

    public MemorySegment tensorData(MemorySegment tensorAddr) {
        var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape(runtimeAddress, tensorAddr, ret));
        long size = retLong(OrtApi.GetTensorShapeElementCount(runtimeAddress, infoAddr, ret))
                * Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType(runtimeAddress, infoAddr, ret))).size();
        return retAddr(OrtApi.GetTensorMutableData(runtimeAddress, tensorAddr, ret))
                .reinterpret(size);
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
        return ret.get(C_POINTER, 0)
                .reinterpret(arena, null);
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
        if (!status.equals(MemorySegment.NULL)) {
            status = status.reinterpret(Long.MAX_VALUE);
            if (status.get(C_INT, 0) != 0) {
                throw new RuntimeException(status.getString(C_INT.byteSize()));
            }
        }
    }
}
