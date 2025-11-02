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

	static final boolean DEBUG = Boolean.getBoolean("oracle.code.onnx.OnnxRuntime.DEBUG");
	private static final String LOG_ID = "onnx-ffm-java";
	private static OnnxRuntime INSTANCE;

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
		try (var libStream = OnnxRuntime.class.getResourceAsStream(libResource)) {
			var libFile = File.createTempFile("libonnxruntime", "");
			Path libFilePath = libFile.toPath();
			Files.copy(Objects.requireNonNull(libStream), libFilePath, StandardCopyOption.REPLACE_EXISTING);
			System.load(libFilePath.toAbsolutePath().toString());
			libFile.deleteOnExit();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private final MemorySegment runtimeAddress, ret, envAddress, defaultAllocatorAddress;

	private OnnxRuntime() {
		var arena = Arena.ofAuto();
		ret = arena.allocate(C_POINTER);
		//  const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t)apiVersion);
		var apiBase = OrtApiBase.reinterpret(OrtGetApiBase(), arena, null);
		runtimeAddress = OrtApi.reinterpret(OrtApiBase.GetApi.invoke(OrtApiBase.GetApi(apiBase), ORT_API_VERSION()), arena, null);
		envAddress = retAddr(OrtApi.CreateEnv.invoke(OrtApi.CreateEnv(runtimeAddress), ORT_LOGGING_LEVEL_VERBOSE(), arena.allocateFrom(LOG_ID), ret));
		defaultAllocatorAddress = retAddr(OrtApi.GetAllocatorWithDefaultOptions.invoke(OrtApi.GetAllocatorWithDefaultOptions(runtimeAddress), ret)).reinterpret(arena, null);
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			OrtApi.ReleaseEnv.invoke(OrtApi.ReleaseEnv(runtimeAddress), envAddress);
		}));
	}

	public static OnnxRuntime getInstance() {
		if (INSTANCE == null) {
			INSTANCE = new OnnxRuntime();
		}
		return INSTANCE;
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

	public Session createSession(Arena arena, String modelPath, SessionOptions options) {
		return new Session(arena, retAddr(OrtApi.CreateSession.invoke(OrtApi.CreateSession(runtimeAddress), envAddress, arena.allocateFrom(modelPath), options.sessionOptionsAddress, ret)));
	}

	public Session createSession(Arena arena, byte[] model, SessionOptions options) {
		return new Session(arena, retAddr(OrtApi.CreateSessionFromArray.invoke(OrtApi.CreateSessionFromArray(runtimeAddress), envAddress, arena.allocateFrom(ValueLayout.JAVA_BYTE, model), model.length, options.sessionOptionsAddress, ret)));
	}

	public MemorySegment createTensor(Arena arena, MemorySegment flatData, Tensor.ElementType elementType, long[] shape) {
		var allocatorInfo = retAddr(OrtApi.AllocatorGetInfo.invoke(OrtApi.AllocatorGetInfo(runtimeAddress), defaultAllocatorAddress, ret));
		return retAddr(OrtApi.CreateTensorWithDataAsOrtValue.invoke(
				OrtApi.CreateTensorWithDataAsOrtValue(runtimeAddress),
				allocatorInfo,
				flatData, flatData.byteSize(),
				shape.length == 0 ? MemorySegment.NULL : autoShape(arena, shape, 8l * flatData.byteSize() / elementType.bitSize()), shape.length,
				elementType.id,
				ret)).reinterpret(arena, value -> OrtApi.ReleaseValue.invoke(OrtApi.ReleaseValue(runtimeAddress), value));
	}

	public Tensor.ElementType tensorElementType(MemorySegment tensorAddr) {
		var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape.invoke(OrtApi.GetTensorTypeAndShape(runtimeAddress), tensorAddr, ret));
		return Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType.invoke(OrtApi.GetTensorElementType(runtimeAddress), infoAddr, ret)));
	}

	public long[] tensorShape(MemorySegment tensorAddr) {
		try (var arena = Arena.ofConfined()) {
			var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape.invoke(OrtApi.GetTensorTypeAndShape(runtimeAddress), tensorAddr, ret));
			long dims = retLong(OrtApi.GetDimensionsCount.invoke(OrtApi.GetDimensionsCount(runtimeAddress), infoAddr, ret));
			var shape = arena.allocate(C_LONG_LONG, dims);
			checkStatus(OrtApi.GetDimensions.invoke(OrtApi.GetDimensions(runtimeAddress), infoAddr, shape, dims));
			return shape.toArray(C_LONG_LONG);
		}
	}

	public MemorySegment tensorData(MemorySegment tensorAddr) {
		var infoAddr = retAddr(OrtApi.GetTensorTypeAndShape.invoke(OrtApi.GetTensorTypeAndShape(runtimeAddress), tensorAddr, ret));
		long size = retLong(OrtApi.GetTensorShapeElementCount.invoke(OrtApi.GetTensorShapeElementCount(runtimeAddress), infoAddr, ret))
				* Tensor.ElementType.fromOnnxId(retInt(OrtApi.GetTensorElementType.invoke(OrtApi.GetTensorElementType(runtimeAddress), infoAddr, ret))).bitSize() / 8;
		return retAddr(OrtApi.GetTensorMutableData.invoke(OrtApi.GetTensorMutableData(runtimeAddress), tensorAddr, ret))
				.reinterpret(size);
	}

	public SessionOptions createSessionOptions(Arena arena) {
		return new SessionOptions(retAddr(OrtApi.CreateSessionOptions.invoke(OrtApi.CreateSessionOptions(runtimeAddress), ret))
				.reinterpret(arena, opts -> OrtApi.ReleaseSessionOptions.invoke(OrtApi.ReleaseSessionOptions(runtimeAddress), opts)));
	}

	public void appendExecutionProvider(Arena arena, SessionOptions sessionOptions, OnnxProvider provider) {
		ExecutionProviderOptions executionProviderOptions = toNativeOptions(arena, provider);

		MemorySegment funcPtr = OrtApi.SessionOptionsAppendExecutionProvider(runtimeAddress);
		var status = OrtApi.SessionOptionsAppendExecutionProvider.invoke(
				funcPtr,
				sessionOptions.getSessionOptionsAddress(),
				arena.allocateFrom(provider.name()),
				executionProviderOptions.keySegment, executionProviderOptions.valSegment, executionProviderOptions.size);
		checkStatus(status);
	}

	public void appendExecutionProvider_V2(Arena arena, SessionOptions sessionOptions, OnnxProvider provider) {
		MemorySegment getEpDevicesFn = OrtApi.GetEpDevices(runtimeAddress);
		MemorySegment devicesOut = arena.allocate(ValueLayout.ADDRESS);
		MemorySegment countOut = arena.allocate(ValueLayout.JAVA_LONG);
		checkStatus(OrtApi.GetEpDevices.invoke(getEpDevicesFn, envAddress, devicesOut, countOut));
		long numDevices = countOut.get(ValueLayout.JAVA_LONG, 0);

		MemorySegment devicesBasePtr = devicesOut.get(ValueLayout.ADDRESS, 0);
		MemorySegment devicesArr = (numDevices > 0 && devicesBasePtr != MemorySegment.NULL)
				? devicesBasePtr.reinterpret(numDevices * ValueLayout.ADDRESS.byteSize())
				: MemorySegment.NULL;

		String target = provider.name();
		var matches = new java.util.ArrayList<MemorySegment>();
		if (devicesArr != MemorySegment.NULL) {
			MemorySegment epNameFn = OrtApi.EpDevice_EpName(runtimeAddress);
			for (int j = 0; j < numDevices; j++) {
				MemorySegment dev = devicesArr.getAtIndex(ValueLayout.ADDRESS, j);
				MemorySegment cstr = OrtApi.EpDevice_EpName.invoke(epNameFn, dev);
				String epName = (cstr == MemorySegment.NULL) ? "" : cstr.getString(0);
				if (target.equals(epName)) matches.add(dev);
			}
		}

		if (matches.isEmpty()) {
			appendExecutionProvider(arena, sessionOptions, provider);
		} else {
			ExecutionProviderOptions executionProviderOptions = toNativeOptions(arena, provider);

			long deviceCount = matches.size();
			MemorySegment deviceArrayPtr = arena.allocate(ValueLayout.ADDRESS, deviceCount);
			for (int j = 0; j < matches.size(); j++)
				deviceArrayPtr.setAtIndex(ValueLayout.ADDRESS, j, matches.get(j));

			MemorySegment functionPtr = OrtApi.SessionOptionsAppendExecutionProvider_V2(runtimeAddress);
			checkStatus(OrtApi.SessionOptionsAppendExecutionProvider_V2.invoke(
					functionPtr,
					sessionOptions.getSessionOptionsAddress(),
					envAddress,
					deviceArrayPtr, deviceCount,
					executionProviderOptions.keySegment(), executionProviderOptions.valSegment(), executionProviderOptions.size()
			));
		}
	}

	private static ExecutionProviderOptions toNativeOptions(Arena arena, OnnxProvider provider) {
		var providerOptions = provider.options();
		MemorySegment keySegment = MemorySegment.NULL;
		MemorySegment valSegment = MemorySegment.NULL;
		int size = 0;

		if (Objects.nonNull(providerOptions) && !providerOptions.isEmpty()) {
			size = providerOptions.size();
			keySegment = arena.allocate(ValueLayout.ADDRESS, size);
			valSegment = arena.allocate(ValueLayout.ADDRESS, size);
			int i = 0;

			for (Map.Entry<String, String> e : providerOptions.entrySet()) {
				keySegment.setAtIndex(ValueLayout.ADDRESS, i, arena.allocateFrom(e.getKey()));
				valSegment.setAtIndex(ValueLayout.ADDRESS, i, arena.allocateFrom(e.getValue()));
				i++;
			}
		}
		ExecutionProviderOptions executionProviderOptions = new ExecutionProviderOptions(keySegment, valSegment, size);
		return executionProviderOptions;
	}

	private record ExecutionProviderOptions(MemorySegment keySegment, MemorySegment valSegment, int size) {}

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
			OrtApi.ReleaseStatus.invoke(OrtApi.ReleaseStatus(runtimeAddress), status);
		}
	}

	public final class Session {

		private final MemorySegment sessionAddress;

		private Session(Arena arena, MemorySegment sessionAddress) {
			this.sessionAddress = sessionAddress.reinterpret(arena,
					session -> OrtApi.ReleaseSession.invoke(OrtApi.ReleaseSession(runtimeAddress), session));
		}

		public int getNumberOfInputs() {
			return retInt(OrtApi.SessionGetInputCount.invoke(OrtApi.SessionGetInputCount(runtimeAddress), sessionAddress, ret));
		}

		public String getInputName(int inputIndex) {
			return retString(OrtApi.SessionGetInputName.invoke(OrtApi.SessionGetInputName(runtimeAddress), sessionAddress, inputIndex, defaultAllocatorAddress, ret));
		}

		public int getNumberOfOutputs() {
			return retInt(OrtApi.SessionGetOutputCount.invoke(OrtApi.SessionGetOutputCount(runtimeAddress), sessionAddress, ret));
		}

		public String getOutputName(int inputIndex) {
			return retString(OrtApi.SessionGetOutputName.invoke(OrtApi.SessionGetOutputName(runtimeAddress), sessionAddress, inputIndex, defaultAllocatorAddress, ret));
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
			checkStatus(OrtApi.Run.invoke(OrtApi.Run(runtimeAddress), sessionAddress, runOptions, inputNames, inputs, (long) inputLen, outputNames, (long) outputLen, outputs));
			var retArr = new Tensor[outputLen];
			for (int i = 0; i < outputLen; i++) {
				var tensorAddr = outputs.getAtIndex(C_POINTER, i)
						.reinterpret(arena, value -> OrtApi.ReleaseValue.invoke(OrtApi.ReleaseValue(runtimeAddress), value));
				retArr[i] = new Tensor(tensorData(tensorAddr).reinterpret(arena, null),
						tensorAddr);
			}
			return List.of(retArr);
		}
	}

	public final class SessionOptions {

		private final MemorySegment sessionOptionsAddress;

		public SessionOptions(MemorySegment sessionOptionsAddress) {
			this.sessionOptionsAddress = sessionOptionsAddress;
			setInterOpNumThreads(1);
		}

		public void setInterOpNumThreads(int numThreads) {
			checkStatus(OrtApi.SetInterOpNumThreads.invoke(OrtApi.SetInterOpNumThreads(runtimeAddress), sessionOptionsAddress, numThreads));
		}

		public void setIntraOpNumThreads(int numThreads) {
			checkStatus(OrtApi.SetIntraOpNumThreads.invoke(OrtApi.SetIntraOpNumThreads(runtimeAddress), sessionOptionsAddress, numThreads));
		}

		public void setSessionExecutionMode(int executionMode) {
			checkStatus(OrtApi.SetSessionExecutionMode.invoke(OrtApi.SetSessionExecutionMode(runtimeAddress), sessionOptionsAddress, executionMode));
		}

		public MemorySegment getSessionOptionsAddress() {
			return sessionOptionsAddress;
		}
	}
}
