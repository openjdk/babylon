import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Deque;
import java.util.ArrayDeque;
import java.util.Set;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Queue;
import java.util.Optional;
import java.util.stream.Stream;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.time.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.FileReader;
import java.lang.reflect.Method;
import java.lang.ref.Cleaner;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment.Scope;
import static java.lang.foreign.ValueLayout.*;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.FloatVector;
import static oneapi.levelzero.ze_api_h.*;
import oneapi.levelzero.ze_api_h;
import oneapi.levelzero.ze_context_desc_t;
import oneapi.levelzero.ze_kernel_desc_t;
import oneapi.levelzero.ze_command_queue_desc_t;
import oneapi.levelzero.ze_command_list_desc_t;
import oneapi.levelzero.ze_command_queue_group_properties_t;
import oneapi.levelzero.ze_event_pool_desc_t;
import oneapi.levelzero.ze_event_desc_t;
import oneapi.levelzero.ze_fence_desc_t;
import oneapi.levelzero.ze_module_desc_t;
import oneapi.levelzero.ze_group_count_t;
import oneapi.levelzero.ze_host_mem_alloc_desc_t;
import oneapi.levelzero.ze_device_mem_alloc_desc_t;
import oneapi.levelzero.ze_device_properties_t;
import oneapi.levelzero.ze_device_compute_properties_t;
import oneapi.levelzero.ze_driver_properties_t;
import oneapi.levelzero.ze_driver_extension_properties_t;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Random;

public class LevelZero {
    public static final AddressLayout driver_handle_t = AddressLayout.ADDRESS;
    private final Arena arena;
    private final MemorySegment driverHandle;
    private final MemorySegment contextHandle;
    private final MemorySegment deviceHandle;
    private final MemorySegment queueHandle;
    private final MemorySegment eventPoolDescription;
    private final String homeDir = System.getProperty("user.home");
    private final String cacheDir = homeDir + "/.triton/cache/";
    private final String addKernelCache = "7961f2e8b433c656051d8638d6a3bb65f43f6cb885525c05d611100dd905aa31";
    private final String softmaxKernelCache = "f0c32acd1173759227ef8e0e8d197c94493b90ebf8d1fc254399ffac6b527d6a";
    private final String matmulKernelCache = "07e17c2833c9c9efea8ccd782af1c3ee05dcac3efb2cb75f2f8a6eecffe381ef";
    private final static VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;
    private Double timeElapsedForRun, timeElapsedForRerun;

    static {
        System.loadLibrary("ze_loader");
    }

    static void debug(String format, Object... args) {
        System.out.printf(format + "%n", args);
    }

    private static void check(int result) {
        if (result != ZE_RESULT_SUCCESS()) {
            throw new RuntimeException(String.format("Call failed: 0x%x (%d)", result, result));
        }
    }

    MemorySegment contextHandle() {
        return contextHandle;
    }

    MemorySegment deviceHandle() {
        return deviceHandle;
    }

    public LevelZero() {
        arena = Arena.ofShared();

        // get driver
        check(zeInit(ZE_INIT_FLAG_GPU_ONLY()));
        MemorySegment driverCount = arena.allocate(Integer.BYTES);
        check(zeDriverGet(driverCount, MemorySegment.NULL));
        debug("driverCount = %d", driverCount.get(JAVA_INT, 0));
        MemorySegment driverHandles = arena.allocate(driverCount.get(JAVA_INT, 0) * driver_handle_t.byteSize(), 8);
        check(zeDriverGet(driverCount, driverHandles));
        driverHandle = driverHandles.get(ADDRESS, 0);

        // create context
        MemorySegment pContextDesc = arena.allocate(ze_context_desc_t.layout());
        ze_context_desc_t.stype(pContextDesc, ZE_STRUCTURE_TYPE_CONTEXT_DESC());
        MemorySegment pContextHandle = arena.allocate(ze_context_handle_t);
        check(zeContextCreate(driverHandle, pContextDesc, pContextHandle));
        contextHandle = pContextHandle.get(ADDRESS, 0);

        // get device
        MemorySegment pDeviceCount = arena.allocate(Integer.BYTES);
        check(zeDeviceGet(driverHandle, pDeviceCount, MemorySegment.NULL));
        int deviceCount = pDeviceCount.get(JAVA_INT, 0);
        assert deviceCount > 0;
        debug("deviceCount = %d", deviceCount);
        MemorySegment deviceHandles = arena.allocate(deviceCount * ze_device_handle_t.byteSize(), 8);
        check(zeDeviceGet(driverHandle, pDeviceCount, deviceHandles));
        for (int i = 0; i < deviceCount; i++) {
            debug("device #%d: %s", i, deviceHandles.get(ze_device_handle_t, i * ze_device_handle_t.byteSize()));
        }
        deviceHandle = deviceHandles.get(ze_device_handle_t, 0 * ze_device_handle_t.byteSize());
        MemorySegment pDeviceProperties = arena.allocate(ze_device_properties_t.layout());
        ze_device_properties_t.stype(pDeviceProperties, ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES());
        check(zeDeviceGetProperties(deviceHandle, pDeviceProperties));
        debug("deviceProperties:\n\ttype = %d\n\tvendorId = %d\n\tmaxMemAllocSize = %d\n\tdeviceId = %d\n\tcoreClockRate = %d",
            ze_device_properties_t.type(pDeviceProperties),
            ze_device_properties_t.vendorId(pDeviceProperties),
            ze_device_properties_t.maxMemAllocSize(pDeviceProperties),
            ze_device_properties_t.deviceId(pDeviceProperties),
            ze_device_properties_t.coreClockRate(pDeviceProperties));

        MemorySegment pDeviceComputeProperties = arena.allocate(ze_device_compute_properties_t.layout());
        ze_device_compute_properties_t.stype(pDeviceComputeProperties, ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES());
        check(zeDeviceGetComputeProperties(deviceHandle, pDeviceComputeProperties));
        debug("deviceProperties:\n\tshared = %d\n\tmaxTotalGroupSize = %d",
            ze_device_compute_properties_t.maxSharedLocalMemory(pDeviceComputeProperties),
            ze_device_compute_properties_t.maxTotalGroupSize(pDeviceComputeProperties));

        // create queue
        MemorySegment pNumQueueGroups = arena.allocate(JAVA_INT, 1);
        check(zeDeviceGetCommandQueueGroupProperties(deviceHandle, pNumQueueGroups, MemorySegment.NULL));
        debug("#Queue Groups: %d", pNumQueueGroups.get(JAVA_INT, 0));
        MemorySegment pGroupProperties = arena.allocate(ze_command_queue_group_properties_t.layout(), pNumQueueGroups.get(JAVA_INT, 0));
        check(zeDeviceGetCommandQueueGroupProperties(deviceHandle, pNumQueueGroups, pGroupProperties));

        MemorySegment pQueueDesc = arena.allocate(ze_command_queue_desc_t.layout());
        ze_command_queue_desc_t.stype(pQueueDesc, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC());
        ze_command_queue_desc_t.index(pQueueDesc, 0);
        ze_command_queue_desc_t.mode(pQueueDesc, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS());
        ze_command_queue_desc_t.ordinal(pQueueDesc, 0);
        MemorySegment pQueueHandle = arena.allocate(ze_command_queue_handle_t);
        check(zeCommandQueueCreate(contextHandle, deviceHandle, pQueueDesc, pQueueHandle));
        queueHandle = pQueueHandle.get(ADDRESS, 0);

        eventPoolDescription = arena.allocate(ze_event_pool_desc_t.layout());
        ze_event_pool_desc_t.stype(eventPoolDescription, ZE_STRUCTURE_TYPE_EVENT_POOL_DESC());
        ze_event_pool_desc_t.count(eventPoolDescription, 20);
        ze_event_pool_desc_t.flags(eventPoolDescription, ZE_EVENT_POOL_FLAG_HOST_VISIBLE());

        timeElapsedForRun = timeElapsedForRerun = 0.0;
    }

    public void clear() {
        check(zeCommandQueueDestroy(queueHandle));
        check(zeContextDestroy(contextHandle));
    }

    public void test(String testName) {
        Object[] args = {};
        Random rand = new Random();
        if (testName.equals("add")) {
            String jsonFileName = cacheDir + addKernelCache + "/add_kernel.json";
            String moduleName = cacheDir + addKernelCache + "/add_kernel.spv";

            int BLOCK_SIZE = 64;
            int elementSize = 4096;
            int gridSize = (elementSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            JSONObject jsonObject = loadJson(jsonFileName);
            String kernelName = jsonObject.getString("name");
            int threads_per_warp = jsonObject.getInt("threads_per_warp");
            int num_warps = jsonObject.getInt("num_warps");
            int shared = jsonObject.getInt("shared");

            float[] input1 = new float[elementSize];
            float[] input2 = new float[elementSize];
            float[] output = new float[elementSize];
            for (int i = 0; i < elementSize; i++) {
                input1[i] = rand.nextFloat();
                input2[i] = rand.nextFloat();
            }
            args = new Object[] {input1, input2, output, elementSize};
            run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);

            float[] expected = Test.add(input1, input2, elementSize);
            Test.check(expected, output);
        } else if (testName.equals("softmax")) {
            String jsonFileName = cacheDir + softmaxKernelCache + "/softmax_kernel.json";
            String moduleName = cacheDir + softmaxKernelCache + "/softmax_kernel.spv";

            JSONObject jsonObject = loadJson(jsonFileName);
            String kernelName = jsonObject.getString("name");
            int threads_per_warp = jsonObject.getInt("threads_per_warp");
            int num_warps = jsonObject.getInt("num_warps");
            int shared = jsonObject.getInt("shared");

            int elementSizeX = 4096, elementSizeY = 64;
            int gridSize = elementSizeX;
            float[] input = new float[elementSizeX * elementSizeY];
            float[] output = new float[elementSizeX * elementSizeY];
            byte[] sharedMemory = new byte[shared]; // use for storing temporary value of max element and sum of exp
            for (int i = 0; i < elementSizeX * elementSizeY; i++) {
                input[i] = rand.nextFloat();
            }
            args = new Object[] {output, input, elementSizeY, elementSizeY, elementSizeY, sharedMemory};
            run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);

            float[] expected = Test.softmax(input, elementSizeX, elementSizeY);
            Test.check(expected, output);
        } else if (testName.equals("matmul")) {
            String jsonFileName = cacheDir + matmulKernelCache + "/matmul_kernel.json";
            String moduleName = cacheDir + matmulKernelCache + "/matmul_kernel.spv";

            JSONObject jsonObject = loadJson(jsonFileName);
            String kernelName = jsonObject.getString("name");
            int threads_per_warp = jsonObject.getInt("threads_per_warp");
            int num_warps = jsonObject.getInt("num_warps");
            int shared = jsonObject.getInt("shared");

            int M = 1024, N = 1024, K = 1024;
            int BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 64;
            int gridSize = ((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
            float[] a = new float[M * K];
            float[] b = new float[K * N];
            float[] c = new float[M * N];
            byte[] sharedMemory = new byte[shared];

            for (int i = 0; i < M * K; i++) {
                a[i] = rand.nextFloat();
            }
            for (int i = 0; i < K * N; i++) {
                b[i] = rand.nextFloat();
            }
            args = new Object[] {a, b, c, M, N, K, K, N, N, sharedMemory};
            run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);

            float[] expected = Test.matmul(a, b, M, N, K);
            Test.check(expected, c);
        } else {
            throw new RuntimeException("Unsupported test: " + testName);
        }
    }

    public void run(String kernelName, String fileName, Object[] args, int threads_per_warp, int num_warps, int shared, int gridSize) {
        debug("=========== run %s ===========", kernelName);
        MemorySegment spirvBinary = loadModule(fileName);
        List<Arg> kernelArgs = collectArgs(args);
        int[] globalSizes = new int[] {gridSize * threads_per_warp * num_warps, 1, 1};
        int[] localSizes = new int[] {threads_per_warp * num_warps, 1, 1};
        KernelGeometry geometry = new KernelGeometry(globalSizes, localSizes);
        debug("geometry = %s", geometry);
        MemorySegment commandListHandle = createCommandList(spirvBinary, kernelName, geometry, kernelArgs, shared, false);
        executeCommandList(commandListHandle);
        check(zeCommandQueueSynchronize(queueHandle, -1L));
        for (int i = 0; i < kernelArgs.size(); i++) {
            copyArgToHost(kernelArgs.get(i), contextHandle);
        }

        for (int i = 0; i < kernelArgs.size(); i++) {
            Arg arg = kernelArgs.get(i);
            MemorySegment dataSegment = arg.dataSegment();
            if (dataSegment != null) {
                check(zeMemFree(contextHandle, dataSegment));
            }
        }
        check(zeCommandListDestroy(commandListHandle));
    }

    public void runRefMatmul(String kernelName, String fileName, Object[] args, int size) {
        debug("=========== run %s ===========", kernelName);
        MemorySegment spirvBinary = loadModule(fileName);
        List<Arg> kernelArgs = collectArgs(args);
        int[] globalSizes = new int[] {size, size, 1};
        int[] localSizes = new int[] {512, 1, 1};
        KernelGeometry geometry = new KernelGeometry(globalSizes, localSizes);
        debug("geometry = %s", geometry);
        MemorySegment commandListHandle = createCommandList(spirvBinary, kernelName, geometry, kernelArgs, 0, true);
        executeCommandList(commandListHandle);
        check(zeCommandQueueSynchronize(queueHandle, -1L));
        for (int i = 0; i < kernelArgs.size(); i++) {
            copyArgToHost(kernelArgs.get(i), contextHandle);
        }

        for (int i = 0; i < kernelArgs.size(); i++) {
            Arg arg = kernelArgs.get(i);
            MemorySegment dataSegment = arg.dataSegment();
            if (dataSegment != null) {
                check(zeMemFree(contextHandle, dataSegment));
            }
        }
        check(zeCommandListDestroy(commandListHandle));
    }

    private List<Arg> collectArgs(Object[] values) {
        List<Arg> args = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            args.add(Arg.createArg(this, "arg" + i, values[i]));
        }
        debug("args = %s", args);
        return args;
    }

    MemorySegment loadModule(String fileName) {
        byte[] data = readBytes(fileName);
        MemorySegment segment = arena.allocate(data.length);
        segment.copyFrom(MemorySegment.ofArray(data));
        return segment;
    }

    byte[] readBytes(String filename) {
        File file = new File(filename);
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] data = new byte[(int) file.length()];
            fis.read(data);
            return data;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    void provisionArg(Arg arg) {
        if (arg.cls() == byte[].class) {
            byte[] array = (byte[])arg.value();
            int segmentSize = array.length;
            arg.setDataSegment(allocateSharedSegment(segmentSize));
            arg.dataSegment().copyFrom(MemorySegment.ofArray(array));
            arg.setSize(8);
            arg.setNeedsCleanup(true);
        }
        else if (arg.cls() == short[].class) {
            short[] array = (short[])arg.value();
            int segmentSize = array.length * Short.BYTES;
            arg.setDataSegment(allocateSharedSegment(segmentSize));
            arg.dataSegment().copyFrom(MemorySegment.ofArray(array));
            arg.setSize(8);
            arg.setNeedsCleanup(true);
        }
        else if (arg.cls() == int[].class) {
            int[] array = (int[])arg.value();
            int segmentSize = array.length * Integer.BYTES;
            arg.setDataSegment(allocateSharedSegment(segmentSize));
            arg.dataSegment().copyFrom(MemorySegment.ofArray(array));
            arg.setSize(8);
            arg.setNeedsCleanup(true);
        }
        else if (arg.cls() == float[].class) {
            float[] array = (float[])arg.value();
            int segmentSize = array.length * Float.BYTES;
            arg.setDataSegment(allocateSharedSegment(segmentSize));
            arg.dataSegment().copyFrom(MemorySegment.ofArray(array));
            arg.setSize(8);
            arg.setNeedsCleanup(true);
        }
        else if (VectorSpecies.class.isAssignableFrom(arg.cls())) {
            arg.setSize(4);
        }
        else if (arg.cls() == Short.class) {
            arg.setSize(2);
        }
        else if (arg.cls() == Integer.class || arg.cls() == Float.class || arg.cls() == Boolean.class) {
            arg.setSize(4);
        }
        else if (arg.cls() == Long.class) {
            arg.setSize(8);
        }
        else if (arg.cls() == GPU.Index.class) {
            MemorySegment pBuffer = arena.allocate(ADDRESS);
            arg.setDataSegment(allocateSharedSegment(24));
            arg.setSize(24);
        }
        else throw new RuntimeException("unsupported type: " + arg.cls());
    }

    void copyArgToHost(Arg arg, MemorySegment contextHandle) {
        if (arg.cls() == short[].class) {
            short[] array = (short[])arg.value();
            MemorySegment arraySegment = MemorySegment.ofArray(array);
            arraySegment.copyFrom(arg.dataSegment());
        }
        else if (arg.cls() == int[].class) {
            int[] array = (int[])arg.value();
            MemorySegment arraySegment = MemorySegment.ofArray(array);
            arraySegment.copyFrom(arg.dataSegment());
        }
        else if (arg.cls() == float[].class) {
            float[] array = (float[])arg.value();
            MemorySegment arraySegment = MemorySegment.ofArray(array);
            arraySegment.copyFrom(arg.dataSegment());
        }
        // else nothing to do
    }

    private MemorySegment createCommandList(MemorySegment spirvModule, String kernelName, KernelGeometry geometry, List<Arg> args, int shared, boolean suggested) {
        Arena arena = Arena.ofShared();
        MemorySegment pCommandListHandle = arena.allocate(ze_command_list_handle_t);
        MemorySegment commandListDesc = arena.allocate(ze_command_list_desc_t.layout());
        ze_command_list_desc_t.stype(eventPoolDescription, ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC());
        ze_command_list_desc_t.commandQueueGroupOrdinal(commandListDesc, 0);
        MemorySegment moduleHandle = createModule(kernelName, spirvModule);
        check(zeCommandListCreate(contextHandle, deviceHandle, commandListDesc, pCommandListHandle));
        MemorySegment commandListHandle = pCommandListHandle.get(ADDRESS, 0);
        MemorySegment kernelHandle = createKernel(moduleHandle, kernelName, geometry, suggested);
        for (int i = 0; i < args.size(); i++) {
            Arg arg = args.get(i);
            setKernelArg(arg, i, commandListHandle, kernelHandle, (shared != 0) && (i == args.size() - 1));
        }
        MemorySegment groupCount = arena.allocate(ze_group_count_t.layout());
        ze_group_count_t.groupCountX(groupCount, (geometry.globalSizes()[0] + geometry.localSizes()[0] - 1) / geometry.localSizes()[0]);
        ze_group_count_t.groupCountY(groupCount, (geometry.globalSizes()[1] + geometry.localSizes()[1] - 1) / geometry.localSizes()[1]);
        ze_group_count_t.groupCountZ(groupCount, (geometry.globalSizes()[2] + geometry.localSizes()[2] - 1) / geometry.localSizes()[2]);
        MemorySegment pKernelWaitHandles = MemorySegment.NULL;
        check(zeCommandListAppendLaunchKernel(commandListHandle, kernelHandle, groupCount, MemorySegment.NULL, 0, pKernelWaitHandles));
        check(zeCommandListClose(commandListHandle));
        return commandListHandle;
    }

    private MemorySegment executeCommandList(MemorySegment commandListHandle) {
        MemorySegment fenceDesc = arena.allocate(ze_fence_desc_t.layout());
        ze_module_desc_t.stype(fenceDesc, ZE_STRUCTURE_TYPE_FENCE_DESC());
        ze_fence_desc_t.flags(fenceDesc, ZE_FENCE_FLAG_SIGNALED());
        MemorySegment pFenceHandle = arena.allocate(ze_fence_handle_t);
        check(zeFenceCreate(queueHandle, fenceDesc, pFenceHandle));
        MemorySegment fenceHandle = pFenceHandle.get(ADDRESS, 0);
        MemorySegment pCommandListHandle = arena.allocate(ze_command_list_handle_t);
        pCommandListHandle.set(ADDRESS, 0, commandListHandle);
        Instant start = Instant.now();
        check(zeCommandQueueExecuteCommandLists(queueHandle, 1, pCommandListHandle, fenceHandle));
        check(zeCommandQueueSynchronize(queueHandle, -1L));
        Instant finish = Instant.now();
        Double timeElapsed = Duration.between(start, finish).toNanos() * 1e-6;
        timeElapsedForRun += timeElapsed;
        debug("time: %f %f\n", timeElapsed, timeElapsedForRun);

        start = Instant.now();
        check(zeCommandQueueExecuteCommandLists(queueHandle, 1, pCommandListHandle, fenceHandle));
        check(zeCommandQueueSynchronize(queueHandle, -1L));
        finish = Instant.now();
        timeElapsed = Duration.between(start, finish).toNanos() * 1e-6;
        timeElapsedForRerun += timeElapsed;
        debug("time for rerun: %f %f\n", timeElapsed, timeElapsedForRerun);
        return fenceHandle;
    }

    private MemorySegment createKernel(MemorySegment moduleHandle, String kernelNameString, KernelGeometry geometry, boolean suggested) {
        MemorySegment kernelDesc = arena.allocate(ze_kernel_desc_t.layout());
        MemorySegment kernelName = arena.allocateFrom(kernelNameString);
        ze_kernel_desc_t.stype(kernelDesc, ZE_STRUCTURE_TYPE_KERNEL_DESC());
        ze_kernel_desc_t.pKernelName(kernelDesc, kernelName);
        debug("name = %s", kernelNameString);
        MemorySegment pKernelHandle = arena.allocate(ze_kernel_handle_t);
        check(zeKernelCreate(moduleHandle, kernelDesc, pKernelHandle));
        int[] globalSizes = geometry.globalSizes();
        int[] localSizes = geometry.localSizes();
        MemorySegment kernelHandle = pKernelHandle.get(ADDRESS, 0);
        if (suggested) {
            MemorySegment pGroupSizeX = arena.allocate(JAVA_INT, localSizes[0]);
            MemorySegment pGroupSizeY = arena.allocate(JAVA_INT, localSizes[1]);
            MemorySegment pGroupSizeZ = arena.allocate(JAVA_INT, localSizes[2]);
            check(zeKernelSuggestGroupSize(kernelHandle, globalSizes[0], globalSizes[1], globalSizes[2], pGroupSizeX, pGroupSizeY, pGroupSizeZ));
            geometry.localSizes()[0] = pGroupSizeX.get(JAVA_INT, 0);
            geometry.localSizes()[1] = pGroupSizeY.get(JAVA_INT, 0);
            geometry.localSizes()[2] = pGroupSizeZ.get(JAVA_INT, 0);
            debug("use suggested group size", geometry.toString());
            check(zeKernelSetGroupSize(kernelHandle, pGroupSizeX.get(JAVA_INT, 0), pGroupSizeY.get(JAVA_INT, 0), pGroupSizeZ.get(JAVA_INT, 0)));
        } else {
            debug("use localSizes", geometry.toString());
            check(zeKernelSetGroupSize(kernelHandle, localSizes[0], localSizes[1], localSizes[2]));
        }
        return kernelHandle;
    }

    private void setKernelArg(Arg arg, int ordinal, MemorySegment commandListHandle, MemorySegment kernelHandle, boolean shared) {
        MemorySegment dataSegment = arg.dataSegment();
        Class<?> cls = arg.cls();
        debug("ordinal = %d, cls = %s, data = %s", ordinal, cls.getSimpleName(), dataSegment);
        if (shared) { // shared memory
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, dataSegment.byteSize(), dataSegment));
        }
        else if (cls == byte[].class || cls == short[].class || cls == int[].class || cls == float[].class || cls.getSimpleName().equals("NativeMemorySegmentImpl")) {
            check(zeCommandListAppendMemoryPrefetch(commandListHandle, dataSegment, dataSegment.byteSize()));
            check(zeCommandListAppendMemAdvise(commandListHandle, deviceHandle, dataSegment, dataSegment.byteSize(), ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION()));
            MemorySegment pDataSegment = arena.allocateFrom(ADDRESS, dataSegment);
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, ADDRESS.byteSize(), pDataSegment));
        }
        else if (cls == Short.class) {
            MemorySegment pArgValue = arena.allocateFrom(JAVA_SHORT, (short)arg.value());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Short.BYTES, pArgValue));
        }
        else if (VectorSpecies.class.isAssignableFrom(cls)) {
            MemorySegment pArgValue = arena.allocateFrom(JAVA_INT, FloatVector.SPECIES_256.length());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Integer.BYTES, pArgValue));
        }
        else if (cls == Integer.class || cls == Boolean.class) {
            MemorySegment pArgValue = arena.allocateFrom(JAVA_INT, (int)arg.value());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Integer.BYTES, pArgValue));
        }
        else if (cls == Long.class) {
            MemorySegment pArgValue = arena.allocateFrom(JAVA_LONG, (long)arg.value());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Long.BYTES, pArgValue));
        }
        else if (cls == Float.class) {
            MemorySegment pArgValue = arena.allocateFrom(JAVA_LONG, Float.floatToIntBits((float)arg.value()));
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Float.BYTES, pArgValue));
        }
        else if (cls == GPU.Index.class) {
            MemorySegment pDataSegment = arena.allocateFrom(ADDRESS, dataSegment);
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, 24, pDataSegment));
        }
        else throw new RuntimeException("unsupported type: " + cls);
    }

    private MemorySegment createModule(String moduleName, MemorySegment spirvCode) {
        MemorySegment pModuleHandle = arena.allocate(ze_module_handle_t);
        MemorySegment moduleDesc = arena.allocate(ze_module_desc_t.layout());
        ze_module_desc_t.stype(moduleDesc, ZE_STRUCTURE_TYPE_MODULE_DESC());
        ze_module_desc_t.format(moduleDesc, ZE_MODULE_FORMAT_IL_SPIRV());
        ze_module_desc_t.pInputModule(moduleDesc, spirvCode);
        ze_module_desc_t.inputSize(moduleDesc, spirvCode.byteSize());
        ze_module_desc_t.pBuildFlags(moduleDesc, arena.allocateFrom(""));
        MemorySegment buildLogHandle = arena.allocate(ze_module_build_log_handle_t);
        check(zeModuleCreate(contextHandle, deviceHandle, moduleDesc, pModuleHandle, buildLogHandle));
        MemorySegment moduleHandle = pModuleHandle.get(ADDRESS, 0);
        return moduleHandle;
    }

    public MemorySegment allocateSharedSegment(long byteSize) {
        return allocateSharedSegment(contextHandle(), deviceHandle(), byteSize, Arena.global());
    }

    public static MemorySegment allocateSharedSegment(MemorySegment contextHandle, MemorySegment deviceHandle, long byteSize, Arena arena) {
        MemorySegment pDeviceMemAllocDesc = arena.allocate(ze_device_mem_alloc_desc_t.layout());
        ze_device_mem_alloc_desc_t.stype(pDeviceMemAllocDesc, ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC());
        ze_device_mem_alloc_desc_t.ordinal(pDeviceMemAllocDesc, 0);
        MemorySegment pHostMemAllocDesc = arena.allocate(ze_host_mem_alloc_desc_t.layout());
        ze_host_mem_alloc_desc_t.stype(pHostMemAllocDesc, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC());
        MemorySegment pBuffer = arena.allocate(ADDRESS);
        check(zeMemAllocShared(contextHandle, pDeviceMemAllocDesc, pHostMemAllocDesc, byteSize, 1, deviceHandle, pBuffer));
        long address = pBuffer.get(JAVA_LONG, 0);
        return MemorySegment.ofAddress(address).reinterpret(byteSize);
    }

    private static record KernelGeometry(int[] globalSizes, int[] localSizes) {
        public KernelGeometry() {
            this(new int[3], new int[] {512, 1, 1});
        }

        @Override
        public String toString() {
            return String.format("global: %s, local: %s", Arrays.toString(globalSizes), Arrays.toString(localSizes));
        }
    }

    public void benchAddKernel() {
        String jsonFileName = cacheDir + addKernelCache + "/add_kernel.json";
        String moduleName = cacheDir + addKernelCache + "/add_kernel.spv";
        JSONObject jsonObject = loadJson(jsonFileName);
        String kernelName = jsonObject.getString("name");
        int threads_per_warp = jsonObject.getInt("threads_per_warp");
        int num_warps = jsonObject.getInt("num_warps");
        int shared = jsonObject.getInt("shared");
        Random rand = new Random();

        Writer writer = new Writer("benchmark/vector_add_benchmark.txt");
        writer.write("elementSize timeElapsed timeElapsedForRerun RTT gb/s \n");

        for (int elementSize = (1 << 12); elementSize <= (1 << 28); elementSize <<= 1) {
            int BLOCK_SIZE = 1024;
            int gridSize = (elementSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            float[] input1 = new float[elementSize];
            float[] input2 = new float[elementSize];
            float[] output = new float[elementSize];
            for (int i = 0; i < elementSize; i++) {
                input1[i] = rand.nextFloat();
                input2[i] = rand.nextFloat();
            }
            Object[] args = new Object[] {input1, input2, output, elementSize};

            // warmup
            run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);
            this.timeElapsedForRun = this.timeElapsedForRerun = (double) 0;

            int nTimes = 10;
            Instant start = Instant.now();
            for (int i = 0; i < nTimes; ++i)
                run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);
            Instant finish = Instant.now();
            Double RTT = Duration.between(start, finish).toNanos() * 1e-6 / nTimes;
            Double timeElapsedForRun = this.timeElapsedForRun / nTimes;
            Double timeElapsedForRerun = this.timeElapsedForRerun / nTimes;
            writer.write(String.format("%d %.4f %.4f %.4f %.4f\n", elementSize, timeElapsedForRun, timeElapsedForRerun, RTT, (4f * 3f * elementSize / timeElapsedForRerun * 1e-6)));
        }
        writer.close();
    }

    public void benchSoftmaxKernel() {
        String jsonFileName = cacheDir + softmaxKernelCache + "/softmax_kernel.json";
        String moduleName = cacheDir + softmaxKernelCache + "/softmax_kernel.spv";
        JSONObject jsonObject = loadJson(jsonFileName);
        String kernelName = jsonObject.getString("name");
        int threads_per_warp = jsonObject.getInt("threads_per_warp");
        int num_warps = jsonObject.getInt("num_warps");
        int shared = jsonObject.getInt("shared");
        Random rand = new Random();

        Writer writer = new Writer("benchmark/softmax_benchmark.txt");
        writer.write("elementSizeX elementSizeY timeElapsed timeElapsedForRerun RTT gb/s \n");

        for (int i = 2; i < 50; i++) {
            int elementSizeX = 4096;
            int elementSizeY = 128 * i;
            int gridSize = elementSizeX;
            float[] input = new float[elementSizeX * elementSizeY];
            float[] output = new float[elementSizeX * elementSizeY];
            byte[] sharedMemory = new byte[shared];
            for (int j = 0; j < elementSizeX * elementSizeY; j++) {
                input[j] = rand.nextFloat();
            }
            Object[] args = new Object[] {output, input, elementSizeY, elementSizeY, elementSizeY, sharedMemory};

            // warmup
            run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);
            this.timeElapsedForRun = this.timeElapsedForRerun = (double) 0;

            int nTimes = 10;
            Instant start = Instant.now();
            for (int j = 0; j < nTimes; ++j)
                run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);
            Instant finish = Instant.now();
            Double RTT = Duration.between(start, finish).toNanos() * 1e-6 / nTimes;
            Double timeElapsedForRun = this.timeElapsedForRun / nTimes;
            Double timeElapsedForRerun = this.timeElapsedForRerun / nTimes;
            writer.write(String.format("%d %d %.4f %.4f %.4f %.4f\n", elementSizeX, elementSizeY, timeElapsedForRun, timeElapsedForRerun, RTT, (4 * 2 * 1e-9 * elementSizeX * elementSizeY / (timeElapsedForRerun * 1e-3))));
        }
        writer.close();
    }

    public void benchMatmulKernel() {
        String jsonFileName = cacheDir + matmulKernelCache + "/matmul_kernel.json";
        String moduleName = cacheDir + matmulKernelCache + "/matmul_kernel.spv";

        JSONObject jsonObject = loadJson(jsonFileName);
        String kernelName = jsonObject.getString("name");
        int threads_per_warp = jsonObject.getInt("threads_per_warp");
        int num_warps = jsonObject.getInt("num_warps");
        int shared = jsonObject.getInt("shared");
        Random rand = new Random();
        int BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 64;


        Writer writer = new Writer("benchmark/matmul_benchmark.txt");
        writer.write("M N K timeElapsed timeElapsedForRerun RTT TFLOPS \n");

        for (int i = 2; i <= 64; i++) {
            int M = 128 * i;
            int N = 128 * i;
            int K = 128 * i;
            int gridSize = ((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M) * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

            float[] a = new float[M * K];
            float[] b = new float[K * N];
            float[] c = new float[M * N];
            byte[] sharedMemory = new byte[shared];
            for (int j = 0; j < M * K; j++) {
                a[j] = rand.nextFloat();
            }
            for (int j = 0; j < K * N; j++) {
                b[j] = rand.nextFloat();
            }
            Object[] args = new Object[] {a, b, c, M, N, K, K, N, N, sharedMemory};

            // warmup
            run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);
            this.timeElapsedForRun = this.timeElapsedForRerun = (double) 0;

            int nTimes = 10;
            Instant start = Instant.now();
            for (int j = 0; j < nTimes; ++j)
                run(kernelName, moduleName, args, threads_per_warp, num_warps, shared, gridSize);
            Instant finish = Instant.now();
            Double RTT = Duration.between(start, finish).toNanos() * 1e-6 / nTimes;
            Double timeElapsedForRun = this.timeElapsedForRun / nTimes;
            Double timeElapsedForRerun = this.timeElapsedForRerun / nTimes;
            writer.write(String.format("%d %d %d %.4f %.4f %.4f %.4f\n", M, N, K, timeElapsedForRun, timeElapsedForRerun, RTT, (2 * 1e-12 * M * N * K / (timeElapsedForRerun * 1e-3))));
        }
        writer.close();
    }

    public static void main(String[] args) {
        LevelZero lz = new LevelZero();
        lz.test("add");
        lz.test("softmax");
        lz.test("matmul");
        lz.benchAddKernel();
        lz.benchSoftmaxKernel();
        lz.benchMatmulKernel();
        lz.clear();
    }


    public static class Arg {
        private final String name;
        private final Object value;
        private final Class<?> cls;
        private int size;
        private boolean needsCleanup;
        private MemorySegment dataSegment;

        public static Arg createArg(LevelZero lz, String name, Object value) {
            Arg arg = new Arg(name, value);
            lz.provisionArg(arg);
            return arg;
        }

        private Arg(String name, Object value) {
            this.name = name;
            this.cls = value.getClass();
            this.value = value;
        }

        public String name() {
            return name;
        }

        public Object value() {
            return value;
        }

        public Class<?> cls() {
            return cls;
        }

        public void setSize(int size) {
            this.size = size;
        }

        public int size() {
            return size;
        }

        public void setDataSegment(MemorySegment segment) {
            dataSegment = segment;
        }

        public MemorySegment dataSegment() {
            return dataSegment;
        }

        public void setNeedsCleanup(boolean needsCleanup) {
            this.needsCleanup = needsCleanup;
        }

        public boolean needsCleanup() {
            return needsCleanup;
        }

        public String toString() {
            return String.format("name = %s, cls = %s", name, cls);
        }
    }

    private JSONObject loadJson(String fileName) {
        StringBuilder jsonString = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName)))
        {
            String line;
            while ((line = br.readLine()) != null) {
                jsonString.append(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        JSONObject jsonObject = new JSONObject(jsonString.toString());
        return jsonObject;
    }

    private class Test {
        public static float[] add(float[] a, float[] b, int SIZE) {
            float[] output = new float[SIZE];
            for (int i = 0; i < SIZE; ++i)
                output[i] = a[i] + b[i];
            return output;
        }
        public static float[] softmax(float[] a, int X, int Y) {
            float[] output = new float[X * Y];
            for (int i = 0; i < X; ++i) {
                float max = Float.MIN_VALUE;
                for (int j = 0; j < Y; ++j) {
                    max = Math.max(max, a[i * Y + j]);
                }
                float sum = 0;
                for (int j = 0; j < Y; ++j) {
                    output[i * Y + j] = (float)Math.exp(a[i * Y + j] - max);
                    sum += output[i * Y + j];
                }
                for (int j = 0; j < Y; ++j) {
                    output[i * Y + j] /= sum;
                }
            }
            return output;
        }
        public static float[] matmul(float[] a, float[] b, int M, int N, int K) {
            float[] output = new float[M * N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float tmp = 0;
                    for (int k = 0; k < K; k++) {
                        tmp += a[i * K + k] * b[k * N + j];
                    }
                    output[i * N + j] = tmp;
                }
            }
            return output;
        }
        public static void check(float[] expected, float[] output) {
            for (int i = 0; i < expected.length; i++) {
                if (Math.abs(expected[i] - output[i]) > 1e-2) {
                    System.out.printf("Mismatch at %d: %f != %f%n", i, expected[i], output[i]);
                    throw new RuntimeException("Mismatch");
                }
            }
            System.out.println("Test passed");
        }
    }

    private class Writer {
        private final String fileName;
        private final FileWriter writer;

        public Writer(String fileName) {
            this.fileName = fileName;
            try {
                writer = new FileWriter(fileName, false);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public void write(String line) {
            try {
                writer.write(line);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public void close() {
            try {
                writer.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
