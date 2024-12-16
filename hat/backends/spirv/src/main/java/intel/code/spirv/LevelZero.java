/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import oneapi.levelzero.ze_api_h;
import static oneapi.levelzero.ze_api_h.*;
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
import oneapi.levelzero.ze_driver_properties_t;
import oneapi.levelzero.ze_driver_extension_properties_t;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.Block;
import jdk.incubator.code.Value;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.callgraph.KernelEntrypoint;
import hat.callgraph.KernelCallGraph;
import intel.code.spirv.SpirvModuleGenerator;
import intel.code.spirv.SpirvOp;
import intel.code.spirv.TranslateToSpirvModel;
import intel.code.spirv.UsmArena;
import java.lang.invoke.MethodHandles;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.AddressLayout;
import static java.lang.foreign.ValueLayout.*;

public class LevelZero {
    static {
        System.loadLibrary("ze_loader");
    }

    public static final AddressLayout driver_handle_t = AddressLayout.ADDRESS;
    public static final AddressLayout context_handle_t = AddressLayout.ADDRESS;
    public static final AddressLayout device_handle_t = AddressLayout.ADDRESS;
    public static final AddressLayout command_queue_handle_t = AddressLayout.ADDRESS;

    private final Arena lzArena;
    private final Arena backendArena;
    private final MemorySegment driverHandle;
    private final MemorySegment contextHandle;
    private final MemorySegment deviceHandle;
    private final MemorySegment queueHandle;
    private final MemorySegment eventPoolDescription;

    public static LevelZero create(Arena javaArena) {
        return new LevelZero(javaArena);
    }

    LevelZero(Arena javaArena) {
        lzArena = javaArena;
        // get driver
        check(zeInit(ZE_INIT_FLAG_GPU_ONLY()));
        int[] numDrivers = new int[1];
        MemorySegment pNumDrivers = lzArena.allocate(JAVA_INT);
        check(zeDriverGet(pNumDrivers, MemorySegment.NULL));
        MemorySegment driverCount = lzArena.allocate(Integer.BYTES);
        check(zeDriverGet(driverCount, MemorySegment.NULL));
        debug("driverCount = %d", driverCount.get(JAVA_INT, 0));
        MemorySegment driverHandles = lzArena.allocate(driverCount.get(JAVA_INT, 0) * driver_handle_t.byteSize(), 8);
        check(zeDriverGet(driverCount, driverHandles));
        driverHandle = driverHandles.get(ADDRESS, 0);

        // create context
        MemorySegment pContextDesc = lzArena.allocate(ze_context_desc_t.layout());
        ze_context_desc_t.stype(pContextDesc, ZE_STRUCTURE_TYPE_CONTEXT_DESC());
        MemorySegment pContextHandle = lzArena.allocate(context_handle_t);
        check(zeContextCreate(driverHandle, pContextDesc, pContextHandle));
        contextHandle = pContextHandle.get(ADDRESS, 0);

        // get device
        MemorySegment pDeviceCount = lzArena.allocate(Integer.BYTES);
        check(zeDeviceGet(driverHandle, pDeviceCount, MemorySegment.NULL));
        int deviceCount = pDeviceCount.get(JAVA_INT, 0);
        assert deviceCount > 0;
        debug("deviceCount = %d", deviceCount);
        MemorySegment deviceHandles = lzArena.allocate(deviceCount * device_handle_t.byteSize(), 8);
        check(zeDeviceGet(driverHandle, pDeviceCount, deviceHandles));
        for (int i = 0; i < deviceCount; i++) {
            debug("device #%d: %s", i, deviceHandles.get(device_handle_t, i * device_handle_t.byteSize()));
        }
        assert deviceCount == 1;
        deviceHandle = deviceHandles.get(ADDRESS, 0);
        MemorySegment pDeviceProperties = lzArena.allocate(ze_device_properties_t.layout());
        ze_device_properties_t.stype(pDeviceProperties, ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES());
        check(zeDeviceGetProperties(deviceHandle, pDeviceProperties));
        debug("deviceProperties:\n\ttype = %d\n\tvendorId = %d\n\tmaxMemAllocSize = %d",
            ze_device_properties_t.type(pDeviceProperties),
            ze_device_properties_t.vendorId(pDeviceProperties),
            ze_device_properties_t.maxMemAllocSize(pDeviceProperties));

        // create queue
        MemorySegment pNumQueueGroups = lzArena.allocate(JAVA_INT, 1);
        check(zeDeviceGetCommandQueueGroupProperties(deviceHandle, pNumQueueGroups, MemorySegment.NULL));
        assert pNumQueueGroups.get(JAVA_INT, 0) == 1;
        MemorySegment pGroupProperties = lzArena.allocate(ze_command_queue_group_properties_t.layout(),
                                                          pNumQueueGroups.get(JAVA_INT, 0));
        check(zeDeviceGetCommandQueueGroupProperties(deviceHandle, pNumQueueGroups, pGroupProperties));

        MemorySegment pQueueDesc = lzArena.allocate(ze_command_queue_desc_t.layout());
        ze_command_queue_desc_t.stype(pQueueDesc, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC());
        ze_command_queue_desc_t.index(pQueueDesc, 0);
        ze_command_queue_desc_t.mode(pQueueDesc, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS());
        ze_command_queue_desc_t.ordinal(pQueueDesc, 0);
        MemorySegment pQueueHandle = lzArena.allocate(command_queue_handle_t);
        check(zeCommandQueueCreate(contextHandle, deviceHandle, pQueueDesc, pQueueHandle));
        queueHandle = pQueueHandle.get(ADDRESS, 0);

        // create event pool (may not use events if use in-order queue)
        eventPoolDescription = lzArena.allocate(ze_event_pool_desc_t.layout().byteSize());
        ze_event_pool_desc_t.stype(eventPoolDescription, ZE_STRUCTURE_TYPE_EVENT_POOL_DESC());
        ze_event_pool_desc_t.count(eventPoolDescription, 20);
        ze_event_pool_desc_t.flags(eventPoolDescription, ZE_EVENT_POOL_FLAG_HOST_VISIBLE());

        backendArena = new UsmArena(contextHandle, deviceHandle);
    }

    public Arena arena() {
        return backendArena;
    }

    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        KernelEntrypoint kernelEntrypoint = kernelCallGraph.entrypoint;
        CoreOp.FuncOp funcOp = kernelEntrypoint.funcOpWrapper().op();
        String kernelName = funcOp.funcName();
        // System.out.println(funcOp.toText());
        MemorySegment spirvBinary = SpirvModuleGenerator.generateModule(kernelName, kernelCallGraph);
        String path = "/tmp/" + kernelName + ".spv";
        SpirvModuleGenerator.writeModuleToFile(spirvBinary, path);
        // System.out.println("generated module \n" + SpirvModuleGenerator.disassembleModule(spirvBinary));
        args[0] = ndRange.kid;
        List<Arg> kernelArgs = collectArgs(funcOp, args);
        int[] globalSizes = new int[] {ndRange.kid.maxX, 1, 1};
        int[] localSizes = new int[] {512, 1, 1};
        KernelGeometry geometry = new KernelGeometry(globalSizes, localSizes);
        MemorySegment commandListHandle = createCommandList(spirvBinary, kernelName, geometry, kernelArgs);
        executeCommandList(commandListHandle);
        check(zeCommandQueueSynchronize(queueHandle, -1L));
        for (int i = 1; i < kernelArgs.size(); i++) {
            copyArgToHost(kernelArgs.get(i), contextHandle);
        }
    }

    private static void check(int result) {
        if (result != ZE_RESULT_SUCCESS()) {
            throw new RuntimeException(String.format("Call failed: 0x%x (%d)", result, result));
        }
    }

    public static void debug(String message, Object... args) {
        System.out.println(String.format(message, args));
    }

    private List<Arg> collectArgs(CoreOp.FuncOp kernelMethod, Object[] values) {
        List<Arg> args = new ArrayList<>();
        List<Block.Parameter> params = kernelMethod.body().entryBlock().parameters();
        for (int i = 0; i < params.size(); i++) {
            args.add(Arg.createArg(this, "arg" + i, params.get(i), values[i]));
        }
        return args;
    }

    void provisionArg(Arg arg) {
        MemorySegment deviceMemAllocDesc = lzArena.allocate(ze_device_mem_alloc_desc_t.layout());
        ze_device_mem_alloc_desc_t.stype(deviceMemAllocDesc, ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC());
        ze_device_mem_alloc_desc_t.ordinal(deviceMemAllocDesc, 0);
        MemorySegment hostMemAllocDesc = lzArena.allocate(ze_host_mem_alloc_desc_t.layout());
        ze_host_mem_alloc_desc_t.stype(hostMemAllocDesc, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC());
        if (arg.value() instanceof Buffer buff) {
            arg.setNeedsCleanup(false);
            arg.setDataSegment(Buffer.getMemorySegment(buff));
            arg.setSize(8);
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
        else if (arg.cls() == KernelContext.class) {
            KernelContext kc = (KernelContext)arg.value();
            MemorySegment segment = lzArena.allocate(8, 8);
            segment.set(JAVA_INT, 0, kc.maxX);
            segment.set(JAVA_INT, 4, kc.x);
            arg.setDataSegment(segment);
            arg.setSize(8);
        }
        else if (arg.cls() == NDRange.class) {
            NDRange kc = (NDRange)arg.value();
            MemorySegment segment = lzArena.allocate(8, 8);
            arg.setDataSegment(segment);
            arg.setSize(8);
        }
        else throw new RuntimeException("unsupported type: " + arg.cls());
    }

    void copyArgToHost(Arg arg, MemorySegment contextHandle) {
        // currently using shared memory
    }

    private MemorySegment createCommandList(MemorySegment spirvModule, String kernelName, KernelGeometry geometry, List<Arg> args) {
        Arena arena = Arena.ofShared();
        MemorySegment pCommandListHandle = lzArena.allocate(ze_command_list_handle_t);
        MemorySegment commandListDesc = lzArena.allocate(ze_command_list_desc_t.layout());
        ze_command_list_desc_t.stype(eventPoolDescription, ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC());
        ze_command_list_desc_t.commandQueueGroupOrdinal(commandListDesc, 0);
        check(zeCommandListCreate(contextHandle, deviceHandle, commandListDesc, pCommandListHandle));
        MemorySegment commandListHandle = pCommandListHandle.get(ADDRESS, 0);
        MemorySegment moduleHandle = createModule(kernelName);
        MemorySegment kernelHandle = createKernel(moduleHandle, kernelName, geometry);
        for (int i = 0; i < args.size(); i++) {
            Arg arg = args.get(i);
            setKernelArg(arg, i, commandListHandle, kernelHandle);
        }
        MemorySegment groupCount = lzArena.allocate(ze_group_count_t.layout());
        ze_group_count_t.groupCountX(groupCount, geometry.globalSizes()[0] / geometry.localSizes()[0]);
        ze_group_count_t.groupCountY(groupCount, geometry.globalSizes()[1] / geometry.localSizes()[1]);
        ze_group_count_t.groupCountZ(groupCount, geometry.globalSizes()[2] / geometry.localSizes()[2]);
        MemorySegment pKernelWaitHandles = MemorySegment.NULL;
        check(zeCommandListAppendLaunchKernel(commandListHandle, kernelHandle, groupCount, MemorySegment.NULL, 0, pKernelWaitHandles));
        check(zeCommandListClose(commandListHandle));
        return commandListHandle;
    }

    private MemorySegment executeCommandList(MemorySegment commandListHandle) {
        MemorySegment fenceDesc = lzArena.allocate(ze_fence_desc_t.layout());
        ze_module_desc_t.stype(fenceDesc, ZE_STRUCTURE_TYPE_FENCE_DESC());
        ze_fence_desc_t.flags(fenceDesc, ZE_FENCE_FLAG_SIGNALED());
        MemorySegment pFenceHandle = lzArena.allocate(ze_fence_handle_t);
        check(zeFenceCreate(queueHandle, fenceDesc, pFenceHandle));
        MemorySegment fenceHandle = pFenceHandle.get(ADDRESS, 0);
        MemorySegment pCommandListHandle = lzArena.allocate(ze_command_list_handle_t);
        pCommandListHandle.set(ADDRESS, 0, commandListHandle);
        check(zeCommandQueueExecuteCommandLists(queueHandle, 1, pCommandListHandle, fenceHandle));
        check(zeCommandQueueSynchronize(queueHandle, -1L));
        return fenceHandle;
    }

    private MemorySegment createKernel(MemorySegment moduleHandle, String kernelNameString, KernelGeometry geometry) {
        MemorySegment kernelDesc = lzArena.allocate(ze_kernel_desc_t.layout());
        MemorySegment kernelName = lzArena.allocateFrom(kernelNameString);
        ze_kernel_desc_t.stype(kernelDesc, ZE_STRUCTURE_TYPE_KERNEL_DESC());
        ze_kernel_desc_t.pKernelName(kernelDesc, kernelName);
        MemorySegment pKernelHandle = lzArena.allocate(ze_kernel_handle_t);
        check(zeKernelCreate(moduleHandle, kernelDesc, pKernelHandle));
        int[] globalSizes = geometry.globalSizes();
        int[] localSizes = geometry.localSizes();
        MemorySegment pGroupSizeX = lzArena.allocate(JAVA_INT, localSizes[0]);
        MemorySegment pGroupSizeY = lzArena.allocate(JAVA_INT, localSizes[1]);
        MemorySegment pGroupSizeZ = lzArena.allocate(JAVA_INT, localSizes[2]);
        MemorySegment kernelHandle = pKernelHandle.get(ADDRESS, 0);
        check(zeKernelSuggestGroupSize(kernelHandle, globalSizes[0], globalSizes[1], globalSizes[2], pGroupSizeX, pGroupSizeY, pGroupSizeZ));
        geometry.localSizes()[0] = pGroupSizeX.get(JAVA_INT, 0);
        geometry.localSizes()[1] = pGroupSizeY.get(JAVA_INT, 0);
        geometry.localSizes()[2] = pGroupSizeZ.get(JAVA_INT, 0);
        check(zeKernelSetGroupSize(kernelHandle, pGroupSizeX.get(JAVA_INT, 0), pGroupSizeY.get(JAVA_INT, 0), pGroupSizeZ.get(JAVA_INT, 0)));
        return kernelHandle;
    }

    private void setKernelArg(Arg arg, int ordinal, MemorySegment commandListHandle, MemorySegment kernelHandle) {
        MemorySegment dataSegment = arg.dataSegment();
        Class<?> cls = arg.cls();
        if (cls == Short.class) {
            MemorySegment pArgValue = lzArena.allocateFrom(JAVA_SHORT, (short)arg.value());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Short.BYTES, pArgValue));
        }
        else if (cls == Integer.class || cls == Boolean.class /*|| VectorSpecies.class.isAssignableFrom(cls)*/) {
            MemorySegment pArgValue = lzArena.allocateFrom(JAVA_INT, (int)arg.value());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Integer.BYTES, pArgValue));
        }
        else if (cls == Long.class) {
            MemorySegment pArgValue = lzArena.allocateFrom(JAVA_LONG, (long)arg.value());
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Long.BYTES, pArgValue));
        }
        else if (cls == Float.class) {
            MemorySegment pArgValue = lzArena.allocateFrom(JAVA_LONG, Float.floatToIntBits((float)arg.value()));
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, Float.BYTES, pArgValue));
        }
        else if (arg.value() instanceof Buffer) {
            MemorySegment pDataSegment = lzArena.allocateFrom(ADDRESS, dataSegment);
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, ADDRESS.byteSize(), pDataSegment));
        }
        else if (arg.value() instanceof KernelContext) {
            MemorySegment pDataSegment = lzArena.allocateFrom(ADDRESS, dataSegment);
            check(zeKernelSetArgumentValue(kernelHandle, ordinal, ADDRESS.byteSize(), pDataSegment));
        }
        else throw new RuntimeException("unsupported type: " + cls);
    }

    private MemorySegment createModule(String moduleName, MemorySegment spirvCode) {
        MemorySegment pModuleHandle = lzArena.allocate(ze_module_handle_t);
        MemorySegment moduleDesc = lzArena.allocate(ze_module_desc_t.layout());
        ze_module_desc_t.stype(moduleDesc, ZE_STRUCTURE_TYPE_MODULE_DESC());
        ze_module_desc_t.format(moduleDesc, ZE_MODULE_FORMAT_IL_SPIRV());
        ze_module_desc_t.pInputModule(moduleDesc, spirvCode);
        ze_module_desc_t.inputSize(moduleDesc, spirvCode.byteSize());
        ze_module_desc_t.pBuildFlags(moduleDesc, lzArena.allocateFrom(""));
        MemorySegment buildLogHandle = lzArena.allocate(ze_module_build_log_handle_t);
        check(zeModuleCreate(contextHandle, deviceHandle, moduleDesc, pModuleHandle, buildLogHandle));
        MemorySegment moduleHandle = pModuleHandle.get(ADDRESS, 0);
        return moduleHandle;
    }

    private MemorySegment createModule(String moduleName)
    {
        try
        {
            MemorySegment codeBytes = MemorySegment.ofArray(Files.readAllBytes(Paths.get("/tmp/" + moduleName + ".spv")));
            MemorySegment spirvCode = lzArena.allocate(codeBytes.byteSize());
            MemorySegment.copy(codeBytes, 0, spirvCode, 0, spirvCode.byteSize());
            MemorySegment pModuleHandle = lzArena.allocate(ze_module_handle_t);
            MemorySegment moduleDesc = lzArena.allocate(ze_module_desc_t.layout());
            ze_module_desc_t.stype(moduleDesc, ZE_STRUCTURE_TYPE_MODULE_DESC());
            ze_module_desc_t.format(moduleDesc, ZE_MODULE_FORMAT_IL_SPIRV());
            ze_module_desc_t.pInputModule(moduleDesc, spirvCode);
            ze_module_desc_t.inputSize(moduleDesc, spirvCode.byteSize());
            ze_module_desc_t.pBuildFlags(moduleDesc, lzArena.allocateFrom(""));
            MemorySegment pbuildLogHandle = lzArena.allocate(ze_module_build_log_handle_t);
            int status = zeModuleCreate(contextHandle, deviceHandle, moduleDesc, pModuleHandle, pbuildLogHandle);
            if (status != ZE_RESULT_SUCCESS()) {
                MemorySegment pSize = lzArena.allocate(JAVA_INT);
                MemorySegment buildLogHandle = pbuildLogHandle.get(ADDRESS, 0);
                zeModuleBuildLogGetString(buildLogHandle, pSize, MemorySegment.NULL);
                MemorySegment buildLog = lzArena.allocate(pSize.get(JAVA_INT, 0));
                zeModuleBuildLogGetString(buildLogHandle, pSize, buildLog);
                System.out.println("Module build log:");
                for (int i = 0; i < pSize.get(JAVA_INT, 0); i += 1) {
                    byte c = buildLog.get(JAVA_BYTE, i);
                    char c1 = (char) (c & 0xFF);
                    System.out.print(c1);
                }
                throw new RuntimeException("failed to create module");
            }
            MemorySegment moduleHandle = pModuleHandle.get(ADDRESS, 0);
            return moduleHandle;
        }
        catch (IOException ioe) {throw new RuntimeException(ioe);}
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

    public static class Arg {
        private final String name;
        private final Value crValue;
        private final Object value;
        private final Class<?> cls;
        private int size;
        private boolean needsCleanup;
        private MemorySegment dataSegment;

        public static Arg createArg(LevelZero levelZero, String name, Value crValue, Object value) {
            Arg arg = new Arg(name, crValue, value);
            levelZero.provisionArg(arg);
            return arg;
        }

        private Arg(String name, Value crValue, Object value) {
            this.name = name;
            this.crValue = crValue;
            this.cls = value.getClass();
            this.value = value;
        }

        public String name() {
            return name;
        }

        public Object value() {
            return value;
        }

        public Object crValue() {
            return crValue;
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

        public int hashCode() {
            return crValue.hashCode();
        }

        public boolean equals(Object other) {
            return other instanceof Arg arg && arg.crValue.equals(arg.crValue);
        }

        public String toString() {
            return String.format("name = %s, hash = %d, cls = %s", name, crValue.hashCode(), cls);
        }
    }
}
