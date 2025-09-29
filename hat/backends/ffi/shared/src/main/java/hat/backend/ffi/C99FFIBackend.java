/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package hat.backend.ffi;

import hat.ComputeRange;
import hat.ThreadMesh;
import hat.NDRange;
import hat.callgraph.CallGraph;
import hat.codebuilders.C99HATKernelBuilder;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.buffer.KernelContext;
import hat.callgraph.KernelCallGraph;
import hat.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.HatMemoryOp;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.BufferState;
import hat.ifacemapper.Schema;
import hat.optools.OpTk;
import hat.phases.HatFinalDetectionPhase;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public abstract class C99FFIBackend extends FFIBackend  implements BufferTracker {

    public C99FFIBackend(String libName, FFIConfig config) {
        super(libName, config);
    }

    public static class CompiledKernel {
        public final C99FFIBackend c99FFIBackend;
        public final KernelCallGraph kernelCallGraph;
        public final BackendBridge.CompilationUnitBridge.KernelBridge kernelBridge;
        public final ArgArray argArray;
        public final KernelContext kernelContext;

        public CompiledKernel(C99FFIBackend c99FFIBackend, KernelCallGraph kernelCallGraph, BackendBridge.CompilationUnitBridge.KernelBridge kernelBridge, Object[] ndRangeAndArgs) {
            this.c99FFIBackend = c99FFIBackend;
            this.kernelCallGraph = kernelCallGraph;
            this.kernelBridge = kernelBridge;
            this.kernelContext = KernelContext.createDefault(kernelCallGraph.computeContext.accelerator);
            ndRangeAndArgs[0] = this.kernelContext;
            this.argArray = ArgArray.create(kernelCallGraph.computeContext.accelerator,kernelCallGraph,  ndRangeAndArgs);
        }

        private void setGlobalMesh(hat.KernelContext kc) {
            kernelContext.maxX(kc.maxX);
            kernelContext.maxY(kc.maxY);
            kernelContext.maxZ(kc.maxZ);
            kernelContext.dimensions(kc.getDimensions());
        }

        private void setGlobalMesh(ThreadMesh threadMesh) {
            kernelContext.maxX(threadMesh.getX());
            kernelContext.maxY(threadMesh.getY());
            kernelContext.maxZ(threadMesh.getZ());
            kernelContext.dimensions(threadMesh.getDims());
        }

        private void setLocalMesh(ThreadMesh threadMesh) {
            kernelContext.lsx(threadMesh.getX());
            kernelContext.lsy(threadMesh.getY());
            kernelContext.lsz(threadMesh.getZ());
            //kernelContext.dimensions(threadMesh.getDims());
        }

        private void setDefaultLocalMesh() {
            kernelContext.lsx(0);
            kernelContext.lsy(0);
            kernelContext.lsz(0);
        }

        private void setupComputeRange(NDRange ndRange) {

            ComputeRange computeRange = ndRange.kid.getComputeRange();
            boolean isComputeRangeDefined = ndRange.kid.hasComputeRange();
            boolean isLocalMeshDefined = ndRange.kid.hasLocalMesh();

            ThreadMesh globalMesh = null;
            ThreadMesh localMesh = null;
            if (isComputeRangeDefined) {
                globalMesh = computeRange.getGlobalMesh();
                localMesh = computeRange.getLocalMesh();
            }

            if (!isComputeRangeDefined) {
                setGlobalMesh(ndRange.kid);
            } else {
                setGlobalMesh(globalMesh);
            }
            if (isComputeRangeDefined && isLocalMeshDefined) {
                setLocalMesh(localMesh);
            } else {
                setDefaultLocalMesh();
            }
        }

        public void dispatch(NDRange ndRange, Object[] args) {
            setupComputeRange(ndRange);
            args[0] = this.kernelContext;
            ArgArray.update(argArray,kernelCallGraph, args);
            kernelBridge.ndRange(this.argArray);
        }
    }

    public Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();

    private void updateListOfSchemas(Op op, List<String> localIfaceList) {
        if (Objects.requireNonNull(op) instanceof HatMemoryOp hatMemoryOp) {
            String klassName = hatMemoryOp.invokeType().toString();
            localIfaceList.add(klassName);
        }
    }

    public <T extends C99HATKernelBuilder<T>> String createCode(KernelCallGraph kernelCallGraph, T builder, Object... args) {

        builder.defines().pragmas().types();
        Set<Schema.IfaceType> already = new LinkedHashSet<>();
        Arrays.stream(args)
                .filter(arg -> arg instanceof Buffer)
                .map(arg -> (Buffer) arg)
                .forEach(ifaceBuffer -> {
                    BoundSchema<?> boundSchema = Buffer.getBoundSchema(ifaceBuffer);
                    boundSchema.schema().rootIfaceType.visitTypes(0, t -> {
                        if (!already.contains(t)) {
                            builder.typedef(boundSchema, t);
                            already.add(t);
                        }
                    });
                });

        List<String> localIFaceList = new ArrayList<>();
        // Traverse the list of reachable functions and append the intrinsics functions found for each of the functions
        if (kernelCallGraph.moduleOp != null) {
            kernelCallGraph.moduleOp.functionTable()
                    .forEach((entryName, f) -> {
                        f.transform(CopyContext.create(), (blockBuilder, op) -> {
                            updateListOfSchemas(op, localIFaceList);
                            blockBuilder.op(op);
                            return blockBuilder;
                        });
                    });
        } else {
            // We take the list from all reachable methods. When we finally merge with the moduleOpWrapper,
            // this else-branch will be deleted.
            kernelCallGraph.kernelReachableResolvedStream().forEach((kernel) -> {
                kernel.funcOp().transform(CopyContext.create(), (blockBuilder, op) -> {
                    updateListOfSchemas(op, localIFaceList);
                    blockBuilder.op(op);
                    return blockBuilder;
                });
            });
        }

        // Traverse the main kernel and append the intrinsics functions found in the main kernel
        kernelCallGraph.entrypoint.funcOp()
                .transform(CopyContext.create(), (blockBuilder, op) -> {
                    updateListOfSchemas(op, localIFaceList);
                    blockBuilder.op(op);
                    return blockBuilder;
                });

        // Dynamically build the schema for the user data type we are creating within the kernel.
        // This is because no allocation was done from the host. This is kernel code, and it is reflected
        // using the code reflection API
        // 1. Add for struct for iface objects
        for (String klassName : localIFaceList) {
            // 1.1 Load the class dynamically
            Class<?> clazz;
            try {
                clazz = Class.forName(klassName);

                // TODO: Contract between the Java interface and the user. We require a method called `create` in order for this to work.
                // 1.2 Obtain the create method
                Method method = clazz.getMethod("create", hat.Accelerator.class);
                method.setAccessible(true);
                Buffer invoke = (Buffer) method.invoke(null, kernelCallGraph.computeContext.accelerator);

                // code gen of the struct
                BoundSchema<?> boundSchema = Buffer.getBoundSchema(invoke);
                boundSchema.schema().rootIfaceType.visitTypes(0, t -> {
                    if (!already.contains(t)) {
                        builder.typedef(boundSchema, t);
                        already.add(t);
                    }
                });
            } catch (NoSuchMethodException | InvocationTargetException | IllegalAccessException |
                     ClassNotFoundException e) {
                throw new RuntimeException(e);
            }
        }

        ScopedCodeBuilderContext buildContext =
                new ScopedCodeBuilderContext(kernelCallGraph.entrypoint.callGraph.computeContext.accelerator.lookup,
                        kernelCallGraph.entrypoint.funcOp());

        // Sorting by rank ensures we don't need forward declarations
        if (CallGraph.noModuleOp) {
            IO.println("NOT using ModuleOp for C99FFIBackend");
            kernelCallGraph.kernelReachableResolvedStream().sorted((lhs, rhs) -> rhs.rank - lhs.rank)
                    .forEach(kernelReachableResolvedMethod -> {
                                HatFinalDetectionPhase finals = new HatFinalDetectionPhase();
                                finals.apply(kernelReachableResolvedMethod.funcOp());
                                // Update the build context for this method to use the right constants-map
                                buildContext.setFinals(finals.getFinalVars());
                                builder.nl().kernelMethod(buildContext, kernelReachableResolvedMethod.funcOp()).nl();
                    });
        } else {
            IO.println("Using ModuleOp for C99FFIBackend");
            kernelCallGraph.moduleOp.functionTable()
                    .forEach((_, funcOp) -> {

                        HatFinalDetectionPhase finals = new HatFinalDetectionPhase();
                        finals.apply(funcOp);

                        // Update the build context for this method to use the right constants-map
                        buildContext.setFinals(finals.getFinalVars());
                        builder.nl().kernelMethod(buildContext, funcOp).nl();
                    });
        }

        // Update the constants-map for the main kernel
        HatFinalDetectionPhase hatFinalDetectionPhase = new HatFinalDetectionPhase();
        hatFinalDetectionPhase.apply(kernelCallGraph.entrypoint.funcOp());
        buildContext.setFinals(hatFinalDetectionPhase.getFinalVars());
        builder.nl().kernelEntrypoint(buildContext, args).nl();

        if (FFIConfig.SHOW_KERNEL_MODEL.isSet(config.bits())) {
            IO.println("Original");
            IO.println(kernelCallGraph.entrypoint.funcOp().toText());
            IO.println("Lowered");
            IO.println(OpTk.lower(kernelCallGraph.entrypoint.funcOp()).toText());
        }
        return builder.toString();
    }


    @Override
    public void preMutate(Buffer b) {
        switch (b.getState()) {
            case BufferState.NO_STATE:
            case BufferState.NEW_STATE:
            case BufferState.HOST_OWNED:
            case BufferState.DEVICE_VALID_HOST_HAS_COPY: {
                if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
                    System.out.println("in preMutate state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
                if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
                    System.out.print("in preMutate state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
                    System.out.println("and switched to " + b.getStateString());
                }
                break;
            }
            default:
                throw new IllegalStateException("Not expecting this state ");
        }
    }

    @Override
    public void postMutate(Buffer b) {
        if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
            System.out.print("in postMutate state = " + b.getStateString() + " no action to take ");
        }
        if (b.getState() != BufferState.NEW_STATE) {
            b.setState(BufferState.HOST_OWNED);
        }
        if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
            System.out.println("and switched to (or stayed on) " + b.getStateString());
        }
    }

    @Override
    public void preAccess(Buffer b) {
        switch (b.getState()) {
            case BufferState.NO_STATE:
            case BufferState.NEW_STATE:
            case BufferState.HOST_OWNED:
            case BufferState.DEVICE_VALID_HOST_HAS_COPY: {
                if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
                    System.out.println("in preAccess state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device

                if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
                    System.out.print("in preAccess state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
                    System.out.println("and switched to " + b.getStateString());
                }
                break;
            }
            default:
                throw new IllegalStateException("Not expecting this state ");
        }
    }


    @Override
    public void postAccess(Buffer b) {
        if (FFIConfig.SHOW_STATE.isSet(config.bits())) {
            System.out.println("in postAccess state = " + b.getStateString());
        }
    }
}
