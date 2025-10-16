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
import hat.Config;
import hat.ThreadMesh;
import hat.NDRange;
import hat.buffer.KernelBufferContext;
import hat.codebuilders.C99HATKernelBuilder;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.callgraph.KernelCallGraph;
import hat.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.HATMemoryOp;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.BufferState;
import hat.ifacemapper.Schema;
import hat.optools.OpTk;
import hat.phases.HATFinalDetectionPhase;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.ClassType;

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

    public C99FFIBackend(String libName, Config config) {
        super(libName, config);
    }

    public static class CompiledKernel {
        public final C99FFIBackend c99FFIBackend;
        public final KernelCallGraph kernelCallGraph;
        public final BackendBridge.CompilationUnitBridge.KernelBridge kernelBridge;
        public final ArgArray argArray;
        public final KernelBufferContext kernelBufferContext;

        public CompiledKernel(C99FFIBackend c99FFIBackend, KernelCallGraph kernelCallGraph, BackendBridge.CompilationUnitBridge.KernelBridge kernelBridge, Object[] ndRangeAndArgs) {
            this.c99FFIBackend = c99FFIBackend;
            this.kernelCallGraph = kernelCallGraph;
            this.kernelBridge = kernelBridge;
            this.kernelBufferContext = KernelBufferContext.createDefault(kernelCallGraph.computeContext.accelerator);
            ndRangeAndArgs[0] = this.kernelBufferContext;
            this.argArray = ArgArray.create(kernelCallGraph.computeContext.accelerator,kernelCallGraph,  ndRangeAndArgs);
        }

        private void setGlobalMesh(hat.KernelContext kc) {
            kernelBufferContext.maxX(kc.maxX);
            kernelBufferContext.maxY(kc.maxY);
            kernelBufferContext.maxZ(kc.maxZ);
            kernelBufferContext.dimensions(kc.getDimensions());
        }

        private void setGlobalMesh(ThreadMesh threadMesh) {
            kernelBufferContext.maxX(threadMesh.getX());
            kernelBufferContext.maxY(threadMesh.getY());
            kernelBufferContext.maxZ(threadMesh.getZ());
            kernelBufferContext.dimensions(threadMesh.getDims());
        }

        private void setLocalMesh(ThreadMesh threadMesh) {
            kernelBufferContext.lsx(threadMesh.getX());
            kernelBufferContext.lsy(threadMesh.getY());
            kernelBufferContext.lsz(threadMesh.getZ());
        }

        private void setDefaultLocalMesh() {
            kernelBufferContext.lsx(0);
            kernelBufferContext.lsy(0);
            kernelBufferContext.lsz(0);
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
            args[0] = this.kernelBufferContext;
            ArgArray.update(argArray,kernelCallGraph, args);
            kernelBridge.ndRange(this.argArray);
        }
    }

    public Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();

    public <T extends C99HATKernelBuilder<T>> String createCode(KernelCallGraph kernelCallGraph, T builder, Object... args) {
        var here = OpTk.CallSite.of(C99FFIBackend.class, "createCode");
        builder.defines().types();
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

        List<TypeElement> localIFaceList = new ArrayList<>();

        kernelCallGraph.getModuleOp()
                .elements()
                .filter(c->Objects.requireNonNull(c) instanceof HATMemoryOp)
                .map(c->((HATMemoryOp)c).invokeType())
                .forEach(localIFaceList::add);

       kernelCallGraph.entrypoint.funcOp()
                .elements()
                .filter(c->Objects.requireNonNull(c) instanceof HATMemoryOp)
                .map(c->((HATMemoryOp)c).invokeType())
                .forEach(localIFaceList::add);

        // Dynamically build the schema for the user data type we are creating within the kernel.
        // This is because no allocation was done from the host. This is kernel code, and it is reflected
        // using the code reflection API
        // 1. Add for struct for iface objects
        for (TypeElement typeElement : localIFaceList) {
            // 1.1 Load the class dynamically
            try {
               Class<?> clazz = (Class<?>)((ClassType)typeElement).resolve(kernelCallGraph.computeContext.accelerator.lookup);//Class.forName(typeElement.toString());
                //System.out.println("!!!!!!For  "+clazz);
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
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }

        ScopedCodeBuilderContext buildContext =
                new ScopedCodeBuilderContext(kernelCallGraph.entrypoint.callGraph.computeContext.accelerator.lookup,
                        kernelCallGraph.entrypoint.funcOp());

        // Sorting by rank ensures we don't need forward declarations
        kernelCallGraph.getModuleOp().functionTable()
                .forEach((_, funcOp) -> {
// TODO: did we just trash the callgraph sidetables?
                    HATFinalDetectionPhase finals = new HATFinalDetectionPhase();
                    finals.apply(funcOp);

                    // Update the build context for this method to use the right constants-map
                    buildContext.setFinals(finals.getFinalVars());
                    builder.nl().kernelMethod(buildContext, funcOp).nl();
                });

        // Update the constants-map for the main kernel
        HATFinalDetectionPhase hatFinalDetectionPhase = new HATFinalDetectionPhase();
        hatFinalDetectionPhase.apply(kernelCallGraph.entrypoint.funcOp());
        buildContext.setFinals(hatFinalDetectionPhase.getFinalVars());
        builder.nl().kernelEntrypoint(buildContext, args).nl();

        if (Config.SHOW_KERNEL_MODEL.isSet(config())) {
            IO.println("Original");
            IO.println(kernelCallGraph.entrypoint.funcOp().toText());
        }
        if (Config.SHOW_LOWERED_KERNEL_MODEL.isSet(config())){
            IO.println("Lowered");
            IO.println(OpTk.lower(here, kernelCallGraph.entrypoint.funcOp()).toText());
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
                if (Config.SHOW_STATE.isSet(config())) {
                    System.out.println("in preMutate state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
                if (Config.SHOW_STATE.isSet(config())) {
                    System.out.print("in preMutate state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (Config.SHOW_STATE.isSet(config())) {
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
        if (Config.SHOW_STATE.isSet(config())) {
            System.out.print("in postMutate state = " + b.getStateString() + " no action to take ");
        }
        if (b.getState() != BufferState.NEW_STATE) {
            b.setState(BufferState.HOST_OWNED);
        }
        if (Config.SHOW_STATE.isSet(config())) {
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
                if (Config.SHOW_STATE.isSet(config())) {
                    System.out.println("in preAccess state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device

                if (Config.SHOW_STATE.isSet(config())) {
                    System.out.print("in preAccess state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (Config.SHOW_STATE.isSet(config())) {
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
        if (Config.SHOW_STATE.isSet(config())) {
            System.out.println("in postAccess state = " + b.getStateString());
        }
    }
}
