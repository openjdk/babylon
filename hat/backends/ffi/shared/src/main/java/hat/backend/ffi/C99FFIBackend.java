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

import hat.NDRange;
import hat.Config;
import hat.KernelContext;
import hat.annotations.Kernel;
import hat.annotations.Preformatted;
import hat.annotations.TypeDef;
import hat.buffer.KernelBufferContext;
import hat.codebuilders.C99HATKernelBuilder;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.callgraph.KernelCallGraph;
import hat.codebuilders.ScopedCodeBuilderContext;
import hat.device.DeviceSchema;
import hat.dialect.HATMemoryOp;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.BufferState;
import hat.ifacemapper.Schema;
import hat.optools.OpTk;
import hat.phases.HATFinalDetectionPhase;
import jdk.incubator.code.TypeElement;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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

        private void setGlobalMesh(NDRange.Global global) {
            kernelBufferContext.gsy(1);
            kernelBufferContext.gsz(1);
            switch (global) {
                case NDRange.Global1D global1D -> {
                    kernelBufferContext.gsx(global1D.x());
                    kernelBufferContext.dimensions(global1D.dimension());
                }
                case NDRange.Global2D global2D -> {
                    kernelBufferContext.gsx(global2D.x());
                    kernelBufferContext.gsy(global2D.y());
                    kernelBufferContext.dimensions(global2D.dimension());
                }
                case NDRange.Global3D global3D -> {
                    kernelBufferContext.gsx(global3D.x());
                    kernelBufferContext.gsy(global3D.y());
                    kernelBufferContext.gsz(global3D.z());
                    kernelBufferContext.dimensions(global3D.dimension());
                }
                case null, default -> {
                    throw new IllegalArgumentException("Unknown global range " + global.getClass());
                }
            }
        }

        private void setLocalMesh(NDRange.Local local) {
            kernelBufferContext.lsy(1);
            kernelBufferContext.lsz(1);
            switch (local) {
                case NDRange.Local1D local1D -> {
                    kernelBufferContext.lsx(local1D.x());
                    kernelBufferContext.dimensions(local1D.dimension());
                }
                case NDRange.Local2D local2D -> {
                    kernelBufferContext.lsx(local2D.x());
                    kernelBufferContext.lsy(local2D.y());
                    kernelBufferContext.dimensions(local2D.dimension());
                }
                case NDRange.Local3D local3D -> {
                    kernelBufferContext.lsx(local3D.x());
                    kernelBufferContext.lsy(local3D.y());
                    kernelBufferContext.lsz(local3D.z());
                    kernelBufferContext.dimensions(local3D.dimension());
                }
                case null, default -> {
                    throw new IllegalArgumentException("Unknown global range " + local.getClass());
                }
            }
        }

        private void setDefaultLocalMesh() {
            kernelBufferContext.lsx(0);
            kernelBufferContext.lsy(0);
            kernelBufferContext.lsz(0);
        }

        private void setupComputeRange(KernelContext kernelContext) {
            NDRange ndRange = kernelContext.getNDRange();
            if (!(ndRange instanceof NDRange.Range range)) {
                throw new IllegalArgumentException("NDRange must be of type NDRange.Range");
            }
            boolean isLocalMeshDefined = kernelContext.hasLocalMesh();
            NDRange.Global global = range.global();
            setGlobalMesh(global);
            if (isLocalMeshDefined) {
                setLocalMesh(range.local());
            } else {
                setDefaultLocalMesh();
            }
        }

        public void dispatch(KernelContext kernelContext, Object[] args) {
            setupComputeRange(kernelContext);
            args[0] = this.kernelBufferContext;
            ArgArray.update(argArray, kernelCallGraph, args);
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

        var annotation = kernelCallGraph.entrypoint.method.getAnnotation(Kernel.class);

        if (annotation!=null){
            var typedef = kernelCallGraph.entrypoint.method.getAnnotation(TypeDef.class);
            if (typedef!=null){
                builder.lineComment("Preformatted typedef body from @Typedef annotation");
                builder.typedefKeyword().space().structKeyword().space().suffix_s(typedef.name()).braceNlIndented(_->
                        builder.preformatted(typedef.body())
                ).suffix_t(typedef.name()).semicolon().nl();
            }
            var preformatted = kernelCallGraph.entrypoint.method.getAnnotation(Preformatted.class);
            if (preformatted!=null){
                builder.lineComment("Preformatted text from @Preformatted annotation");
                builder.preformatted(preformatted.value());
            }
            builder.lineComment("Preformatted code body from @Kernel annotation");
            builder.preformatted(annotation.value());
        } else {
            List<TypeElement> localIFaceList = new ArrayList<>();

            kernelCallGraph.getModuleOp()
                    .elements()
                    .filter(c -> Objects.requireNonNull(c) instanceof HATMemoryOp)
                    .map(c -> ((HATMemoryOp) c).invokeType())
                    .forEach(localIFaceList::add);

            kernelCallGraph.entrypoint.funcOp()
                    .elements()
                    .filter(c -> Objects.requireNonNull(c) instanceof HATMemoryOp)
                    .map(c -> ((HATMemoryOp) c).invokeType())
                    .forEach(localIFaceList::add);

            // Dynamically build the schema for the user data type we are creating within the kernel.
            // This is because no allocation was done from the host. This is kernel code, and it is reflected
            // using the code reflection API
            // 1. Add for struct for iface objects
            Set<String> typedefs = new HashSet<>();
            for (TypeElement typeElement : localIFaceList) {
                // 1.1 Load the class dynamically
                try {
                    //Class<?> clazz = (Class<?>) ((ClassType) typeElement).resolve(kernelCallGraph.computeContext.accelerator.lookup);
                    Class<?> clazz = Class.forName(typeElement.toString());

                    //System.out.println("!!!!!!For  "+clazz);
                    // TODO: Contract between the Java interface and the user. We require a method called `create` in order for this to work.
                    // 1.2 Obtain the create method

                    Method method = clazz.getMethod("create", hat.Accelerator.class);
                    method.setAccessible(true);
                    Buffer invoke = (Buffer) method.invoke(null, kernelCallGraph.computeContext.accelerator);
                    if (invoke != null) {
                        // code gen of the struct
                        BoundSchema<?> boundSchema = Buffer.getBoundSchema(invoke);
                        boundSchema.schema().rootIfaceType.visitTypes(0, t -> {
                            if (!already.contains(t)) {
                                builder.typedef(boundSchema, t);
                                already.add(t);
                            }
                        });
                    } else {
                        // new approach
                        Field schemaField = clazz.getDeclaredField("schema");
                        schemaField.setAccessible(true);
                        Object schema = schemaField.get(schemaField);

                        Class<?> deviceSchemaClass = Class.forName(DeviceSchema.class.getName());
                        Method toTextMethod = deviceSchemaClass.getDeclaredMethod("toText");
                        toTextMethod.setAccessible(true);
                        String toText = (String) toTextMethod.invoke(schema);

                        if (toText != null) {
                            // From here is text processing
                            String[] split = toText.split(">");
                            // Each item is a data struct
                            for (String s : split) {
                                // curate: remove first character
                                s = s.substring(1);
                                String dsName = s.split(":")[0];
                                if (typedefs.contains(dsName)) {
                                    continue;
                                }
                                typedefs.add(dsName);
                                // sanitize dsName
                                dsName = sanitize(dsName);
                                builder.typedefKeyword()
                                        .space()
                                        .structKeyword()
                                        .space()
                                        .suffix_s(dsName)
                                        .obrace()
                                        .nl();

                                String[] members = s.split(";");

                                int j = 0;
                                builder.in();
                                for (int i = 0; i < members.length; i++) {
                                    String member = members[i];
                                    String[] field = member.split(":");
                                    if (i == 0) { j = 1;}
                                    String isArray = field[j++];
                                    String type = field[j++];
                                    String name = field[j++];
                                    String lenValue = "";
                                    if (isArray.equals("[")) {
                                        lenValue = field[j];
                                    }
                                    j = 0;
                                    if (typedefs.contains(type))
                                        type = sanitize(type) + "_t";
                                    else
                                        type = sanitize(type);

                                    builder.typeName(type)
                                            .space()
                                            .identifier(name);

                                    if (isArray.equals("[")) {
                                        builder.space()
                                                .osbrace()
                                                .identifier(lenValue)
                                                .csbrace();
                                    }
                                    builder.semicolon().nl();
                                }
                                builder.out();
                                builder.cbrace().suffix_t(dsName).semicolon().nl();
                            }
                        } else {
                            throw new RuntimeException("[ERROR] Could not find valid device schema ");
                        }
                    }
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
                        HATFinalDetectionPhase finals = new HATFinalDetectionPhase(kernelCallGraph.entrypoint.callGraph.computeContext.accelerator);
                        finals.apply(funcOp);

                        // Update the build context for this method to use the right constants-map
                        buildContext.setFinals(finals.getFinalVars());
                        builder.nl().kernelMethod(buildContext, funcOp).nl();
                    });

            // Update the constants-map for the main kernel
            HATFinalDetectionPhase hatFinalDetectionPhase = new HATFinalDetectionPhase(kernelCallGraph.entrypoint.callGraph.computeContext.accelerator);
            hatFinalDetectionPhase.apply(kernelCallGraph.entrypoint.funcOp());
            buildContext.setFinals(hatFinalDetectionPhase.getFinalVars());
            builder.nl().kernelEntrypoint(buildContext, args).nl();

            if (config().showKernelModel()) {
                IO.println("Original");
                IO.println(kernelCallGraph.entrypoint.funcOp().toText());
            }
            if (config().showLoweredKernelModel()) {
                IO.println("Lowered");
                IO.println(OpTk.lower(here, kernelCallGraph.entrypoint.funcOp()).toText());
            }
        }
        return builder.toString();
    }


    private String sanitize(String s) {
        String[] split1 = s.split("\\.");
        if (split1.length == 1) {
            return s;
        }
        s = split1[split1.length - 1];
        if (s.split("\\$").length > 1) {
            s = sanitize(s.split("\\$")[1]);
        }
        return s;
    }

    @Override
    public void preMutate(Buffer b) {
        switch (b.getState()) {
            case BufferState.NO_STATE:
            case BufferState.NEW_STATE:
            case BufferState.HOST_OWNED:
            case BufferState.DEVICE_VALID_HOST_HAS_COPY: {
                if (config().showState()) {
                    System.out.println("in preMutate state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
                if (config().showState()) {
                    System.out.print("in preMutate state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (config().showState()) {
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
        if (config().showState()) {
            System.out.print("in postMutate state = " + b.getStateString() + " no action to take ");
        }
        if (b.getState() != BufferState.NEW_STATE) {
            b.setState(BufferState.HOST_OWNED);
        }
        if (config().showState()) {
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
                if (config().showState()) {
                    System.out.println("in preAccess state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device

                if (config().showState()) {
                    System.out.print("in preAccess state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (config().showState()) {
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
        if (config().showState()) {
            System.out.println("in postAccess state = " + b.getStateString());
        }
    }
}
