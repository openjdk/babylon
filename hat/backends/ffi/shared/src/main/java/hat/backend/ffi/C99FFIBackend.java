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

import hat.Config;
import hat.KernelContext;
import hat.NDRange;
import hat.annotations.Kernel;
import hat.annotations.Preformatted;
import hat.annotations.TypeDef;
import hat.buffer.ArgArray;
import hat.buffer.KernelBufferContext;
import hat.callgraph.IfaceDataDag;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.MethodCallDag;
import hat.codebuilders.C99HATKernelBuilder;
import hat.codebuilders.C99VecAndMatHandler;
import hat.device.DeviceSchema;
import hat.device.NonMappableIface;
import hat.types.BF16;
import hat.types.F16;
import jdk.incubator.code.CodeTransformer;
import optkl.IfaceValue;
import optkl.codebuilders.ScopedCodeBuilderContext;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.BufferState;
import optkl.ifacemapper.BufferTracker;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

public abstract class C99FFIBackend extends FFIBackend implements BufferTracker {
    public C99FFIBackend(Arena arena, MethodHandles.Lookup lookup, String libName, Config config) {
        super(arena, lookup, libName, config);
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
            this.kernelBufferContext = KernelBufferContext.createDefault(kernelCallGraph.computeCallGraph.computeContext.accelerator());
            ndRangeAndArgs[0] = this.kernelBufferContext;
            this.argArray = ArgArray.create(kernelCallGraph.computeCallGraph.computeContext.accelerator(), kernelCallGraph, ndRangeAndArgs);
        }

        public void dispatch(KernelContext kernelContext, Object[] args) {
            kernelBufferContext.gsy(1);
            kernelBufferContext.gsz(1);
            switch (kernelContext.ndRange.global()) {
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
                    throw new IllegalArgumentException("Unknown global range " + kernelContext.ndRange.global().getClass());
                }
            }
            if (kernelContext.ndRange.hasLocal()) {
                kernelBufferContext.lsy(1);
                kernelBufferContext.lsz(1);
                switch (kernelContext.ndRange.local()) {
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
                        throw new IllegalArgumentException("Unknown global range " + kernelContext.ndRange.local().getClass());
                    }
                }
            } else {
                kernelBufferContext.lsx(0);
                kernelBufferContext.lsy(0);
                kernelBufferContext.lsz(0);
            }
            args[0] = this.kernelBufferContext;
            ArgArray.update(argArray, kernelCallGraph, args);
            kernelBridge.ndRange(this.argArray);
        }
    }

    public Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();

    static Type nameToTypeOrThrow(String name) {
        return switch (name){
            case "void"->void.class;
            case "boolean"->boolean.class;
            case "byte"->byte.class;
            case "short"->short.class;
            case "char"->char.class;
            case "int"->int.class;
            case "float"->float.class;
            case "double"->double.class;
            case "long"->long.class;
            default -> {
                try {
                    if (Class.forName(name) instanceof Class<?> clazz) {
                        yield clazz;
                    } else {
                        throw new RuntimeException("Not a class");
                    }
                }catch (ClassNotFoundException classNotFoundException){
                    throw new RuntimeException("Not a class");
                }
            }
        };

    }

    private <T extends C99HATKernelBuilder<T>> void generateDeviceTypeStructs(T builder, String toText, Set<Type> types) {
        // From here is text processing
        String[] split = toText.split(">");
        // Each item is a data struct
        for (String ss : split) {
            // curate: remove first character
            final var finalS = ss.substring(1);
            String dsName = finalS.split(":")[0];
            if (types.add(nameToTypeOrThrow(dsName))) {
                builder.typedefStruct(sanitize(dsName), _ -> {
                    String[] members = finalS.split(";");
                    builder.indent(_ -> {
                        for (int i = 0, j = 1; i < members.length; i++, j = 0) {
                            String[] field = members[i].split(":");
                            final boolean isArray = field[j++].equals("[");
                            String rawtype = field[j++];
                            Type rawTypeClass = nameToTypeOrThrow(rawtype);
                            builder.type(sanitize(rawtype) + ((types.contains(rawTypeClass) ? "_t" : ""))).sp().id(field[j++]);
                            if (isArray) {
                                final String lenValue = field[j];
                                builder.sp().sbrace(_ -> builder.id(lenValue));
                            }
                            builder.semicolon().nl();

                        }
                    });
                });
                builder.semicolon().nl().nl();
            }

        }
    }

    public <T extends C99HATKernelBuilder<T>> String createCode(KernelCallGraph kernelCallGraph, T builder, Object... args) {
        builder.defines().types();

        var visitedAlready = new HashSet<Schema.IfaceType>();
        Arrays.stream(args)
                .filter(arg -> arg instanceof Buffer)
                .map(arg -> (Buffer) arg)
                .forEach(ifaceBuffer -> {
                    BoundSchema<?> boundSchema = MappableIface.getBoundSchema(ifaceBuffer);
                    boundSchema.schema().rootIfaceType.visitUniqueTypes(t -> {
                        if (visitedAlready.add(t)) { // true first time we see this type
                            builder.typedef(boundSchema, t);
                        }
                    });
                });


        var kernelAnnotation = kernelCallGraph.callDag.entryPoint.method().getAnnotation(Kernel.class);
        if (kernelAnnotation != null) {
            // If we find a kernelAnnotation we can't trust the data in kernelCallGraph's state.
            kernelCallGraph.usesAtomics = true;
            kernelCallGraph.accessedFP16Classes.addAll(List.of(F16.class,BF16.class));//usesFp16 = true;
            kernelCallGraph.usesBarrier = true;

            var typedefAnnotation = kernelCallGraph.callDag.entryPoint.method().getAnnotation(TypeDef.class);
            if (typedefAnnotation != null) {
                builder.lineComment("Preformatted typedef body from @Typedef annotation");
                builder.typedefStruct(typedefAnnotation.name(), _ -> builder.preformatted(typedefAnnotation.body())).semicolon().nl();
            }
            var preformattedAnnotation = kernelCallGraph.callDag.entryPoint.method().getAnnotation(Preformatted.class);
            if (preformattedAnnotation != null) {
                builder.lineComment("Preformatted text from @Preformatted annotation");
                builder.preformatted(preformattedAnnotation.value());
            }
            builder.lineComment("Preformatted code body from @Kernel annotation");
            builder.preformatted(kernelAnnotation.value());
        } else {
            Set<Type> types = new HashSet<>();
            if (!kernelCallGraph.accessedFP16Classes.isEmpty()) {
                // Add HAT reserved types
                types.add(F16.class);
                types.add(BF16.class);
            }

            kernelCallGraph.accessedIfaceClasses.stream()
                    .filter(NonMappableIface.class::isAssignableFrom)
                    .map(c->(Class<NonMappableIface>)c)
                    .forEach(c-> {
                        var ifaceDataDag = new IfaceDataDag<NonMappableIface>(dag -> {
                            var entryPoint = dag.getNode(c);
                            dag.methodsWithIfaceReturnTypes(c)
                                    .forEach(ifaceInfo ->
                                            dag.addEdge(entryPoint, dag.getNode(ifaceInfo.clazz())) // this recurses with each added class
                                    );
                                  });
                        Consumer<IfaceDataDag.IfaceInfo<NonMappableIface>> dump = ifaceValue -> {
                            try {
                                Field schemaField = c.getDeclaredField("deviceSchema");
                                schemaField.setAccessible(true);
                                var s = schemaField.get(schemaField);
                                if (s instanceof DeviceSchema<?> deviceSchema) {
                                  //  System.out.println("typedef "+deviceSchema.clazz.getSimpleName()+"{");
                                   // System.out.println("}");
                                }
                            }catch (NoSuchFieldException|IllegalAccessException e) {
                                throw new RuntimeException(e);
                            }
                        };
                        if (ifaceDataDag.isDag()){
                       //     System.out.println(String.join(" -> ",ifaceDataDag.rankOrdered.stream().map(IfaceDataDag.IfaceInfo::dotName).toList()));
                            ifaceDataDag.rankOrdered.forEach(dump);
                        }else {
                          //  var node = ifaceDataDag.getNode(c);
                         //   System.out.println(node.dotName());
                            ifaceDataDag.rankOrdered.forEach(dump);
                        }
                    });

            // Dynamically build the schema for the user data type we are creating within the kernel.
            // This is because no allocation was done from the host. This is kernel code, and it is reflected
            // using the code reflection API
            // 1. Add for struct for iface objects

            kernelCallGraph.accessedIfaceClasses.stream().filter(NonMappableIface.class::isAssignableFrom).forEach(
                    c -> {
                        try {
                            Field schemaField = c.getDeclaredField("deviceSchema");
                            schemaField.setAccessible(true);
                            var s = schemaField.get(schemaField);
                            if (s instanceof DeviceSchema<?> deviceSchema) {
                                // <1> We are creating text form of DeviceType schema
                                String toText = deviceSchema.toText();
                                if (toText != null) {
                                    // <2> just to then parse the text from above.
                                    // Lets get the model in a cleaner form
                                    generateDeviceTypeStructs(builder, toText, types);
                                } else {
                                    throw new RuntimeException("[ERROR] Could not find valid device schema ");
                                }
                            } else if (s instanceof Schema<?> schema) {
                                throw new RuntimeException("found " + schema + " in NonMappableIface " + c.getName());
                            }
                        } catch (ReflectiveOperationException e) {
                            throw new RuntimeException(e);
                        }
                    }
            );

            var buildContext = new ScopedCodeBuilderContext(kernelCallGraph.lookup(), kernelCallGraph.callDag.entryPoint.funcOp());

            if (!kernelCallGraph.accessedVecClasses.isEmpty()) {
                C99VecAndMatHandler.createVecFunctions(builder);
            }


            kernelCallGraph.callDag.rankOrdered.stream()
                    .filter(m->m instanceof MethodCallDag.OtherMethodCall)
                    .forEach(m -> builder.nl().kernelMethod(buildContext, m.funcOp()).nl());
            builder.nl().kernelEntrypoint(buildContext).nl();

            if (config().showKernelModel()) {
                IO.println("Non Lowered");
                IO.println(kernelCallGraph.callDag.entryPoint.funcOp().toText());
            }
            if (config().showLoweredKernelModel()) {
                IO.println("Lowered");
                IO.println(kernelCallGraph.callDag.entryPoint.funcOp().transform(CodeTransformer.LOWERING_TRANSFORMER).toText());
            }
        }
        return builder.toString();
    }

    private String sanitize(String s) {
        String[] split1 = s.split("\\.");
        if (split1.length != 1) {
            s = split1[split1.length - 1];
            if (s.split("\\$").length > 1) {
                int last = s.lastIndexOf("$");
                s = s.substring(last + 1);
            }
        }
        return s;
    }

    @Override
    public void preMutate(MappableIface b) {
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
    public void postMutate(MappableIface b) {
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
    public void preAccess(MappableIface b) {
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
    public void postAccess(MappableIface b) {
        if (config().showState()) {
            System.out.println("in postAccess state = " + b.getStateString());
        }
    }
}
