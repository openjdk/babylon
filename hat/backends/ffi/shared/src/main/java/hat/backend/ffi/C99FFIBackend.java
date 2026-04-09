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
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.BufferState;
import optkl.ifacemapper.BufferTracker;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
            // Do we really need this?  We never actually read these
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
                    case null, default -> throw new IllegalArgumentException("Unknown global range " + kernelContext.ndRange.local().getClass());
                }
            } else {
                kernelBufferContext.lsx(0);
                kernelBufferContext.lsy(0);
                kernelBufferContext.lsz(0);
            }

            // Set Tile
            kernelBufferContext.tlx(0);
            kernelBufferContext.tly(0);
            kernelBufferContext.tlz(0);
            if (kernelContext.ndRange.hasTile()) {
                switch (kernelContext.ndRange.tile()) {
                    case NDRange.Tile1D tile1D -> kernelBufferContext.tlx(tile1D.x());
                    case NDRange.Tile2D tile2D -> {
                        kernelBufferContext.tlx(tile2D.x());
                        kernelBufferContext.tly(tile2D.y());
                    }
                    case NDRange.Tile3D tile3D -> {
                        kernelBufferContext.tlx(tile3D.x());
                        kernelBufferContext.tly(tile3D.y());
                        kernelBufferContext.tlz(tile3D.z());
                    }
                    case null, default -> throw new IllegalArgumentException("Unknown global range " + kernelContext.ndRange.tile().getClass());
                }
            }


            // Set warp
            kernelBufferContext.wsx(false);
            kernelBufferContext.wsy(false);
            kernelBufferContext.wsz(false);
            if (kernelContext.ndRange.hasWarp()) {
                switch (kernelContext.ndRange.warp()) {
                    case NDRange.Warp1D warp1D -> kernelBufferContext.wsx(warp1D.x());
                    case NDRange.Warp2D warp2D -> {
                        kernelBufferContext.wsx(warp2D.x());
                        kernelBufferContext.wsy(warp2D.y());
                    }
                    case NDRange.Warp3D warp3D -> {
                        kernelBufferContext.wsx(warp3D.x());
                        kernelBufferContext.wsy(warp3D.y());
                        kernelBufferContext.wsz(warp3D.z());
                    }
                    case null, default -> throw new IllegalArgumentException("Unknown global range " + kernelContext.ndRange.warp().getClass());
                }
            }

            args[0] = this.kernelBufferContext;
            ArgArray.update(argArray, kernelCallGraph, args);
            kernelBridge.ndRange(this.argArray);
        }
    }

    public Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();


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
            kernelCallGraph.accessedFP16Classes.addAll(List.of(F16.class, BF16.class));
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
            Set<Class<?>> typedeffed = new HashSet<>();
            typedeffed.add(F16.class);
            typedeffed.add(BF16.class);
            kernelCallGraph.accessedNonMappableIfaceClasses.stream()
                    .filter(c->!typedeffed.contains(c))
                    .map(c->(Class<NonMappableIface>) c) // why do we need to do this.
                    .forEach(c -> {
                        // We create a dag of iface references rooted at c
                        var ifaceDataDag = new IfaceDataDag<NonMappableIface>(dag -> {
                            var entryPoint = dag.getNode(c);
                            dag.methodsWithIfaceReturnTypes(c).forEach(ifaceInfo ->
                                 dag.addEdge(entryPoint, dag.getNode(ifaceInfo.clazz())) // this recurses with each added class
                            );
                        });
                        // Now we can generate typedefs in rankOrder (so inner typedefs first)
                        if (ifaceDataDag.isDag()) {
                            ifaceDataDag.rankOrdered.stream()
                                    .filter(ifaceInfo -> !typedeffed.contains(ifaceInfo.clazz()))
                                    .forEach(ifaceInfo -> typedeffed.add(
                                            DeviceSchema.getDeviceSchemaOrThrow(ifaceInfo.clazz()).typedef(builder).clazz()
                                    )
                            );
                        } else  {
                            typedeffed.add(DeviceSchema.getDeviceSchemaOrThrow(c).typedef(builder).clazz());
                        }
                    });

            // This is a slight hack for Shader support.
            if (!kernelCallGraph.accessedVecClasses.isEmpty()) {
                C99VecAndMatHandler.createVecFunctions(builder);
            }

            kernelCallGraph.callDag.rankOrdered.stream()
                    .filter(m -> m instanceof MethodCallDag.OtherMethodCall)
                    .forEach(m -> builder.nl().kernelMethod( m.funcOp()).nl());

            builder.nl().kernelEntrypoint().nl();

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
