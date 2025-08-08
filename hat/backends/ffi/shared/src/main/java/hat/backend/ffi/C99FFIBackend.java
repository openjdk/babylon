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
import hat.codebuilders.C99HATKernelBuilder;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.buffer.KernelContext;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.BufferState;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpWrapper;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
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
        public final KernelContext kernelContext;

        public CompiledKernel(C99FFIBackend c99FFIBackend, KernelCallGraph kernelCallGraph, BackendBridge.CompilationUnitBridge.KernelBridge kernelBridge, Object[] ndRangeAndArgs) {
            this.c99FFIBackend = c99FFIBackend;
            this.kernelCallGraph = kernelCallGraph;
            this.kernelBridge = kernelBridge;
            this.kernelContext = KernelContext.create(kernelCallGraph.computeContext.accelerator, 0, 0, 0, 0, 0, 0);
            ndRangeAndArgs[0] = this.kernelContext;
            this.argArray = ArgArray.create(kernelCallGraph.computeContext.accelerator,kernelCallGraph,  ndRangeAndArgs);
        }

        public void dispatch(NDRange ndRange, Object[] args) {
          //  long ns = System.nanoTime();
            kernelContext.maxX(ndRange.kid.maxX);
            kernelContext.maxY(ndRange.kid.maxY);
            kernelContext.maxZ(ndRange.kid.maxZ);
            kernelContext.dimensions(ndRange.kid.getDimensions());
            args[0] = this.kernelContext;
            ArgArray.update(argArray,kernelCallGraph, args);
            //System.out.println("argupdate  "+((System.nanoTime()-ns)/1000)+" us");
           // ns = System.nanoTime();
            kernelBridge.ndRange(this.argArray);
           // System.out.println("dispatch time "+((System.nanoTime()-ns)/1000)+" us");
        }
    }

    public Map<KernelCallGraph, CompiledKernel> kernelCallGraphCompiledCodeMap = new HashMap<>();

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

        // Sorting by rank ensures we don't need forward declarations
        if (Boolean.getBoolean("moduleOp")) {
            kernelCallGraph.moduleOpWrapper.functionTable()
                    .forEach((_, funcOp) -> builder.nl().kernelMethod(new FuncOpWrapper(kernelCallGraph.computeContext.accelerator.lookup, funcOp)).nl());
        } else {
            kernelCallGraph.kernelReachableResolvedStream().sorted((lhs, rhs) -> rhs.rank - lhs.rank)
                    .forEach(kernelReachableResolvedMethod -> builder.nl().kernelMethod(kernelReachableResolvedMethod).nl());
        }

        builder.nl().kernelEntrypoint(kernelCallGraph.entrypoint, args).nl();

        if (config.isSHOW_KERNEL_MODEL()) {
            System.out.println("Original");
            System.out.println(kernelCallGraph.entrypoint.funcOpWrapper().op().toText());
            System.out.println("Lowered");
            System.out.println(kernelCallGraph.entrypoint.funcOpWrapper().lower().op().toText());
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
                if (config.isSHOW_STATE()) {
                    System.out.println("in preMutate state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
                if (config.isSHOW_STATE()) {
                    System.out.print("in preMutate state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (config.isSHOW_STATE()) {
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
        if (config.isSHOW_STATE()) {
            System.out.print("in postMutate state = " + b.getStateString() + " no action to take ");
        }
        if (b.getState() != BufferState.NEW_STATE) {
            b.setState(BufferState.HOST_OWNED);
        }
        if (config.isSHOW_STATE()) {
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
                if (config.isSHOW_STATE()) {
                    System.out.println("in preAccess state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                backendBridge.getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device

                if (config.isSHOW_STATE()) {
                    System.out.print("in preAccess state = " + b.getStateString() + " we pulled from device ");
                }
                b.setState(BufferState.DEVICE_VALID_HOST_HAS_COPY);
                if (config.isSHOW_STATE()) {
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
        if (config.isSHOW_STATE()) {
            System.out.println("in postAccess state = " + b.getStateString());
        }
    }
}
