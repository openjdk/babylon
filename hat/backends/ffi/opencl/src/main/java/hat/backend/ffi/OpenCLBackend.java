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


import hat.ComputeContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.BufferState;

import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class OpenCLBackend extends C99FFIBackend implements BufferTracker {

    final OpenCLConfig config;

    final MethodHandle getBackend_MH;

    public long getBackend(int configBits) {
        try {
            backendHandle = (long) getBackend_MH.invoke(configBits);
        } catch (Throwable throwable) {
            throw new IllegalStateException(throwable);
        }
        return backendHandle;
    }

    public OpenCLBackend(String configSpec) {
        this(OpenCLConfig.of(configSpec));
    }

    public OpenCLBackend(OpenCLConfig config) {
        super("opencl_backend");
        this.config = config;
        getBackend_MH = nativeLibrary.longFunc("getOpenCLBackend", JAVA_INT);
        getBackend(this.config.bits());
        if (this.config.isINFO()) {
            System.out.println("CONFIG = " + this.config);
            info();
        }
    }


    public OpenCLBackend() {
        this(OpenCLConfig.of());
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        // System.out.println("OpenCL backend received computeContext minimizing = "+ config.isMINIMIZE_COPIES());
        injectBufferTracking(computeContext.computeCallGraph.entrypoint, config.isSHOW_COMPUTE_MODEL(), config.isMINIMIZE_COPIES());
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        //System.out.println("OpenCL backend dispatching kernel " + kernelCallGraph.entrypoint.method);
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createCode(kernelCallGraph, new OpenCLHatKernelBuilder(), args, config.isSHOW_KERNEL_MODEL());
            if (config.isSHOW_CODE()) {
                System.out.println(code);
            }
            long programHandle = compileProgram(code);
            if (programOK(programHandle)) {
                long kernelHandle = getKernel(programHandle, kernelCallGraph.entrypoint.method.getName());
                return new CompiledKernel(this, kernelCallGraph, code, kernelHandle, args);
            } else {
                throw new IllegalStateException("opencl failed to compile ");
            }
        });
        compiledKernel.dispatch(ndRange, args);

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
                getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
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
                getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device

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
/*
    @Override
    public void preEscape(Buffer b) {
        switch (b.getState()) {
            case BufferState.NO_STATE:
            case BufferState.NEW_STATE:
            case BufferState.HOST_OWNED:
            case BufferState.DEVICE_VALID_HOST_HAS_COPY: {
                if (config.isSHOW_STATE()) {
                    System.out.println("in preEscape state = " + b.getStateString() + " no action to take");
                }
                break;
            }
            case BufferState.DEVICE_OWNED: {
                getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
                if (config.isSHOW_STATE()) {
                    System.out.print("in preEscape state = " + b.getStateString() + " we pulled from device ");
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
    public void postEscape(Buffer b) {
        if (config.isSHOW_STATE()) {
            System.out.print("in postEscape state = " + b.getStateString() + " we pulled from device ");
        }
        b.setState(BufferState.HOST_OWNED);
        if (config.isSHOW_STATE()) {
            System.out.println("and switched to " + b.getStateString());
        }
    } */
}
