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

import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class OpenCLBackend extends C99FFIBackend implements BufferTracker {

    final Config config;

    final MethodHandle getBackend_MH;
    public long getBackend(int mode, int platform, int device, int unused) {
        try {
            backendHandle = (long) getBackend_MH.invoke(mode, platform, device, unused);
        } catch (Throwable throwable) {
            throw new IllegalStateException(throwable);
        }
        return backendHandle;
    }
    public OpenCLBackend(String configSpec) {
        this(Config.of(configSpec));
    }
    public OpenCLBackend(Config config) {
        super("opencl_backend");
        this.config = config;
        getBackend_MH  =  nativeLibrary.longFunc("getOpenCLBackend",JAVA_INT,JAVA_INT, JAVA_INT, JAVA_INT);
        getBackend(config.bits(),0, 0, 0 );
        if (config.isINFO()) {
            System.out.println("CONFIG = "+config);
            info();
        }
    }


    public OpenCLBackend() {
        this(Config.of().or(Config.GPU()));
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        //System.out.println("OpenCL backend received computeContext");
        injectBufferTracking(computeContext.computeCallGraph.entrypoint, config.isSHOW_COMPUTE_MODEL());
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
        compiledKernel.dispatch(ndRange,args);

    }

    @Override
    public void preMutate(Buffer b) {
        if (config.isMINIMIZE_COPIES()) {
            if (b.isDeviceDirty()) {
                if (!b.isHostChecked()) {
                    getBufferFromDeviceIfDirty(b);// calls through FFI and might block when fetching from device
                    b.setHostChecked();
                }
                b.clearDeviceDirty();
            }
        }
    }

    @Override
    public void postMutate(Buffer b) {
        if (config.isMINIMIZE_COPIES()) {
            b.setHostDirty();
        }
    }

    @Override
    public void preAccess(Buffer b) {
        if (config.isMINIMIZE_COPIES()) {
            if (b.isDeviceDirty() && !b.isHostChecked()) {
                getBufferFromDeviceIfDirty(b); // calls through FFI and might block when fetching from device
                // We don't call clearDeviceDirty() if we did then 'just reading on the host' would force copy in next dispatch
                //so buffer is still considered deviceDirty
                b.setHostChecked();
            }
        }
    }

    @Override
    public void postAccess(Buffer b) {
       // a no op buffer may well still be deviceDirty
    }

    @Override
    public void preEscape(Buffer b) {
        if (config.isMINIMIZE_COPIES()) {
            if (b.isDeviceDirty()) {
                if (!b.isHostChecked()) {
                    getBufferFromDeviceIfDirty(b);
                    b.setHostChecked();
                }
               // b.clearDeviceDirty();
            }
        }//  we have to assume the escapee is about to be accessed
    }

    @Override
    public void postEscape(Buffer b) {
        if (config.isMINIMIZE_COPIES()) {
            b.setHostDirty(); // We have no choice but to assume escapee was modified by the call
        }
    }
}
