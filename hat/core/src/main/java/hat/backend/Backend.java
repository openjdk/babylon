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
package hat.backend;


import hat.ComputeContext;
import hat.Config;
import hat.KernelContext;
import hat.NDRange;
//import hat.backend.java.JavaMultiThreadedBackend;
//import hat.backend.java.JavaSequentialBackend;
import hat.buffer.BufferAllocator;
import hat.callgraph.KernelCallGraph;

import java.util.ServiceLoader;
import java.util.function.Predicate;

public  abstract class Backend implements BufferAllocator {
    private final Config config;

    public Config config(){
        return config;
    }

    protected Backend(Config config){
        this.config = config;
    }

    final public String getName() {
        return this.getClass().getName();
    }

    final public static Predicate<Backend> PROPERTY = (backend) -> {
        String requestedBackendName = System.getProperty("hat.backend");
        if (requestedBackendName == null) {
            throw new IllegalStateException("Expecting property hat.backend to name a Backend class");
        }
        String backendName = backend.getName();
        return (backendName.equals(requestedBackendName));
    };
    public static Predicate<Backend> FIRST = backend -> true;

    public static Backend getBackend(Predicate<Backend> backendPredicate) {
        return ServiceLoader.load(Backend.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .filter(backendPredicate)
                .findFirst().orElseThrow();
    }

    public abstract void computeContextHandoff(ComputeContext computeContext);

    public abstract void dispatchCompute(ComputeContext computeContext, Object... args);

    public abstract void dispatchKernel(KernelCallGraph kernelCallGraph, KernelContext kernelContext, Object... args);
}
