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
import hat.NDRange;
import hat.callgraph.KernelCallGraph;

import java.lang.foreign.Arena;
import java.util.ServiceLoader;
import java.util.function.Predicate;

public interface Backend {
    Arena arena();

    default String getName() {
        return this.getClass().getName();
    }

    Predicate<Backend> PROPERTY = (backend) -> {
        String requestedBackendName = System.getProperty("hat.backend");
        if (requestedBackendName == null) {
            throw new IllegalStateException("Expecting property hat.backend to name a Backend class");
        }
        String backendName = backend.getName();
        return (backendName.equals(requestedBackendName));
    };
    Predicate<Backend> FIRST = backend -> true;
    Predicate<Backend> FIRST_NATIVE = backend -> backend instanceof NativeBackend nativeBackend && nativeBackend.isAvailable();
    Predicate<Backend> JAVA_MULTITHREADED = backend -> backend instanceof JavaMultiThreadedBackend;
    Predicate<Backend> JAVA_SEQUENTIAL = backend -> backend instanceof JavaSequentialBackend;

    static Backend getBackend(Predicate<Backend> backendPredicate) {
        return ServiceLoader.load(Backend.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .filter(backendPredicate)
                .findFirst().orElseThrow();
    }

    void computeContextHandoff(ComputeContext computeContext);

    void dispatchCompute(ComputeContext computeContext, Object... args);

    void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args);
}
