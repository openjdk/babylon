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
import hat.buffer.BackendConfig;
import hat.callgraph.KernelCallGraph;
import hat.NDRange;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;

public class MockBackend extends NativeBackend {

    interface  MockConfig extends BackendConfig {
        // See backends/mock/include/mock_backend.h
        //  class MockConfig{
        //       public:
        //         boolean gpu;
        //         boolean junk;
        //   };
        static MockConfig create(Arena arena, MethodHandles.Lookup lookup, boolean gpu) {
            MockConfig config = SegmentMapper.of(lookup, MockConfig.class,
                    JAVA_BOOLEAN.withName("gpu"),
                    JAVA_BOOLEAN.withName("junk")
            ).allocate(arena);
            config.gpu(gpu);
            return config;
        }
        boolean gpu();
        void gpu(boolean gpu );
        boolean junk();
        void junk(boolean junk );
    }
    public MockBackend()  {
        super("mock_backend");
        getBackend(MockConfig.create(arena(),MethodHandles.lookup(), true));
    }

    @Override
    public void computeContextHandoff(ComputeContext computeContext){
        System.out.println("Mock backend recieved closed closure");
        System.out.println("Mock backend will mutate  "+ computeContext.computeCallGraph.entrypoint + computeContext.computeCallGraph.entrypoint.method);
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        System.out.println("Mock dispatch kernel");
        // Here we receive a callgraph from the kernel entrypoint
        // The first time we see this we need to convert the kernel entrypoint 
        // and rechable methods to a form that our mock backend can execute. 
        kernelCallGraph.kernelReachableResolvedStream().forEach(kr -> {
           
        });
    }
}
