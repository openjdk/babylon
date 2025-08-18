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
package hat.buffer;

import hat.Accelerator;
import hat.ifacemapper.Schema;

public interface KernelContext extends Buffer {

    // ----------------------------------------------------------------------|
    //| OpenCL            | CUDA                                  | HAT      |
    //| ----------------- | ------------------------------------- |--------- |
    //| get_global_id(0)  | blockIdx.x *blockDim.x + threadIdx.x  | gix      |
    //| get_global_size(0)| gridDim.x * blockDim.x                | gsx      |
    //| get_local_id(0)   | threadIdx.x                           | lix      |
    //| get_local_size(0) | blockDim.x                            | lsx      |
    //| get_group_id(0)   | blockIdx.x                            | bix      |
    //| get_num_groups(0) | gridDim.x                             | bsx      |
    // ----------------------------------------------------------------------|
    interface MeshBuffer extends Struct {
        int x();
        void x(int x);

        int y();
        void y(int y);

        int z();
        void z(int z);

        int maxX();
        void maxX(int maxX);

        int maxY();
        void maxY(int maxY);

        int maxZ();
        void maxZ(int maxZ);

        int dimensions();
        void dimensions(int numDimensions);
    }

    MeshBuffer globalMesh();

    MeshBuffer localMesh();

    Schema<KernelContext> schema = Schema.of(KernelContext.class,
            kernelBufferContext -> kernelBufferContext
                    .field("globalMesh", f -> f.fields("x","maxX", "y", "maxY", "z", "maxZ", "dimensions"))
                    .field("localMesh", f -> f.fields("x","maxX", "y", "maxY", "z", "maxZ", "dimensions"))
            );

    private static void setDefaultMesh(MeshBuffer meshBuffer) {
        meshBuffer.x(0);
        meshBuffer.maxX(0);
        meshBuffer.y(0);
        meshBuffer.maxY(0);
        meshBuffer.z(0);
        meshBuffer.maxZ(0);
        meshBuffer.dimensions(3);
    }

    static KernelContext createDefault(Accelerator accelerator) {
        KernelContext kernelBufferContext =  schema.allocate(accelerator);
        setDefaultMesh(kernelBufferContext.globalMesh());
        setDefaultMesh(kernelBufferContext.localMesh());
        return kernelBufferContext;
    }
}
