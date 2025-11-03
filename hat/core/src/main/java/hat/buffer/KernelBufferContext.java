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

public interface KernelBufferContext extends Buffer {

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

    int dimensions();
    void dimensions(int numDimensions);

    // Global: new names
    int gix();
    void gix(int gix);
    int giy();
    void giy(int giy);
    int giz();
    void giz(int giz);

    int gsx();
    void gsx(int gsx);
    int gsy();
    void gsy(int gsy);
    int gsz();
    void gsz(int gsz);

    // Local accesses
    int lix();
    void lix(int lix);
    int liy();
    void liy(int liy);
    int liz();
    void liz(int liz);

    // Local group size / block size
    int lsx();
    void lsx(int lsx);
    int lsy();
    void lsy(int lsy);
    int lsz();
    void lsz(int lsz);

    // Block ID
    int bix();
    void bix(int bix);
    int biy();
    void biy(int biy);
    int biz();
    void biz(int biz);

    Schema<KernelBufferContext> schema = Schema.of(KernelBufferContext.class,
            kernelContext -> kernelContext
                    .fields(
                            "dimensions",  // Dimension (1D, 2D or 3D)
                            "gix", "giy", "giz",   // global thread-id accesses
                            "gsx", "gsy", "gsz",   // global sizes
                            "lix", "liy", "liz",   // local (thread-ids)
                            "lsx", "lsy", "lsz",   // block size
                            "bix", "biy", "biz"    // block id
                    ));

    static KernelBufferContext createDefault(Accelerator accelerator) {
        KernelBufferContext kernelBufferContext =  schema.allocate(accelerator);
        
        kernelBufferContext.dimensions(3);

        kernelBufferContext.gix(0);
        kernelBufferContext.giy(0);
        kernelBufferContext.giz(0);

        kernelBufferContext.gsy(0);
        kernelBufferContext.giy(0);
        kernelBufferContext.giz(0);

        kernelBufferContext.lix(0);
        kernelBufferContext.liy(0);
        kernelBufferContext.liz(0);

        kernelBufferContext.lsx(0);
        kernelBufferContext.lsy(0);
        kernelBufferContext.lsz(0);

        kernelBufferContext.bix(0);
        kernelBufferContext.biy(0);
        kernelBufferContext.biz(0);

        return kernelBufferContext;
    }
}
