/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Reflect;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.Schema;
import optkl.util.carriers.ArenaAndLookupCarrier;

public interface DispatchContext extends Buffer {

    @Reflect
    default void  schema(){
        type();
        dimensions();          // Dimension (1D, 2D or 3D)
        gsx(); gsy(); gsz();   // global sizes
        lsx(); lsy(); lsz();   // local sizes
        bsx(); bsy(); bsz();   // block sizes
        tlx(); tly(); tlz();   // tile sizes
        wsx(); wsy(); wsz();   // warp sizes
    }
    Schema<DispatchContext> schema = Schema.of(DispatchContext.class);

    // ----------------------------------------------------------------------|
    // Mapping between OpenCL, CUDA and HAT                                  |
    // ----------------------------------------------------------------------|
    //| OpenCL            | CUDA                                  | HAT      |
    //| ----------------- | ------------------------------------- |--------- |
    //| get_global_size(0)| gridDim.x * blockDim.x                | gsx      |
    //| get_local_size(0) | blockDim.x                            | lsx      |
    //| get_num_groups(0) | gridDim.x                             | bsx      |
    // ----------------------------------------------------------------------|

    int type(); //0 kernel, 1 tile, 2 tensor

    void type(int type);
    int dimensions();

    void dimensions(int dimensions);


    int gsx();
    void gsx(int gsx);
    int gsy();
    void gsy(int gsy);
    int gsz();
    void gsz(int gsz);


    // Local group size / block size
    int lsx();
    void lsx(int lsx);
    int lsy();
    void lsy(int lsy);
    int lsz();
    void lsz(int lsz);


    int bsx();
    void bsx(int bsx);
    int bsy();
    void bsy(int bsy);
    int bsz();
    void bsz(int bsz);

    // Tile size
    int tlx();
    void tlx(int tlx);
    int tly();
    void tly(int tly);
    int tlz();
    void tlz(int tlz);

    // Warp Size
    int wsx();
    void wsx(int wsx);
    int wsy();
    void wsy(int wsy);
    int wsz();
    void wsz(int wsz);

    static DispatchContext createDefault(ArenaAndLookupCarrier cc) {
        DispatchContext dispatchContext = BoundSchema.of(cc ,schema).allocate();
        dispatchContext.type(0); // default to kernel
        // Set default value for each construct
        dispatchContext.dimensions(3);


        dispatchContext.gsy(0);
        dispatchContext.gsx(0);
        dispatchContext.gsz(0);

        dispatchContext.lsx(0);
        dispatchContext.lsy(0);
        dispatchContext.lsz(0);

        dispatchContext.bsx(0);
        dispatchContext.bsy(0);
        dispatchContext.bsz(0);

        dispatchContext.tlx(0);
        dispatchContext.tly(0);
        dispatchContext.tlz(0);

        dispatchContext.wsx(0);
        dispatchContext.wsy(0);
        dispatchContext.wsz(0);

        return dispatchContext;
    }
}
