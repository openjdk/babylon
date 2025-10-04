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

import hat.codebuilders.C99HATKernelBuilder;
import hat.codebuilders.ScopedCodeBuilderContext;

import jdk.incubator.code.Op;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {
    private CudaHATKernelBuilder threadDimId(int id) {
        return keyword(switch(id){
            case 0->"x";
            case 1->"y";
            case 2->"z";
            default -> throw new RuntimeException("Thread Dimension not supported");
        });
    }
    @Override
    public CudaHATKernelBuilder defines() {
        return self()
               //   .hashDefine("HAT_CUDA")
                 // .hashIfdef("HAT_CUDA", _ ->
                   //     indent(_ -> self()
                .hashDefine("_GLOBAL_MEM", _ -> {})
                .hashDefine("_LOCAL_MEM", _ -> keyword("__shared__"))
                .hashDefine("_FUNC", _->externC().space().keyword("__device__").space().keyword("inline"))
                .hashDefine("_KERNEL", _->externC().space().keyword("__global__"))
                .hashDefine("_gix()", _ -> paren(_->blockId(0).asterisk().localSize(0).plus().localId(0)))
                .hashDefine("_giy()", _ -> paren(_->blockId(1).asterisk().localSize(1).plus().localId(1)))
                .hashDefine("_giz()", _ -> paren(_->blockId(2).asterisk().localSize(2).plus().localId(2)))
                .hashDefine("_lix()", _ -> keyword("threadIdx").dot().threadDimId(0))
                .hashDefine("_liy()", _ -> keyword("threadIdx").dot().threadDimId(1))
                .hashDefine("_liz()", _ -> keyword("threadIdx").dot().threadDimId(2))
                .hashDefine("_gsx()", _ -> keyword("gridDim").dot().threadDimId(0).asterisk().localSize(0))
                .hashDefine("_gsy()", _ -> keyword("gridDim").dot().threadDimId(1).asterisk().localSize(1))
                .hashDefine("_gsz()", _ -> keyword("gridDim").dot().threadDimId(2).asterisk().localSize(2))
                .hashDefine("_lsx()", _ -> keyword("blockDim").dot().threadDimId(0))
                .hashDefine("_lsy()", _ -> keyword("blockDim").dot().threadDimId(1))
                .hashDefine("_lsz()", _ -> keyword("blockDim").dot().threadDimId(2))
                .hashDefine("_bix()", _ -> keyword("blockIdx").dot().threadDimId(0))
                .hashDefine("_biy()", _ -> keyword("blockIdx").dot().threadDimId(1))
                .hashDefine("_biz()", _ -> keyword("blockIdx").dot().threadDimId(2))
                .hashDefine("_barrier()", _->keyword("__syncthreads").ocparen());
                    //    )
        //);
    }
    /*
    @Override
   public CudaHATKernelBuilder kernelPrefix() {
       return externC().space().keyword("__global__").space();
   }

    @Override
    public CudaHATKernelBuilder functionPrefix() {
        return externC().space().keyword("__device__").space().keyword("inline").space();
    }

    @Override
    public CudaHATKernelBuilder globalPtrPrefix() {
        return self();
    }

    @Override
    public CudaHATKernelBuilder localPtrPrefix() {
        return keyword("__shared__").space();
    } */


    @Override
    public CudaHATKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name){
        return identifier("atomicAdd").paren(_ -> ampersand().recurse(buildContext, instanceResult.op()).rarrow().identifier(name).comma().literal(1));
    }
}
