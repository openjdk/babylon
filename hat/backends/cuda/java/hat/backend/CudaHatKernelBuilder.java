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

import hat.backend.c99codebuilders.C99HatKernelBuilder;

import java.lang.reflect.code.type.JavaType;

public class CudaHatKernelBuilder extends C99HatKernelBuilder<CudaHatKernelBuilder> {

    @Override
    public CudaHatKernelBuilder defines() {
        return this
                .hashDefine("NDRANGE_CUDA")
                .hashDefine("__global");
    }

    @Override
    public CudaHatKernelBuilder pragmas() {
        return self();
    }

    public CudaHatKernelBuilder globalId(){
        return identifier("blockIdx").dot().identifier("x")
                .asterisk()
                .identifier("blockDim").dot().identifier("x")
                .plus()
                .identifier("threadIdx").dot().identifier("x");
    }
    @Override
    public CudaHatKernelBuilder globalSize(){
        return identifier("gridDim").dot().identifier("x")
                .asterisk()
                .identifier("blockDim").dot().identifier("x");
    }


    @Override
    public CudaHatKernelBuilder kernelDeclaration(String name) {
        return externC().space().keyword("__global__").space().voidType().space().identifier(name);
    }

    @Override
    public CudaHatKernelBuilder functionDeclaration(JavaType javaType, String name) {
        return externC().space().keyword("__device__").space().keyword("inline").space().type(javaType).space().identifier(name);
    }

    @Override
    public CudaHatKernelBuilder globalPtrPrefix() {
        return self();
    }
}
