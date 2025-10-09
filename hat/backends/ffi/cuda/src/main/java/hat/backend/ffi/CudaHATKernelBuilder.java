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

import hat.NDRange;
import hat.codebuilders.C99HATKernelBuilder;
import hat.codebuilders.ScopedCodeBuilderContext;

import hat.dialect.*;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {


    @Override
    public CudaHATKernelBuilder defines() {
        return self();  // nor this
    }

    private CudaHATKernelBuilder threadDimId(int id) {
        return keyword(switch(id){
            case 0->"x";
            case 1->"y";
            case 2->"z";
            default -> throw new RuntimeException("Thread Dimension not supported");
        });
    }

    @Override
    public CudaHATKernelBuilder globalId(int id) {
        return paren(_->blockId(id).asterisk().localSize(id).plus().localId(id));
    }

    @Override
    public CudaHATKernelBuilder localId(int id) {
        return keyword("threadIdx").dot().threadDimId(id);
    }

    @Override
    public CudaHATKernelBuilder globalSize(int id) {
        return keyword("gridDim").dot().threadDimId(id).asterisk().localSize(id);
    }

    @Override
    public CudaHATKernelBuilder localSize(int id) {
        return keyword("blockDim").dot().threadDimId(id);
    }

    @Override
    public CudaHATKernelBuilder blockId(int id) {
        return keyword("blockIdx").dot().threadDimId(id);
    }

    @Override
    public CudaHATKernelBuilder kernelDeclaration(CoreOp.FuncOp funcOp) {
        return externC().space().keyword("__global__").space().voidType().space().funcName(funcOp);
    }

    @Override
    public CudaHATKernelBuilder functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp) {
        return externC().space().keyword("__device__").space().keyword("inline").space().type(codeBuilderContext,javaType).space().funcName(funcOp);
    }

    @Override
    public CudaHATKernelBuilder globalPtrPrefix() {
        return self();
    }

    @Override
    public CudaHATKernelBuilder localPtrPrefix() {
        return keyword("__shared__");
    }


    @Override
    public CudaHATKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name){
        return identifier("atomicAdd").paren(_ -> {
             ampersand().recurse(buildContext, instanceResult.op());
             rarrow().identifier(name).comma().literal(1);
        });
    }

    @Override
    public CudaHATKernelBuilder syncBlockThreads() {
        return keyword("__syncthreads").ocparen();
    }

    @Override
    public CudaHATKernelBuilder generateVectorStore(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {
        blockComment("Store Vector Not Implemented");
        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorBinary(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {
        blockComment("Binary Vector Not Implemented");
        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorLoad(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {
        blockComment("Load Vector Not Implemented");
        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        blockComment("Select Vector Not Implemented");
        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        blockComment("Select Vector Not Implemented");
        return self();
    }
}
