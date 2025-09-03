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
import hat.codebuilders.HATCodeBuilderContext;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {

    public CudaHATKernelBuilder(NDRange ndRange) {
        super(ndRange);
    }

    @Override
    public CudaHATKernelBuilder defines() {
        return this
                .hashDefine("NDRANGE_CUDA")
                .hashDefine("__global");
    }

    @Override
    public CudaHATKernelBuilder pragmas() {
        return self();
    }

    private String buildThreadDimId(int id) {
        String threadDimId;
        if (id == 0) {
            threadDimId = "x";
        } else if (id == 1) {
            threadDimId = "y";
        } else if (id == 2) {
            threadDimId = "z";
        } else {
            throw new RuntimeException("Thread Dimension not supported");
        }
        return threadDimId;
    }

    @Override
    public CudaHATKernelBuilder globalId(int id) {
        String threadDimId = buildThreadDimId(id);
        return identifier("blockIdx").dot().identifier(threadDimId)
                .asterisk()
                .identifier("blockDim").dot().identifier(threadDimId)
                .plus()
                .identifier("threadIdx").dot().identifier(threadDimId);
    }

    @Override
    public CudaHATKernelBuilder localId(int id) {
        String threadDimId = buildThreadDimId(id);
        return identifier("threadIdx").dot().identifier(threadDimId);
    }

    @Override
    public CudaHATKernelBuilder globalSize(int id) {
        String threadDimId = buildThreadDimId(id);
        return identifier("gridDim").dot().identifier(threadDimId)
                .asterisk()
                .identifier("blockDim").dot().identifier(threadDimId);
    }

    @Override
    public CudaHATKernelBuilder localSize(int id) {
        String threadDimId = buildThreadDimId(id);
        return identifier("blockDim").dot().identifier(threadDimId);
    }

    @Override
    public CudaHATKernelBuilder blockId(int id) {
        String threadDimId = buildThreadDimId(id);
        return identifier("blockIdx").dot().identifier(threadDimId);
    }

    @Override
    public CudaHATKernelBuilder kernelDeclaration(CoreOp.FuncOp funcOp) {
        return externC().space().keyword("__global__").space().voidType().space().identifier(funcOp.funcName());
    }

    @Override
    public CudaHATKernelBuilder functionDeclaration(HATCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp) {
        return externC().space().keyword("__device__").space().keyword("inline").space().type(codeBuilderContext,javaType).space().identifier(funcOp.funcName());
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
    public CudaHATKernelBuilder atomicInc(HATCodeBuilderContext buildContext, Op.Result instanceResult, String name){
        return identifier("atomicAdd").paren(_ -> {
             ampersand().recurse(buildContext, instanceResult.op());
             rarrow().identifier(name).comma().literal(1);
        });
    }

    @Override
    public CudaHATKernelBuilder emitPrivateDeclaration(String typeStructName, String varName) {
        return suffix_t(typeStructName)
                .space()
                .emitText(varName).nl();
    }

    @Override
    public CudaHATKernelBuilder emitLocalDeclaration(String typeName, String varName) {
        return localPtrPrefix()
                .space()
                .suffix_t(typeName)
                .space()
                .identifier(varName);
    }

    @Override
    public CudaHATKernelBuilder syncBlockThreads() {
        return identifier("__syncthreads()");
    }
}
