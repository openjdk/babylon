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
package hat.backend.jextracted;

import hat.NDRange;
import hat.codebuilders.C99HATKernelBuilder;
import hat.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.HatVSelectLoadOp;
import hat.dialect.HatVSelectStoreOp;
import hat.dialect.HatVectorBinaryOp;
import hat.dialect.HatVectorLoadOp;
import hat.dialect.HatVectorStoreView;
import hat.dialect.HatVectorVarLoadOp;
import hat.dialect.HatVectorVarOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

public class OpenCLHatKernelBuilder extends C99HATKernelBuilder<OpenCLHatKernelBuilder> {

    @Override
    public OpenCLHatKernelBuilder defines(){
        return hashIfndef("NULL", _ -> hashDefine("NULL", "0"))
                .pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable")
                .pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable");
    }

    @Override
    public OpenCLHatKernelBuilder globalId(int id) {
        return identifier("get_global_id").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHatKernelBuilder localId(int id) {
        return identifier("get_local_id").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHatKernelBuilder globalSize(int id) {
        return identifier("get_global_size").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHatKernelBuilder localSize(int id) {
        return identifier("get_local_size").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHatKernelBuilder blockId(int id) {
        return identifier("get_group_id").oparen().literal(id).cparen();
    }


    @Override
    public OpenCLHatKernelBuilder kernelDeclaration(CoreOp.FuncOp funcOp) {
        return keyword("__kernel").space().voidType().space().funcName(funcOp);
    }

    @Override
    public OpenCLHatKernelBuilder functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType type, CoreOp.FuncOp funcOp) {
        return keyword("inline").space().type(codeBuilderContext,type).space().funcName(funcOp);
    }

    @Override
    public OpenCLHatKernelBuilder globalPtrPrefix() {
        return keyword("__global");
    }

    @Override
    public OpenCLHatKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name){
          return identifier("atomic_inc").paren(_ -> {
              ampersand().recurse(buildContext, instanceResult.op());
              rarrow().identifier(name);
          });
    }

    @Override
    public OpenCLHatKernelBuilder localPtrPrefix() {
        return keyword("__local");
    }

    @Override
    public OpenCLHatKernelBuilder syncBlockThreads() {
        return identifier("barrier").oparen().identifier("CLK_LOCAL_MEM_FENCE").cparen().semicolon();
    }

    @Override
    public OpenCLHatKernelBuilder generateVectorStore(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder generateVectorStore");
    }

    @Override
    public OpenCLHatKernelBuilder generateVectorBinary(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder generateVectorBinary");
    }

    @Override
    public OpenCLHatKernelBuilder generateVectorLoad(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder generateVectorLoad");
    }

    @Override
    public OpenCLHatKernelBuilder generateVectorSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder generateVectorSelectLoadOp");
    }

    @Override
    public OpenCLHatKernelBuilder generateVectorSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder generateVectorSelectStoreOp");
    }

    @Override
    public OpenCLHatKernelBuilder hatVectorVarOp(ScopedCodeBuilderContext buildContext, HatVectorVarOp hatVectorVarOp) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder hatVectorVarOp");
    }

    @Override
    public OpenCLHatKernelBuilder hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HatVectorVarLoadOp hatVectorVarLoadOp) {
        throw new RuntimeException("implement OpenCLHatKernelBuilder hatVectorVarLoadOp");
    }
}
