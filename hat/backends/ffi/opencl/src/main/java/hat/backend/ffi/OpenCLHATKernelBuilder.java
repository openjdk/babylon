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

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    public OpenCLHATKernelBuilder(NDRange ndRange) {
        super(ndRange);
    }

    @Override
    public OpenCLHATKernelBuilder defines() {
        hashDefine("NDRANGE_OPENCL");
        hashIfndef("NULL", _ -> hashDefine("NULL", "0"));
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder pragmas() {
        return self().
                pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable").
                pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable");
    }

    @Override
    public OpenCLHATKernelBuilder globalId(int id) {
        return identifier("get_global_id").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHATKernelBuilder localId(int id) {
        return identifier("get_local_id").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHATKernelBuilder globalSize(int id) {
        return identifier("get_global_size").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHATKernelBuilder localSize(int id) {
        return identifier("get_local_size").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHATKernelBuilder blockId(int id) {
        return identifier("get_group_id").oparen().literal(id).cparen();
    }

    @Override
    public OpenCLHATKernelBuilder syncBlockThreads() {
        return identifier("barrier").oparen().identifier("CLK_LOCAL_MEM_FENCE").cparen().semicolon();
    }

    @Override
    public OpenCLHATKernelBuilder kernelDeclaration(CoreOp.FuncOp funcOp) {
        return keyword("__kernel").space().voidType().space().identifier(funcOp.funcName());
    }

    @Override
    public OpenCLHATKernelBuilder functionDeclaration(HATCodeBuilderContext codeBuilderContext, JavaType type, CoreOp.FuncOp funcOp) {
        return keyword("inline").space().type(codeBuilderContext,type).space().identifier(funcOp.funcName());
    }

    @Override
    public OpenCLHATKernelBuilder globalPtrPrefix() {
        return keyword("__global");
    }

    @Override
    public OpenCLHATKernelBuilder localPtrPrefix() {
        return keyword("__local");
    }

    @Override
    public OpenCLHATKernelBuilder atomicInc(HATCodeBuilderContext buildContext, Op.Result instanceResult, String name){
          return identifier("atomic_inc").paren(_ ->
              ampersand().recurse(buildContext, instanceResult.op()).rarrow().identifier(name)
          );
    }

}
