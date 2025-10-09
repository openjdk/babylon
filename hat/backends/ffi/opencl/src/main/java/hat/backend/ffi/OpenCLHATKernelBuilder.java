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
import hat.dialect.*;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {


    @Override
    public OpenCLHATKernelBuilder defines() {
        return self()
              //  .hashDefine("HAT_OPENCL")
              //  .hashIfdef("HAT_OPENCL", _ ->
                //        indent(_ -> self()
                                .hashIfndef("NULL", _ -> hashDefine("NULL", "0"))
                                .pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable")
                                .pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable")
                                .hashDefine("_gix()", _ -> paren(_ -> identifier("get_global_id").paren(_ -> intConstZero())))
                                .hashDefine("_giy()", _ -> paren(_ -> identifier("get_global_id").paren(_ -> intConstOne())))
                                .hashDefine("_giz()", _ -> paren(_ -> identifier("get_global_id").paren(_ -> intConstTwo())))
                                .hashDefine("_lix()", _ -> paren(_ -> identifier("get_local_id").paren(_ -> intConstZero())))
                                .hashDefine("_liy()", _ -> paren(_ -> identifier("get_local_id").paren(_ -> intConstOne())))
                                .hashDefine("_liz()", _ -> paren(_ -> identifier("get_local_id").paren(_ -> intConstTwo())))
                                .hashDefine("_gsx()", _ -> paren(_ -> identifier("get_global_size").paren(_ -> intConstZero())))
                                .hashDefine("_gsy()", _ -> paren(_ -> identifier("get_global_size").paren(_ -> intConstOne())))
                                .hashDefine("_gsz()", _ -> paren(_ -> identifier("get_global_size").paren(_ -> intConstTwo())))
                                .hashDefine("_lsx()", _ -> paren(_ -> identifier("get_local_size").paren(_ -> intConstZero())))
                                .hashDefine("_lsy()", _ -> paren(_ -> identifier("get_local_size").paren(_ -> intConstOne())))
                                .hashDefine("_lsz()", _ -> paren(_ -> identifier("get_local_size").paren(_ -> intConstTwo())))
                                .hashDefine("_bix()", _ -> paren(_ -> identifier("get_group_id").paren(_ -> intConstZero())))
                                .hashDefine("_biy()", _ -> paren(_ -> identifier("get_group_id").paren(_ -> intConstOne())))
                                .hashDefine("_biz()", _ -> paren(_ -> identifier("get_group_id").paren(_ -> intConstTwo())))
                                .hashDefine("_barrier()", _->identifier("barrier").oparen().identifier("CLK_LOCAL_MEM_FENCE").cparen());
               //         )
               // );
    }

    @Override
    public OpenCLHATKernelBuilder globalId(int id) {
        switch (id) {
            case 0 -> identifier("_gix()");
            case 1 -> identifier("_giy()");
            case 2 -> identifier("_giz()");
            default -> throw new RuntimeException("globalId id = " + id);
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder localId(int id) {
        switch (id) {
            case 0 -> identifier("_lix()");
            case 1 -> identifier("_liy()");
            case 2 -> identifier("_liz()");
            default -> throw new RuntimeException("localId id = " + id);
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder globalSize(int id) {
        switch (id) {
            case 0 -> identifier("_gsx()");
            case 1 -> identifier("_gsy()");
            case 2 -> identifier("_gsz()");
            default -> throw new RuntimeException("globalSize id = " + id);
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder localSize(int id) {
        switch (id) {
            case 0 -> identifier("_lsx()");
            case 1 -> identifier("_lsy()");
            case 2 -> identifier("_lsz()");
            default -> throw new RuntimeException("localSize id = " + id);
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder blockId(int id) {
        switch (id) {
            case 0 -> identifier("_bix()");
            case 1 -> identifier("_biy()");
            case 2 -> identifier("_biz()");
            default -> throw new RuntimeException("blockId id = " + id);
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder syncBlockThreads() {
        return identifier("_barrier").ocparen();
    }

    @Override
    public OpenCLHATKernelBuilder generateVectorStore(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {
        Value dest = hatVectorStoreView.operands().get(0);
        Value index = hatVectorStoreView.operands().get(2);

        identifier("vstore" + hatVectorStoreView.storeN())
                .oparen()
                .varName(hatVectorStoreView)
                .comma()
                .space()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (dest instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        rarrow().identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder generateVectorBinary(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {
        // TODO: generalize type using the dialect node
        typeName("float4")
                .space()
                .varName(hatVectorBinaryOp)
                .space().equals().space();

        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        if (op1 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        identifier(hatVectorBinaryOp.operationType().symbol()).space();

        if (op2 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder generateVectorLoad(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {
        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        typeName(hatVectorLoadOp.buildType())
                .space()
                .varName(hatVectorLoadOp)
                .space().equals().space()
                .identifier("vload" + hatVectorLoadOp.loadN())
                .oparen()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (source instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        rarrow().identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen();

        return self();
    }

    @Override
    public OpenCLHATKernelBuilder generateVectorSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        identifier(hatVSelectLoadOp.varName())
                .dot()
                .identifier(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder generateVectorSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        identifier(hatVSelectStoreOp.varName())
                .dot()
                .identifier(hatVSelectStoreOp.mapLane())
                .space().equals().space();
        if (hatVSelectStoreOp.resultValue() != null) {
            // We have detected a direct resolved result (resolved name)
            varName(hatVSelectStoreOp.resultValue());
        } else {
            // otherwise, we traverse to resolve the expression
            Value storeValue = hatVSelectStoreOp.operands().get(1);
            if (storeValue instanceof Op.Result r) {
                recurse(buildContext, r.op());
            }
        }
        return self();
    }

    public OpenCLHATKernelBuilder kernelPrefix() {
        return keyword("__kernel").space();
    }


    @Override
    public OpenCLHATKernelBuilder kernelDeclaration(CoreOp.FuncOp funcOp) {
        return kernelPrefix().voidType().space().identifier(funcOp.funcName());
    }

    public OpenCLHATKernelBuilder functionPrefix() {
        return keyword("inline").space();
    }

    @Override
    public OpenCLHATKernelBuilder functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType type, CoreOp.FuncOp funcOp) {
        return functionPrefix().type(codeBuilderContext, type).space().identifier(funcOp.funcName());
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
    public OpenCLHATKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        return identifier("atomic_inc").paren(_ ->
                ampersand().recurse(buildContext, instanceResult.op()).rarrow().identifier(name)
        );
    }


}
