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
import hat.codebuilders.CodeBuilder;
import hat.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.HATF16ConvOp;
import hat.dialect.HATF16ToFloatConvOp;
import hat.dialect.HATVectorBinaryOp;
import hat.dialect.HATVectorLoadOp;
import hat.dialect.HATVectorOfOp;
import hat.dialect.HATVectorSelectLoadOp;
import hat.dialect.HATVectorSelectStoreOp;
import hat.dialect.HATVectorStoreView;
import hat.dialect.HATVectorVarOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    @Override
    public OpenCLHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_OPENCL")
                //  .hashIfdef("HAT_OPENCL", _ ->
                //        indent(_ -> self()
                .hashIfndef("NULL", _ -> hashDefine("NULL", "0"))
                .pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable")
                .pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable")
                .pragma("OPENCL", "EXTENSION", "cl_khr_fp16", ":", "enable")                      // Enable Half type
                .hashDefine("HAT_FUNC", _ -> keyword("inline"))
                .hashDefine("HAT_KERNEL", _ -> keyword("__kernel"))
                .hashDefine("HAT_GLOBAL_MEM", _ -> keyword("__global"))
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__local"))
                .hashDefine("HAT_GIX", _ -> paren(_ -> identifier("get_global_id").paren(_ -> intConstZero())))
                .hashDefine("HAT_GIY", _ -> paren(_ -> identifier("get_global_id").paren(_ -> intConstOne())))
                .hashDefine("HAT_GIZ", _ -> paren(_ -> identifier("get_global_id").paren(_ -> intConstTwo())))
                .hashDefine("HAT_LIX", _ -> paren(_ -> identifier("get_local_id").paren(_ -> intConstZero())))
                .hashDefine("HAT_LIY", _ -> paren(_ -> identifier("get_local_id").paren(_ -> intConstOne())))
                .hashDefine("HAT_LIZ", _ -> paren(_ -> identifier("get_local_id").paren(_ -> intConstTwo())))
                .hashDefine("HAT_GSX", _ -> paren(_ -> identifier("get_global_size").paren(_ -> intConstZero())))
                .hashDefine("HAT_GSY", _ -> paren(_ -> identifier("get_global_size").paren(_ -> intConstOne())))
                .hashDefine("HAT_GSZ", _ -> paren(_ -> identifier("get_global_size").paren(_ -> intConstTwo())))
                .hashDefine("HAT_LSX", _ -> paren(_ -> identifier("get_local_size").paren(_ -> intConstZero())))
                .hashDefine("HAT_LSY", _ -> paren(_ -> identifier("get_local_size").paren(_ -> intConstOne())))
                .hashDefine("HAT_LSZ", _ -> paren(_ -> identifier("get_local_size").paren(_ -> intConstTwo())))
                .hashDefine("HAT_BIX", _ -> paren(_ -> identifier("get_group_id").paren(_ -> intConstZero())))
                .hashDefine("HAT_BIY", _ -> paren(_ -> identifier("get_group_id").paren(_ -> intConstOne())))
                .hashDefine("HAT_BIZ", _ -> paren(_ -> identifier("get_group_id").paren(_ -> intConstTwo())))
                .hashDefine("HAT_BARRIER", _ -> identifier("barrier").oparen().identifier("CLK_LOCAL_MEM_FENCE").cparen());
        //         )
        // );
    }

    @Override
    public OpenCLHATKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        return identifier("atomic_inc").paren(_ -> ampersand().recurse(buildContext, instanceResult.op()).rarrow().identifier(name));
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HATVectorStoreView hatVectorStoreView) {
        Value dest = hatVectorStoreView.operands().get(0);
        Value index = hatVectorStoreView.operands().get(2);

        identifier("vstore" + hatVectorStoreView.vectorN())
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
        either(hatVectorStoreView.isSharedOrPrivate(), CodeBuilder::dot, CodeBuilder::rarrow);
        identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HATVectorBinaryOp hatVectorBinaryOp) {

        oparen();
        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        if (op1 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        space().identifier(hatVectorBinaryOp.operationType().symbol()).space();

        if (op2 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        cparen();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HATVectorLoadOp hatVectorLoadOp) {
        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        identifier("vload" + hatVectorLoadOp.vectorN())
                .oparen()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (source instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        either(hatVectorLoadOp.isSharedOrPrivate(), CodeBuilder::dot, CodeBuilder::rarrow);
        identifier("array").osbrace();
        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        csbrace().cparen();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HATVectorSelectLoadOp hatVSelectLoadOp) {
        identifier(hatVSelectLoadOp.varName())
                .dot()
                .identifier(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HATVectorSelectStoreOp hatVSelectStoreOp) {
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

    @Override
    public OpenCLHATKernelBuilder hatF16ConvOp(ScopedCodeBuilderContext buildContext, HATF16ConvOp hatF16ConvOp) {
        oparen().halfType().cparen();
        Value initValue = hatF16ConvOp.operands().getFirst();
        if (initValue instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorVarOp(ScopedCodeBuilderContext buildContext, HATVectorVarOp hatVectorVarOp) {
        typeName(hatVectorVarOp.buildType())
                .space()
                .varName(hatVectorVarOp)
                .space().equals().space();

        Value operand = hatVectorVarOp.operands().getFirst();
        if (operand instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder genVectorIdentifier(ScopedCodeBuilderContext builderContext, HATVectorOfOp hatVectorOfOp) {
        oparen().identifier(hatVectorOfOp.buildType()).cparen().oparen();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatF16ToFloatConvOp(ScopedCodeBuilderContext builderContext, HATF16ToFloatConvOp hatF16ToFloatConvOp) {
        oparen().halfType().cparen();
        Value initValue = hatF16ToFloatConvOp.operands().getFirst();
        if (initValue instanceof Op.Result r) {
            recurse(builderContext, r.op());
        }
        if (!hatF16ToFloatConvOp.isLocal()) {
            rarrow().identifier("value");
        }
        return self();
    }
}
