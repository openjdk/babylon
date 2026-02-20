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

import hat.codebuilders.C99HATKernelBuilder;
import hat.dialect.HATF16Op;
import hat.dialect.HATVectorOp;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.util.HashMap;
import java.util.Map;

public class OpenCLJExtractedHATKernelBuilder extends C99HATKernelBuilder<OpenCLJExtractedHATKernelBuilder> {

    protected OpenCLJExtractedHATKernelBuilder(ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(scopedCodeBuilderContext);
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_OPENCL")
                .hashIfndef("NULL", _ -> hashDefine("NULL", "0"))
                .pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable")
                .pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable")
                .pragma("OPENCL", "EXTENSION", "cl_khr_fp16", ":", "enable")                      // Enable Half type
                .hashDefine("HAT_FUNC", _ -> keyword(""))
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
    public OpenCLJExtractedHATKernelBuilder atomicInc( Op.Result instanceResult, String name) {
        return identifier("atomic_inc").paren(_ -> ampersand().recurse( instanceResult.op()).rarrow().identifier(name));
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatVectorStoreOp( HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        Value dest = hatVectorStoreView.operands().get(0);
        Value index = hatVectorStoreView.operands().get(2);

        identifier("vstore" + hatVectorStoreView.vectorShape().lanes())
                .oparen()
                .varName(hatVectorStoreView)
                .comma()
                .space()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (dest instanceof Op.Result r) {
            recurse( r.op());
        }
        either(hatVectorStoreView instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
        identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse( r.op());
        }

        csbrace().cparen();
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatBinaryVectorOp( HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {

        oparen();
        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        if (op1 instanceof Op.Result r) {
            recurse( r.op());
        }
        space().identifier(hatVectorBinaryOp.operationType().symbol()).space();

        if (op2 instanceof Op.Result r) {
            recurse( r.op());
        }
        cparen();
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatVectorLoadOp( HATVectorOp.HATVectorLoadOp hatVectorLoadOp) {
        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        identifier("vload" + hatVectorLoadOp.vectorShape().lanes())
                .oparen()
                .intConstZero()
                .comma()
                .space()
                .ampersand();

        if (source instanceof Op.Result r) {
            recurse(r.op());
        }

        either(hatVectorLoadOp instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
        identifier("array").osbrace();
        if (index instanceof Op.Result r) {
            recurse( r.op());
        }
        csbrace().cparen();
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatSelectLoadOp( HATVectorOp.HATVectorSelectLoadOp hatVSelectLoadOp) {
        identifier(hatVSelectLoadOp.varName())
                .dot()
                .identifier(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatSelectStoreOp( HATVectorOp.HATVectorSelectStoreOp hatVSelectStoreOp) {
        identifier(hatVSelectStoreOp.varName())
                .dot()
                .identifier(hatVSelectStoreOp.mapLane())
                .space().equals().space();
        if (hatVSelectStoreOp.resolvedName() != null) {
            // We have detected a direct resolved result (resolved name)
            varName(hatVSelectStoreOp.resolvedName());
        } else if (hatVSelectStoreOp.operands().get(1) instanceof Op.Result r) {
                recurse( r.op());

        }
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatF16ConvOp( HATF16Op.HATF16ConvOp hatF16ConvOp) {
        paren(_->typeName("half"));
        if (hatF16ConvOp.operands().getFirst() instanceof Op.Result r) {
            recurse( r.op());
        }
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatVectorVarOp( HATVectorOp.HATVectorVarOp hatVectorVarOp) {
        typeName(hatVectorVarOp.buildType())
                .space()
                .varName(hatVectorVarOp)
                .space().equals().space();

        Value operand = hatVectorVarOp.operands().getFirst();
        if (operand instanceof Op.Result r) {
            recurse( r.op());
        }
        return self();
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder genVectorIdentifier( HATVectorOp.HATVectorOfOp hatVectorOfOp) {
        return paren(_->identifier(hatVectorOfOp.buildType()));
    }

    @Override
    public OpenCLJExtractedHATKernelBuilder hatF16ToFloatConvOp( HATF16Op.HATF16ToFloatConvOp hatF16ToFloatConvOp) {
        return paren(_-> f16Type()).identifier(hatF16ToFloatConvOp.varName());
    }

    private static final Map<String, String> MATH_FUNCTIONS = new HashMap<>();
    static {
        MATH_FUNCTIONS.put("maxf", "max");
        MATH_FUNCTIONS.put("maxd", "max");
        MATH_FUNCTIONS.put("maxf16", "MAX_HAT");
        MATH_FUNCTIONS.put("minf", "min");
        MATH_FUNCTIONS.put("mind", "min");
        MATH_FUNCTIONS.put("minf16", "MIN_HAT");

        MATH_FUNCTIONS.put("expf", "exp");
        MATH_FUNCTIONS.put("expd", "exp");
        MATH_FUNCTIONS.put("expf16", "half_exp");

        MATH_FUNCTIONS.put("cosf", "cos");
        MATH_FUNCTIONS.put("cosd", "cos");
        MATH_FUNCTIONS.put("sinf", "sin");
        MATH_FUNCTIONS.put("sind", "sin");
        MATH_FUNCTIONS.put("tanf", "tan");
        MATH_FUNCTIONS.put("tand", "tan");

        MATH_FUNCTIONS.put("native_cosf", "native_cos");
        MATH_FUNCTIONS.put("native_sinf", "native_sin");
        MATH_FUNCTIONS.put("native_tanf", "native_tan");
        MATH_FUNCTIONS.put("native_expf", "native_exp");

        MATH_FUNCTIONS.put("sqrtf", "sqrt");
        MATH_FUNCTIONS.put("sqrtd", "sqrt");
    }

    @Override
    protected String mapMathIntrinsic(String hatMathIntrinsicName) {
        return MATH_FUNCTIONS.getOrDefault(hatMathIntrinsicName, hatMathIntrinsicName);
    }

}
