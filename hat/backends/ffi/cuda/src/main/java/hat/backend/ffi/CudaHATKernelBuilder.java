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
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.*;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.util.List;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {

    private CudaHATKernelBuilder half2float() {
        return identifier("__half2float");
    }

    private CudaHATKernelBuilder __nv_bfloat16() {
        return identifier("__nv_bfloat16");
    }

    private CudaHATKernelBuilder __bfloat162float() {
        return identifier("__bfloat162float");
    }

    private CudaHATKernelBuilder reinterpret_cast() {
        return keyword("reinterpret_cast");
    }
    private CudaHATKernelBuilder threadIdx() {
        return keyword("threadIdx");
    }
    private CudaHATKernelBuilder threadIdxX() {return threadIdx().dot().identifier("x");}
    private CudaHATKernelBuilder threadIdxY() {return threadIdx().dot().identifier("y");}
    private CudaHATKernelBuilder threadIdxZ() {return threadIdx().dot().identifier("z");}
    private CudaHATKernelBuilder gridDim() {
        return keyword("gridDim");
    }
    private CudaHATKernelBuilder gridDimX() {return gridDim().dot().identifier("x");}
    private CudaHATKernelBuilder gridDimY() {return gridDim().dot().identifier("y");}
    private CudaHATKernelBuilder gridDimZ() {return gridDim().dot().identifier("z");}
    private CudaHATKernelBuilder blockDim() {return keyword("blockDim");}
    private CudaHATKernelBuilder blockDimX() {return blockDim().dot().identifier("x");}
    private CudaHATKernelBuilder blockDimY() {return blockDim().dot().identifier("y");}
    private CudaHATKernelBuilder blockDimZ() {return blockDim().dot().identifier("z");}
    private CudaHATKernelBuilder blockIdx() {return keyword("blockIdx");}
    private CudaHATKernelBuilder blockIdxX() {return blockIdx().dot().identifier("x");}
    private CudaHATKernelBuilder blockIdxY() {return blockIdx().dot().identifier("y");}
    private CudaHATKernelBuilder blockIdxZ() {return blockIdx().dot().identifier("z");}
    @Override
    public CudaHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_CUDA")
                .hashDefine("HAT_GLOBAL_MEM", _ -> {})
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__shared__"))
                .hashDefine("HAT_FUNC", _->externC().space().keyword("__device__").space())//.keyword("inline"))
                .hashDefine("HAT_KERNEL", _->externC().space().keyword("__global__"))
                .hashDefine("HAT_GIX", _ -> paren(_-> HAT_BIX().asterisk().HAT_LSX().plus().HAT_LIX()))
                .hashDefine("HAT_GIY", _ -> paren(_-> HAT_BIY().asterisk().HAT_LSY().plus().HAT_LIY()))
                .hashDefine("HAT_GIZ", _ -> paren(_-> HAT_BIZ().asterisk().HAT_LSZ().plus().HAT_LIZ()))
                .hashDefine("HAT_LIX", _ -> threadIdxX())
                .hashDefine("HAT_LIY", _ -> threadIdxY())
                .hashDefine("HAT_LIZ", _ -> threadIdxZ())
                .hashDefine("HAT_GSX", _ -> gridDimX().asterisk().HAT_LSX())
                .hashDefine("HAT_GSY", _ -> gridDimY().asterisk().HAT_LSY())
                .hashDefine("HAT_GSZ", _ -> gridDimZ().asterisk().HAT_LSZ())
                .hashDefine("HAT_LSX", _ -> blockDimX())
                .hashDefine("HAT_LSY", _ -> blockDimY())
                .hashDefine("HAT_LSZ", _ -> blockDimZ())
                .hashDefine("HAT_BIX", _ -> blockIdxX())
                .hashDefine("HAT_BIY", _ -> blockIdxY())
                .hashDefine("HAT_BIZ", _ -> blockIdxZ())
                .hashDefine("HAT_BARRIER", _->keyword("__syncthreads").ocparen())
                .includeSys("cuda_fp16.h", "cuda_bf16.h")
                .hashDefine("BFLOAT16", _->keyword("__nv_bfloat16"))
                .typedefSingleValueStruct("F16", "half")
                .typedefSingleValueStruct("BF16",  "BFLOAT16");
    }

    @Override
    public CudaHATKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name){
        return identifier("atomicAdd").paren(_ -> ampersand().recurse(buildContext, instanceResult.op()).rarrow().identifier(name).comma().literal(1));
    }

    @Override
    public CudaHATKernelBuilder hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        Value dest = hatVectorStoreView.operands().get(0);
        Value index = hatVectorStoreView.operands().get(2);
        keyword("reinterpret_cast")
                .lt()
                .typeName(hatVectorStoreView.buildType())
                .space()
                .asterisk()
                .gt()
                .oparen()
                .ampersand();

        if (dest instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        either(hatVectorStoreView.isSharedOrPrivate(), CodeBuilder::dot, CodeBuilder::rarrow);
        identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen().osbrace().intConstZero().csbrace()
                .space().equals().space();
        // if the value to be stored is an operation, recurse on the operation
        if (hatVectorStoreView.operands().get(1) instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
            recurse(buildContext, r.op());
        } else {
            varName(hatVectorStoreView);
        }

        return self();
    }

    @Override
    public CudaHATKernelBuilder hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {

        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        final String postFixOp1 = "_1";
        final String postFixOp2 = "_2";

        if (op1 instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp1) {
            typeName(hatVectorBinaryOp1.buildType()).space()
                            .identifier(hatVectorBinaryOp.varName() + postFixOp1)
                                    .semicolon().nl();
            hatVectorBinaryOp1.varName(hatVectorBinaryOp.varName() + postFixOp1);
            recurse(buildContext, hatVectorBinaryOp1);
        }

        if (op2 instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp2) {
            typeName(hatVectorBinaryOp2.buildType()).space()
                    .identifier(hatVectorBinaryOp.varName() + postFixOp2)
                    .semicolon().nl();
            hatVectorBinaryOp2.varName(hatVectorBinaryOp.varName() + postFixOp2);
            recurse(buildContext, hatVectorBinaryOp2);
        }

        for (int i = 0; i < hatVectorBinaryOp.vectorN(); i++) {

           identifier(hatVectorBinaryOp.varName())
                   .dot()
                   .identifier(hatVectorBinaryOp.mapLane(i))
                   .space().equals().space();

            if (op1 instanceof Op.Result r) {
                if (!(r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp1)) {
                    recurse(buildContext, r.op());
                } else {
                    identifier(hatVectorBinaryOp1.varName());
                }
            }
            dot().identifier(hatVectorBinaryOp.mapLane(i)).space();
            identifier(hatVectorBinaryOp.operationType().symbol()).space();

            if (op2 instanceof Op.Result r) {
                if (!(r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp2)) {
                    recurse(buildContext, r.op());
                } else {
                    identifier(hatVectorBinaryOp2.varName());
                }
            }
            dot().identifier(hatVectorBinaryOp.mapLane(i)).semicolon().nl();
        }

        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorLoadOp hatVectorLoadOp) {
        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        reinterpret_cast()
                .lt()
                .typeName(hatVectorLoadOp.buildType())
                .space()
                .asterisk()
                .gt()
                .oparen()
                .ampersand();

        if (source instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        either(hatVectorLoadOp.isSharedOrPrivate(), CodeBuilder::dot, CodeBuilder::rarrow);
        identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen().osbrace().intConstZero().csbrace();

        return self();
    }

    @Override
    public CudaHATKernelBuilder hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorSelectLoadOp hatVSelectLoadOp) {
        identifier(hatVSelectLoadOp.varName())
                .dot()
                .identifier(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorSelectStoreOp hatVSelectStoreOp) {
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
    public CudaHATKernelBuilder hatF16ConvOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16ConvOp hatF16ConvOp) {
        oparen();
        ReducedFloatType reducedFloatType = hatF16ConvOp.reducedFloatType();
        generateReduceFloatType(reducedFloatType);
        cparen().obrace();

        buildReducedFloatType(reducedFloatType);
        oparen();
        Value param =  hatF16ConvOp.operands().getFirst();
        if (param instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        cparen().cbrace();
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatF16ToFloatConvOp(ScopedCodeBuilderContext builderContext, HATF16Op.HATF16ToFloatConvOp hatF16ToFloatConvOp) {
        buildReducedFloatType(hatF16ToFloatConvOp.reducedFloatType());
        oparen();
        Value param =  hatF16ToFloatConvOp.operands().getFirst();
        if (param instanceof Op.Result r) {
            recurse(builderContext, r.op());
        }
        if (!hatF16ToFloatConvOp.isLocal()) {
            rarrow().identifier("value");
        } else if (!hatF16ToFloatConvOp.wasFloat()) {
            dot().identifier("value");
        }
        cparen();
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorVarOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorVarOp hatVectorVarOp) {
        Value operand = hatVectorVarOp.operands().getFirst();
        typeName(hatVectorVarOp.buildType())
                .space()
                .varName(hatVectorVarOp);

        if (operand instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
            semicolon().nl();
        } else {
            space().equals().space();
        }

        if (operand instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder genVectorIdentifier(ScopedCodeBuilderContext builderContext, HATVectorOp.HATVectorOfOp hatVectorOfOp) {
        composeIdentifier("make_", hatVectorOfOp.buildType());
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatF16BinaryOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16BinaryOp hatF16BinaryOp) {
        Value op1 = hatF16BinaryOp.operands().get(0);
        Value op2 = hatF16BinaryOp.operands().get(1);
        ReducedFloatType reducedFloatType = hatF16BinaryOp.reducedFloatType();
        List<Boolean> references = hatF16BinaryOp.references();
        byte f32Mixed = hatF16BinaryOp.getByteFloatRepresentation();

        paren( _-> generateReduceFloatType(reducedFloatType)).obrace().oparen();

        if (f32Mixed == HATF16Op.HATF16BinaryOp.LAST_OP) {
            generateReducedFloatConversionToFloat(reducedFloatType).oparen();
        }

        if (op1 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        if (references.getFirst()) {
            rarrow().identifier("value");
        } else if (op1 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
            dot().identifier("value");
        }

        if (f32Mixed == HATF16Op.HATF16BinaryOp.LAST_OP) {
            cparen();
        }

        space().identifier(hatF16BinaryOp.binaryOperationType().symbol()).space();

        if (f32Mixed == HATF16Op.HATF16BinaryOp.FIRST_OP) {
            generateReducedFloatConversionToFloat(reducedFloatType).oparen();
        }

        if (op2 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        if (references.get(1)) {
            rarrow().identifier("value");
        } else if (op2 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
            dot().identifier("value");
        }

        if (f32Mixed == HATF16Op.HATF16BinaryOp.FIRST_OP) {
            // close the pending parenthesis
            cparen();
        }

        cparen().cbrace();
        return self();
    }

    private CudaHATKernelBuilder buildReducedFloatType(ReducedFloatType reducedFloatType) {
        switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> half2float();
            case ReducedFloatType.BFloat16 _ -> __nv_bfloat16();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        }
        return self();
    }

    private CudaHATKernelBuilder generateReduceFloatType(ReducedFloatType reducedFloatType) {
        switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> f16Type();
            case ReducedFloatType.BFloat16 _ -> bf16Type();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        }
        return self();
    }

    private CudaHATKernelBuilder generateReducedFloatConversionToFloat(ReducedFloatType reducedFloatType) {
        switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ ->  half2float();
            case ReducedFloatType.BFloat16 _ ->  __bfloat162float();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        }
        return self();
    }
}
