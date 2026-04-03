/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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

import hat.callgraph.KernelCallGraph;
import hat.codebuilders.C99HATKernelBuilder;
import hat.dialect.HATF16Op;
import hat.dialect.HATVectorOp;
import optkl.OpHelper;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.util.HashMap;
import java.util.Map;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    protected OpenCLHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraph,scopedCodeBuilderContext);
    }

    public OpenCLHATKernelBuilder vstore(int dims) {
        return id("vstore" + dims);
    }

    public OpenCLHATKernelBuilder vload(int dims) {
        return id("vload" + dims);
    }

    @Override
    public OpenCLHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_OPENCL")
                .hashIfndef("NULL", _ -> hashDefine("NULL", "0"))
                .when(kernelCallGraph.usesAtomics,_->pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable"))
                .when(kernelCallGraph.usesAtomics,_->pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable"))
                /*.when(kernelCallGraph.usesFp16,_->*/.pragma("OPENCL", "EXTENSION", "cl_khr_fp16", ":", "enable")//)                      // Enable Half type
                .hashDefine("HAT_FUNC", _ -> keyword(""))
                .hashDefine("HAT_KERNEL", _ -> keyword("__kernel"))
                .hashDefine("HAT_GLOBAL_MEM", _ -> keyword("__global"))
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__local"))
                .when(kernelCallGraph.accessedKcFields.contains("gix"), _->hashDefine("HAT_GIX", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKcFields.contains("giy"), _->hashDefine("HAT_GIY", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKcFields.contains("giz"), _->hashDefine("HAT_GIZ", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKcFields.contains("lix"), _->hashDefine("HAT_LIX", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKcFields.contains("liy"), _->hashDefine("HAT_LIY", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKcFields.contains("liz"), _->hashDefine("HAT_LIZ", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKcFields.contains("gsx"), _->hashDefine("HAT_GSX", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKcFields.contains("gsy"), _->hashDefine("HAT_GSY", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKcFields.contains("gsz"), _->hashDefine("HAT_GSZ", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKcFields.contains("lsx"), _->hashDefine("HAT_LSX", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKcFields.contains("lsy"), _->hashDefine("HAT_LSY", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKcFields.contains("lsz"), _->hashDefine("HAT_LSZ", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKcFields.contains("bix"), _->hashDefine("HAT_BIX", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKcFields.contains("biy"), _->hashDefine("HAT_BIY", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKcFields.contains("biz"), _->hashDefine("HAT_BIZ", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKcFields.contains("bsx"), _->hashDefine("HAT_BSX", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKcFields.contains("bsy"), _->hashDefine("HAT_BSY", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKcFields.contains("bsz"), _->hashDefine("HAT_BSZ", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.usesFp16, _->maxMacro("MAX_HAT"))
                .when(kernelCallGraph.usesFp16, _->minMacro("MIN_HAT"))
                .when(kernelCallGraph.usesBarrier, _->hashDefine("HAT_BARRIER", _ -> id("barrier").oparen().id("CLK_LOCAL_MEM_FENCE").cparen()))
                /*.when(callgraphState.usesFp16,_->*/.hashDefine("BFLOAT16", _ -> keyword("ushort"))//)
                /*.when(callgraphState.usesFp16,_->*/.typedefSingleValueStruct("F16",  "half")//)
                /*.when(callgraphState.usesFp16,_->*/.typedefSingleValueStruct("BF16",  "BFLOAT16")//)
                /*.when(callgraphState.usesFp16,_->*/.unionBfloat16()//)
                /*.when(callgraphState.usesFp16,_->*/.build_builtin_bfloat16ToFloat("bf16")//)
                /*.when(callgraphState.usesFp16,_->*/.build_builtin_float2bfloat16("f")/*)*/;
    }

    @Override
    public OpenCLHATKernelBuilder atomicInc( Op.Result instanceResult, String name) {
        return id("atomic_inc").paren(_ -> ampersand().recurse( instanceResult.op()).rarrow().id(name));
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorStoreOp( HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        vstore(hatVectorStoreView.vectorShape().lanes()).paren(_-> {
            // if the value to be stored is an operation, recurse on the operation
            if (hatVectorStoreView.operands().get(1).result().op() instanceof HATVectorOp.HATVectorBinaryOp binOp) {
                recurse(binOp);
            } else {
                varName(hatVectorStoreView);
            }
            csp().intConstZero().csp().ampersand().recurse(hatVectorStoreView.operands().get(0).result().op());
            either(hatVectorStoreView instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ ->recurse(hatVectorStoreView.operands().get(2).result().op()));
        });
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatBinaryVectorOp( HATVectorOp.HATVectorBinaryOp binOp) {
        return paren(_-> {
            recurse(binOp.operands().get(0).result().op());
            sp().id(binOp.operationType().symbol()).sp();
            recurse(binOp.operands().get(1).result().op());
        });
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorLoadOp( HATVectorOp.HATVectorLoadOp hatVectorLoadOp) {
        vload(hatVectorLoadOp.vectorShape().lanes()).paren(_-> {
            intConstZero().comma().sp().ampersand();
            recurse( hatVectorLoadOp.operands().get(0).result().op());
            either(hatVectorLoadOp instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ -> recurse( hatVectorLoadOp.operands().get(1).result().op()));
        });
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatSelectLoadOp( HATVectorOp.HATVectorSelectLoadOp hatVSelectLoadOp) {
        if (hatVSelectLoadOp.operands().getFirst().result().op() instanceof HATVectorOp.HATVectorLoadOp vLoadOp) {
            recurse( vLoadOp);
        } else {
            id(hatVSelectLoadOp.varName());
        }
        dot().id(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatSelectStoreOp( HATVectorOp.HATVectorSelectStoreOp hatVSelectStoreOp) {
        if (hatVSelectStoreOp.operands().getFirst().result().op() instanceof HATVectorOp.HATVectorLoadOp vLoadOp) {
            recurse( vLoadOp);
        } else {
            id(hatVSelectStoreOp.varName());
        }
        dot().id(hatVSelectStoreOp.mapLane()).sp().equals().sp();
        return either (hatVSelectStoreOp.resolvedName() != null,
                _-> varName(hatVSelectStoreOp.resolvedName()),
                _-> recurse(hatVSelectStoreOp.operands().get(1).result().op())
        );
    }

    @Override
    public OpenCLHATKernelBuilder hatF16ConvOp( HATF16Op.HATF16ConvOp hatF16ConvOp) {
        ReducedFloatType reducedFloatType = hatF16ConvOp.reducedFloatType();
        return paren(_->{
            switch(reducedFloatType){
                case ReducedFloatType.HalfFloat _ -> f16Type();
                case ReducedFloatType.BFloat16 _-> bf16Type();
                default ->   throw new RuntimeException("What is ths reducedType");
            }
        }).brace(_-> {
            var firstOperandOp = hatF16ConvOp.operands().getFirst().result().op();
            either (reducedFloatType instanceof ReducedFloatType.BFloat16,
                    _-> builtin_float2bfloat16().paren(_-> recurse(firstOperandOp)),
                    _-> recurse( firstOperandOp)
            );
        });
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorVarOp( HATVectorOp.HATVectorVarOp hatVectorVarOp) {
        type(hatVectorVarOp.buildType()).sp().varName(hatVectorVarOp).sp().equals().sp();
        recurse( hatVectorVarOp.operands().getFirst().result().op());
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder genVectorIdentifier( HATVectorOp.HATVectorOfOp hatVectorOfOp) {
        return paren(_-> id(hatVectorOfOp.buildType()));
    }

    @Override
    public OpenCLHATKernelBuilder hatF16ToFloatConvOp( HATF16Op.HATF16ToFloatConvOp hatF16ToFloatConvOp) {
        // Type conversions: half,bfloat16 -> float
        ReducedFloatType reducedFloatType = hatF16ToFloatConvOp.reducedFloatType();

        if (reducedFloatType instanceof ReducedFloatType.HalfFloat) {// half -> float
            paren(_->f32Type());
        } else if (reducedFloatType instanceof ReducedFloatType.BFloat16) {// bfloat16 -> float
            builtin_bfloat16ToFloat();
        }
        parenWhen(reducedFloatType instanceof ReducedFloatType.BFloat16,_-> {
            recurse(hatF16ToFloatConvOp.operands().getFirst().result().op());
            if (!hatF16ToFloatConvOp.isLocal()) {
                rarrow().id("value");
            } else if (!hatF16ToFloatConvOp.wasFloat()) {
                dot().id("value");
            } else{
                throw new RuntimeException("Can we get here");
            }
        });
        return self();
    }

    // Mapping between API function names and OpenCL intrinsics for the math operations
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
