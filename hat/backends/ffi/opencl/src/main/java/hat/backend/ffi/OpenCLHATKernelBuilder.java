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
import hat.types.BF16;
import hat.types.F16;
import hat.types.S16ImplOfF16;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import jdk.incubator.code.Op;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static optkl.IfaceValue.Vector.getVectorShape;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    protected OpenCLHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraph, scopedCodeBuilderContext);
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
                .when(kernelCallGraph.accessedKernelContextFields.contains("gix"), _->hashDefine("HAT_GIX", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("giy"), _->hashDefine("HAT_GIY", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("giz"), _->hashDefine("HAT_GIZ", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("lix"), _->hashDefine("HAT_LIX", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("liy"), _->hashDefine("HAT_LIY", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("liz"), _->hashDefine("HAT_LIZ", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("gsx"), _->hashDefine("HAT_GSX", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("gsy"), _->hashDefine("HAT_GSY", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("gsz"), _->hashDefine("HAT_GSZ", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("lsx"), _->hashDefine("HAT_LSX", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("lsy"), _->hashDefine("HAT_LSY", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("lsz"), _->hashDefine("HAT_LSZ", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("bix"), _->hashDefine("HAT_BIX", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("biy"), _->hashDefine("HAT_BIY", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("biz"), _->hashDefine("HAT_BIZ", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstTwo()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("bsx"), _->hashDefine("HAT_BSX", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstZero()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("bsy"), _->hashDefine("HAT_BSY", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstOne()))))
                .when(kernelCallGraph.accessedKernelContextFields.contains("bsz"), _->hashDefine("HAT_BSZ", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstTwo()))))
                .when(!kernelCallGraph.accessedFP16Classes.isEmpty(), _->maxMacro("MAX_HAT"))
                .when(!kernelCallGraph.accessedFP16Classes.isEmpty(), _->minMacro("MIN_HAT"))
                .when(kernelCallGraph.usesBarrier, _ ->hashDefine("HAT_BARRIER", _ -> id("barrier").oparen().id("CLK_LOCAL_MEM_FENCE").cparen()))
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
    public OpenCLHATKernelBuilder generateVectorLoad(JavaOp.InvokeOp invokeOp, IfaceValue.Vector.Shape vectorShape, boolean deviceAllocated) {
        vload(vectorShape.lanes()).paren(_ -> {
            intConstZero().comma().sp().ampersand();
            recurseResultOrThrow(invokeOp.operands().get(0));
            either(deviceAllocated, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ -> recurseResultOrThrow(invokeOp.operands().get(1)));
        });
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorStoreOp(JavaOp.InvokeOp invokeOp, IfaceValue.Vector.Shape vectorShape, String name, boolean deviceAllocated) {
        vstore(vectorShape.lanes()).paren(_-> {
            // if the value to be stored is an operation, recurse on the operation
            if (invokeOp.operands().get(1).asResult().op() instanceof HATVectorOp.HATVectorBinaryOp binOp) {
                recurse(binOp);
            } else {
                varName(name);
            }
            csp().intConstZero().csp().ampersand().recurseResultOrThrow(invokeOp.operands().get(0));
            either(deviceAllocated, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ ->recurseResultOrThrow(invokeOp.operands().get(2)));
        });
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorStoreOp( HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        vstore(hatVectorStoreView.vectorShape().lanes()).paren(_-> {
            // if the value to be stored is an operation, recurse on the operation
            if (hatVectorStoreView.operands().get(1).asResult().op() instanceof HATVectorOp.HATVectorBinaryOp binOp) {
                recurse(binOp);
            } else {
                varName(hatVectorStoreView);
            }
            csp().intConstZero().csp().ampersand().recurseResultOrThrow(hatVectorStoreView.operands().get(0));
            either(hatVectorStoreView instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ ->recurseResultOrThrow(hatVectorStoreView.operands().get(2)));
        });
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatBinaryVectorOp( HATVectorOp.HATVectorBinaryOp binOp) {
        return paren(_-> {
            recurseResultOrThrow(binOp.operands().get(0));
            sp().id(binOp.operationType().symbol()).sp();
            recurseResultOrThrow(binOp.operands().get(1));
        });
    }

    @Override
    public OpenCLHATKernelBuilder hatVectorLoadOp( HATVectorOp.HATVectorLoadOp hatVectorLoadOp) {
        vload(hatVectorLoadOp.vectorShape().lanes()).paren(_-> {
            intConstZero().comma().sp().ampersand();
            recurseResultOrThrow( hatVectorLoadOp.operands().get(0));
            either(hatVectorLoadOp instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ -> recurseResultOrThrow( hatVectorLoadOp.operands().get(1)));
        });
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatSelectStoreOp(OpHelper.Invoke invoke, InvokeVar invokeVar) {
        if (invoke.op().operands().getFirst().declaringElement() instanceof HATVectorOp.HATVectorLoadOp vLoadOp) {
            recurse(vLoadOp);
        } else {
            id(invokeVar.name());
        }
        dot().id(mapLane(invokeVar.laneIdx())).assign();
        String resolvedName = invokeVar.resolveName();
        return either (resolvedName != null,
                _-> varName(resolvedName),
                _-> recurseResultOrThrow(invoke.op().operands().get(1))
        );
    }

    @Override
    public OpenCLHATKernelBuilder hatF16ConvOp(JavaOp.InvokeOp invokeOp, Class<?> reduceFloatType) {
        return paren(_-> f16OrBF16(reduceFloatType)).brace(_->
                either (BF16.class.isAssignableFrom(reduceFloatType),
                        _-> builtin_float2bfloat16().paren(_-> recurseResultOrThrow(invokeOp.operands().getFirst())),
                        _-> recurseResultOrThrow(invokeOp.operands().getFirst())
                ));
    }

    @Override
    public OpenCLHATKernelBuilder genVectorIdentifier(IfaceValue.Vector.Shape vectorShape) {
        return paren(_-> id(vectorShape.codeType().toString() + vectorShape.lanes()));
    }

    @Override
    public OpenCLHATKernelBuilder hatF16ToFloatConvOp(OpHelper.Invoke invoke, Class<?> reducedFloatType, boolean wasFloat, boolean isF16Local) {
        if (F16.class.isAssignableFrom(reducedFloatType)) {// half -> float
            paren(_->f32Type());
        } else if (BF16.class.isAssignableFrom(reducedFloatType)) {// bfloat16 -> float
            builtin_bfloat16ToFloat();
        }
        parenWhen(BF16.class.isAssignableFrom(reducedFloatType),_-> {
            recurseResultOrThrow(invoke.op().operands().getFirst());
            if (!isF16Local) {
                rarrow();
            } else if (!wasFloat) {
                dot();
            } else{
                throw new RuntimeException("Can we get here");
            }
            id("value");
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

    private Class<?> reduceFloatType(Optional<OpHelper.Invoke> invoke) {
        if (invoke.isPresent() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().refType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }

    private Class<?> reduceFloatTypeFromReturnType(Optional<OpHelper.Invoke> invoke) {
        if (invoke.isPresent() &&  S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().returnType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }

    @Override
    protected OpenCLHATKernelBuilder varOpForNarrowType(CoreOp.VarOp varOp) {
        // obtain the category:
        Value first = varOp.operands().getFirst();
        Class<?> narrowCategory;
        if (first.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
            // Find the category - This is the generic case, when ALL custom ops are removed
            Stream<OpHelper.Invoke> stream = OpHelper.Invoke.stream(kernelCallGraph.lookup(), invokeOp);
            Optional<OpHelper.Invoke> invoke = stream.findFirst();
            narrowCategory = reduceFloatType(invoke);
            if (narrowCategory == null && isMathLib(invoke)) {
                narrowCategory = reduceFloatTypeFromReturnType(invoke);
            }
        } else if (first.declaringElement() instanceof HATF16Op.HATF16BinaryOp hatf16BinaryOp) {
            narrowCategory = hatf16BinaryOp.float16Class();
//        } else if (first.declaringElement() instanceof HATF16Op.HATF16ConvOp hatf16ConvOp) {
//            narrowCategory = hatf16ConvOp.float16Class();
        } else {
            throw new IllegalStateException("Expected an invoke, but found: " + first.declaringElement().getClass());
        }
        if (narrowCategory == null) {
            throw new IllegalStateException("Narrow type can't be null: ");
        }
        f16OrBF16(narrowCategory).sp().assign(
                _ -> id(varOp.varName()),
                _ -> recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op()));
        return self();
    }

    @Override
    protected OpenCLHATKernelBuilder varOpForVectors(CoreOp.VarOp varOp) {
        // build VectorType
        VarType resultType = varOp.resultType();
        if (!(resultType.valueType() instanceof PrimitiveType)) {
            IfaceValue.Vector.Shape vectorShape = null;
            if (resultType.valueType() instanceof ClassType classType) {
                vectorShape = getVectorShape(kernelCallGraph.lookup(), classType);
            } else if (resultType.valueType() instanceof VarType varType) {
                vectorShape = getVectorShape(kernelCallGraph.lookup(), varType.valueType());
            }
            if (vectorShape == null) {
                // guarantee we don't have a null shape. Otherwise. we can't generate the correct code
                throw new IllegalStateException("Could not find vector shape");
            }
            // Emit
            type(vectorShape.codeType().toString() + vectorShape.lanes());
            sp().varName(varOp).sp().equals().sp();
            recurseResultOrThrow(varOp.operands().getFirst());
        }
        return self();
    }

    @Override
    protected OpenCLHATKernelBuilder varOpInit(CoreOp.VarOp varOp) {
        suffix_t((ClassType) varOp.varValueType()).sp()
                .assign(_ -> id(varOp.varName()),
                        _ -> recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op()));
        return self();
    }

    @Override
    protected OpenCLHATKernelBuilder varOpLocalMemory(CoreOp.VarOp varOp) {
        HAT_LOCAL_MEM().sp();
        return varOpPrivateMemory(varOp);
    }

    @Override
    protected OpenCLHATKernelBuilder varOpPrivateMemory(CoreOp.VarOp varOp) {
        VarType resultType = varOp.resultType();
        if (resultType.valueType() instanceof VarType varType) {
            suffix_t((ClassType) varType.valueType());
        } else if (resultType.valueType() instanceof ClassType classType) {
            suffix_t(classType);
        }
        return sp().varName(varOp);
    }

}
