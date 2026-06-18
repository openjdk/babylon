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
import hat.dialect.BinaryOpEnum;
import hat.phases.HATPhaseUtils;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.codebuilders.ScopedCodeBuilderContext;
import jdk.incubator.code.Op;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static hat.phases.HATPhaseUtils.isMathLib;
import static optkl.IfaceValue.Vector.getVectorShape;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    private static final String EXTENSION = "EXTENSION";
    private static final String OPENCL = "OPENCL";
    private static final String ENABLE = "enable";
    private static final String GLOBAL_ID = "get_global_id";
    private static final String LOCAL_ID = "get_local_id";
    private static final String GLOBAL_SIZE = "get_global_size";
    private static final String LOCAL_SIZE = "get_local_size";
    private static final String GROUP_ID = "get_group_id";
    private static final String NUM_GROUPS = "get_num_groups";

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

    protected OpenCLHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraph, scopedCodeBuilderContext);
    }

    @Override
    public OpenCLHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_OPENCL")
                .hashIfndef("NULL", _ -> hashDefine("NULL", "0"))
                .when(useAtomic(),_->pragma(OPENCL, EXTENSION, "cl_khr_global_int32_base_atomics", ":", ENABLE))
                .when(useAtomic(),_->pragma(OPENCL, EXTENSION, "cl_khr_local_int32_base_atomics", ":", ENABLE))
                .when(useS16Types(), _ -> pragma(OPENCL, EXTENSION, "cl_khr_fp16", ":", ENABLE))
                .hashDefine("HAT_FUNC", _ -> keyword(""))
                .hashDefine("HAT_KERNEL", _ -> keyword("__kernel"))
                .hashDefine("HAT_GLOBAL_MEM", _ -> keyword("__global"))
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__local"))

                // General macros
                .when(useVectors() || useS16Types(), _ -> concatMacro())
                .when(useVectors() || useS16Types(), _ -> prefixMacro())

                // Vector macros
                .when(useVectors(), _ -> defineVectorAccessMacro("VECTOR_0",false))
                .when(useVectors(), _ -> defineVectorAccessMacro("VECTOR_1",true))
                .when(useVectors(), _ -> defineMacroVLoadN())
                .when(useVectors(), _ -> defineMacroVStoreN())
                .when(useVectors(), _ -> defineMacroVectorOf(2))
                .when(useVectors(), _ -> defineMacroVectorOf(3))
                .when(useVectors(), _ -> defineMacroVectorOf(4))
                .when(useVectors(), _ -> defineMacroVectorSelectLoad(VSELECT_LOAD))
                .when(useVectors(), _ -> defineMacroVectorSelectStore(VSELECT_STORE))

                // Narrow types macros
                .when(useS16Types(), _ ->defineMacroBF16Of())
                .when(useS16Types(), _ ->defineMacroF162Float(F16_TO_FLOAT_0, false))
                .when(useS16Types(), _ ->defineMacroF162Float(F16_TO_FLOAT_1, true))
                .when(useS16Types(), _ -> defineMacroF16Of())
                .when(useS16Types(), _ ->defineMacroBF162Float(BF16_TO_FLOAT_0, false))
                .when(useS16Types(), _ ->defineMacroBF162Float(BF16_TO_FLOAT_1, true))

                // Thread access macros
                .when(useThreadConstruct("gix"), _-> hashDefine("HAT_GIX", _ -> paren(_ -> id(GLOBAL_ID).paren(_ -> intConstZero()))))
                .when(useThreadConstruct("giy"), _-> hashDefine("HAT_GIY", _ -> paren(_ -> id(GLOBAL_ID).paren(_ -> intConstOne()))))
                .when(useThreadConstruct("giz"), _-> hashDefine("HAT_GIZ", _ -> paren(_ -> id(GLOBAL_ID).paren(_ -> intConstTwo()))))
                .when(useThreadConstruct("lix"), _-> hashDefine("HAT_LIX", _ -> paren(_ -> id(LOCAL_ID).paren(_ -> intConstZero()))))
                .when(useThreadConstruct("liy"), _-> hashDefine("HAT_LIY", _ -> paren(_ -> id(LOCAL_ID).paren(_ -> intConstOne()))))
                .when(useThreadConstruct("liz"), _-> hashDefine("HAT_LIZ", _ -> paren(_ -> id(LOCAL_ID).paren(_ -> intConstTwo()))))
                .when(useThreadConstruct("gsx"), _-> hashDefine("HAT_GSX", _ -> paren(_ -> id(GLOBAL_SIZE).paren(_ -> intConstZero()))))
                .when(useThreadConstruct("gsy"), _-> hashDefine("HAT_GSY", _ -> paren(_ -> id(GLOBAL_SIZE).paren(_ -> intConstOne()))))
                .when(useThreadConstruct("gsz"), _-> hashDefine("HAT_GSZ", _ -> paren(_ -> id(GLOBAL_SIZE).paren(_ -> intConstTwo()))))
                .when(useThreadConstruct("lsx"), _-> hashDefine("HAT_LSX", _ -> paren(_ -> id(LOCAL_SIZE).paren(_ -> intConstZero()))))
                .when(useThreadConstruct("lsy"), _-> hashDefine("HAT_LSY", _ -> paren(_ -> id(LOCAL_SIZE).paren(_ -> intConstOne()))))
                .when(useThreadConstruct("lsz"), _-> hashDefine("HAT_LSZ", _ -> paren(_ -> id(LOCAL_SIZE).paren(_ -> intConstTwo()))))
                .when(useThreadConstruct("bix"), _-> hashDefine("HAT_BIX", _ -> paren(_ -> id(GROUP_ID).paren(_ -> intConstZero()))))
                .when(useThreadConstruct("biy"), _-> hashDefine("HAT_BIY", _ -> paren(_ -> id(GROUP_ID).paren(_ -> intConstOne()))))
                .when(useThreadConstruct("biz"), _-> hashDefine("HAT_BIZ", _ -> paren(_ -> id(GROUP_ID).paren(_ -> intConstTwo()))))
                .when(useThreadConstruct("bsx"), _-> hashDefine("HAT_BSX", _ -> paren(_ -> id(NUM_GROUPS).paren(_ -> intConstZero()))))
                .when(useThreadConstruct("bsy"), _-> hashDefine("HAT_BSY", _ -> paren(_ -> id(NUM_GROUPS).paren(_ -> intConstOne()))))
                .when(useThreadConstruct("bsz"), _-> hashDefine("HAT_BSZ", _ -> paren(_ -> id(NUM_GROUPS).paren(_ -> intConstTwo()))))

                // Math Functions
                .when(useS16Types(), _->maxMacro("MAX_HAT"))
                .when(useS16Types(), _->minMacro("MIN_HAT"))

                // Barrier
                .when(useBarrier(), _ ->hashDefine("HAT_BARRIER", _ -> id("barrier").oparen().id("CLK_LOCAL_MEM_FENCE").cparen()))

                // S16 transformations macros
                .when(useS16Types(), _ -> hashDefine("BFLOAT16", _ -> keyword("ushort")))
                .when(useS16Types(), _ -> typedefSingleValueStruct("F16",  "half"))
                .when(useS16Types(), _ -> typedefSingleValueStruct("BF16",  "BFLOAT16"))
                .when(useS16Types(), _ -> unionBfloat16())
                .when(useS16Types(), _ -> build_builtin_bfloat16ToFloat("bf16"))
                .when(useS16Types(), _ -> build_builtin_float2bfloat16("f"));
    }

    @Override
    public OpenCLHATKernelBuilder atomicInc( Op.Result instanceResult, String name) {
        return id("atomic_inc").paren(_ -> ampersand().recurse( instanceResult.op()).rarrow().id(name));
    }

    private OpenCLHATKernelBuilder defineMacroVectors(String macroName, List<String> params, String construct, Consumer<C99HATKernelBuilder> codeBuilder) {
        return macroNoParenthesis(macroName, params, _ -> id(CONCAT)
                .paren(_ -> id(construct).comma().id(N))
                .paren( _ -> {
                        codeBuilder.accept(self());
                        intConstZero().comma().sp().id(CONCAT)
                        .paren(_ -> id(VECTOR).comma().id(IS_LOCAL))
                        .paren( _ -> id(ADDDR).comma().id(INDEX));
                }));
    }

    /**
     * <code>
     *     #define VLOADN(N, addr, index, isLocal) CONCAT(vload, N)(0, CONCAT(VECTOR_, isLocal)(addr, index))
     * </code>
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroVLoadN() {
        return defineMacroVectors(VLOADN, getMacroVectorParamsLoad(), VLOAD,  _ -> self());
    }

    /**
     * <code>
     *     #define VSTOREN(N, addr, index, isLocal, vectorVal) CONCAT(vstore, N)((vectorVal), 0, CONCAT(VECTOR_, isLocal)(addr, index))
     * </code>
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroVStoreN() {
        return defineMacroVectors(VSTOREN, getMacroVectorParamsStore(), VSTORE,  _ ->  id(VECTOR_VAL).comma().sp());
    }

    /**
     * <code>
     *     #define VECTOR_OF4(elementType, p0, p1, p2, p3) (CONCAT(elementType,4))(p0,p1,p2,p3)
     * </code>
     *
     * @param lanes
     *  vector width
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroVectorOf(int lanes) {
        List<String> params = new ArrayList<>();
        params.add(ELEMENT_TYPE);
        IntStream.range(0, lanes).mapToObj(i -> "p" + i).forEach(params::add);
        return macroNoParenthesis(VECTOR_OF + lanes, params, _ -> {
            paren(_ -> id(CONCAT).paren(_ -> id(ELEMENT_TYPE).comma().id(String.valueOf(lanes))));
            paren(_ -> {
                for (int i = 1; i < params.size(); i++) {
                    id(params.get(i));
                    either((i < params.size() - 1), _ -> comma(), _ -> self());
                }
            });
        });
    }

    /**
     * <code>
     * #define F16_OF(val) (F16_t){(val)}
     * </code>
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroF16Of() {
        List<String> params = List.of("val");
        return macroNoParenthesis(C99HATKernelBuilder.F16_OF, params, _ ->
                paren(_ -> f16Type())
                        .brace(_ ->
                                paren(_-> id("val"))));
    }

    /**
     * <code>
     *  #define BF16_OF(val) (BF16_t){floatTobfloat16(val)}
     * </code>
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroBF16Of() {
        List<String> params = List.of("val");
        return macroNoParenthesis(C99HATKernelBuilder.BF16_OF, params, _ ->
                paren(_ -> bf16Type())
                        .brace(_ -> builtin_float2bfloat16().paren(_-> id("val"))));
    }

    /**
     * <code>
     *     #define F16_TO_FLOAT_0(val) (float)(val->value)
     *     #define F16_TO_FLOAT_1(val) (float)(val.value)
     * </code>
     *
     * @param name
     *       Name of the OpenCL Macro
     * @param isLocal
     *       Flag to indicate if access is from a local/private region or global region.
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroF162Float(String name, boolean isLocal) {
        List<String> params = List.of("val");
        return macroNoParenthesis(name, params, _ ->
                paren(_ -> f32Type())
                        .paren(_-> id("val")
                                .dotOrArrow(isLocal)
                                .id(VALUE)));
    }

    /**
     * <code>
     *     #define BF16_TO_FLOAT_0(val) (bfloat16Tofloat(val->value))
     *     #define BF16_TO_FLOAT_1(val) (bfloat16Tofloat(val.value))
     * </code>
     * @param name
     *     Name of the OpenCL Macro
     * @param isLocal
     *     Flag to indicate if access is from a local/private region or global region.
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder defineMacroBF162Float(String name, boolean isLocal) {
        List<String> params = List.of("val");
        return macroNoParenthesis(name, params, _ ->
                paren(_ -> builtin_bfloat16ToFloat()
                        .paren(_-> id("val")
                                .dotOrArrow(isLocal)
                                .id(VALUE))));
    }

    @Override
    public OpenCLHATKernelBuilder hatBinaryVectorOp(OpHelper.Invoke binOp) {
        return paren(_-> {
            recurseResultOrThrow(binOp.op().operands().get(0));
            sp().id(BinaryOpEnum.of(binOp.op()).symbol()).sp();
            recurseResultOrThrow(binOp.op().operands().get(1));
        });
    }

    @Override
    protected String mapMathIntrinsic(String hatMathIntrinsicName) {
        return MATH_FUNCTIONS.getOrDefault(hatMathIntrinsicName, hatMathIntrinsicName);
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
            narrowCategory = HATPhaseUtils.reduceFloatType(invoke);
            if (narrowCategory == null && isMathLib(invoke)) {
                narrowCategory = HATPhaseUtils.reduceFloatTypeFromReturnType(invoke);
            }
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
