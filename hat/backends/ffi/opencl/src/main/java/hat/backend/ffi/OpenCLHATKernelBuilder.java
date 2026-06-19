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
import hat.types.F16;
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
import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static hat.phases.HATPhaseUtils.isMathLib;
import static optkl.IfaceValue.Vector.getVectorShape;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    @FunctionalInterface
    private interface CodeGenAction {
        void apply() throws IllegalStateException;
    }

    private final Map<String, CodeGenAction> tensorTypeTable;

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
        tensorTypeTable = new HashMap<>();
        tensorTypeTable.put("loadF16", this::f16Type);
        tensorTypeTable.put("load", this::f32Type);
        tensorTypeTable.put("loadF32", this::f32Type);
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

    @Override
    protected OpenCLHATKernelBuilder varOpTensor(CoreOp.VarOp varOp) {
        return recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op());
    }

    @Override
    protected OpenCLHATKernelBuilder hatWarpSize() {
        return constant("1");
    }

    private OpenCLHATKernelBuilder generateHatTensorCreate(List<Integer> shape, Object klass, String varTensorName, Value v) {
        final int sizeToAllocate = shape.get(0) * shape.get(1);
        switch (klass) {
            case ClassType classType when OpHelper.isAssignable(scopedCodeBuilderContext.lookup(), classType, F16.class) -> f16Type();
            case PrimitiveType primitiveType when primitiveType.equals(PrimitiveType.FLOAT) -> type("float");
            case null, default -> {
                // When we derive the type for tensors that are not accumulators
                if (v.declaringElement() instanceof CoreOp.VarOp tensorVarOp) {
                    Value tensorVar = tensorVarOp.result();
                    String loadVariance = findLoadVariance(tensorVar, tensorVarOp);
                    CodeGenAction typeFunction = tensorTypeTable.getOrDefault(loadVariance, null);
                    if (typeFunction == null) {
                        throw new IllegalStateException("Load Type not supported:" + typeFunction);
                    }
                    typeFunction.apply();
                }
            }
        }
        return sp().varName(varTensorName).sbrace(_-> intValue(sizeToAllocate));
    }

    private OpenCLHATKernelBuilder createTensor(OpHelper.Invoke tensorCreateOp) {
        Value v = tensorCreateOp.op().result().uses().getFirst();
        Value shapeValue;
        String varTensorName;
        if (v.declaringElement() instanceof CoreOp.VarOp tensorVarOp) {
            shapeValue = findShape(tensorVarOp.result(), tensorVarOp.result());
            varTensorName = tensorVarOp.varName();
        } else {
            throw new IllegalStateException("Value not supported");
        }
        List<Integer> shape = obtainShapeTensor(shapeValue);
        return generateHatTensorCreate(shape, null, varTensorName, v);
    }

    private OpenCLHATKernelBuilder createTensorAccumulator(OpHelper.Invoke tensorCreateOp) {
        Value v = tensorCreateOp.op().result().uses().getFirst();
        Value shapeValue = tensorCreateOp.op().operands().getFirst();
        List<Integer> shape = obtainShapeTensor(shapeValue);
        Object klass = null;
        Value classOperand = tensorCreateOp.op().operands().getLast();
        if (classOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
            klass = constantOp.value();
        }
        String varTensorName = null;
        if (v.declaringElement() instanceof CoreOp.VarOp tensorVarOp) {
            varTensorName = tensorVarOp.varName();
        }
        return generateHatTensorCreate(shape, klass, varTensorName, v);
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorCreateOperation(OpHelper.Invoke invoke) {
        List<Value> operands = invoke.op().operands();
        if (operands.isEmpty()) {
            return createTensor(invoke);
        } else {
            return createTensorAccumulator(invoke);
        }
    }

    private String generateVariableName(String prefix) {
        String vocab = "abcdefghijklmnopqrstuvxyz";
        Random r = new Random();
        StringBuilder varA = new StringBuilder(prefix);
        for (int i = 0; i < 3; i++) {
            varA.append(vocab.charAt(r.nextInt(vocab.length())));
        }
        return varA.toString();
    }

    private static final String INDEX_PREFIX = "index_$";

    /**
     * Code example being generated:
     *
     * <p>
     *     <code>
     *        for (int index_$fzm = 0;index_$fzm < shape[0];index_$fzm++) {
     *            for (int index_$ups = 0;index_$ups < shape[1];index_$ups++) {
     *               acc[index_$fzm*16+index_$ups] = initValue;
     *       }};
     *     </code>
     * </p>
     *
     * @param shape
     * @param tensorVarOp
     * @param initValue
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder emitFillOperationForAccummulator(List<Integer> shape, CoreOp.VarOp tensorVarOp, float initValue) {
        String prefix = INDEX_PREFIX;
        int from = 0;
        int toLoopA = shape.getFirst();
        int toLoopB = shape.get(1);
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).assign().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(toLoopA).semicolon();
            id(varA).plusplus();
        }).sp().brace(_ -> {
            in().nl().forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).assign().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(toLoopB).semicolon();
                id(varB).plusplus();
            }).sp().in();

            brace(_ -> nl()
                    .id(tensorVarOp.varName())
                    .sbrace(_ ->
                            id(varA).mul()
                                    .intValue(toLoopB)
                                    .plus()
                                    .id(varB))
                    .assign()
                    .constant(Float.toString(initValue)).id("f")
                    .semicolon().nl()).out().out();
        });
        return self();
    }

    /**
     * Code example being generated:
     *
     * <p>
     *     <code>
     *       for (int m = 0; m < SHAPE_1; m++)
     *           for (int n = 0; n < SHAPE_2; n++)
     *             tensor[m * SHAPE_2 + n] = initValue;
     *     </code>
     * </p>
     *
     * @param tensorFillOp
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    @Override
    public OpenCLHATKernelBuilder hatTensorFill(OpHelper.Invoke tensorFillOp) {
        // 1. Access to the variable name
        var tensorValue = tensorFillOp.op().operands().getFirst();
        CoreOp.VarOp tensorVarOp = findTensorVarOp(tensorValue);
        if (tensorVarOp == null) {
            throw new IllegalStateException("[Error][Codegen] Expected a CoreOp.VarOp, but found `null` instead");
        }

        // 2. Access the shape
        // Second parameters: analysis of the shape
        List<Integer> shape = getShapeFromTensorCreateValue(tensorVarOp.operands().getFirst());

        // 3. Access the layout
        var tensorInitValue = tensorFillOp.op().operands().get(1);
        float initValue = getValueConstantTensor(tensorInitValue);

        // 4. Generate the fill operation
        emitFillOperationForAccummulator(shape, tensorVarOp, initValue);
        return self();
    }

    /**
     * Example of code being generated:
     *
     * <p>
     * <code>
     *  for (int m = 0; m < WMMA_M; m++) {
     *    for (int n = 0; n < WMMA_N; n++) {
     *       float sum = acc[m][n];
     *       for (int k = 0; k < WMMA_K; k++) {
     *         F16_t ha = a_frag[m * WMMA_K + k];
     *         F16_t hb = b_frag[k * WMMA_N + n];
     *         F16_t result = (F16_t){(ha.value * hb.value)};
     *         sum += (float)(result.value);
     *       }
     *       acc[m][n] = sum;
     *    }
     * }
     * </code>
     * </p>
     *
     * @param shape
     * @param tensorA
     * @param tensorB
     * @param tensorC
     * @param result
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder generateTensorMMA(List<Integer> shape, CoreOp.VarOp tensorA, CoreOp.VarOp tensorB, CoreOp.VarOp tensorC, CoreOp.VarOp result) {
        String prefix = INDEX_PREFIX;
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        String varC = generateVariableName(prefix);
        String acc = generateVariableName("sum_");
        final int from = 0;
        final int M = shape.get(0);
        final int N = shape.get(1);
        final int K = shape.get(2);

        // ---------------------------
        // Shapes:
        // tensor A   with shape: MxK
        // tensor B   with shape: KxN
        // tensor acc with shape: MxN
        // ---------------------------

        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).assign().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(M).semicolon();
            id(varA).plusplus();
        }).sp().brace(_ -> {
            in().nl().forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).assign().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(N).semicolon();
                id(varB).plusplus();
            }).in();

            brace(_ -> {
                nl().f32Type().sp().id(acc).assign().id(tensorC.varName()).sbrace(_ -> {
                    id(varA).mul().intValue(N).sp().plus().id(varB);
                }).semicolon().nl();

                forKeyword().sp().paren(_ -> {
                    s32Type().sp().id(varC).assign().intValue(from).semicolon();
                    id(varC).sp().lt().sp().intValue(K).semicolon();
                    id(varC).plusplus();
                }).sp().in();

                brace(_ -> {
                    nl();
                    String ha = generateVariableName("ha_");
                    String hb = generateVariableName("hb_");
                    String resultTensor = generateVariableName("h_res_");
                    f16Type().sp().id(ha).assign().id(tensorA.varName()).sbrace(_ -> id(varA).mul().intValue(K).sp().plus().id(varC)).semicolon().nl();
                    f16Type().sp().id(hb).assign().id(tensorB.varName()).sbrace(_ -> id(varC).mul().intValue(N).sp().plus().id(varB)).semicolon().nl();
                    f16Type().sp().id(resultTensor).assign().paren(_ -> f16Type()).brace(_ -> paren(_ -> id(ha).dot().id(VALUE).mul().id(hb).dot().id(VALUE))).semicolon().nl();
                    id(acc).sp().plusEquals().cast(_ -> f32Type()).paren(_ -> id(resultTensor).dot().id(VALUE)).semicolon().nl();
                }).nl().out();

                id(result.varName()).sbrace(_ -> id(varA).sp().mul().sp().intValue(N).sp().plus().sp().id(varB)).assign().id(acc).semicolon().nl();

            }).semicolon().nl();

        }).out().out();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorMMA(OpHelper.Invoke tensorMMAOp) {
        var resulTensorValue = tensorMMAOp.op().operands().getFirst();
        var tensorAValue = tensorMMAOp.op().operands().get(1);
        var tensorBValue = tensorMMAOp.op().operands().get(2);
        var tensorCValue = tensorMMAOp.op().operands().get(3);
        var tensorA = findTensorVarOp(tensorAValue);
        var tensorB = findTensorVarOp(tensorBValue);
        var tensorC = findTensorVarOp(tensorCValue);
        var tensorResult = findTensorVarOp(resulTensorValue);
        if (tensorA == null || tensorB == null || tensorC == null || tensorResult == null) {
            throw new IllegalStateException("[Error][CodeGen] Expected a tensorValue, but found `null` instead");
        }
        List<Integer> shape = getShapeFromTensorVarOp(tensorResult);
        return generateTensorMMA(shape, tensorA, tensorB, tensorC, tensorResult);
    }

    private CoreOp.VarOp findTensorVarOp(OpHelper.Invoke tensorLoadOp) {
        var tensorStoreLoadValue = tensorLoadOp.op().result().uses().getFirst();
        if (tensorStoreLoadValue.declaringElement() instanceof CoreOp.VarAccessOp.VarStoreOp tensorStoreLoadOp) {
            Value first = tensorStoreLoadOp.operands().getFirst();
            if (first.declaringElement() instanceof CoreOp.VarOp tensorVarOp) {
                return tensorVarOp;
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    /**
     * Code example being generated:
     *
     * <p>
     * <code>
     *      for (int m = 0; m < WMMA_M; m++) {
     *         int rowA = aRow + m;
     *         for (int n = 0; n < WMMA_N; n++) {
     *           int colA = aCol + n;
     *           int idxA = rowA + colA * lda;
     *           HAT_GLOBAL_MEM F16Impl_t* ha = &matrixA->array[idxA];
     *           F16_t r = (F16_t){ha->value};
     *           tensorA[m * WMMA_M + n] = r;
     *         }
     *     }
     * </code>
     * </p>
     *
     * @param shape
     * @param iIndexValue
     * @param jIndexValue
     * @param isColumnMajor
     * @param leadingDimension
     * @param ptrValue
     * @param tensorVarOp
     *
     */
    private void generateTensorLoad(List<Integer> shape, Value iIndexValue, Value jIndexValue, boolean isColumnMajor, Value leadingDimension, Value ptrValue, CoreOp.VarOp tensorVarOp, int tensorOrder) {

        String prefix = INDEX_PREFIX;
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        final int from = 0;
        final int M;
        final int N;

        // ---------------------------
        // Shapes:
        // tensor A   with shape: MxK
        // tensor B   with shape: KxN
        // ---------------------------
        String matrixOrder = tensorOrderTable.get(tensorOrder);
        switch (matrixOrder) {
            case TENSOR_MATRIX_A -> {
                M = shape.get(0); // M
                N = shape.get(2); // K
            }
            case TENSOR_MATRIX_B -> {
                M = shape.get(2); // K
                N = shape.get(1); // N
            }
            case null, default -> throw new IllegalStateException("Tensor load matrix order not detected");
        }

        // We need to get if tensorA or tensorB is being loaded

        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).assign().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(M).semicolon();
            id(varA).plusplus();
        }).in();

        String row = generateVariableName("row_");

        brace(_ -> {
            nl().s32Type().sp().id(row).assign();

            if (iIndexValue instanceof Op.Result r) {
                recurse(r.op());
            }
            plus().id(varA).semicolon().nl();

            forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).assign().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(N).semicolon();
                id(varB).plusplus();
            }).sp().in();

            String col = generateVariableName("col_");

            brace(_ -> {
                nl().s32Type().sp().id(col).assign();

                if (jIndexValue instanceof Op.Result r) {
                    recurse(r.op());
                }
                plus().id(varB).semicolon().nl();

                String index = generateVariableName(INDEX_PREFIX);
                s32Type().sp().id(index).assign();

                String aVal = row;
                String bVal = col;
                if (isColumnMajor) {
                    aVal = col;
                    bVal = row;
                }

                id(aVal).sp().mul().sp();
                if (leadingDimension instanceof Op.Result r) {
                    recurse(r.op());
                }
                sp().plus().id(bVal).semicolon().nl();


                // TODO: We assume a load from global memory. In
                // future version, we will process loads from other
                // memory regions of the accelerator

                String ha = generateVariableName("ha_");
                id("HAT_GLOBAL_MEM F16Impl_t").asterisk().sp().id(ha).assign().ampersand();

                if (ptrValue instanceof  Op.Result r) {
                    recurse(r.op());
                }
                rarrow().id("array").sbrace( _ -> id(index)).semicolon().nl();

                String r = generateVariableName("r_");
                f16Type().sp().id(r).assign().cast( _ -> f16Type()).brace( _-> id(ha).rarrow().id("value")).semicolon().nl();

                // store into the acc
                emitText(tensorVarOp.varName()).sbrace( _ -> id(varA).sp().mul().intValue(N).sp().plus().id(varB));
                equals().sp().id(r).semicolon().nl();
            }).out();
        }).out();
    }

    /**
     * Code example being generated:
     *
     * <p>
     * <code>
     *     for (int m = 0; m < WMMA_M; m++) {
     int rowB = bRow + m;
     *          for (int n = 0; n < WMMA_N; n++) {
     *            int colB = bCol + n;
     *            int idxB = rowB + colB * ldb;
     *            HAT_GLOBAL_MEM F16Impl_t* hb = &matrixB->array[idxB];
     *            F16_t r = (F16_t){hb->value};
     *            b_frag[m * WMMA_M + n] = r;
     *          }
     *      }
     * </code>
     * </p>
     *
     * @param tensorLoadOp
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    @Override
    protected OpenCLHATKernelBuilder hatTensorLoad(OpHelper.Invoke tensorLoadOp) {
        List<Value> operands = tensorLoadOp.op().operands();
        var ptrValue = operands.getFirst();
        var iIndexValue = operands.get(1);
        var jIndexValue = operands.get(2);
        var leadingDimension = operands.get(3);
        CoreOp.VarOp tensorVarOp = findTensorVarOp(tensorLoadOp);
        List<Integer> shape;
        if (tensorVarOp != null) {
            shape = obtainShapeTensor(operands.get(4));
        } else {
            throw new IllegalStateException("[Error][CodeGen] Expected to see an instance of tensorVarOp but `null` found");
        }

        // Obtain if tensorA or tensorB is being loaded.
        // This is important to get the loop-bounds correct if matrices are not square
        int tensorOrder = getTensorOrder(tensorVarOp.result());

        boolean isColumnMajor = false;
        if (tensorLoadOp.op().operands().size() > 5) {
            isColumnMajor = isColumnMajor(operands.get(5));
        }

        generateTensorLoad(shape, iIndexValue, jIndexValue, isColumnMajor, leadingDimension, ptrValue, tensorVarOp, tensorOrder);
        return self();
    }

    /**
     * Example of code being generated:
     *
     * <p>
     * <code>
     *  for (int m = 0; m < WMMA_M; m++) {
     *   int rowC = cRow + m;
     *   for (int n = 0; n < WMMA_N; n++) {
     *      int colC = cCol + n;
     *      int idxC = (cRow) + (cCol) * ldc;
     *      matrixC->array[idxC] = acc[m * 16 + n];
     *   }
     * }
     * </code>
     * </p>
     *
     * @param shape
     * @param iIndexValue
     * @param jIndexValue
     * @param isColumnMajor
     * @param leadingDimension
     * @param ptrValue
     * @param tensorVarOp
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    private OpenCLHATKernelBuilder generateTensorStore(List<Integer> shape, Value iIndexValue, Value jIndexValue, boolean isColumnMajor, Value leadingDimension, Value ptrValue, CoreOp.VarOp tensorVarOp) {
        String prefix = INDEX_PREFIX;
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        final int from = 0;
        // Output is MxN, given the shape in a M,N,K triplet
        final int M = shape.get(0);
        final int N = shape.get(1);

        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).assign().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(M).semicolon();
            id(varA).plusplus();
        }).in();

        String row = generateVariableName("row_");

        brace(_ -> {
            nl().s32Type().sp().id(row).assign();

            if (iIndexValue instanceof Op.Result r) {
                recurse(r.op());
            }
            plus().id(varA).semicolon().nl();

            forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).assign().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(N).semicolon();
                id(varB).plusplus();
            }).sp().in();

            String col = generateVariableName("col_");

            brace(_ -> {
                nl().s32Type().sp().id(col).assign();

                if (jIndexValue instanceof Op.Result r) {
                    recurse(r.op());
                }
                plus().id(varB).semicolon().nl();

                String index = generateVariableName(INDEX_PREFIX);
                s32Type().sp().id(index).assign();

                String aVal = row;
                String bVal = col;
                if (isColumnMajor) {
                    aVal = col;
                    bVal = row;
                }

                id(aVal).sp().mul().sp();
                if (leadingDimension instanceof Op.Result r) {
                    recurse(r.op());
                }
                sp().plus().id(bVal).semicolon().nl();

                // TODO: We assume a load from global memory. In
                // future version, we will process loads from other
                // memory regions of the accelerator
                if (ptrValue instanceof  Op.Result r) {
                    recurse(r.op());
                }
                rarrow().id("array").sbrace( _ -> id(index)).assign();
                id(tensorVarOp.varName()).sbrace( _ -> id(varA).mul().intValue(N).plus().id(varB));
                semicolon().nl();
            }).out();
        }).out();
        return self();
    }

    /**
     * Code example being generated:
     *
     * <p>
     * <code>
     *  for (int m = 0; m < WMMA_M; m++) {
     *  `int rowC = cRow + m;
     *   for (int n = 0; n < WMMA_N; n++) {
     *      int colC = cCol + n;
     *      int idxC = (cRow) + (cCol) * ldc;
     *      matrixC->array[idxC] = acc[m * 16 + n];
     *   }
     * }
     * </code>
     * </p>
     *
     * @param tensorStoreOp
     *
     * @return {@link OpenCLHATKernelBuilder}
     */
    @Override
    protected OpenCLHATKernelBuilder hatTensorStore(OpHelper.Invoke tensorStoreOp) {
        // 1. We need the global ptr
        // 2. We need the indexes (i, j)
        // 3. We need leading dimension
        // 4. We need the name of the tensor
        // 5. We need the shape
        // 6. We need the access layout

        List<Value> operands = tensorStoreOp.op().operands();
        var ptrValue = operands.getFirst();
        var iIndexValue = operands.get(1);
        var jIndexValue = operands.get(2);
        var tensorValue = operands.get(3);
        var leadingDimension = operands.get(4);

        CoreOp.VarOp tensorVarOp = findTensorVarOp(tensorValue);
        if (tensorVarOp == null) {
            throw new IllegalStateException("[Error][CodeGen] Expected to find a tensorVarOp, but `null` instead.");
        }

        var shape = getShapeFromTensorVarOp(tensorVarOp);

        boolean isColumnMajor = false;
        if (tensorStoreOp.op().operands().size() > 5) {
            isColumnMajor = isColumnMajor(operands.get(5));
        }

        generateTensorStore(shape, iIndexValue, jIndexValue, isColumnMajor, leadingDimension, ptrValue, tensorVarOp);
        return self();
    }
}
