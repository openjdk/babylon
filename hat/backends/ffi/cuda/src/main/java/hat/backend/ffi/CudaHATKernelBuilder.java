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

import hat.DType;
import hat.callgraph.KernelCallGraph;
import hat.codebuilders.C99HATKernelBuilder;
import hat.codetypes.ConstantType;
import hat.codetypes.PtrType;
import hat.codetypes.TensorType;
import hat.dialect.ArithMathOps;
import hat.dialect.BinaryOpEnum;
import hat.dialect.TileOps;
import hat.phases.HATFP16Phase;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.FuncOpParams;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.OpHelper.Invoke;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.types.BF16;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.SequencedSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static hat.phases.HATPhaseUtils.isArrayReference;
import static hat.phases.HATPhaseUtils.isMathLib;
import static hat.phases.HATPhaseUtils.isOperandF32;
import static hat.phases.HATPhaseUtils.isVectorBinaryOperation;
import static hat.phases.HATPhaseUtils.mapLane;
import static hat.phases.HATPhaseUtils.reduceFloatType;
import static hat.phases.HATPhaseUtils.reduceFloatTypeFromReturnType;
import static jdk.incubator.code.dialect.core.CoreOp.VarOp;
import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.Invoke.invoke;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {

    // Mapping between API function names and CUDA intrinsics for the math operations
    protected static final Map<String, String> MATH_FUNCTIONS = new HashMap<>();

    static {
        MATH_FUNCTIONS.put("maxf", "max");
        MATH_FUNCTIONS.put("maxd", "max");
        MATH_FUNCTIONS.put("maxf16", "MAX_HAT");
        MATH_FUNCTIONS.put("minf", "min");
        MATH_FUNCTIONS.put("mind", "min");
        MATH_FUNCTIONS.put("minf16", "MIN_HAT");

        MATH_FUNCTIONS.put("expf", "expf");
        MATH_FUNCTIONS.put("expd", "exp");
        MATH_FUNCTIONS.put("expf16", "hexp");

        MATH_FUNCTIONS.put("cosf", "cosf");
        MATH_FUNCTIONS.put("cosd", "cos");
        MATH_FUNCTIONS.put("sinf", "sinf");
        MATH_FUNCTIONS.put("sind", "sin");
        MATH_FUNCTIONS.put("tanf", "tanf");
        MATH_FUNCTIONS.put("tand", "tan");

        MATH_FUNCTIONS.put("native_cosf", "__cosf");
        MATH_FUNCTIONS.put("native_sinf", "__sinf");
        MATH_FUNCTIONS.put("native_tanf", "__tanf");
        MATH_FUNCTIONS.put("native_expf", "__expf");

        MATH_FUNCTIONS.put("sqrtf", "sqrtf");
        MATH_FUNCTIONS.put("sqrtd", "sqrt");
    }

    private final Map<Op, String> mapVectorName;
    private final Deque<String> stack;
    private static final int CUDA_WARP_SIZE = 32;
    private final boolean isTile;

    protected CudaHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext, boolean isTile) {
        super(kernelCallGraph, scopedCodeBuilderContext);
        stack = new ConcurrentLinkedDeque<>();
        mapVectorName = new ConcurrentHashMap<>();
        this.isTile = isTile;
    }

    private CudaHATKernelBuilder half2float() {
        return id("__half2float");
    }

    private CudaHATKernelBuilder float2half() {
        return id("__float2half");
    }

    private CudaHATKernelBuilder nvBFloat16() {
        return id("__nv_bfloat16");
    }

    private CudaHATKernelBuilder bfloat162float() {
        return id("__bfloat162float");
    }

    private CudaHATKernelBuilder reinterpretCast() {
        return keyword("reinterpret_cast");
    }

    private CudaHATKernelBuilder threadIdx() {
        return keyword("threadIdx");
    }

    private CudaHATKernelBuilder threadIdxX() {
        return threadIdx().dot().id("x");
    }

    private CudaHATKernelBuilder threadIdxY() {
        return threadIdx().dot().id("y");
    }

    private CudaHATKernelBuilder threadIdxZ() {
        return threadIdx().dot().id("z");
    }

    private CudaHATKernelBuilder gridDim() {
        return keyword("gridDim");
    }

    private CudaHATKernelBuilder gridDimX() {
        return gridDim().dot().id("x");
    }

    private CudaHATKernelBuilder gridDimY() {
        return gridDim().dot().id("y");
    }

    private CudaHATKernelBuilder gridDimZ() {
        return gridDim().dot().id("z");
    }

    private CudaHATKernelBuilder blockDim() {
        return keyword("blockDim");
    }

    private CudaHATKernelBuilder blockDimX() {
        return blockDim().dot().id("x");
    }

    private CudaHATKernelBuilder blockDimY() {
        return blockDim().dot().id("y");
    }

    private CudaHATKernelBuilder blockDimZ() {
        return blockDim().dot().id("z");
    }

    private CudaHATKernelBuilder blockIdx() {
        return keyword("blockIdx");
    }

    private CudaHATKernelBuilder blockIdxX() {
        return blockIdx().dot().id("x");
    }

    private CudaHATKernelBuilder blockIdxY() {
        return blockIdx().dot().id("y");
    }

    private CudaHATKernelBuilder blockIdxZ() {
        return blockIdx().dot().id("z");
    }

    @Override
    protected CudaHATKernelBuilder hatWarpSize() {
        return id("HAT_WRS");
    }

    @Override
    public CudaHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_CUDA")
                .hashDefine("HAT_GLOBAL_MEM", _ -> {})
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__shared__"))
                .hashDefine("HAT_FUNC", _ -> externC().sp().keyword("__device__").sp())
                .hashDefine("HAT_KERNEL", _ -> externC().sp().either(!isTile, _ -> keyword("__global__"), _ -> keyword("__tile_global__")))

                // threads
                .hashDefine("HAT_GIX", _ -> paren(_ -> HAT_BIX().asterisk().HAT_LSX().plus().HAT_LIX()))
                .hashDefine("HAT_GIY", _ -> paren(_ -> HAT_BIY().asterisk().HAT_LSY().plus().HAT_LIY()))
                .hashDefine("HAT_GIZ", _ -> paren(_ -> HAT_BIZ().asterisk().HAT_LSZ().plus().HAT_LIZ()))
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
                .hashDefine("HAT_BSX", _ -> gridDimX())
                .hashDefine("HAT_BSY", _ -> gridDimY())
                .hashDefine("HAT_BSZ", _ -> gridDimZ())
                .hashDefine("HAT_WRS", _ -> paren( _ -> intValue(CUDA_WARP_SIZE)))

                // Barrier
                .when(useBarrier(), _ -> hashDefine("HAT_BARRIER", _ -> keyword("__syncthreads").ocparen()))

                // Math
                .when(useS16Types(), _ -> maxMacro("MAX_HAT"))
                .when(useS16Types(), _ -> minMacro("MIN_HAT"))
                .when(isTile, _ -> ceilDiv("ceilDiv"))

                // General Macros
                .when(useVectors(), _ -> concatMacro().prefixMacro())

                // Vectors
                .when(useVectors(), _ -> defineVectorAccessMacro("VECTOR_0",false))
                .when(useVectors(), _ -> defineVectorAccessMacro("VECTOR_1",true))
                .when(useVectors(), _ -> defineMacroVLoadN())
                .when(useVectors(), _ -> defineMacroVStoreN())
                .when(useVectors(), _ -> defineMacroVectorOf(2))
                .when(useVectors(), _ -> defineMacroVectorOf(3))
                .when(useVectors(), _ -> defineMacroVectorOf(4))
                .when(useVectors(), _ -> defineMacroVectorSelectLoad(VSELECT_LOAD))
                .when(useVectors(), _ -> defineMacroVectorSelectStore(VSELECT_STORE))

                // S16 types
                .when(useS16Types() || isTile, _ -> defineMacroF16Of(F16_OF))
                .when(useS16Types() || isTile, _ -> defineMacroBF16Of(BF16_OF))
                .when(useS16Types() || isTile, _ -> defineMacroF162Float(F16_TO_FLOAT_0, false))
                .when(useS16Types() || isTile, _ -> defineMacroF162Float(F16_TO_FLOAT_1, true))
                .when(useS16Types() || isTile, _ -> defineMacroBF162Float(BF16_TO_FLOAT_0, false))
                .when(useS16Types() || isTile, _ -> defineMacroBF162Float(BF16_TO_FLOAT_1, true))
                .when(useS16Types() || isTile, _ -> includeSys("cuda_fp16.h", "cuda_bf16.h"))
                .when(useS16Types() || isTile, _ -> hashDefine("BFLOAT16", _ -> keyword("__nv_bfloat16")))
                .when(useS16Types() || isTile, _ -> typedefSingleValueStruct("F16", "half"))
                .when(useS16Types() || isTile, _ -> typedefSingleValueStruct("BF16", "BFLOAT16"))

                // Tensor Macros
                .when(useTensors(), _ -> includeSys("mma.h"))
                .when(useTensors(), _ -> defineFragmentCreate(MACRO_FRAGMENT_CREATE))
                .when(useTensors(), _ -> defineMacroTensorFill(MACRO_FRAGMENT_FILL))
                .when(useTensors(), _ -> defineMacroTensorMMA(MACRO_FRAGMENT_MMA))
                .when(useTensors(), _ -> defineMacroTensorLoadF16(MACRO_FRAGMENT_LOAD_F16))
                .when(useTensors(), _ -> defineMacroTensorStore(MACRO_FRAGMENT_STORE))

                // tile
                .when(isTile, _ -> include("cuda_tile.h"))
                .when(isTile, _-> id("namespace ct = cuda::tiles").semicolon().nl())
                .when(isTile, _-> namespace("ct::literals"));
    }

    @Override
    public CudaHATKernelBuilder atomicInc(Op.Result instanceResult, String name) {
        return id("atomicAdd").paren(_ -> ampersand().recurseResultOrThrow(instanceResult).rarrow().id(name).comma().literal(1));
    }

    /**
     * <code>
     *     #define VLOADN(N, addr, index, isLocal) reinterpret_cast<CONCAT(float, N) *>(CONCAT(VECTOR_, isLocal)(addr, index))[0]
     * </code>
     *
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroVLoadN() {
        List<String> params = getMacroVectorParamsLoad();
        return macroNoParenthesis(VLOADN, params, _ ->
                reinterpretCast().lt().id(CONCAT).paren(_ -> f32Type().comma().sp().id(N)).sp().asterisk().gt()
                .paren( _ -> id(CONCAT).paren( _ -> id(VECTOR).comma().sp().id(IS_LOCAL))
                .paren( _ -> id(ADDDR).comma().sp().id(INDEX)))
                .sbrace( _ -> intConstZero()));
    }

    /**
     * <code>
     *     #define VSTOREN(N, a, index, isLocal, vectorVal) reinterpret_cast<CONCAT(float, N)*>(CONCAT(VECTOR_, isLocal)(a, index))[0] = vectorVal
     * </code>
     *
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroVStoreN() {
        List<String> params = getMacroVectorParamsStore();
        return macroNoParenthesis(VSTOREN, params, _ ->
                reinterpretCast().lt().id(CONCAT).paren(_ -> f32Type().comma().sp().id(N)).sp().asterisk().gt()
                        .paren( _ -> id(CONCAT).paren( _ -> id(VECTOR).comma().sp().id(IS_LOCAL))
                                .paren( _ -> id(ADDDR).comma().sp().id(INDEX)))
                        .sbrace( _ -> intConstZero()).sp().equals().sp().id(VECTOR_VAL));
    }

    /**
     * <code>
     *    #define VECTOR_OF2(elementType, p0, p1) (PREFIX(make_,CONCAT(elementType,2)))(p0,p1)
     *    #define VECTOR_OF3(elementType, p0, p1, p2) (PREFIX(make_,CONCAT(elementType,3)))(p0,p1,p2)
     *    #define VECTOR_OF4(elementType, p0, p1, p2, p3) (PREFIX(make_,CONCAT(elementType,4)))(p0,p1,p2,p3)
     * </code>
     * @param lanes
     *    Vector width
     *
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroVectorOf(int lanes) {
        List<String> params = new ArrayList<>();
        params.add(ELEMENT_TYPE);
        IntStream.range(0, lanes).mapToObj(i -> "p" + i).forEach(params::add);
        return macroNoParenthesis(VECTOR_OF + lanes, params, _ -> {
            paren(_ -> id(PREFIX).paren(_ ->
                    id(MAKE_).comma().id(CONCAT).paren(_ -> id(ELEMENT_TYPE).comma().id(String.valueOf(lanes)))));
            paren(_ -> {
                for (int i = 1; i < params.size(); i++) {
                    id(params.get(i));
                    either((i < params.size() - 1), _ -> comma(), _ -> self());
                }
            });
        });
    }

    private CudaHATKernelBuilder defineS16macro(String name, Consumer<CudaHATKernelBuilder> type, Consumer<CudaHATKernelBuilder> buildFunction) {
        List<String> params = List.of("val");
        return macroNoParenthesis(name, params, _ ->
                paren(_ -> type.accept(self()))
                        .brace(_ -> {
                            buildFunction.accept(self());
                            paren(_-> id("val"));
                        }));
    }

    /**
     * <code>
     *    #define F16_OF(val) (F16_t){__float2half(val)}
     * </code>
     * @param name
     *     Name of the CUDA Macro
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroF16Of(String name) {
        return defineS16macro(name, _ -> f16Type(), _ -> float2half());
    }

    /**
     * <code>
     *    #define BF16_OF(val) (BF16_t){__nv_bfloat16(val)}
     * </code>
     * @param name
     *    Name of the CUDA Macro
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroBF16Of(String name) {
        return defineS16macro(name, _ -> bf16Type(), _ -> nvBFloat16());
    }

    private CudaHATKernelBuilder defineMacroS16Conversion(String name, Consumer<CudaHATKernelBuilder> type, boolean isLocal) {
        List<String> params = List.of("val");
        return macroNoParenthesis(name, params, _ ->
                paren(_ -> type.accept(self()))
                        .paren(_-> id("val")
                                .dotOrArrow(isLocal)
                                .id(VALUE)));
    }

    /**
     * <code>
     *    #define F16_TO_FLOAT_0(val) (__half2float)(val->value)
     *    #define F16_TO_FLOAT_1(val) (__half2float)(val.value)
     * </code>
     * @param name
     *    Name of the CUDA Macro
     * @param isLocal
     *    Flag to indicate if the parameter corresponds to a variable in private/shared or global region.
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroF162Float(String name, boolean isLocal) {
        return defineMacroS16Conversion(name, _ -> half2float(), isLocal);
    }

    /**
     * <code>
     *     #define BF16_TO_FLOAT_0(val) (__bfloat162float(val->value))
     *     #define BF16_TO_FLOAT_1(val) (__bfloat162float(val.value))
     * </code>
     * @param name
     *     Name of the CUDA Macro
     * @param isLocal
     *     Flag to indicate if the parameter corresponds to a variable in private/shared or global region.
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroBF162Float(String name, boolean isLocal) {
        return defineMacroS16Conversion(name, _ -> bfloat162float(), isLocal);
    }

    private static final String INVALID = "INVALID";

    /**
     * List of macros using __VA_ARGS__:
     *
     * <p>
     * <code>
     * #define FRAGMENT_CREATE_3(type, name, size) type name[size]
     * #define FRAGMENT_CREATE_7(kind, size, m, n, k, type, name) nvcuda::wmma::fragment<nvcuda::wmma::kind, m, n, k, type> name
     * #define FRAGMENT_CREATE_8(kind, size, m, n, k, type, layout, name) nvcuda::wmma::fragment<nvcuda::wmma::kind, m, n, k, type, layout> name
     * #define FRAGMENT_CREATE_SELECT(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
     * #define FRAGMENT_CREATE(...) FRAGMENT_CREATE_SELECT(__VA_ARGS__,FRAGMENT_CREATE_8,FRAGMENT_CREATE_7,INVALID,INVALID,INVALID,FRAGMENT_CREATE_3)(__VA_ARGS__)
     * </code>
     * </p>
     *
     * @param macroName
     * @return {@link CudaHATKernelBuilder}
     */
    public CudaHATKernelBuilder defineFragmentCreate(String macroName) {
        List<String> params = List.of("type",  "name", "size");
        macroNoParenthesis(macroName + "_3", params, _ ->
                id(params.getFirst()).sp().id(params.get(1)).sbrace(_ -> id(params.get(2))));

        List<String> params1  = List.of("kind", "size", "m", "n", "k", "type", "name");
        macroNoParenthesis(macroName + "_7", params1, _ ->
                id(WMMA_FRAGMENT_BASE)
                    .ltgt(_ ->
                        id(WMMA_PREFIX).id("kind")
                            .comma().sp()
                            .id("m")
                            .comma().sp()
                            .id("n")
                            .comma().sp()
                            .id("k")
                            .comma().sp()
                            .id("type")).sp().id("name"));

        List<String> params2 = List.of("kind", "size", "m", "n", "k", "type", "layout", "name");
        macroNoParenthesis(macroName + "_8", params2, _ ->
                id(WMMA_FRAGMENT_BASE)
                        .ltgt(_ ->
                                id(WMMA_PREFIX).id("kind")
                                        .comma().sp()
                                        .id("m")
                                        .comma().sp()
                                        .id("n")
                                        .comma().sp()
                                        .id("k")
                                        .comma().sp()
                                        .id("type")
                                        .comma().sp()
                                        .id("layout")).sp().id("name"));

        List<String> params3 = List.of("_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "NAME", "...");
        macroNoParenthesis("FRAGMENT_CREATE_SELECT", params3, _ ->
                id("NAME"));

        List<String> params4 = List.of("...");
        macroNoParenthesis("FRAGMENT_CREATE", params4, _ ->
                id("FRAGMENT_CREATE_SELECT").paren(_ ->
                                id("__VA_ARGS__").comma()
                                        .id("FRAGMENT_CREATE_8").comma()
                                        .id("FRAGMENT_CREATE_7").comma()
                                        .id(INVALID).comma()
                                        .id(INVALID).comma()
                                        .id(INVALID).comma()
                                        .id("FRAGMENT_CREATE_3"))
                        .paren( _ -> id("__VA_ARGS__")));
        return self();
    }

    /**
     * Example of code being generated:
     * <code>
     *   nvcuda::wmma::fill_fragment(acc, initValue);
     * </code>
     *
     * @param name
     *
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroTensorFill(String name) {
        List<String> params = paramsOfTensorFillMacro();
        return macroNoParenthesis(name, params, _ ->
                paren( _ ->
                        id(WMMA_FILL_TENSOR).paren( _->
                                id(params.get(2)).comma().id(params.get(5)))));
    }

    /**
     * Example of code being generated:
     * <code>
     *  nvcuda::wmma::mma_sync(acc,tensorA,tensorB,acc);
     * </code>
     * @param macroName
     *
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder defineMacroTensorMMA(String macroName) {
        // Args: "i", "j", "k", "acc", "tensorA", "tensorB", "tensorC", "tensorResult", "M", "N", "K";
        List<String> params = paramsOfTensorMMAMacro();
        List<String> cudaMMAArgs = new ArrayList<>();
        cudaMMAArgs.add(params.get(7)); // tensorResult
        cudaMMAArgs.add(params.get(4)); // tensorA
        cudaMMAArgs.add(params.get(5)); // tensorB
        cudaMMAArgs.add(params.get(6)); // tensorC
        return macroNoParenthesis(macroName, params, _ ->
                paren( _ ->
                        id(WMMA_MMA_TENSOR).paren( _-> commaSeparated(cudaMMAArgs, this::id))));
    }

    /**
     * Example of code being generated:
     *
     * <p>
     * <code>
     * wmma::load_matrix_sync(a_frag, matrix->array + headSize + aRow + aCol * lda, lda);
     * </code>
     * </p>
     *
     * @param macroName Macro name
     * @return {@link CudaHATKernelBuilder}
     */
    public CudaHATKernelBuilder defineMacroTensorLoadF16(String macroName) {
        List<String> params = paramsOfTensorLoad();
        return macroNoParenthesis(macroName, params, _ ->
                sp().backslash().nl()
                        .id(WMMA_LOAD_TENSOR)
                        .paren(_ -> {
                            id("tensorToLoad").comma();
                            paren(_ -> type("half").asterisk());
                            id("reference")
                                    .rarrow().id(ARRAY)
                                    .sp().plus().sp()
                                    .id("iIndexValue").plus().paren(_ -> id("jIndexValue").mul().id("leadingDimension"))
                                    .comma()
                                    .id("leadingDimension");
                        }));
    }

    /**
     * Example of code being generated:
     * <code>
     *  nvcuda::wmma::store_matrix_sync(matrixC->array + cCol+(cRow*ldc), acc,ldc,nvcuda::wmma::mem_row_major);
     * </code>
     * @param macroName
     * @return {@link CudaHATKernelBuilder}
     */
    public CudaHATKernelBuilder defineMacroTensorStore(String macroName) {
        List<String> params = List.of("M", "N", "varA", "varB", "iIndexValue", "jIndexValue", "isColumnMajor", "leadingDimension", "reference", "tensorToStore", "memAccessLayout");
        return macroNoParenthesis(macroName, params, _ ->
                sp().backslash().nl()
                .id(WMMA_STORE_TENSOR).paren(_ ->
                id("reference").rarrow().id(ARRAY)
                        .sp().plus().sp()
                        .id("iIndexValue").plus().paren(_ -> id("jIndexValue").mul().id("leadingDimension"))
                        .comma()
                        .id("tensorToStore")
                        .comma()
                        .id("leadingDimension")
                        .comma()
                        .id("memAccessLayout")));
    }

    private void recurseVectorOperand(JavaOp.InvokeOp invokeOp, String postfix) {
        Invoke invoke = invoke(scopedCodeBuilderContext.lookup(), invokeOp);
        IfaceValue.Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
        String type = vectorShape.codeType().toString() + vectorShape.lanes();
        String current = stack.peek();
        type(type).sp().id(current + postfix).semicolon().nl();
        stack.push(current + postfix);
        mapVectorName.put(invokeOp, current + postfix);
        recurse(invokeOp);
    }

    private CudaHATKernelBuilder generateHATBinaryVectorOperation(OpHelper.Invoke invoke, String nameVector) {
        Value op1 = invoke.op().operands().get(0);
        Value op2 = invoke.op().operands().get(1);
        IfaceValue.Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
        for (int lane = 0; lane < vectorShape.lanes(); lane++) {
            id(nameVector).dot().id(mapLane(lane)).sp().equals().sp();
            if (op1 instanceof Op.Result r) {
                if (!(r.op() instanceof JavaOp.InvokeOp invokeOp && isVectorBinaryOperation(invoke(scopedCodeBuilderContext.lookup(), invokeOp)))) {
                    recurse(r.op());
                } else {
                    id(mapVectorName.get(invokeOp));
                }
            }
            dot().id(mapLane(lane)).sp();
            id(BinaryOpEnum.of(invoke.op()).symbol()).sp();
            if (op2 instanceof Op.Result r) {
                if (!(r.op() instanceof JavaOp.InvokeOp invokeOp && isVectorBinaryOperation(invoke(scopedCodeBuilderContext.lookup(), invokeOp)))) {
                    recurse(r.op());
                } else {
                    id(mapVectorName.get(invokeOp));
                }
            }
            dot().id(mapLane(lane)).semicolon().nl();
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatBinaryVectorOp(OpHelper.Invoke invoke) {

        Value op1 = invoke.op().operands().get(0);
        Value op2 = invoke.op().operands().get(1);

        final String postFixOp1 = "_1";
        final String postFixOp2 = "_2";

        SequencedSet<Op.Result> uses = invoke.op().result().uses();
        String nameVector = null;
        for (Op.Result result : uses) {
            if (result.declaringElement() instanceof CoreOp.VarOp varOp) {
                // This means we have a vector declaration that we need to operate on using
                // the individual components
                stack.push(varOp.varName());
                nameVector = varOp.varName();
            }
        }

        if (nameVector != null) {
            // We add the name on the stack to process pending
            // vector operations as operands
            stack.push(nameVector);
        } else {
            // it must be already in the haspMap
            nameVector = mapVectorName.get(invoke.op());
        }

        if (nameVector == null) {
            // main name can't be null
            // This is only triggered for VectorArrayViews
            // which means that probably we need a check in the ArrayViews
            return self();
        }

        if (op1 instanceof Op.Result r && r.op() instanceof JavaOp.InvokeOp invokeOp && isVectorBinaryOperation(invoke(scopedCodeBuilderContext.lookup(), invokeOp))) {
            recurseVectorOperand(invokeOp, postFixOp1);
        }

        if (!stack.isEmpty()) {
            stack.pop();
        }

        if (op2 instanceof Op.Result r && r.op() instanceof JavaOp.InvokeOp invokeOp && isVectorBinaryOperation(invoke(scopedCodeBuilderContext.lookup(), invokeOp))) {
            recurseVectorOperand(invokeOp, postFixOp2);
        }

        if (!stack.isEmpty()) {
            stack.pop();
        }
        return generateHATBinaryVectorOperation(invoke, nameVector);
    }

    @Override
    public CudaHATKernelBuilder hatF16BinaryOp(Invoke invoke, Class<?> reducedFloatType) {
        Value op1 = invoke.op().operands().get(0);
        Value op2 = invoke.op().operands().get(1);
        boolean isFirstOperandReference = isArrayReference(scopedCodeBuilderContext.lookup(), op1);
        boolean isSecondOperandReference = isArrayReference(scopedCodeBuilderContext.lookup(), op2);

        final byte f32Mixed;
        if (!isFirstOperandReference && isOperandF32(op1)) {
            f32Mixed = HATFP16Phase.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(op2)) {
            f32Mixed = HATFP16Phase.LAST_OP;
        } else {
            f32Mixed = 0x00;
        }
        paren(_ -> f16OrBF16(reducedFloatType));
        brace(_ ->
                paren(_ -> {
                    if (f32Mixed == HATFP16Phase.LAST_OP) {
                        s16ToFloat(reducedFloatType).oparen();
                    }
                    recurseResultOrThrow(op1);
                    if (isFirstOperandReference) {
                        rarrow().id(VALUE);
                    } else if (op1 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
                        dot().id(VALUE);
                    }
                    if (f32Mixed == HATFP16Phase.LAST_OP) {
                        cparen();
                    }
                    sp().id(matchSymbol(invoke.name())).sp();
                    if (f32Mixed == HATFP16Phase.FIRST_OP) {
                        s16ToFloat(reducedFloatType).oparen();
                    }
                    recurseResultOrThrow(op2);
                    if (isSecondOperandReference) {
                        rarrow().id(VALUE);
                    } else if (op2 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
                        dot().id(VALUE);
                    }
                    if (f32Mixed == HATFP16Phase.FIRST_OP) {
                        cparen();
                    }

                })
        );
        return self();
    }

    private CudaHATKernelBuilder s16ToFloat(Class<?> float16Class) {
        if (F16.class.isAssignableFrom(float16Class)) {
            return half2float();
        } else if (BF16.class.isAssignableFrom(float16Class)) {
            return bfloat162float();
        } else {
            throw new IllegalStateException("Unexpected value: " + float16Class);
        }
    }

    @Override
    protected String mapMathIntrinsic(String hatMathIntrinsicName) {
        return MATH_FUNCTIONS.getOrDefault(hatMathIntrinsicName, hatMathIntrinsicName);
    }

    @Override
    protected CudaHATKernelBuilder varOpForNarrowType(CoreOp.VarOp varOp) {
        Value first = varOp.operands().getFirst();
        Class<?> narrowCategory;
        if (first.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
            // Find the category - This is the generic case, when ALL custom ops are removed
            Stream<Invoke> stream = Invoke.stream(kernelCallGraph.lookup(), invokeOp);
            Optional<Invoke> invoke = stream.findFirst();
            narrowCategory = reduceFloatType(invoke);
            if (narrowCategory == null && isMathLib(invoke)) {
                narrowCategory = reduceFloatTypeFromReturnType(invoke);
            }
        } else {
            throw new IllegalStateException("Expected an invoke, but found: " + first.declaringElement().getClass());
        }
        if (narrowCategory == null) {
            throw new IllegalStateException("Narrow type can't be null: ");
        }
        // handle narrow types (F16 and BFloat)
        return f16OrBF16(narrowCategory).sp().assign(
                _ -> id(varOp.varName()),
                _ -> recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op()));
    }

    @Override
    protected CudaHATKernelBuilder varOpForVectors(CoreOp.VarOp varOp) {
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

            type(vectorShape.codeType().toString() + vectorShape.lanes()).sp().varName(varOp);
            Value operand = varOp.operands().getFirst();
            if (operand instanceof Op.Result r && r.op() instanceof JavaOp.InvokeOp invokeOp && isVectorBinaryOperation(invoke(scopedCodeBuilderContext().lookup(), invokeOp))) {
                semicolon().nl();
            } else {
                assign();
            }
            return recurseResultOrThrow(operand);
        }
        return self();
    }

    @Override
    protected CudaHATKernelBuilder varOpInit(CoreOp.VarOp varOp) {
        return suffix_t((ClassType) varOp.varValueType()).sp()
                .assign(_ -> id(varOp.varName()),
                        _ -> recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op()));
    }

    @Override
    protected CudaHATKernelBuilder varOpLocalMemory(CoreOp.VarOp varOp) {
        return HAT_LOCAL_MEM().sp().varOpPrivateMemory(varOp);
    }

    @Override
    protected CudaHATKernelBuilder varOpPrivateMemory(CoreOp.VarOp varOp) {
        VarType resultType = varOp.resultType();
        if (resultType.valueType() instanceof VarType varType) {
            suffix_t((ClassType) varType.valueType());
        } else if (resultType.valueType() instanceof ClassType classType) {
            suffix_t(classType);
        }
        return sp().varName(varOp);
    }

    @Override
    protected CudaHATKernelBuilder varOpTile(VarOp varOp) {
        return id("auto")
                .sp()
                .varName(varOp)
                .assign()
                .recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op());
    }

    public static final String WMMA_MEM_COL_MAJOR = "nvcuda::wmma::mem_col_major";
    public static final String WMMA_MEM_ROW_MAJOR = "nvcuda::wmma::mem_row_major";
    public static final String WMMA_STORE_TENSOR = "nvcuda::wmma::store_matrix_sync";
    public static final String WMMA_LOAD_TENSOR = "nvcuda::wmma::load_matrix_sync";
    public static final String WMMA_MMA_TENSOR = "nvcuda::wmma::mma_sync";
    public static final String WMMA_FILL_TENSOR = "nvcuda::wmma::fill_fragment";
    public static final String WMMA_COL_MAJOR = "nvcuda::wmma::col_major";
    public static final String WMMA_ROW_MAJOR = "nvcuda::wmma::row_major";
    public static final String WMMA_FRAGMENT_BASE = "nvcuda::wmma::fragment";
    public static final String WMMA_PREFIX = "nvcuda::wmma::";

    private CudaHATKernelBuilder generateCreateTensor(List<Integer> shape, String matrixOrder, String type, Value access, String tensorVar) {
        // Params: "kind", "size", "m", "n", "k", "type", "layout", "name";
        // call the macro with the right args
        id(MACRO_FRAGMENT_CREATE).paren(_ -> {
            id(matrixOrder).comma()
                    .id(ZERO).comma().sp()           // For the CUDA backend, this value is not used
                    .intValue(shape.getFirst()).comma().sp()
                    .intValue(shape.get(1)).comma().sp()
                    .intValue(shape.get(2)).comma().sp()
                    .type(type).sp().comma();
            if (!matrixOrder.equals(TENSOR_ACC)) {
                if (access == null) {
                    id(WMMA_ROW_MAJOR);
                } else if (access.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
                    // Expecting an invokeOp
                    var invoke = invoke(scopedCodeBuilderContext().lookup(), invokeOp);
                    if (invoke != null && invoke.resultTypeIs(Tensor.ColumMajor.class)) {
                        id(WMMA_COL_MAJOR);
                    } else if (invoke != null && invoke.resultTypeIs(Tensor.RowMajor.class)) {
                        id(WMMA_ROW_MAJOR);
                    } else {
                        throw new IllegalStateException("[Error]");
                    }
                }
                comma().sp();
            }

            id(tensorVar);
        });
        return self();
    }


    private static final Map<String, String> tensorTypeTable = new HashMap<>();
    static {
        tensorTypeTable.put("loadF16", "half");
        tensorTypeTable.put("load",    "float");
        tensorTypeTable.put("loadF32", "float");
    }

    private CudaHATKernelBuilder generateTensorAccumulateCreate(Invoke tensorCreate) {
        // tensor declaration for the accumulator
        Value shapeValue = tensorCreate.op().operands().getFirst();
        List<Integer> shape = obtainShapeTensor(shapeValue);
        Value classOperand = tensorCreate.op().operands().get(1);
        Object klass = null;
        if (classOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
            klass = constantOp.value();
        }
        String tensorType = null;
        if (klass != null) {
            switch (klass) {
                case ClassType classType when OpHelper.isAssignable(scopedCodeBuilderContext.lookup(), classType, F16.class) -> tensorType = "half";
                case PrimitiveType primitiveType when primitiveType.equals(PrimitiveType.FLOAT) -> tensorType = "float";
                default -> throw new IllegalStateException("Type class not supported for Tensors: " + klass);
            }
        }
        Value v = tensorCreate.op().result().uses().getFirst();
        VarOp tensorVarOp;
        if (v.declaringElement() instanceof CoreOp.VarOp varOp) {
            tensorVarOp = varOp;
        } else {
            throw new IllegalStateException("Expected a VarOp");
        }
        Value valueAccessLayout = tensorCreate.op().operands().getLast();
        return generateCreateTensor(shape, TENSOR_ACC, tensorType, valueAccessLayout, tensorVarOp.varName());
    }

    private CudaHATKernelBuilder generateTensorCreate(Invoke tensorCreate) {
        Value v = tensorCreate.op().result().uses().getFirst();
        // Find the declaration value of the tensor
        // otherwise, we have to inspect the shape from the TensorLoadOp
        if (v.declaringElement() instanceof VarOp tensorVarOp) {
            String matrixOrder = tensorOrderTable.get(TENSOR_ORDER_DEFAULT);
            Value tensorValue = tensorVarOp.result();
            // Inspect the code-model to reach the MMA op and determine the ordering of matrices
            int indexOrdering = getTensorOrder(tensorValue);
            if (tensorOrderTable.containsKey(indexOrdering)) {
                matrixOrder = tensorOrderTable.get(indexOrdering);
            }

            var shapeValue = findShape(tensorVarOp.result(), tensorVarOp.result());
            var shape = obtainShapeTensor(shapeValue);
            String loadVariance = findLoadVariance(tensorValue, tensorVarOp);
            var type = tensorTypeTable.getOrDefault(loadVariance, null);
            var valueAccessLayout = findAccessLayout(tensorValue, tensorVarOp);

            if (shape.size() != 3) {
                throw new IllegalStateException("Tensor Shape must have 3 values" + type);
            }
            if (type == null) {
                throw new IllegalStateException("Load Type not supported:" + type);
            }
            return generateCreateTensor(shape, matrixOrder, type, valueAccessLayout, tensorVarOp.varName());
        } else {
            throw new IllegalStateException("Value not supported");
        }
    }

    @Override
    public CudaHATKernelBuilder hatTensorCreateOperation(Invoke tensorCreate) {
        if (tensorCreate.op().operands().isEmpty()) {
            // this corresponds to a tensor declaration for the input data
            return generateTensorCreate(tensorCreate);
        } else {
            // generate accumulate for the tensors
            return generateTensorAccumulateCreate(tensorCreate);
        }
    }

    /**
     * Example of code being generated:
     *
     * <p>
     * <code>
     *     wmma::load_matrix_sync(a_frag, matrix->array + headSize + aRow + aCol * lda, lda);
     * </code>
     * </p>
     *
     * @param tensorLoad
     *      Invoke node that represents the tensor load operation
     *
     * @return {@link CudaHATKernelBuilder}
     */
    @Override
    protected CudaHATKernelBuilder hatTensorLoad(OpHelper.Invoke tensorLoad) {
        List<Value> operands = tensorLoad.op().operands();
        var ptrValue = operands.getFirst();
        var iIndexValue = operands.get(1);
        var jIndexValue = operands.get(2);
        var leadingDimension = operands.get(3);
        CoreOp.VarOp tensorVarOp = findTensorVarOp(tensorLoad);
        List<Integer> shape;
        if (tensorVarOp != null) {
            shape = obtainShapeTensor(operands.get(4));
        } else {
            throw new IllegalStateException("[Error][CodeGen] Expected to see an instance of tensorVarOp but `null` found");
        }

        // Obtain if tensorA or tensorB is being loaded.
        // This is important to get the loop-bounds correct if matrices are not square
        int tensorOrder = getTensorOrder(tensorVarOp.result());

        boolean isColumnMajor;
        if (tensorLoad.op().operands().size() == 6) {
            isColumnMajor = isColumnMajor(operands.getLast());
        } else {
            isColumnMajor = false;
        }

        String varA = generateVariableName(INDEX_PREFIX);
        String varB = generateVariableName(INDEX_PREFIX);
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

        // Switch indexes when the memory access layout is not in column major
        if (!isColumnMajor) {
            Value tmp = iIndexValue;
            iIndexValue = jIndexValue;
            jIndexValue = tmp;
        }

        // params:         List<String> params = List.of("M", "N", "varA", "varB", "iIndexValue", "jIndexValue", "isColumnMajor", "leadingDimension", "reference", "tensorToLoad");
        Value finalIIndexValue = iIndexValue;
        Value finalJIndexValue = jIndexValue;
        return id(MACRO_FRAGMENT_LOAD_F16).paren(_ ->
                intValue(M).comma().sp()
                        .intValue(N).comma().sp()
                        .id(varA).comma().sp()
                        .id(varB).comma().sp()
                        .recurseResultOrThrow(finalIIndexValue).comma().sp()
                        .recurseResultOrThrow(finalJIndexValue).comma().sp()
                        .id(String.valueOf(isColumnMajor)).comma().sp()
                        .recurseResultOrThrow(leadingDimension).comma().sp()
                        .recurseResultOrThrow(ptrValue).comma().sp()
                        .id(tensorVarOp.varName()));

    }

    @Override
    protected CudaHATKernelBuilder hatTileAlignOperation(Invoke invoke) {
        return tileContext().id("assume_aligned").paren( _ ->
                recurseResultOrThrow(invoke.op().operands().getFirst()).rarrow().id(ARRAY)
                .comma().sp()
                .recurseResultOrThrow(invoke.op().operands().get(1)).id("_ic"));
    }

    private int obtainShapeDimensions(Value value) {
        int shapeValues;
        if (value.asResult().op().resultType().equals(JavaType.INT)) {
            shapeValues = 1;
        } else {
            // we expect an invoke that describes the shape.
            while (!(value.declaringElement() instanceof JavaOp.InvokeOp invokeOp)) {
                if (Objects.requireNonNull(value.asResult().op()) instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    value = varLoadOp.varOperand();
                } else {
                    throw new IllegalStateException("Unexpected value: " + value.asResult().op());
                }
            }
            shapeValues = invokeOp.operands().size();
        }
        return shapeValues;
    }

    private void genExtentSize(Value value) {
        switch (value.declaringElement()) {
            case VarOp varOp -> genExtentSize(varOp.operands().getFirst());
            case JavaOp.InvokeOp invokeOp -> recurseResultOrThrow(invokeOp.operands().getFirst()).rarrow().id(LENGTH);
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> genExtentSize(varLoadOp.operands().getFirst());
            case null, default -> throw new IllegalStateException("Expected a VarOp");
        }
    }

    private CudaHATKernelBuilder genTileConstantShape(Value value, int argIndex) {
        if (value.declaringElement() instanceof JavaOp.InvokeOp invokeOp && invokeOp.invokeReference().name().equals("shape")) {
            return genTileConstantShape(invokeOp.operands().get(argIndex), argIndex);
        } else if (value.declaringElement() instanceof CoreOp.ConstantOp constant) {
            Object value1 = constant.value();
            if (value1 instanceof Integer i) {
                id(i + "_ic");
            } else {
                throw new IllegalStateException("Expected a integer value to specify a tile shape");
            }
        } else {
            throw new IllegalStateException("Expected a ConstantOp for obtaining the Tile Shape, but found: " + value.declaringElement().getClass());
        }
        return self();
    }

    private CudaHATKernelBuilder tileBlockId() {
        return tileContext().id("bid").ocparen();
    }

    /**
     * Example of code being generated:
     *
     * <p>
     * <code>
     *     store_matrix_sync(matrix->array + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
     * </code>
     * </p>
     *
     * @param tensorStore
     *      Invoke node that represents the tensor store operation
     *
     * @return {@link CudaHATKernelBuilder}
     */
    @Override
    protected CudaHATKernelBuilder hatTensorStore(OpHelper.Invoke tensorStore) {
        List<Value> operands = tensorStore.op().operands();
        // Access layout is the last operand
        boolean isColumnMajor;
        // Since the Access Layout is an optional parameter, we check
        if (tensorStore.op().operands().size() == 6) {
            isColumnMajor = isColumnMajor(operands.getLast());
        } else {
            isColumnMajor = false;
        }

        Value reference = operands.getFirst();
        Value iIndex = operands.get(1);
        Value jIndex = operands.get(2);
        Value tensorToStore = operands.get(3);
        Value ldSize = operands.get(4);
        CoreOp.VarOp tensorVarOp = findVarOpOrThrow(tensorToStore);

        var shape = getShapeFromTensorVarOp(tensorVarOp);
        // Output is MxN, given the shape in an (M,N,K) triplet
        final int M = shape.get(0);
        final int N = shape.get(1);
        String varA = generateVariableName(INDEX_PREFIX);
        String varB = generateVariableName(INDEX_PREFIX);

        // Switch indexes when the memory access layout is not in column major
        if (!isColumnMajor) {
            Value tmp = iIndex;
            iIndex = jIndex;
            jIndex = tmp;
        }

        Value finalIIndex = iIndex;
        Value finalJIndex = jIndex;
        return id(MACRO_FRAGMENT_STORE).paren(_ ->
                intValue(M).comma().sp()
                .intValue(N).comma().sp()
                .id(varA).comma().sp()
                .id(varB).comma().sp()
                .recurseResultOrThrow(finalIIndex).comma().sp()
                .recurseResultOrThrow(finalJIndex).comma().sp()
                .id(String.valueOf(isColumnMajor)).comma().sp()
                .recurseResultOrThrow(ldSize).comma().sp()
                .recurseResultOrThrow(reference).comma().sp()
                .id(tensorVarOp.varName()).comma().sp()
                .either(isColumnMajor,
                _ -> id(WMMA_MEM_COL_MAJOR),
                _ -> id(WMMA_MEM_ROW_MAJOR)));
    }

    protected static final String ARRAY = "array";
    protected static final String LENGTH = "length";

    public CudaHATKernelBuilder restrict() {
        return typeModifier("__restrict__");
    }

    public CudaHATKernelBuilder declareParam(FuncOpParams.Info param) {
        if (this.isTile) {
            // inspect type of parameter
            CodeType type = param.parameter.type();
            type(type).sp();
            if (!(type instanceof PrimitiveType)) {
                restrict().sp();
            }
            return varName(param.varOp);
        }
        return type((JavaType) param.parameter.type()).sp().varName(param.varOp);
    }

    @Override
    public CudaHATKernelBuilder type(CodeType codeType) {
        if (codeType instanceof PtrType ptrType && ptrType.rType() instanceof JavaType javaType) {
            return type(javaType);
        } else {
            return super.type(codeType);
        }
    }

    @Override
    public CudaHATKernelBuilder tileConstantOp(ArithMathOps.ConstantOp constantOp) {
        if (constantOp.value() instanceof Integer val) {
            return intConst(val);
        }
        throw new UnsupportedOperationException("Constant type not supported: " + constantOp.value());
    }

    @Override
    public CudaHATKernelBuilder tileIdOp(TileOps.TileIDOp tileIdOp) {
        switch (tileIdOp.dimension()) {
            case 0 -> tileBlockId().dot().id("x");
            case 1 -> tileBlockId().dot().id("y");
            case 2 -> tileBlockId().dot().id("z");
            default -> throw new UnsupportedOperationException("Tile ID Operation is not supported yet.");
        }
        return self();
    }

    private CudaHATKernelBuilder generateAlignedReference(Value ref) {
        if (ref.declaringElement() instanceof JavaOp.InvokeOp invokeOp && invokeOp.invokeReference().name().equals(ALIGN)) {
            return recurseResultOrThrow(invokeOp.operands().getFirst());
        } else {
            if (ref instanceof Op.Result r) {
                return generateAlignedReference(r.op().operands().getFirst());
            }
        }
        throw new IllegalStateException("Reference not supported: " + ref);
    }

    private CudaHATKernelBuilder genTileSize(CodeType resultType, Value ptr) {
        if (resultType instanceof ConstantType constantType && constantType.value() instanceof TensorType tensorType) {
            CodeType tt = tensorType.elementType();
            if (tt.equals(DType.TENSOR_F32_TYPE) || tt.equals(DType.Float)) { // this check is due to type equivalence
                generateAlignedReference(ptr).rarrow().id("m");
            } else if (tt.equals(DType.TENSOR_2D_F32_TYPE) || tt.equals(DType.TENSOR_2D_F16_TYPE)) {
                generateAlignedReference(ptr).rarrow().id("m").comma().sp().generateAlignedReference(ptr).rarrow().id("n");
            } else {
                throw new UnsupportedOperationException("Tensor Type not supported yet: " + tt);
            }
        }
        return self();
    }

    private static final String ALIGN = "align";

    private CudaHATKernelBuilder genTileSize(Value ptr, int dim) {
        if (ptr.declaringElement() instanceof JavaOp.InvokeOp invokeOp && invokeOp.invokeReference().name().equals(ALIGN)) {
            generateAlignedReference(ptr).rarrow();
            switch (dim) {
                case 0 -> { return id("m"); }
                case 1 -> { return id("n"); }
                default -> throw new IllegalStateException("Unexpected value: " + dim);
            }
        } else {
            if (ptr instanceof Op.Result r) {
                return genTileSize(r.op().operands().getFirst(), dim);
            }
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder tileLoadOp(TileOps.LoadOp tileLoadOp) {
        List<Value> operands = tileLoadOp.operands();
        Value ptr = operands.get(0);
        Value dimension = operands.get(1);
        List<Object> dims = tileLoadOp.dims();
        CodeType resultType = tileLoadOp.resultType();

        boolean isDivisible16 = dims.stream().filter(dim -> dim instanceof Integer).map(dim -> (Integer) dim).noneMatch(i -> i % 16 != 0);

        partitionView().brace(_ -> {
                tensorSpan().brace(_ -> {
                    recurseResultOrThrow(ptr);
                    comma().sp().tensorExtent().paren(_ -> {
                        if (isDivisible16) {
                            // if input is divisible by 16, then we can emit the following optimization
                            commaSpaceSeparated(dims, x -> { tensorAssumeDivisible(16).paren(_ -> id(x.toString())); });
                        } else {
                            genTileSize(resultType, ptr);
                        }
                    });
                }).comma();

            // Process shapes: We assume shapes are constants.
            if (resultType instanceof ConstantType constantType && constantType.value() instanceof TensorType tt) {
                tileShape().brace(_ -> {
                    List<Integer> shapeList = tt.shape();
                    literalIC(shapeList.getFirst());
                    for (int i = 1; i < shapeList.size(); i++) {
                        comma().literalIC(shapeList.get(i));
                    }
                });
            }
        });
        return dot().tileLoad().paren( _ -> recurseResultOrThrow(dimension));
    }

    @Override
    public CudaHATKernelBuilder tileAddOp(ArithMathOps.AddOp tileAddOp) {
        return recurseResultOrThrow(tileAddOp.operands().getFirst())
                .plus()
                .recurseResultOrThrow(tileAddOp.operands().get(1));
    }

    @Override
    public CudaHATKernelBuilder tileStoreOp(TileOps.StoreOp tileStoreOp) {
        List<Value> operands = tileStoreOp.operands();
        Value inputReference = operands.get(0);
        Value blockId = operands.get(1);
        Value tensor = operands.get(2);
        List<Object> dims = tileStoreOp.dims();

        boolean isDivisible16 = dims.stream().filter(dim -> dim instanceof Integer).map(dim -> (Integer) dim).noneMatch(i -> i % 16 != 0);

        return partitionView().brace( _ -> {
                tensorSpan().brace( _ -> {
                recurseResultOrThrow(inputReference);
                comma().tensorExtent().brace(_ -> {
                    if (isDivisible16) {
                        // if input is divisible by 16, then we can emit the following optimization
                        commaSpaceSeparated(dims, x -> tensorAssumeDivisible(16).paren(_ -> id(x.toString())));
                    } else {
                        genTileSize(tensor.type(), inputReference);
                    }
                });
            }).comma();
            CodeType tensorType = tensor.type();
            if (tensorType instanceof ConstantType constantType && constantType.value() instanceof TensorType tt) {
                tileShape().brace(_ -> {
                    List<Integer> shape = tt.shape();
                    literalIC(shape.getFirst());
                    for (int i = 1; i < shape.size(); i++) {
                        comma().literalIC(shape.get(i));
                    }
                });
            } else {
                // At this point, since the Tile Dialect was built after the type check and shape propagation,
                // we know this error can't occur. If something unexpected happens, we throw an error.
                throw new UnsupportedOperationException("[codegen] tensor store shape not supported yet.");
            }
        }).dot().tileStore().paren( _ -> recurseResultOrThrow(tensor).comma().sp().recurseResultOrThrow(blockId));
    }

    @Override
    public CudaHATKernelBuilder tileNumOp(TileOps.TileNumOp tileNumOp) {
        List<Value> operands = tileNumOp.operands();
        Value ptr = operands.getFirst();
        Value dimension = operands.get(1);
        Value shape = operands.get(2);

        int dimValue;
        if (dimension.declaringElement() instanceof CoreOp.ConstantOp constantOp && constantOp.value() instanceof Integer dim) {
            dimValue = dim;
        } else {
            throw new UnsupportedOperationException("[codegen] dimension number not supported yet.");
        }

        final Value tileExtent;
        if (dimValue > 0 && shape.declaringElement() instanceof TileOps.TileShapeOp shapeOp) {
            if (dimValue >= shapeOp.operands().size()) {
                throw new UnsupportedOperationException("[codegen] dimension number not supported yet.");
            }
            tileExtent = shapeOp.operands().get(dimValue);
        } else {
            tileExtent = shape;
        }

        return paren(_ ->
                genTileSize(ptr, dim)
                        .sp()
                        .plus()
                        .recurseResultOrThrow(tileExtent)
                        .sp()
                        .minus()
                        .intConst(1)
        ).div().recurseResultOrThrow(tileExtent);
    }

    @Override
    public CudaHATKernelBuilder tileFullOp(TileOps.TileFullOp tileFullOp) {
        List<Value> operands = tileFullOp.operands();
        Value shape = operands.getFirst();
        Value initValue = operands.get(1);
        return tileFull()
                .ltgt(_ -> {
                    cudaTile().ltgt(_ -> {
                        if (tileFullOp.resultType() instanceof ConstantType constantType && constantType.value() instanceof TensorType tt) {
                            type(tt.elementType().toString());
                        } else {
                            throw new UnsupportedOperationException("[codegen] tile full operation not supported yet: " + tileFullOp.resultType());
                        }
                        comma().sp().tileShape().ltgt(_ -> recurseResultOrThrow(shape));
                    });
                }).paren(_ -> {
                    recurseResultOrThrow(initValue);

                    // Add "f" if the init value is float to avoid type conversion in the low SASS code
                    if (initValue.type() instanceof PrimitiveType primitiveType && primitiveType.equals(JavaType.FLOAT)) {
                        id("f");
                    }
                });
    }

    @Override
    public CudaHATKernelBuilder tileShapeOp(TileOps.TileShapeOp tileShapeOp) {
        return commaSpaceSeparated(tileShapeOp.operands(), v -> {
            recurseResultOrThrow(v).id("_ic");
        });
    }

    @Override
    public CudaHATKernelBuilder tileSumOp(TileOps.TileSumOp tileSumOp) {
        List<Value> operands = tileSumOp.operands();
        Value tensor = operands.getFirst();
        Value dimension = operands.get(1);
        return tileSum().paren(_ ->
                recurseResultOrThrow(tensor).comma().sp().recurseResultOrThrow(dimension).id("_ic"));
    }

    @Override
    public CudaHATKernelBuilder tileIndexOp(TileOps.TileIndexOp tileIndexOp) {
        return commaSpaceSeparated(tileIndexOp.operands(), this::recurseResultOrThrow);
    }

    @Override
    public CudaHATKernelBuilder tileTransposeOp(ArithMathOps.TransposeOp tileTransposeOp) {
        return tileTranspose().paren(_-> recurseResultOrThrow(tileTransposeOp.operands().getFirst()));
    }

    @Override
    public CudaHATKernelBuilder tileZerosOp(TileOps.TileZerosOp tileZerosOp) {
        tileZeros().ltgt(_ -> {
            cudaTile().ltgt(_ -> {
            if (tileZerosOp.resultType() instanceof ConstantType constantType && constantType.value() instanceof TensorType tt) {
                if (tt.elementType().equals(DType.TENSOR_2D_F32_TYPE) || tt.elementType().equals(DType.TENSOR_F32_TYPE)) {
                    f32Type();
                } else {
                    type(tt.elementType().toString());
                }
            } else {
                throw new UnsupportedOperationException("[codegen] tile full operation not supported yet: " + tileZerosOp.resultType());
            }
            comma().sp().tileShape().ltgt(_ -> {
                CodeType codeType = tileZerosOp.resultType();
                if (codeType instanceof ConstantType constantType1 && constantType1.value() instanceof TensorType tensorType) {
                    List<Integer> shape = tensorType.shape();
                    commaSpaceSeparated(shape, this::intValue);
                } else {
                    throw new IllegalStateException("[codegen] ct::zero shape not recognized");
                }
                });
            });
        }).paren( _ ->{});

        return self();
    }

    // CUDA Tile Constructs
    private CudaHATKernelBuilder tileContext() {
        return id("ct").colon().colon();
    }

    private CudaHATKernelBuilder partitionView() {
        return tileContext().id("partition_view");
    }

    private CudaHATKernelBuilder tensorSpan() {
        return tileContext().id("tensor_span");
    }

    private CudaHATKernelBuilder tensorExtent() {
        return tileContext().id("extents");
    }

    private CudaHATKernelBuilder tensorAssumeDivisible(int div) {
        return tileContext().id("assume_divisible").ltgt(_ -> {
            intValue(div);
        });
    }

    private CudaHATKernelBuilder cudaTile() {
        return tileContext().id("tile");
    }

    private CudaHATKernelBuilder tileFull() {
        return tileContext().id("full");
    }

    private CudaHATKernelBuilder tileZeros() {
        return tileContext().id("zeros");
    }

    private CudaHATKernelBuilder tileSum() {
        return tileContext().id("sum");
    }

    private CudaHATKernelBuilder tileTranspose() {
        return tileContext().id("transpose");
    }

    private CudaHATKernelBuilder tileShape() {
        return tileContext().id("shape");
    }

    private CudaHATKernelBuilder tileMMA() {
        return tileContext().id("mma");
    }

    private CudaHATKernelBuilder tileCeilDiv() {
        return tileContext().id("ceildiv");
    }

    private CudaHATKernelBuilder tileMin() {
        return tileContext().id("min");
    }

    private CudaHATKernelBuilder tileMax() {
        return tileContext().id("max");
    }

    private CudaHATKernelBuilder tileIRange() {
        return tileContext().id("irange");
    }

    private CudaHATKernelBuilder tileLoad() {
        return id("load");
    }

    private CudaHATKernelBuilder tileStore() {
        return id("store");
    }

    @Override
    public CudaHATKernelBuilder tileMMAOp(ArithMathOps.MMAOp tileMMAOp) {
        return tileMMA().paren(_ -> commaSpaceSeparated(tileMMAOp.operands(), this::recurseResultOrThrow));
    }

    @Override
    public CudaHATKernelBuilder cDivOp(ArithMathOps.CDivOp cDivOp) {
        return tileCeilDiv().paren(_ -> commaSpaceSeparated(cDivOp.operands(), this::recurseResultOrThrow));
    }

    @Override
    public CudaHATKernelBuilder minOp(ArithMathOps.MinOp minOp) {
        return tileMin().paren(_ -> commaSpaceSeparated(minOp.operands(), this::recurseResultOrThrow));
    }

    @Override
    public CudaHATKernelBuilder tileIrangeOp(TileOps.TileIrangeOp tileIrangeOp) {
        Value startIndex = tileIrangeOp.operands().getFirst();
        Value endIndex = tileIrangeOp.operands().getLast();
        return tileIRange().paren(_ -> {
           recurseResultOrThrow(startIndex).comma().sp().recurseResultOrThrow(endIndex);
        });
    }
}
