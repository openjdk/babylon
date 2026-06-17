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

import hat.callgraph.KernelCallGraph;
import hat.codebuilders.C99HATKernelBuilder;
import hat.dialect.BinaryOpEnum;
import hat.phases.HATFP16Phase;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.OpHelper.Invoke;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.types.BF16;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.util.*;
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
    private static final Map<String, String> MATH_FUNCTIONS = new HashMap<>();

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

    protected CudaHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraph, scopedCodeBuilderContext);
        stack = new ConcurrentLinkedDeque<>();
        mapVectorName = new ConcurrentHashMap<>();
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
        return constant("32");
    }

    @Override
    public CudaHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_CUDA")
                .hashDefine("HAT_GLOBAL_MEM", _ -> {})
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__shared__"))
                .hashDefine("HAT_FUNC", _ -> externC().sp().keyword("__device__").sp())//.keyword("inline"))
                .hashDefine("HAT_KERNEL", _ -> externC().sp().keyword("__global__"))

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

                // Barrier
                .when(useBarrier(), _ -> hashDefine("HAT_BARRIER", _ -> keyword("__syncthreads").ocparen()))

                // Math
                .when(useS16Types(), _ -> maxMacro("MAX_HAT"))
                .when(useS16Types(), _ ->minMacro("MIN_HAT"))

                // General Macros
                .when(useVectors() || useVectors(), _ -> concatMacro())
                .when(useVectors() || useVectors(), _ -> prefixMacro())

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
                .when(useS16Types(), _ -> defineMacroF16Of(F16_OF))
                .when(useS16Types(), _ -> defineMacroBF16Of(BF16_OF))
                .when(useS16Types(), _ -> defineMacroF162Float(F16_TO_FLOAT_0, false))
                .when(useS16Types(), _ -> defineMacroF162Float(F16_TO_FLOAT_1, true))
                .when(useS16Types(), _ -> defineMacroBF162Float(BF16_TO_FLOAT_0, false))
                .when(useS16Types(), _ -> defineMacroBF162Float(BF16_TO_FLOAT_1, true))
                .when(useS16Types(), _ -> includeSys("cuda_fp16.h", "cuda_bf16.h"))
                .when(useS16Types(), _ -> hashDefine("BFLOAT16", _ -> keyword("__nv_bfloat16")))
                .when(useS16Types(), _ -> typedefSingleValueStruct("F16", "half"))
                .when(useS16Types(), _ -> typedefSingleValueStruct("BF16", "BFLOAT16"))
                .when(useTensors(), _ -> includeSys("mma.h")); // only enable if tensor views are used
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
    protected CudaHATKernelBuilder varOpTensor(CoreOp.VarOp varOp) {
        recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op());
        sp().id(varOp.varName());
        return self();
    }

    public static final String WMMA_MEM_COL_MAJOR = "nvcuda::wmma::mem_col_major";
    public static final String WMMA_MEM_ROW_MAJOR = "nvcuda::wmma::mem_row_major";
    public static final String WMMA_STORE_TENSOR = "nvcuda::wmma::store_matrix_sync";
    public static final String WMMA_LOAD_TENSOR = "nvcuda::wmma::load_matrix_sync";
    public static final String WMMA_MMA_TENSOR = "nvcuda::wmma::mma_sync";
    public static final String WMMA_FILL_TENSOR = "nvcuda::wmma::fill_fragment";
    public static final String WMMA_COL_MAJOR = "nvcuda::wmma::col_major";
    public static final String WMMA_ROW_MAJOR = "nvcuda::wmma::row_major";
    public static final String WMMA_FRAGMENT_BASE = "nvcuda::wmma::fragment<nvcuda::wmma::";

    private CudaHATKernelBuilder generateCreateTensor(List<Integer> shape, String matrixOrder, String type, Value access) {
        id(WMMA_FRAGMENT_BASE)
                .id(matrixOrder)
                .comma().sp()
                .intValue(shape.getFirst())
                .comma().sp()
                .intValue(shape.get(1))
                .comma().sp()
                .intValue(shape.get(2))
                .comma().sp()
                .type(type);

        if (matrixOrder.equals(TENSOR_ACC)) {
            gt();
        } else {// infer from the last parameter
            comma();
            if (access == null) {
                id(WMMA_ROW_MAJOR);
            } else if (access.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
                // Expecting an invokeOp
                var invoke = invoke(scopedCodeBuilderContext().lookup(), invokeOp);
                if (invoke.resultTypeIs(Tensor.ColumMajor.class)) {
                    id(WMMA_COL_MAJOR);
                } else if (invoke.resultTypeIs(Tensor.RowMajor.class)) {
                    id(WMMA_ROW_MAJOR);
                } else {
                    throw new IllegalStateException("[Error]");
                }
            }
            gt();
        }
        return self();
    }

    private static final String TENSOR_MATRIX_A = "matrix_a";
    private static final String TENSOR_MATRIX_B = "matrix_b";
    private static final String TENSOR_ACC = "accumulator";

    private int getTensorOrder(Value tensorValue) {
        return getTensorOrder(tensorValue, tensorValue);
    }

    private int getTensorOrder(Value tensorValue, Value v) {
        return v instanceof Op.Result r ? getTensorOrder(tensorValue, r.op()) : -1;
    }

    // We traverse the usages of the op until we find the MMA operation.
    // Once the MMA is found, then we compare if the arguments (VarLoadOp) contains the
    // reference to the var declartion being analyzed. In that case, we return its index.
    private int getTensorOrder(Value tensorValue, Op op) {
        int operandIndex = -1;
        switch (op) {
            case JavaOp.InvokeOp tensorMMAOp when tensorMMAOp.invokeReference().name().equals("mma") -> {
                List<Value> operands = tensorMMAOp.operands();
                for (Value argument : operands) {
                    operandIndex++;
                    if (argument.declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                            && varLoadOp.operands().getFirst().equals(tensorValue)) {
                        return operandIndex;
                    }
                }
            }
            default -> {
                for (Op.Result use : op.result().uses()) {
                    if ((operandIndex = getTensorOrder(tensorValue, use)) != -1) {
                        return operandIndex;
                    }
                }
            }
        }
        return operandIndex;
    }

    private String findLoadVariance(Value tensorVar, Value v) {
        return v instanceof Op.Result r ? findLoadVariance(tensorVar, r.op()) : null;
    }

    private String findLoadVariance(Value tensorVar, Op op) {
        String varianceName = null;
        switch (op) {
            case CoreOp.VarAccessOp.VarStoreOp storeLoadOp -> {
                Value tensorToStore = storeLoadOp.operands().getFirst();
                if (tensorToStore.equals(tensorVar)) {
                    Value value = storeLoadOp.operands().get(1);
                    if (value.declaringElement() instanceof JavaOp.InvokeOp tensorLoadOp) {
                        return tensorLoadOp.invokeReference().name();
                    }
                }
            }
            default -> {
                for (Op.Result use : op.result().uses()) {
                    if ((varianceName = findLoadVariance(tensorVar, use)) != null) {
                        return varianceName;
                    }
                }
            }
        }
        return varianceName;
    }

    private Value findAccessLayout(Value tensorVar, Value v) {
        return v instanceof Op.Result r ? findAccessLayout(tensorVar, r.op()) : null;
    }

    private Value findAccessLayout(Value tensorVar, Op op) {
        Value valueLayout = null;
        switch (op) {
            case CoreOp.VarAccessOp.VarStoreOp storeLoadOp -> {
                Value tensorToStore = storeLoadOp.operands().getFirst();
                if (tensorToStore.equals(tensorVar)) {
                    Value value = storeLoadOp.operands().get(1);
                    if (value.declaringElement() instanceof JavaOp.InvokeOp tensorLoadOp) {
                        if (tensorLoadOp.operands().size() == INDEX_ACCESS + 1) {
                            return tensorLoadOp.operands().getLast();
                        } else {
                            return null;
                        }
                    }
                }
            }
            default -> {
                for (Op.Result use : op.result().uses()) {
                    if ((valueLayout = findAccessLayout(tensorVar, use)) != null) {
                        return valueLayout;
                    }
                }
            }
        }
        return valueLayout;
    }

    public Value findShape(Value tensorVar, Value v) {
        return v instanceof Op.Result r ? findShape(tensorVar, r.op()) : null;
    }

    // ABI
    private static final int INDEX_LOAD = 0;
    private static final int INDEX_ROW = 1;
    private static final int INDEX_COL = 2;
    private static final int INDEX_LDD = 3;
    private static final int INDEX_SHAPE = 4;
    private static final int INDEX_ACCESS = 5;

    private Value findShape(Value tensorVar, Op op) {
        Value shape = null;
        switch (op) {
            case CoreOp.VarAccessOp.VarStoreOp storeLoadOp -> {
                Value tensorToStore = storeLoadOp.operands().getFirst();
                if (tensorToStore.equals(tensorVar)) {
                    Value value = storeLoadOp.operands().get(1);
                    if (value.declaringElement() instanceof JavaOp.InvokeOp tensorLoadOp) {
                        return tensorLoadOp.operands().get(INDEX_SHAPE);
                    }
                }
            }
            default -> {
                for (Op.Result use : op.result().uses()) {
                    if ((shape = findShape(tensorVar, use)) != null) {
                        return shape;
                    }
                }
            }
        }
        return shape;
    }

    private static final Map<Integer, String> tensorOrderTable = new HashMap<>();
    private static final int DEFAULT_TENSOR_ORDERING = -1;
    static {
        tensorOrderTable.put(1, TENSOR_MATRIX_A);
        tensorOrderTable.put(2, TENSOR_MATRIX_B);
        tensorOrderTable.put(-1, TENSOR_MATRIX_A); // We set one by default
    }

    private static final Map<String, String> tensorTypeTable = new HashMap<>();
    static {
        tensorTypeTable.put("loadF16", "half");
        tensorTypeTable.put("load",    "float");
        tensorTypeTable.put("loadF32", "float");
    }

    private CudaHATKernelBuilder generateTensorAccumulateCreate(Invoke tensorCreateOp) {
        // tensor declaration for the accumulator
        Value shapeValue = tensorCreateOp.op().operands().getFirst();
        List<Integer> shape = obtainShapeTensor(shapeValue);
        Value classOperand = tensorCreateOp.op().operands().get(1);
        Object klass = null;
        if (classOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
            klass = constantOp.value();
        }
        String tensorType = null;
        if (klass != null) {
            switch (klass) {
                case ClassType classType when classType.toClassName().equals(F16.class.getCanonicalName()) ->
                        tensorType = "half";
                case PrimitiveType primitiveType when primitiveType.equals(PrimitiveType.FLOAT) -> tensorType = "float";
                default -> throw new IllegalStateException("Type class not supported for Tensors: " + klass);
            }
        }
        Value valueAccessLayout = tensorCreateOp.op().operands().getLast();
        return generateCreateTensor(shape, TENSOR_ACC, tensorType, valueAccessLayout);
    }

    private CudaHATKernelBuilder generateTensorCreate(Invoke tensorCreateOp) {
        Value v = tensorCreateOp.op().result().uses().getFirst();

        // Find the declaration value of the tensor
        String matrixOrder = tensorOrderTable.get(DEFAULT_TENSOR_ORDERING);
        Value shapeValue;
        String type;
        Value valueAccessLayout;
        List<Integer> shape;
        // otherwise, we have to inspect the shape from the TensorLoadOp
        if (v.declaringElement() instanceof VarOp tensorVarOp) {
            Value tensorValue = tensorVarOp.result();
            // Inspect the code-model to reach the MMA op and determine the ordering of matrices
            int indexOrdering = getTensorOrder(tensorValue);
            if (tensorOrderTable.containsKey(indexOrdering)) {
                matrixOrder = tensorOrderTable.get(indexOrdering);
            }

            shapeValue = findShape(tensorVarOp.result(), tensorVarOp.result());
            shape = obtainShapeTensor(shapeValue);
            String loadVariance = findLoadVariance(tensorValue, tensorVarOp);
            type = tensorTypeTable.getOrDefault(loadVariance, null);
            valueAccessLayout = findAccessLayout(tensorValue, tensorVarOp);

            if (shape.size() != 3) {
                throw new IllegalStateException("Tensor Shape must have 3 values" + type);
            }
            if (type == null) {
                throw new IllegalStateException("Load Type not supported:" + type);
            }
        } else {
            throw new IllegalStateException("Value not supported");
        }

        return generateCreateTensor(shape, matrixOrder, type, valueAccessLayout);
    }

    @Override
    public CudaHATKernelBuilder hatTensorCreateOperation(Invoke tensorCreateOp) {
        if (tensorCreateOp.op().operands().isEmpty()) {
            // this corresponds to a tensor declaration for the input data
            return generateTensorCreate(tensorCreateOp);
        } else {
            // generate accumulate for the tensors
            return generateTensorAccumulateCreate(tensorCreateOp);
        }
    }

    @Override
    public CudaHATKernelBuilder hatTensorFill(OpHelper.Invoke tensorFillOp) {
        id(WMMA_FILL_TENSOR).paren( _-> {
            List<Value> operands = tensorFillOp.op().operands();
            recurseResultOrThrow(operands.getFirst())
                    .comma()
                    .recurseResultOrThrow(operands.get(1));
        });
        return self();
    }

    private static CoreOp.VarOp findTensorVarOp(Value varLoadOp) {
        return switch (varLoadOp.declaringElement()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp2 -> findTensorVarOp(varLoadOp2.operands().getFirst());
            case CoreOp.VarOp varOp -> varOp;
            case null, default -> null;
        };
    }

    @Override
    public CudaHATKernelBuilder hatTensorMMA(Invoke tensorMMAOp) {
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
        List<VarOp> operands = List.of(tensorResult, tensorA, tensorB, tensorC);
        return id(WMMA_MMA_TENSOR).paren( _-> commaSeparated(operands, va -> id(va.varName())));
    }

    private CudaHATKernelBuilder generateLoadTensor(OpHelper.Invoke tensorLoadOp, boolean isColumnMajor, String tensorName) {
        // First operand is the reference to global memory
        List<Value> operands = tensorLoadOp.op().operands();
        Value reference = operands.getFirst();
        id(WMMA_LOAD_TENSOR)
                .paren(_ -> {
                    id(tensorName).comma();
                    paren(_ -> type("half").asterisk());
                    recurseResultOrThrow(reference);
                    rarrow().id(ARRAY)
                            .sp().plus().sp()
                            .indexForTensor(isColumnMajor, operands.get(1), operands.get(2), operands.get(3))
                            .comma();
                    recurseResultOrThrow(operands.get(3));
                });

        return self();
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
     * @param tensorLoadOp
     *
     * @return {@link CudaHATKernelBuilder}
     */
    @Override
    protected CudaHATKernelBuilder hatTensorLoad(OpHelper.Invoke tensorLoadOp) {
        // Find name tensor of the first argument
        String tensorName = "";
        SequencedSet<Op.Result> uses = tensorLoadOp.op().result().uses();
        VarOp tensorVarOp = null;
        for (Op.Result result : uses) {
            if (result.declaringElement() instanceof CoreOp.VarAccessOp.VarStoreOp storeLoadOp) {
                // obtain first arg from tensorStoreOp
                Value first = storeLoadOp.operands().getFirst();
                if (first.declaringElement() instanceof VarOp varOp) {
                    tensorVarOp = varOp;
                    tensorName = tensorVarOp.varName();
                }
            }
        }

        boolean isColumnMajor = false;
        if (tensorVarOp != null && tensorLoadOp.op().operands().size() > 5) {
            Value value = tensorLoadOp.op().operands().getLast();
            isColumnMajor = isColumnMajor(value);
        }
        return generateLoadTensor(tensorLoadOp, isColumnMajor, tensorName);
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
     * @param operands
     * @param isColumnMajor
     *
     * @return {@link CudaHATKernelBuilder}
     */
    private CudaHATKernelBuilder generateStoreTensor(List<Value> operands, boolean isColumnMajor) {
        Value reference = operands.getFirst();
        id(WMMA_STORE_TENSOR).paren(_ -> {
            Value iIndex = operands.get(1);
            Value jIndex = operands.get(2);
            Value tensorToStore = operands.get(3);
            Value ldSize = operands.get(4);

            CoreOp.VarOp tensorVarOp = findTensorVarOp(tensorToStore);
            assert tensorVarOp != null;

            recurseResultOrThrow(reference)
                    .rarrow().id(ARRAY)
                    .sp().plus().sp()
                    .indexForTensor(isColumnMajor, iIndex, jIndex, ldSize)
                    .comma()
                    .id(tensorVarOp.varName())
                    .comma()
                    .recurseResultOrThrow(ldSize)
                    .comma();

            if (isColumnMajor) {
                id(WMMA_MEM_COL_MAJOR);
            } else {
                id(WMMA_MEM_ROW_MAJOR);
            }
        });
        return self();
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
     * @param tensorStoreOp
     *
     * @return {@link CudaHATKernelBuilder}
     */
    @Override
    protected CudaHATKernelBuilder hatTensorStore(OpHelper.Invoke tensorStoreOp) {
        List<Value> operands = tensorStoreOp.op().operands();
        // Access layout is the last operand
        final boolean isColumnMajor;
        // Since the Access Layout is an optional parameter, we check
        if (tensorStoreOp.op().operands().size() == 6) {
            isColumnMajor = isColumnMajor(operands.getLast());
        } else {
            // use row major by default
            isColumnMajor = false;
        }
        return generateStoreTensor(operands, isColumnMajor);
    }

    private static final String ARRAY = "array";
}
