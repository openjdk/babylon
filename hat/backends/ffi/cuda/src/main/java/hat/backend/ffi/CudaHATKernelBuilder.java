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
import hat.dialect.HATF16Op;
import hat.dialect.HATTensorOp;
import hat.dialect.HATVectorOp;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.types.BF16;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedSet;

import static optkl.OpHelper.Invoke.invoke;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {

    protected CudaHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraph, scopedCodeBuilderContext);
    }

    @Override
    protected CudaHATKernelBuilder hatWarpSize() {
        return constant("32");
    }

    private CudaHATKernelBuilder half2float() {
        return id("__half2float");
    }

    private CudaHATKernelBuilder __nv_bfloat16() {
        return id("__nv_bfloat16");
    }

    private CudaHATKernelBuilder __bfloat162float() {
        return id("__bfloat162float");
    }

    private CudaHATKernelBuilder reinterpret_cast() {
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
    public CudaHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_CUDA")
                .hashDefine("HAT_GLOBAL_MEM", _ -> {
                })
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__shared__"))
                .hashDefine("HAT_FUNC", _ -> externC().sp().keyword("__device__").sp())//.keyword("inline"))
                .hashDefine("HAT_KERNEL", _ -> externC().sp().keyword("__global__"))
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
                .hashDefine("HAT_BARRIER", _ -> keyword("__syncthreads").ocparen())
                .maxMacro("MAX_HAT")
                .minMacro("MIN_HAT")
                .includeSys("cuda_fp16.h", "cuda_bf16.h")
                .hashDefine("BFLOAT16", _ -> keyword("__nv_bfloat16"))
                .typedefSingleValueStruct("F16", "half")
                .typedefSingleValueStruct("BF16", "BFLOAT16")
                .includeSys("mma.h"); // only enable if tensor views are used
    }

    @Override
    public CudaHATKernelBuilder atomicInc(Op.Result instanceResult, String name) {
        return id("atomicAdd").paren(_ -> ampersand().recurseResultOrThrow(instanceResult).rarrow().id(name).comma().literal(1));
    }

    @Override
    public CudaHATKernelBuilder hatVectorStoreOp(HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        Value dest = hatVectorStoreView.operands().get(0);
        Value index = hatVectorStoreView.operands().get(2);
        keyword("reinterpret_cast").ltgt(_ -> type(hatVectorStoreView.buildType()).sp().asterisk());
        paren(_ -> {
            ampersand().recurseResultOrThrow(dest);
            either(hatVectorStoreView instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
            id(ARRAY).sbrace(_ -> recurseResultOrThrow(index));
        });
        sbrace(_ -> intConstZero());
        sp().equals().sp();
        // if the value to be stored is an operation, recurse on the operation
        if (hatVectorStoreView.operands().get(1) instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
            recurse(r.op());
        } else {
            varName(hatVectorStoreView);
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatBinaryVectorOp(HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {

        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        final String postFixOp1 = "_1";
        final String postFixOp2 = "_2";

        if (op1 instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp1) {
            type(hatVectorBinaryOp1.buildType()).sp()
                    .id(hatVectorBinaryOp.varName() + postFixOp1)
                    .semicolon().nl();
            hatVectorBinaryOp1.varName(hatVectorBinaryOp.varName() + postFixOp1);
            recurse(hatVectorBinaryOp1);
        }

        if (op2 instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp2) {
            type(hatVectorBinaryOp2.buildType()).sp()
                    .id(hatVectorBinaryOp.varName() + postFixOp2)
                    .semicolon().nl();
            hatVectorBinaryOp2.varName(hatVectorBinaryOp.varName() + postFixOp2);
            recurse(hatVectorBinaryOp2);
        }

        for (int i = 0; i < hatVectorBinaryOp.vectorShape().lanes(); i++) {
            // this is where varName is null
            id(hatVectorBinaryOp.varName()).dot().id(hatVectorBinaryOp.mapLane(i)).sp().equals().sp();

            if (op1 instanceof Op.Result r) {
                if (!(r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp1)) {
                    recurse(r.op());
                } else {
                    id(hatVectorBinaryOp1.varName());
                }
            }
            dot().id(hatVectorBinaryOp.mapLane(i)).sp();
            id(hatVectorBinaryOp.operationType().symbol()).sp();

            if (op2 instanceof Op.Result r) {
                if (!(r.op() instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp2)) {
                    recurse(r.op());
                } else {
                    id(hatVectorBinaryOp2.varName());
                }
            }
            dot().id(hatVectorBinaryOp.mapLane(i)).semicolon().nl();
        }

        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorLoadOp(HATVectorOp.HATVectorLoadOp hatVectorLoadOp) {
        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        reinterpret_cast().ltgt(_ -> type(hatVectorLoadOp.buildType()).sp().asterisk());
        paren(_ -> {
            ampersand();recurseResultOrThrow(source);
            either(hatVectorLoadOp instanceof HATVectorOp.Shared, CodeBuilder::dot, CodeBuilder::rarrow);
            id(ARRAY).sbrace(_ -> recurseResultOrThrow(index));
        });
        sbrace(_ -> intConstZero());
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatSelectLoadOp(HATVectorOp.HATVectorSelectLoadOp hatVSelectLoadOp) {
        id(hatVSelectLoadOp.varName()).dot().id(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatSelectStoreOp(HATVectorOp.HATVectorSelectStoreOp hatVSelectStoreOp) {
        id(hatVSelectStoreOp.varName()).dot().id(hatVSelectStoreOp.mapLane()).sp().equals().sp();
        if (hatVSelectStoreOp.resolvedName() != null) {
            // We have detected a direct resolved result (resolved name)
            varName(hatVSelectStoreOp.resolvedName());
        } else {
            // otherwise, we traverse to resolve the expression
            recurseResultOrThrow(hatVSelectStoreOp.operands().get(1));
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatF16ConvOp(HATF16Op.HATF16ConvOp hatF16ConvOp) {
        var float16Class = hatF16ConvOp.float16Class();
        paren(_ -> f16OrBF16(float16Class)).brace(_ -> {
            buildFloat16Class(float16Class);
            paren(_ ->
                recurseResultOrThrow(hatF16ConvOp.operands().getFirst())
            );
        });
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatF16ToFloatConvOp(HATF16Op.HATF16ToFloatConvOp hatF16ToFloatConvOp) {
        buildFloat16Class(hatF16ToFloatConvOp.float16Class());
        paren(_ -> {
            recurseResultOrThrow(hatF16ToFloatConvOp.operands().getFirst());
            if (!hatF16ToFloatConvOp.isLocal()) {
                rarrow().id("value");
            } else if (!hatF16ToFloatConvOp.wasFloat()) {
                dot().id("value");
            }
        });
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorVarOp(HATVectorOp.HATVectorVarOp hatVectorVarOp) {
        type(hatVectorVarOp.buildType()).sp().varName(hatVectorVarOp);
        Value operand = hatVectorVarOp.operands().getFirst();
        if (operand instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
            semicolon().nl();
        } else {
            assign();
        }
        return recurseResultOrThrow(operand);
    }

    @Override
    public CudaHATKernelBuilder genVectorIdentifier(HATVectorOp.HATVectorOfOp hatVectorOfOp) {
        return id("make_" + hatVectorOfOp.buildType());
    }

    @Override
    public CudaHATKernelBuilder hatF16BinaryOp(HATF16Op.HATF16BinaryOp hatF16BinaryOp) {

        Value op1 = hatF16BinaryOp.operands().get(0);
        Value op2 = hatF16BinaryOp.operands().get(1);
        boolean isFirstOperandReference = isArrayReference(op1);
        boolean isSecondOperandReference = isArrayReference(op2);

        final byte f32Mixed;
        if (!isFirstOperandReference && isOperandF32(op1)) {
            f32Mixed = HATF16Op.HATF16BinaryOp.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(op2)) {
            f32Mixed = HATF16Op.HATF16BinaryOp.LAST_OP;
        } else {
            f32Mixed = 0x00;
        }
        var float16Class = hatF16BinaryOp.float16Class();
        paren(_ -> f16OrBF16(float16Class));
        brace(_ ->
                paren(_ -> {
                    if (f32Mixed == HATF16Op.HATF16BinaryOp.LAST_OP) {
                        generateFloat16ConversionToFloat(float16Class).oparen();
                    }
                    recurseResultOrThrow(op1);
                    if (isFirstOperandReference) {
                        rarrow().id("value");
                    } else if (op1 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
                        dot().id("value");
                    }
                    if (f32Mixed == HATF16Op.HATF16BinaryOp.LAST_OP) {
                        cparen();
                    }
                    sp().id(hatF16BinaryOp.binaryOperationType().symbol()).sp();
                    if (f32Mixed == HATF16Op.HATF16BinaryOp.FIRST_OP) {
                        generateFloat16ConversionToFloat(float16Class).oparen();
                    }
                    recurseResultOrThrow(op2);
                    if (isSecondOperandReference) {
                        rarrow().id("value");
                    } else if (op2 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
                        dot().id("value");
                    }
                    if (f32Mixed == HATF16Op.HATF16BinaryOp.FIRST_OP) {
                        cparen();
                    }

                })
        );
        return self();
    }

    private CudaHATKernelBuilder buildFloat16Class(Class<?> float16Class) {
        if (F16.class.isAssignableFrom(float16Class)) {
            return half2float();
        } else if (BF16.class.isAssignableFrom(float16Class)) {
            return __nv_bfloat16();
        } else {
            throw new IllegalStateException("Unexpected value: " + float16Class);
        }
    }

    private CudaHATKernelBuilder generateFloat16ConversionToFloat(Class<?> float16Class) {
        if (F16.class.isAssignableFrom(float16Class)) {
            return half2float();
        } else if (BF16.class.isAssignableFrom(float16Class)) {
            return __bfloat162float();
        } else {
            throw new IllegalStateException("Unexpected value: " + float16Class);
        }
    }

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

    @Override
    protected String mapMathIntrinsic(String hatMathIntrinsicName) {
        return MATH_FUNCTIONS.getOrDefault(hatMathIntrinsicName, hatMathIntrinsicName);
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

    @Override
    public CudaHATKernelBuilder hatTensorVarOp(HATTensorOp.TensorVarOp tensorVarOp) {
        recurse(OpHelper.asResultOrThrow(tensorVarOp.operands().getFirst()).op());
        sp().id(tensorVarOp.varName());
        return self();
    }

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
            if (access.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
                // Expecting an invokeOp
                var invoke = invoke(scopedCodeBuilderContext().lookup(), invokeOp);
                comma();
                if (invoke.resultTypeIs(Tensor.ColumMajor.class)) {
                    id(WMMA_COL_MAJOR);
                } else if (invoke.resultTypeIs(Tensor.RowMajor.class)) {
                    id(WMMA_ROW_MAJOR);
                } else {
                    throw new CUDACodeGenException("[Error]");
                }
                gt();
            }
        }
        return self();
    }

    private static final String TENSOR_MATRIX_A = "matrix_a";
    private static final String TENSOR_MATRIX_B = "matrix_b";
    private final String TENSOR_ACC = "accumulator";

    private String getMatrixOrder(Value valueParameter) {
        if (valueParameter instanceof Op.Result r && r.op() instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
            FieldRef fieldRef = fieldLoadOp.fieldReference();
            return switch (fieldRef.name()) {
                case "FIRST" -> TENSOR_MATRIX_A;
                case "SECOND" -> TENSOR_MATRIX_B;
                default -> TENSOR_ACC;
            };
        }
        return null;
    }

    @Override
    public CudaHATKernelBuilder hatTensorCreateOp(HATTensorOp.TensorCreateOp tensorCreateOp) {
        // infer first parameter
        List<Value> operands = tensorCreateOp.operands();
        Value first = operands.getFirst();
        // The first operand  gives us the matrix order or accumulator
        String matrixOrder = getMatrixOrder(first);

        // Second parameters: analysis of the shape
        List<Integer> shape = new ArrayList<>();

        Value second = operands.get(1);
        if (second.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
            List<Value> shapeOperands = invokeOp.operands();
            for (Value shapeOperand : shapeOperands) {
                if (shapeOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
                    shape.add((int) constantOp.value());
                } else {
                    throw new CUDACodeGenException("Error: expected to find a ConstantOp, but found a " + shapeOperand.declaringElement().getClass());
                }
            }
        } else {
            throw new CUDACodeGenException("InvokeOp expected, but found: " + second.declaringElement().getClass());
        }
        if (shape.size() != 3) {
            throw new CUDACodeGenException("Shape must have three values");
        }

        // The third parameter is the type. It could be `half` or `float` as first implementation
        // This parameter is another constant with the type
        Value classOperand = operands.get(2);
        Object klass = null;
        if (classOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
            klass = constantOp.value();
        }

        String type = "";
        if (klass instanceof ClassType classType && classType.toClassName().equals(F16.class.getCanonicalName())) {
            type = "half";
        } else if (klass instanceof PrimitiveType primitiveType && primitiveType.equals(PrimitiveType.FLOAT)) {
            type = "float";
        }

        Value access = operands.getLast();
        return generateCreateTensor(shape, matrixOrder, type, access);
    }

    @Override
    public CudaHATKernelBuilder hatTensorVarLoadOp(HATTensorOp.TensorVarLoadOp hatTensorVarLoadOp) {
        Value operand = hatTensorVarLoadOp.operands().getFirst();
        if (operand instanceof Op.Result r && r.op() instanceof HATTensorOp.TensorVarOp tensorVarOp) {
            varName(tensorVarOp.varName());
        } else {
            throw new CUDACodeGenException("[ERROR] Expected HATTensorVarOp");
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatTensorFillOp(HATTensorOp.TensorFillOp tensorFillOp) {
        id(WMMA_FILL_TENSOR).paren( _-> {
            List<Value> operands = tensorFillOp.operands();
            recurseValueOrThrough(operands.getFirst())
                    .comma()
                    .recurseValueOrThrough(operands.get(1));
        });
        return self();
    }

    @Override
    protected CUDACodeGenException launchBackendException(String message) {
        return new CUDACodeGenException(message);
    }

    @Override
    public CudaHATKernelBuilder hatTensorMMAOp(HATTensorOp.TensorMMAOp tensorMMAOp) {
        id(WMMA_MMA_TENSOR).paren( _-> commaSeparated(tensorMMAOp.operands(), this::recurseValueOrThrough));
        return self();
    }

    private CudaHATKernelBuilder generateLoadTensor(HATTensorOp.TensorLoadOp tensorLoadOp, boolean isColumnMajor, String tensorName) {
        // First operand is the reference to global memory
        List<Value> operands = tensorLoadOp.operands();
        Value reference = operands.getFirst();
        id(WMMA_LOAD_TENSOR)
                .paren(_ -> {
                    id(tensorName).comma();
                    paren(_ -> type("half").asterisk());
                    recurseValueOrThrough(reference);
                    rarrow().id(ARRAY)
                            .sp().plus().sp()
                            .indexForTensor(isColumnMajor, operands.get(1), operands.get(2), operands.get(3))
                            .comma();
                    recurseValueOrThrough(operands.get(3));
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
    public CudaHATKernelBuilder hatTensorLoadOp(HATTensorOp.TensorLoadOp tensorLoadOp) {
        // Find name tensor of the first argument
        String tensorName = "";
        SequencedSet<Op.Result> uses = tensorLoadOp.result().uses();
        HATTensorOp.TensorVarOp tensorVarOp = null;
        for (Op.Result result : uses) {
            if (result.declaringElement() instanceof HATTensorOp.TensorStoreLoadOp storeLoadOp) {
                // obtain first arg from tensorStoreOp
                Value first = storeLoadOp.operands().getFirst();
                if (first.declaringElement() instanceof HATTensorOp.TensorVarOp varOp) {
                    tensorVarOp = varOp;
                    tensorName = tensorVarOp.varName();
                }
            }
        }

        boolean isColumnMajor = true;
        if (tensorVarOp != null) {
            Value value = tensorVarOp.operands().getFirst();
            if (value.declaringElement() instanceof HATTensorOp.TensorCreateOp createOp) {
                Value tensorLayout = createOp.operands().getLast();
                isColumnMajor = isColumnMajor(tensorLayout);
            }
        }
        return generateLoadTensor(tensorLoadOp, isColumnMajor, tensorName);
    }

    @Override
    public CudaHATKernelBuilder hatTensorStoreLoadOp(HATTensorOp.TensorStoreLoadOp hatTensorStoreLoadOp) {
        List<Value> operands = hatTensorStoreLoadOp.operands();
        recurseValueOrThrough(operands.getLast());
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

            recurseValueOrThrough(reference)
                    .rarrow().id(ARRAY)
                    .sp().plus().sp()
                    .indexForTensor(isColumnMajor, iIndex, jIndex, ldSize)
                    .comma()
                    .recurseValueOrThrough(tensorToStore)
                    .comma()
                    .recurseValueOrThrough(ldSize)
                    .comma();

            if (isColumnMajor) {
                id(WMMA_MEM_COL_MAJOR);
            } else {
                id(WMMA_MEM_ROW_MAJOR);
            }
        });
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatTensorStoreOp(HATTensorOp.TensorStoreOp tensorStoreOp) {
        List<Value> operands = tensorStoreOp.operands();
        // Access layout is the last operand
        final boolean isColumnMajor = isColumnMajor(operands.get(5));
        return generateStoreTensor(operands, isColumnMajor);
    }

    private static final String ARRAY = "array";
}
