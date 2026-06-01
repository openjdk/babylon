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
import hat.dialect.HATVectorOp;
import hat.phases.HATFP16Phase;
import hat.types.F16;
import hat.types.S16ImplOfF16;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.OpHelper.Invoke;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.types.BF16;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

import static optkl.IfaceValue.Vector.getVectorShape;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {

    protected CudaHATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraph, scopedCodeBuilderContext);
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
                .typedefSingleValueStruct("BF16", "BFLOAT16");
    }

    @Override
    public CudaHATKernelBuilder atomicInc(Op.Result instanceResult, String name) {
        return id("atomicAdd").paren(_ -> ampersand().recurseResultOrThrow(instanceResult).rarrow().id(name).comma().literal(1));
    }

    @Override
    public CudaHATKernelBuilder hatVectorStoreOp(JavaOp.InvokeOp invokeOp, IfaceValue.Vector.Shape vectorShape, String name, boolean deviceAllocated) {
        Value dest = invokeOp.operands().get(0);
        Value index = invokeOp.operands().get(2);
        keyword("reinterpret_cast").ltgt(_ -> type(vectorShape.codeType().toString() + vectorShape.lanes()).sp().asterisk());
        paren(_ -> {
            ampersand().recurseResultOrThrow(dest);
            either(deviceAllocated, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ -> recurseResultOrThrow(index));
        });
        sbrace(_ -> intConstZero());
        sp().equals().sp();
        // if the value to be stored is an operation, recurse on the operation
        if (invokeOp.operands().get(1) instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
            recurse(r.op());
        } else {
            varName(name);
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorStoreOp(JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp, IfaceValue.Vector.Shape vectorShape, boolean isLocal, String name) {
        Op.Result dest = OpHelper.resultFromFirstOperandOrThrow(arrayStoreOp);
        Value index = arrayStoreOp.operands().get(1);
        keyword("reinterpret_cast").ltgt(_ -> type(vectorShape.codeType().toString() + vectorShape.lanes()).sp().asterisk());
        paren(_ -> {
            ampersand().recurseResultOrThrow(dest);
            either(isLocal, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ -> recurseResultOrThrow(index));
        });
        sbrace(_ -> intConstZero());
        sp().equals().sp();
        // if the value to be stored is an operation, recurse on the operation
        if (arrayStoreOp.operands().get(1) instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
            recurse(r.op());
        } else {
            varName(name);
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

    public String buildTypeVector(IfaceValue.Vector.Shape vectorShape) {
        return vectorShape.codeType().toString() + vectorShape.lanes();
    }

    @Override
    public CudaHATKernelBuilder generateVectorLoad(Value source, Value index, IfaceValue.Vector.Shape vectorShape, boolean deviceAllocated) {
        reinterpret_cast().ltgt(_ -> type(buildTypeVector(vectorShape)).sp().asterisk());
        paren(_ -> {
            ampersand();
            recurseResultOrThrow(source);
            either(deviceAllocated, CodeBuilder::dot, CodeBuilder::rarrow);
            id("array").sbrace(_ -> recurseResultOrThrow(index));
        });
        sbrace(_ -> intConstZero());
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatSelectStoreOp(Invoke invoke, InvokeVar invokeVar) {
        id(invokeVar.name()).dot().id(mapLane(invokeVar.laneIdx())).sp().equals().sp();
        String resolvedName = invokeVar.resolveName();
        if (resolvedName != null) {
            // We have detected a direct resolved result (resolved name)
            varName(resolvedName);
        } else {
            // otherwise, we traverse to resolve the expression
            recurseResultOrThrow(invoke.op().operands().get(1));
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatF16ConvOp(JavaOp.InvokeOp invokeOp, Class<?> reducedFloatType) {
        paren(_ -> f16OrBF16(reducedFloatType)).brace(_ -> {
            buildFloat16Class(reducedFloatType);
            paren(_ ->
                    recurseResultOrThrow(invokeOp.operands().getFirst())
            );
        });
        return self();
    }

    private static final String VALUE = "value";

    @Override
    public CudaHATKernelBuilder hatF16ToFloatConvOp(Invoke invoke, Class<?> reducedFloatType, boolean wasFloat, boolean isF16Local) {
        buildFloat16Class(reducedFloatType);
        paren(_ -> {
            recurseResultOrThrow(invoke.op().operands().getFirst());
            if (!isF16Local) {
                rarrow().id(VALUE);
            } else if (!wasFloat) {
                dot().id(VALUE);
            }
        });
        return self();
    }

    @Override
    public CudaHATKernelBuilder genVectorIdentifier(IfaceValue.Vector.Shape vectorShape) {
        return id("make_" + vectorShape.codeType().toString() + vectorShape.lanes());
    }

    @Override
    public CudaHATKernelBuilder hatF16BinaryOp(Invoke invoke, Class<?> reducedFloatType) {
        Value op1 = invoke.op().operands().get(0);
        Value op2 = invoke.op().operands().get(1);
        boolean isFirstOperandReference = isArrayReference(op1);
        boolean isSecondOperandReference = isArrayReference(op2);

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
                        generateFloat16ConversionToFloat(reducedFloatType).oparen();
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
                        generateFloat16ConversionToFloat(reducedFloatType).oparen();
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

    private Class<?> reduceFloatType(Optional<Invoke> invoke) {
        if (invoke.isPresent() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().refType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }

    private Class<?> reduceFloatTypeFromReturnType(Optional<Invoke> invoke) {
        if (invoke.isPresent() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().returnType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
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
            if (operand instanceof Op.Result r && r.op() instanceof HATVectorOp.HATVectorBinaryOp) {
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
                .assign(
                        _ -> id(varOp.varName()),
                        _ -> recurse(OpHelper.asResultOrThrow(varOp.operands().getFirst()).op()));
    }

    @Override
    protected CudaHATKernelBuilder varOpLocalMemory(CoreOp.VarOp varOp) {
        HAT_LOCAL_MEM().sp();
        return varOpPrivateMemory(varOp);
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

}
