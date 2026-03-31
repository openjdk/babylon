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
import hat.dialect.HATTensorOp;
import hat.dialect.HATVectorOp;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.OpHelper;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Op;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class OpenCLHATKernelBuilder extends C99HATKernelBuilder<OpenCLHATKernelBuilder> {

    protected OpenCLHATKernelBuilder(KernelCallGraph.State kernelCallGraphState, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(kernelCallGraphState,scopedCodeBuilderContext);
    }

    @Override
    protected OpenCLHATKernelBuilder hatWarpSize() {
        return constant("1");
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
                .when(callgraphState.usesAtomics,_->pragma("OPENCL", "EXTENSION", "cl_khr_global_int32_base_atomics", ":", "enable"))
                .when(callgraphState.usesAtomics,_->pragma("OPENCL", "EXTENSION", "cl_khr_local_int32_base_atomics", ":", "enable"))
                /*.when(callgraphState.usesFp16,_->*/.pragma("OPENCL", "EXTENSION", "cl_khr_fp16", ":", "enable")//)                      // Enable Half type
                .hashDefine("HAT_FUNC", _ -> keyword(""))
                .hashDefine("HAT_KERNEL", _ -> keyword("__kernel"))
                .hashDefine("HAT_GLOBAL_MEM", _ -> keyword("__global"))
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__local"))
                .when(callgraphState.accessedKcFields.contains("gix"), _->hashDefine("HAT_GIX", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstZero()))))
                .when(callgraphState.accessedKcFields.contains("giy"), _->hashDefine("HAT_GIY", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstOne()))))
                .when(callgraphState.accessedKcFields.contains("giz"), _->hashDefine("HAT_GIZ", _ -> paren(_ -> id("get_global_id").paren(_ -> intConstTwo()))))
                .when(callgraphState.accessedKcFields.contains("lix"), _->hashDefine("HAT_LIX", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstZero()))))
                .when(callgraphState.accessedKcFields.contains("liy"), _->hashDefine("HAT_LIY", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstOne()))))
                .when(callgraphState.accessedKcFields.contains("liz"), _->hashDefine("HAT_LIZ", _ -> paren(_ -> id("get_local_id").paren(_ -> intConstTwo()))))
                .when(callgraphState.accessedKcFields.contains("gsx"), _->hashDefine("HAT_GSX", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstZero()))))
                .when(callgraphState.accessedKcFields.contains("gsy"), _->hashDefine("HAT_GSY", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstOne()))))
                .when(callgraphState.accessedKcFields.contains("gsz"), _->hashDefine("HAT_GSZ", _ -> paren(_ -> id("get_global_size").paren(_ -> intConstTwo()))))
                .when(callgraphState.accessedKcFields.contains("lsx"), _->hashDefine("HAT_LSX", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstZero()))))
                .when(callgraphState.accessedKcFields.contains("lsy"), _->hashDefine("HAT_LSY", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstOne()))))
                .when(callgraphState.accessedKcFields.contains("lsz"), _->hashDefine("HAT_LSZ", _ -> paren(_ -> id("get_local_size").paren(_ -> intConstTwo()))))
                .when(callgraphState.accessedKcFields.contains("bix"), _->hashDefine("HAT_BIX", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstZero()))))
                .when(callgraphState.accessedKcFields.contains("biy"), _->hashDefine("HAT_BIY", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstOne()))))
                .when(callgraphState.accessedKcFields.contains("biz"), _->hashDefine("HAT_BIZ", _ -> paren(_ -> id("get_group_id").paren(_ -> intConstTwo()))))
                .when(callgraphState.accessedKcFields.contains("bsx"), _->hashDefine("HAT_BSX", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstZero()))))
                .when(callgraphState.accessedKcFields.contains("bsy"), _->hashDefine("HAT_BSY", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstOne()))))
                .when(callgraphState.accessedKcFields.contains("bsz"), _->hashDefine("HAT_BSZ", _ -> paren(_ -> id("get_num_groups").paren(_ -> intConstTwo()))))
                .when(callgraphState.usesFp16, _->maxMacro("MAX_HAT"))
                .when(callgraphState.usesFp16, _->minMacro("MIN_HAT"))
                .when(callgraphState.usesBarrier, _->hashDefine("HAT_BARRIER", _ -> id("barrier").oparen().id("CLK_LOCAL_MEM_FENCE").cparen()))
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

    @Override
    public OpenCLHATKernelBuilder hatTensorVarOp(HATTensorOp.TensorVarOp tensorVarOp) {
        recurse(OpHelper.asResultOrThrow(tensorVarOp.operands().getFirst()).op());
        // We don't need to generate the name at this point, but rather during tensor create.
        // That's the place we know all information, including type, shape, and name
        // sp().id(tensorVarOp.varName());
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorCreateOp(HATTensorOp.TensorCreateOp tensorCreateOp) {
        List<Value> operands = tensorCreateOp.operands();

        // Second parameters: analysis of the shape
        int[] shape = new int[3];
        Value second = operands.get(1);
        if (second.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
            List<Value> shapeOperands = invokeOp.operands();
            for (int i = 0; i < shapeOperands.size(); i++) {
                Value shapeOperand = shapeOperands.get(i);
                if (shapeOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
                    shape[i] = (int) constantOp.value();
                }
            }
        }

        // The third parameter is the type. It could be `half` or `float` as first implementation
        // This parameter is another constant with the type
        Value classOperand = operands.get(2);
        Object klass = null;
        if (classOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
            klass = constantOp.value();
        }

        var tensorVarValue = tensorCreateOp.result().uses().getFirst();
        String varTensorName = null;
        if (tensorVarValue.declaringElement() instanceof HATTensorOp.TensorVarOp tensorVarOp) {
            varTensorName = tensorVarOp.varName();
        }
        int size = shape[0] * shape[1];
        if (klass instanceof ClassType classType && classType.toClassName().equals(F16.class.getCanonicalName())) {
            f16Type();
        } else if (klass instanceof PrimitiveType primitiveType && primitiveType == PrimitiveType.FLOAT) {
            type("float");
        } else {
            throw new RuntimeException("[ERROR] Codegen. Type " + klass + " not expected");
        }
        sp().varName(varTensorName).sbrace(_-> constant(Integer.toString(size)));
        return self();
    }

    static HATTensorOp.TensorVarOp findTensorVarOp(Value varLoadOp) {
        return switch (varLoadOp.declaringElement()) {
            case HATTensorOp.TensorVarLoadOp tensorVarLoadOp -> findTensorVarOp(tensorVarLoadOp.operands().getFirst());
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp2 -> findTensorVarOp(varLoadOp2.operands().getFirst());
            case HATTensorOp.TensorVarOp tensorVarOp -> tensorVarOp;
            case null, default -> null;
        };
    }

    static float getValueConstantTensor(Value v) {
        if ((v instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp constant)) {
            Object valueConstant = constant.value();
            return (float) valueConstant;

        } else if (v instanceof Op.Result r) {
            return getValueConstantTensor(r.op().operands().getFirst());
        }
        return -1.0f;
    }

    private int[] getShapeFromTensorCreateValue(Value tensorCreateValue) {
        if (tensorCreateValue.declaringElement() instanceof HATTensorOp.TensorCreateOp tensorCreateOp) {
            // Second parameters: analysis of the shape
            int[] shape = new int[3];
            Value second = tensorCreateOp.operands().get(1);
            if (second.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> shapeOperands = invokeOp.operands();
                for (int i = 0; i < shapeOperands.size(); i++) {
                    Value shapeOperand = shapeOperands.get(i);
                    if (shapeOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
                        shape[i] = (int) constantOp.value();
                    }
                }
            }
            return shape;
        }
        return new int[]{};
    }

    private int[] getShapeFromTensorVarOp(HATTensorOp.TensorVarOp tensorVarOp) {
        Value tensorCreateValueOp = tensorVarOp.operands().getFirst();
        if (tensorCreateValueOp.declaringElement() instanceof HATTensorOp.TensorCreateOp tensorCreateOp) {
            // Second parameters: analysis of the shape
            int[] shape = new int[3];
            Value second = tensorCreateOp.operands().get(1);
            if (second.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> shapeOperands = invokeOp.operands();
                for (int i = 0; i < shapeOperands.size(); i++) {
                    Value shapeOperand = shapeOperands.get(i);
                    if (shapeOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
                        shape[i] = (int) constantOp.value();
                    }
                }
            }
            return shape;
        }
        return new int[]{};
    }

    private boolean isColumnMajorFromVarOp(HATTensorOp.TensorVarOp tensorVarOp) {
        Value tensorCreateValueOp = tensorVarOp.operands().getFirst();
        if (tensorCreateValueOp.declaringElement() instanceof HATTensorOp.TensorCreateOp tensorCreateOp) {
            // Parameter 3 defines the access layout
            Value valueLayout = tensorCreateOp.operands().get(3);
            return isColumnMajor(valueLayout);
        }
        return false;
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

    private OpenCLHATKernelBuilder emitForLoopWithBound(int from, int to, HATTensorOp.TensorVarOp tensorVarOp, float initValue) {
        //  emitText("for (int m = 0; m < " + shape[0] + "; m++) { " +
        //           "for (int n = 0; n < " + shape[1] + "; n++) { " +
        //            tensorVarOp.varName() + "[m * " + shape[0] + " + n] = " + initValue + "f;" + "}" + "}");
        String prefix = "index_$";
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).sp().equals().sp().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(to).semicolon();
            id(varA).plusplus();
        }).sp().brace(_ -> {
            in().nl().forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).sp().equals().sp().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(to).semicolon();
                id(varB).plusplus();
            }).sp().in();

            brace(_ -> {
                nl().id(tensorVarOp.varName()).sbrace(_ ->
                        id(varA).mul()
                                .id(Integer.toString(to))
                                .plus()
                                .id(varB))
                                .sp().equals().sp()
                                .constant(Float.toString(initValue)).id("f")
                                .semicolon().nl();
            }).out().out();
        });
        return self();
    }

    private OpenCLHATKernelBuilder emitTensorFill(int[] shape, HATTensorOp.TensorVarOp tensorVarOp, float initValue) {

        return emitForLoopWithBound(0, shape[0], tensorVarOp, initValue);
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorFillOp(HATTensorOp.TensorFillOp tensorFillOp) {

        /*
         From this expression:

         Tensor.fill(acc, 0.0f);

         We need to generate the following skeleton
                for (int m = 0; m < SHAPE_1; m++)
                    for (int n = 0; n < SHAPE_2; n++)
                        tensor[m * SHAPE_1 + n] = initValue;
         */

        // 1. Access to the variable name
        var tensorValue = tensorFillOp.operands().getFirst();
        HATTensorOp.TensorVarOp tensorVarOp = findTensorVarOp(tensorValue);
        if (tensorVarOp == null) {
            throw new RuntimeException("[Error][Codegen] Expected a tensorVarOp, but found `null` instead");
        }

        // 2. Access the shape
        // Second parameters: analysis of the shape
        Value tensorAccDecl = tensorVarOp.operands().getFirst();
        int[] shape = getShapeFromTensorCreateValue(tensorAccDecl);

        // 3. Access the layout
        var tensorInitValue = tensorFillOp.operands().get(1);
        float initValue = getValueConstantTensor(tensorInitValue);

        emitTensorFill(shape, tensorVarOp, initValue);

        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorVarLoadOp(HATTensorOp.TensorVarLoadOp hatTensorVarLoadOp) {
        Value operand = hatTensorVarLoadOp.operands().getFirst();
        if (operand instanceof Op.Result r && r.op() instanceof HATTensorOp.TensorVarOp tensorVarOp) {
            varName(tensorVarOp.varName());
        } else {
            throw new RuntimeException("[ERROR] Expected HATTensorVarOp");
        }
        return self();
    }

    private OpenCLHATKernelBuilder generateTensorMMA(int[] shape, HATTensorOp.TensorVarOp tensorA, HATTensorOp.TensorVarOp tensorB, HATTensorOp.TensorVarOp tensorC, HATTensorOp.TensorVarOp result) {

        String prefix = "index_$";
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        String varC = generateVariableName(prefix);
        String acc = generateVariableName("sum_");
        final int from = 0;
        final int to = shape[0];

        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).sp().equals().sp().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(to).semicolon();
            id(varA).plusplus();
        }).sp().brace(_ -> {
            in().nl().forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).sp().equals().sp().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(to).semicolon();
                id(varB).plusplus();
            }).in();

            brace(_ -> {
                nl().f32Type().sp().id(acc).sp().equals().sp().id(tensorC.varName()).sbrace( _-> {
                    id(varA).mul().id(Integer.toString(shape[0])).sp().plus().id(varB);
                }).semicolon().nl();

                forKeyword().sp().paren(_ -> {
                    s32Type().sp().id(varC).sp().equals().sp().intValue(from).semicolon();
                    id(varC).sp().lt().sp().intValue(to).semicolon();
                    id(varC).plusplus();
                }).sp().in();

                brace(_ -> {
                    nl();
                    String ha = generateVariableName("ha_");
                    String hb = generateVariableName("hb_");
                    String resultTensor = generateVariableName("h_res_");
                    f16Type().sp().id(ha).sp().equals().sp().id(tensorA.varName()).sbrace( _ -> id(varA).mul().id(Integer.toString(shape[0])).sp().plus().id(varC)).semicolon().nl();
                    f16Type().sp().id(hb).sp().equals().sp().id(tensorB.varName()).sbrace( _ -> id(varC).mul().id(Integer.toString(shape[0])).sp().plus().id(varB)).semicolon().nl();
                    f16Type().sp().id(resultTensor).sp().equals().sp().paren( _ -> f16Type()).brace( _ -> paren( _ -> id(ha).dot().id("value").mul().id(hb).dot().id("value"))).semicolon().nl();
                    id(acc).sp().plusEquals().cast( _ -> f32Type()).paren( _-> id(resultTensor).dot().id("value")).semicolon().nl();
                }).nl().out();

                id(result.varName()).sbrace( _ -> id(varA).mul().id(Integer.toString(shape[0])).sp().plus().id(varB)).sp().equals().sp().id(acc).semicolon().nl();

            }).semicolon().nl();

        }).out().out();

//        emitText("for (int m = 0; m < " + shape[0] + "; m++) {").nl();
//        emitText("for (int n = 0; n < " + shape[1] + "; n++) {").nl();
//        emitText(" float sum = ").nl();
//        emitText(tensorC.varName()).emitText("[m * " + shape[0] + " + n]").semicolon().nl();
//        emitText("for (int k = 0; k < " + shape[2] + "; k++) {").nl();
//        emitText("F16_t ha = " + tensorA.varName() + "[m * " + shape[0] + " + k];").nl();
//        emitText("F16_t hb = " + tensorB.varName() + "[k * " + shape[2] + " + n];").nl();
//        emitText("F16_t result = (F16_t){(ha.value * hb.value)};").nl();
//        emitText("sum += (float)(result.value);").nl();
//        cbrace().nl();
//        emitText(result.varName()+ "[ m * " + shape[0] + " + n] = sum;").nl();
//        cbrace().nl();
//        cbrace().nl();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorMMAOp(HATTensorOp.TensorMMAOp tensorMMAOp) {
        // Tensor.mma(acc, tensorA, tensorB, acc)
        var resulTensorValue = tensorMMAOp.operands().getFirst();
        var tensorAValue = tensorMMAOp.operands().get(1);
        var tensorBValue = tensorMMAOp.operands().get(2);
        var tensorCValue = tensorMMAOp.operands().get(3);
        var tensorA = findTensorVarOp(tensorAValue);
        var tensorB = findTensorVarOp(tensorBValue);
        var tensorC = findTensorVarOp(tensorCValue);
        var tensorResult = findTensorVarOp(resulTensorValue);
        if (tensorA == null || tensorB == null || tensorC == null || tensorResult == null) {
            throw new RuntimeException("[Error][CodeGen] Expected a tensorValue, but found `null` instead");
        }
        int[] shape = getShapeFromTensorVarOp(tensorA);
        return generateTensorMMA(shape, tensorA, tensorB, tensorC, tensorResult);
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorStoreLoadOp(HATTensorOp.TensorStoreLoadOp storeLoadOp) {
        List<Value> operands = storeLoadOp.operands();
        if (operands.getLast() instanceof Op.Result r) {
            recurse(r.op());
        }
        return self();
    }

    private HATTensorOp.TensorVarOp findTensorVarOp(HATTensorOp.TensorLoadOp tensorLoadOp) {
        var tensorStoreLoadValue = tensorLoadOp.result().uses().getFirst();
        if (tensorStoreLoadValue.declaringElement() instanceof HATTensorOp.TensorStoreLoadOp tensorStoreLoadOp) {
            Value first = tensorStoreLoadOp.operands().getFirst();
            if (first.declaringElement() instanceof HATTensorOp.TensorVarOp tensorVarOp) {
                return tensorVarOp;
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    private OpenCLHATKernelBuilder generateTensorLoad(int[] shape, Value iIndexValue, Value jIndexValue, boolean isColumnMajor, Value leadingDimension, Value ptrValue, HATTensorOp.TensorVarOp tensorVarOp) {

        // Example being generated:
        //        emitText("""
        //            for (int m = 0; m < WMMA_M; m++) {
        //                int rowA = aRow + m;
        //                for (int n = 0; n < WMMA_N; n++) {
        //                    int colA = aCol + n;
        //                    int idxA = rowA + colA * lda;
        //                    HAT_GLOBAL_MEM F16Impl_t* ha = &matrixA->array[idxA];
        //                    F16_t r = (F16_t){ha->value};
        //                    tensorA[m * WMMA_M + n] = r;
        //                }
        //            }
        //            """);

        String prefix = "index_$";
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        final int to = shape[0];
        final int from = 0;

        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).sp().equals().sp().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(to).semicolon();
            id(varA).plusplus();
        }).in();

        String row = generateVariableName("row_");

        brace(_ -> {
            nl().s32Type().sp().id(row).sp().equals().sp();

            if (iIndexValue instanceof Op.Result r) {
                recurse(r.op());
            }
            plus().id(varA).semicolon().nl();

            forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).sp().equals().sp().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(to).semicolon();
                id(varB).plusplus();
            }).sp().in();

            String col = generateVariableName("col_");

            brace(_ -> {
                nl().s32Type().sp().id(col).sp().equals().sp();

                if (jIndexValue instanceof Op.Result r) {
                    recurse(r.op());
                }
                plus().id(varB).semicolon().nl();

                String index = generateVariableName("index_");
                s32Type().sp().id(index).sp().equals().sp().id(row);

                if (isColumnMajor) plus();
                else mul();
                id(col);
                if (isColumnMajor) mul();
                else plus();
                if (leadingDimension instanceof Op.Result r) {
                    recurse(r.op());
                }
                semicolon().nl();

                // TODO: We assume a load from global memory. In
                // future version, we will process loads from other
                // memory regions of the accelerator

                String ha = generateVariableName("ha_");
                id("HAT_GLOBAL_MEM F16Impl_t").asterisk().sp().id(ha).sp().equals().sp().ampersand();

                if (ptrValue instanceof  Op.Result r) {
                    recurse(r.op());
                }
                rarrow().id("array").sbrace( _ -> id(index)).semicolon().nl();

                String r = generateVariableName("r_");
                f16Type().sp().id(r).sp().equals().sp().cast( _ -> f16Type()).brace( _-> id(ha).rarrow().id("value")).semicolon().nl();

                // store into the acc
                emitText(tensorVarOp.varName()).sbrace( _ -> id(varA).mul().id(Integer.toString(shape[0])).plus().id(varB));
                equals().sp().id(r).semicolon().nl();
            }).out();
        }).out();


//        emitText("for (int m = 0; m < " + shape[0] + "; m++) {").nl();
//        emitText(" int rowA = ");
//        if (iIndexValue instanceof Op.Result r) {
//            recurse(r.op());
//        }
//        emitText(" + m;").nl();
//        emitText("for (int n = 0; n < " + shape[1] + "; n++) {").nl();
//        emitText(" int colA = ");
//        if (jIndexValue instanceof Op.Result r) {
//            recurse(r.op());
//        }
//        emitText(" + n;").nl();
//
//        emitText(" int idxA = ").id("rowA");
//        if (isColumnMajor) plus(); else mul();
//        id("colA");
//        if (isColumnMajor) mul(); else plus();
//        if (leadingDimension instanceof Op.Result r) {
//            recurse(r.op());
//        }
//        semicolon().nl();
//
//        emitText("HAT_GLOBAL_MEM F16Impl_t* ha = &");
//
//        if (ptrValue instanceof  Op.Result r) {
//            recurse(r.op());
//        }
//        emitText("->array[idxA]").semicolon().nl();
//
//        emitText("F16_t r = (F16_t){ha->value};").nl();
//
//        // store into the acc
//        emitText(tensorVarOp.varName()).sbrace( _ -> id("m").mul().id(Integer.toString(shape[0])).plus().id("n"));
//        equals().sp().id("r").semicolon().nl();
//        cbrace().cbrace().nl();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorLoadOp(HATTensorOp.TensorLoadOp tensorLoadOp) {
        // tensorA = Tensor.load(matrixA, aRow, aCol, lda);

        // 1. We need the global ptr
        // 2. We need the indexes (i, j)
        // 3. We need leading dimension
        // 4. We need the name of the tensor
        // 5. We need the shape
        // 6. We need the access layout

        List<Value> operands = tensorLoadOp.operands();
        var ptrValue = operands.getFirst();
        var iIndexValue = operands.get(1);
        var jIndexValue = operands.get(2);
        var leadingDimension = operands.get(3);
        HATTensorOp.TensorVarOp tensorVarOp = findTensorVarOp(tensorLoadOp);
        int[] shape;
        boolean isColumnMajor;
        if (tensorVarOp != null) {
            shape = getShapeFromTensorVarOp(tensorVarOp);
            isColumnMajor = isColumnMajorFromVarOp(tensorVarOp);
        } else {
            throw new RuntimeException("[Error][CodeGen] Expected to see an instance of tensorVarOp but `null` found");
        }
        generateTensorLoad(shape, iIndexValue, jIndexValue, isColumnMajor, leadingDimension, ptrValue, tensorVarOp);
        return self();
    }

    private OpenCLHATKernelBuilder generateTensorStore(int[] shape, Value iIndexValue, Value jIndexValue, boolean isColumnMajor, Value leadingDimension, Value ptrValue, HATTensorOp.TensorVarOp tensorVarOp) {
        // Example being generated
        // emitText("""
        //     for (int m = 0; m < WMMA_M; m++) {
        //     int rowC = cRow + m;
        //       for (int n = 0; n < WMMA_N; n++) {
        //         int colC = cCol + n;
        //         int idxC = (cRow) + (cCol) * ldc;
        //         matrixC->array[idxC] = acc[m * 16 + n];
        //        }
        //     }
        //     """);

        String prefix = "index_$";
        String varA = generateVariableName(prefix);
        String varB = generateVariableName(prefix);
        final int to = shape[0];
        final int from = 0;

        forKeyword().sp().paren(_ -> {
            s32Type().sp().id(varA).sp().equals().sp().intValue(from).semicolon();
            id(varA).sp().lt().sp().intValue(to).semicolon();
            id(varA).plusplus();
        }).in();

        String row = generateVariableName("row_");

        brace(_ -> {
            nl().s32Type().sp().id(row).sp().equals().sp();

            if (iIndexValue instanceof Op.Result r) {
                recurse(r.op());
            }
            plus().id(varA).semicolon().nl();

            forKeyword().sp().paren(_ -> {
                s32Type().sp().id(varB).sp().equals().sp().intValue(from).semicolon();
                id(varB).sp().lt().sp().intValue(to).semicolon();
                id(varB).plusplus();
            }).sp().in();

            String col = generateVariableName("col_");

            brace(_ -> {
                nl().s32Type().sp().id(col).sp().equals().sp();

                if (jIndexValue instanceof Op.Result r) {
                    recurse(r.op());
                }
                plus().id(varB).semicolon().nl();

                String index = generateVariableName("index_");
                s32Type().sp().id(index).sp().equals().sp().id(row);

                if (isColumnMajor) plus();
                else mul();
                id(col);
                if (isColumnMajor) mul();
                else plus();
                if (leadingDimension instanceof Op.Result r) {
                    recurse(r.op());
                }
                semicolon().nl();

                if (ptrValue instanceof  Op.Result r) {
                    recurse(r.op());
                }
                rarrow().id("array").sbrace( _ -> id(index)).sp().equals().sp();
                id(tensorVarOp.varName()).sbrace( _ -> id(varA).mul().id(Integer.toString(shape[0])).plus().id(varB));
                semicolon().nl();
            }).out();
        }).out();


//        emitText("for (int m = 0; m < " + shape[0] + "; m++) {").nl();
//        emitText(" int rowC = ");
//        if (iIndexValue instanceof Op.Result r) {
//            recurse(r.op());
//        }
//        emitText(" + m;").nl();
//        emitText("for (int n = 0; n < " + shape[1] + "; n++) {").nl();
//        emitText(" int colC = ");
//        if (jIndexValue instanceof Op.Result r) {
//            recurse(r.op());
//        }
//        emitText(" + n;").nl();
//
//        emitText(" int idxC = ").id("rowC");
//        if (isColumnMajor) plus(); else mul();
//        id("colC");
//        if (isColumnMajor) mul(); else plus();
//        if (leadingDimension instanceof Op.Result r) {
//            recurse(r.op());
//        }
//        semicolon().nl();
//
//        if (ptrValue instanceof  Op.Result r) {
//            recurse(r.op());
//        }
//        emitText("->array[idxC]").sp().equals().sp();
//
//        emitText(tensorVarOp.varName()).sbrace( _ -> id("m").mul().id(Integer.toString(shape[0])).plus().id("n"));
//        semicolon().nl();
//
//        cbrace().cbrace().nl();
        return self();
    }

    @Override
    public OpenCLHATKernelBuilder hatTensorStoreOp(HATTensorOp.TensorStoreOp tensorStoreOp) {
        // Tensor.store(matrixC, cRow, cCol, acc, ldc, Tensor.ofColumnMajor());

        // 1. We need the global ptr
        // 2. We need the indexes (i, j)
        // 3. We need leading dimension
        // 4. We need the name of the tensor
        // 5. We need the shape
        // 6. We need the access layout

        List<Value> operands = tensorStoreOp.operands();
        var ptrValue = operands.getFirst();
        var iIndexValue = operands.get(1);
        var jIndexValue = operands.get(2);
        var tensorValue = operands.get(3);
        var leadingDimension = operands.get(4);

        HATTensorOp.TensorVarOp tensorVarOp = findTensorVarOp(tensorValue);
        if (tensorVarOp == null) {
            throw new RuntimeException("[Error][CodeGen] Expected to find a tensorVarOp, but `null` instead.");
        }

        int[] shape = getShapeFromTensorVarOp(tensorVarOp);

        Value accessLayout = operands.get(5);
        final boolean isColumnMajor = isColumnMajor(accessLayout);

        generateTensorStore(shape, iIndexValue, jIndexValue, isColumnMajor, leadingDimension, ptrValue, tensorVarOp);
        return self();
    }

}
