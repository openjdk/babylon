/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;

public class TranslateToSpirvModel  {
    private Map<Block, Block.Builder> blockMap;    // Java block to spirv block builder
    private Map<Value, Value> valueMap;            // Java model Value to Spirv model Value

    public static SpirvOps.FuncOp translateFunction(CoreOp.FuncOp func) {
        CoreOp.FuncOp lowFunc = lowerMethod(func);
        TranslateToSpirvModel instance = new TranslateToSpirvModel();
        Body.Builder bodyBuilder = instance.translateBody(lowFunc.body(), lowFunc, null);
        return new SpirvOps.FuncOp(lowFunc.funcName(), lowFunc.invokableType(), bodyBuilder);
    }

    private TranslateToSpirvModel() {
        blockMap = new HashMap<>();
        valueMap = new HashMap<>();
    }

    private Body.Builder translateBody(Body body, Op parentOp, Body.Builder parentBody) {
        Body.Builder bodyBuilder = Body.Builder.of(parentBody, body.bodyType());
        Block.Builder spirvBlock = bodyBuilder.entryBlock();
        blockMap.put(body.entryBlock(), spirvBlock);
        List<Block> blocks = body.blocks();
        // map Java blocks to spirv blocks
        for (Block b : blocks.subList(1, blocks.size()))  {
            Block.Builder loweredBlock = spirvBlock.block(b.parameterTypes());
            blockMap.put(b, loweredBlock);
            spirvBlock = loweredBlock;
        }
        // map entry block parameters to spirv function parameter
        spirvBlock = bodyBuilder.entryBlock();
        List<SpirvOp> paramOps = new ArrayList<>();
        List<SpirvOps.VariableOp> varOps = new ArrayList<>();
        Block entryBlock = body.entryBlock();
        int paramCount = entryBlock.parameters().size();
        for (int i = 0; i < paramCount; i++) {
            Block.Parameter bp = entryBlock.parameters().get(i);
            assert entryBlock.ops().get(i) instanceof CoreOp.VarOp;
            SpirvOp funcParam = new SpirvOps.FunctionParameterOp(bp.type(), List.of());
            spirvBlock.op(funcParam);
            valueMap.put(bp, funcParam.result());
            paramOps.add(funcParam);
        }
        // SPIR-V Variable ops must be the first ops in a function's entry block and do not include initialization.
        // Emit all SPIR-V Variable ops first and emit initializing stores afterward, at the CR model VarOp position.
        for (int i = 0; i < paramCount; i++) {
            CoreOp.VarOp jvop = (CoreOp.VarOp)entryBlock.ops().get(i);
            TypeElement resultType = new PointerType(jvop.varType(), StorageType.CROSSWORKGROUP);
            SpirvOps.VariableOp svop = new SpirvOps.VariableOp((String)jvop.attributes().get(""), resultType, jvop.varType());
            spirvBlock.op(svop);
            valueMap.put(jvop.result(), svop.result());
            varOps.add(svop);
        }
        // add non-function-parameter variables
        for (int bi = 0; bi < body.blocks().size(); bi++)  {
            Block block = body.blocks().get(bi);
            spirvBlock = blockMap.get(block);
            List<Op> ops = block.ops();
            for (int i = (bi == 0 ? paramCount : 0); i < ops.size(); i++) {
                if (bi > 0) spirvBlock = blockMap.get(block);
                Op op = ops.get(i);
                if (op instanceof CoreOp.VarOp jvop) {
                    TypeElement resultType = new PointerType(jvop.varType(), StorageType.CROSSWORKGROUP);
                    SpirvOps.VariableOp svop = new SpirvOps.VariableOp((String)jvop.attributes().get(""), resultType, jvop.varType());
                    bodyBuilder.entryBlock().op(svop);
                    valueMap.put(jvop.result(), svop.result());
                    varOps.add(svop);
                }
            }
        }
        for (int bi = 0; bi < body.blocks().size(); bi++)  {
            Block block = body.blocks().get(bi);
            spirvBlock = blockMap.get(block);
            for (Op op : block.ops()) {
                switch (op) {
                    case CoreOp.ReturnOp rop -> {
                        spirvBlock.op(new SpirvOps.ReturnOp(rop.resultType(), mapOperands(rop)));
                    }
                    case CoreOp.VarOp vop -> {
                        Value dest = valueMap.get(vop.result());
                        Value value = valueMap.get(vop.operands().get(0));
                        // init variable here; declaration has been moved to top of function
                        spirvBlock.op(new SpirvOps.StoreOp(dest, value));
                    }
                    case CoreOp.VarAccessOp.VarLoadOp vlo -> {
                        List<Value> operands = mapOperands(vlo);
                        SpirvOps.LoadOp load = new SpirvOps.LoadOp(vlo.resultType(), operands);
                        spirvBlock.op(load);
                        valueMap.put(vlo.result(), load.result());
                    }
                    case CoreOp.VarAccessOp.VarStoreOp vso -> {
                        Value dest = valueMap.get(vso.varOp().result());
                        Value value = valueMap.get(vso.operands().get(1));
                        spirvBlock.op(new SpirvOps.StoreOp(dest, value));
                    }
                    case CoreOp.ArrayAccessOp.ArrayLoadOp alo -> {
                        Value array = valueMap.get(alo.operands().get(0));
                        Value index = valueMap.get(alo.operands().get(1));
                        TypeElement arrayType = array.type();
                        SpirvOps.ConvertOp convert = new SpirvOps.ConvertOp(JavaType.type(long.class), List.of(index));
                        spirvBlock.op(new SpirvOps.LoadOp(arrayType, List.of(array)));
                        spirvBlock.op(convert);
                        SpirvOp ibac = new SpirvOps.InBoundAccessChainOp(arrayType, List.of(array, convert.result()));
                        spirvBlock.op(ibac);
                        SpirvOp load = new SpirvOps.LoadOp(alo.resultType(), List.of(ibac.result()));
                        spirvBlock.op(load);
                        valueMap.put(alo.result(), load.result());
                    }
                    case CoreOp.ArrayAccessOp.ArrayStoreOp aso -> {
                        Value array = valueMap.get(aso.operands().get(0));
                        Value index = valueMap.get(aso.operands().get(1));
                        TypeElement arrayType = array.type();
                        SpirvOp ibac = new SpirvOps.InBoundAccessChainOp(arrayType, List.of(array, index));
                        spirvBlock.op(ibac);
                        spirvBlock.op(new SpirvOps.StoreOp(ibac.result(), valueMap.get(aso.operands().get(2))));
                    }
                    case CoreOp.ArrayLengthOp alo -> {
                        Op len = new SpirvOps.ArrayLengthOp(JavaType.INT, List.of(valueMap.get(alo.operands().get(0))));
                        spirvBlock.op(len);
                        valueMap.put(alo.result(), len.result());
                    }
                    case CoreOp.AddOp aop -> {
                        TypeElement type = aop.operands().get(0).type();
                        List<Value> operands = mapOperands(aop);
                        SpirvOp addOp;
                        if (isIntegerType(type)) addOp = new SpirvOps.IAddOp(type, operands);
                        else if (isFloatType(type)) addOp = new SpirvOps.FAddOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(addOp);
                        valueMap.put(aop.result(), addOp.result());
                     }
                    case CoreOp.SubOp sop -> {
                        TypeElement  type = sop.operands().get(0).type();
                        List<Value> operands = mapOperands(sop);
                        SpirvOp subOp;
                        if (isIntegerType(type)) subOp = new SpirvOps.ISubOp(type, operands);
                        else if (isFloatType(type)) subOp = new SpirvOps.FSubOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(subOp);
                        valueMap.put(sop.result(), subOp.result());
                     }
                    case CoreOp.MulOp mop -> {
                        TypeElement type = mop.operands().get(0).type();
                        List<Value> operands = mapOperands(mop);
                        SpirvOp mulOp;
                        if (isIntegerType(type)) mulOp = new SpirvOps.IMulOp(type, operands);
                        else if (isFloatType(type)) mulOp = new SpirvOps.FMulOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(mulOp);
                        valueMap.put(mop.result(), mulOp.result());
                    }
                    case CoreOp.DivOp dop -> {
                        TypeElement type = dop.operands().get(0).type();
                        List<Value> operands = mapOperands(dop);
                        SpirvOp divOp;
                        if (isIntegerType(type)) divOp = new SpirvOps.IDivOp(type, operands);
                        else if (isFloatType(type)) divOp = new SpirvOps.FDivOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(divOp);
                        valueMap.put(dop.result(), divOp.result());
                    }
                    case CoreOp.ModOp mop -> {
                        TypeElement type = mop.operands().get(0).type();
                        List<Value> operands = mapOperands(mop);
                        SpirvOp modOp = new SpirvOps.ModOp(type, operands);
                        spirvBlock.op(modOp);
                        valueMap.put(mop.result(), modOp.result());
                    }
                    case CoreOp.EqOp eqop -> {
                        TypeElement type = eqop.operands().get(0).type();
                        List<Value> operands = mapOperands(eqop);
                        SpirvOp seqop;
                        if (isIntegerType(type)) seqop = new SpirvOps.IEqualOp(type, operands);
                        else if (isFloatType(type)) seqop = new SpirvOps.FEqualOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(seqop);
                        valueMap.put(eqop.result(), seqop.result());
                    }
                    case CoreOp.NeqOp neqop -> {
                        TypeElement type = neqop.operands().get(0).type();
                        List<Value> operands = mapOperands(neqop);
                        SpirvOp sneqop;
                        if (isIntegerType(type)) sneqop = new SpirvOps.INotEqualOp(type, operands);
                        else if (isFloatType(type)) sneqop = new SpirvOps.FNotEqualOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(sneqop);
                        valueMap.put(neqop.result(), sneqop.result());
                    }
                    case CoreOp.LtOp ltop -> {
                        TypeElement type = ltop.operands().get(0).type();
                        List<Value> operands = mapOperands(ltop);
                        SpirvOp sltop = new SpirvOps.LtOp(type, operands);
                        spirvBlock.op(sltop);
                        valueMap.put(ltop.result(), sltop.result());
                    }
                    case CoreOp.InvokeOp inv -> {
                        List<Value> operands = mapOperands(inv);
                        SpirvOp spirvCall = new SpirvOps.CallOp(inv.invokeDescriptor(), operands);
                        spirvBlock.op(spirvCall);
                        valueMap.put(inv.result(), spirvCall.result());
                    }
                    case CoreOp.ConstantOp cop -> {
                        SpirvOp scop = new SpirvOps.ConstantOp(cop.resultType(), cop.value());
                        spirvBlock.op(scop);
                        valueMap.put(cop.result(), scop.result());
                    }
                    case CoreOp.ConvOp cop -> {
                        List<Value> operands = mapOperands(cop);
                        SpirvOp scop = new SpirvOps.ConvertOp(cop.resultType(), operands);
                        spirvBlock.op(scop);
                        valueMap.put(cop.result(), scop.result());
                    }
                    case CoreOp.FieldAccessOp.FieldLoadOp flo -> {
                        SpirvOp load = new SpirvOps.FieldLoadOp(flo.resultType(), flo.fieldDescriptor(), mapOperands(flo));
                        spirvBlock.op(load);
                        valueMap.put(flo.result(), load.result());
                    }
                    case CoreOp.BranchOp bop -> {
                        Block.Reference successor = blockMap.get(bop.branch().targetBlock()).successor();
                        spirvBlock.op(new SpirvOps.BranchOp(successor));
                    }
                    case CoreOp.ConditionalBranchOp cbop -> {
                        Block trueBlock = cbop.trueBranch().targetBlock();
                        Block falseBlock = cbop.falseBranch().targetBlock();
                        Block.Reference spvTrueBlock = blockMap.get(trueBlock).successor();
                        Block.Reference spvFalseBlock = blockMap.get(falseBlock).successor();
                        spirvBlock.op(new SpirvOps.ConditionalBranchOp(spvTrueBlock, spvFalseBlock, mapOperands(cbop)));
                    }
                    default -> unsupported("op", op.getClass());
                }
            } //ops
        } // blocks
        return bodyBuilder;
    }

    private RuntimeException unsupported(String message, Object value) {
        return new RuntimeException("Unsupported " + message + ": " + value);
    }

    private static CoreOp.FuncOp lowerMethod(CoreOp.FuncOp fop) {
        CoreOp.FuncOp lfop = fop.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop)  {
                return lop.lower(block);
            }
            else {
                block.op(op);
                return block;
            }
        });
        return lfop;
    }

    private List<Value> mapOperands(Op op) {
        List<Value> operands = new ArrayList<>();
        for (Value javaValue : op.operands()) {
            Value spirvValue = valueMap.get(javaValue);
            assert spirvValue != null : "no value mapping from %s" + javaValue;
            operands.add(spirvValue);
        }
        return operands;
    }

    private boolean isIntegerType(TypeElement type) {
        return type.equals(JavaType.INT) || type.equals(JavaType.LONG);
    }

    private boolean isFloatType(TypeElement type) {
        return type.equals(JavaType.FLOAT) || type.equals(JavaType.DOUBLE);
    }
}