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

import java.util.Set;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;
import intel.code.spirv.SpirvOp.PhiOp;

public class TranslateToSpirvModel  {
    private Map<Block, Block.Builder> blockMap;    // Java block to spirv block builder
    private Map<Value, Value> valueMap;            // Java model Value to Spirv model Value

    public static SpirvOp.FuncOp translateFunction(CoreOp.FuncOp func) {
        CoreOp.FuncOp lowFunc = lowerMethod(func);
        TranslateToSpirvModel instance = new TranslateToSpirvModel();
        Body.Builder bodyBuilder = instance.translateBody(lowFunc.body(), lowFunc, null);
        SpirvOp.FuncOp tempFunc = new SpirvOp.FuncOp(lowFunc.funcName(), lowFunc.invokableType(), bodyBuilder);
        SpirvOp.FuncOp spirvFunc = instance.addPhiOps(tempFunc);
        return spirvFunc;
    }

    public TranslateToSpirvModel() {
        blockMap = new HashMap<>();
        valueMap = new HashMap<>();
    }

    private PhiMap buildPhiMap(SpirvOp.FuncOp func) {
        PhiMap phiMap = PhiMap.create(); // Map<Block, List<List<PhiMap.Predecessor>>>>
        // populate map with (predecessor, parameter value) pairs for each block parameter
        for (Block block : func.body().blocks()) {
            for (Op op : block.ops()) {
                switch (op) {
                    case SpirvOp.BranchOp bop: {
                        Block targetBlock = bop.branch().targetBlock();
                        for (int i = 0; i < bop.branch().arguments().size(); i++) {
                            Value arg = bop.branch().arguments().get(i);
                            phiMap.addPredecessor(targetBlock, i, new PhiMap.Predecessor(bop.parent(), arg));
                        }
                        break;
                    }
                    case SpirvOp.ConditionalBranchOp cbop: {
                        Block trueBlock = cbop.trueBranch().targetBlock();
                        Block falseBlock = cbop.falseBranch().targetBlock();
                        for (int i = 0; i < cbop.trueBranch().arguments().size(); i++) {
                            Value arg = cbop.trueBranch().arguments().get(i);
                            phiMap.addPredecessor(trueBlock, i, new PhiMap.Predecessor(cbop.parent(), arg));
                        }
                        for (int i = 0; i < cbop.falseBranch().arguments().size(); i++) {
                            Value arg = cbop.falseBranch().arguments().get(i);
                            phiMap.addPredecessor(falseBlock, i, new PhiMap.Predecessor(cbop.parent(), arg));
                        }
                        break;
                    }
                    default:
                }
            }
        }
        return phiMap;
    }

    private SpirvOp.FuncOp addPhiOps(SpirvOp.FuncOp func) {
        PhiMap phiMap = buildPhiMap(func);
        SpirvOp.FuncOp tfunc = func.transform((builder, op) -> {
            Block block = op.parent();
            if (phiMap.containsBlock(block)) {
                for (int i = 0; i < block.parameters().size(); i++) {
                    List<PhiMap.Predecessor> inPredecessors = phiMap.getPredecessors(block, i);
                    List<PhiOp.Predecessor> outPredecessors = new ArrayList<>();
                    Block.Parameter param = block.parameters().get(i);
                    for (PhiMap.Predecessor predecessor : inPredecessors) {
                        Block.Builder sourceBuilder = builder.context().getBlock(predecessor.block());
                        Block.Reference sourceRef = sourceBuilder.isEntryBlock() ? null : sourceBuilder.successor();
                        outPredecessors.add(new PhiOp.Predecessor(sourceRef, builder.context().getValue(predecessor.value())));
                    }
                    Op.Result phiResult = builder.op(new SpirvOp.PhiOp(param.type(), outPredecessors));
                    builder.context().mapValue(param, phiResult);
                }
                phiMap.removeBlock(block);
            }
            builder.op(op);
            return builder;
        });
        return tfunc;
    }

    private Body.Builder translateBody(Body body, Op parentOp, Body.Builder ancestorBody) {
        Body.Builder bodyBuilder = Body.Builder.of(ancestorBody, body.bodyType());
        Block.Builder spirvBlock = bodyBuilder.entryBlock();
        blockMap.put(body.entryBlock(), spirvBlock);
        List<Block> blocks = body.blocks();
        // map Java blocks to spirv blocks
        for (Block b : blocks.subList(1, blocks.size()))  {
            Block.Builder loweredBlock = spirvBlock.block();
            for (TypeElement paramType : b.parameterTypes()) {
                loweredBlock.parameter(paramType);
            }
            spirvBlock = loweredBlock;
            for (int i = 0; i < b.parameters().size(); i++) {
                Block.Parameter param = b.parameters().get(i);
                valueMap.put(param, spirvBlock.parameters().get(i));
            }
            blockMap.put(b, spirvBlock);
        }
        // map entry block parameters to spirv function parameter
        spirvBlock = bodyBuilder.entryBlock();
        List<SpirvOp> paramOps = new ArrayList<>();
        Block entryBlock = body.entryBlock();
        int paramCount = entryBlock.parameters().size();
        for (int i = 0; i < paramCount; i++) {
            Block.Parameter bp = entryBlock.parameters().get(i);
            assert entryBlock.ops().get(i) instanceof CoreOp.VarOp;
            SpirvOp funcParam = new SpirvOp.FunctionParameterOp(bp.type(), List.of());
            spirvBlock.op(funcParam);
            valueMap.put(bp, funcParam.result());
            paramOps.add(funcParam);
        }
        // emit all SpirvOp.VariableOps as first ops in entry block
        for (Block block : body.blocks()) {
            for (Op op : block.ops()) {
                if (op instanceof CoreOp.VarOp jvop) {
                    TypeElement resultType = new PointerType(jvop.varValueType(), StorageType.CROSSWORKGROUP);
                    SpirvOp.VariableOp svop = new SpirvOp.VariableOp((String)jvop.attributes().get(""), resultType, jvop.varValueType());
                    bodyBuilder.entryBlock().op(svop);
                    valueMap.put(jvop.result(), svop.result());
                }
            }
        }
        for (Block block : body.blocks()) {
            spirvBlock = blockMap.get(block);
            for (Op op : block.ops()) {
                switch (op) {
                    case CoreOp.VarOp vop -> {
                        Value dest = valueMap.get(vop.result());
                        Value value = valueMap.get(vop.operands().get(0));
                        // init variable here; declaration has been moved to top of function
                        SpirvOp.StoreOp store = new SpirvOp.StoreOp(dest, value);
                        spirvBlock.op(store);
                    }
                    case CoreOp.ReturnOp rop -> {
                        if (rop.operands().size() > 0) {
                            spirvBlock.op(new SpirvOp.ReturnValueOp(rop.resultType(), mapOperands(rop)));
                        }
                        else {
                            spirvBlock.op(new SpirvOp.ReturnOp(rop.resultType()));
                        }
                    }
                    case CoreOp.VarAccessOp.VarLoadOp vlo -> {
                        List<Value> operands = mapOperands(vlo);
                        Op.Result loadResult = spirvBlock.op(new SpirvOp.LoadOp(vlo.resultType(), operands));
                        valueMap.put(vlo.result(), loadResult);
                    }
                    case CoreOp.VarAccessOp.VarStoreOp vso -> {
                        Value dest = valueMap.get(vso.varOp().result());
                        Value value = valueMap.get(vso.operands().get(1));
                        spirvBlock.op(new SpirvOp.StoreOp(dest, value));
                    }
                    case CoreOp.ArrayAccessOp.ArrayLoadOp alo -> {
                        Value array = valueMap.get(alo.operands().get(0));
                        Value index = valueMap.get(alo.operands().get(1));
                        TypeElement arrayType = array.type();
                        SpirvOp.ConvertOp convert = new SpirvOp.ConvertOp(JavaType.type(long.class), List.of(index));
                        spirvBlock.op(new SpirvOp.LoadOp(arrayType, List.of(array)));
                        spirvBlock.op(convert);
                        SpirvOp ibac = new SpirvOp.InBoundsAccessChainOp(arrayType, List.of(array, convert.result()));
                        spirvBlock.op(ibac);
                        Op.Result loadResult = spirvBlock.op(new SpirvOp.LoadOp(alo.resultType(), List.of(ibac.result())));
                        valueMap.put(alo.result(), loadResult);
                    }
                    case CoreOp.ArrayAccessOp.ArrayStoreOp aso -> {
                        Value array = valueMap.get(aso.operands().get(0));
                        Value index = valueMap.get(aso.operands().get(1));
                        TypeElement arrayType = array.type();
                        SpirvOp ibac = new SpirvOp.InBoundsAccessChainOp(arrayType, List.of(array, index));
                        spirvBlock.op(ibac);
                        SpirvOp.StoreOp store = new SpirvOp.StoreOp(ibac.result(), valueMap.get(aso.operands().get(2)));
                        spirvBlock.op(store);
                    }
                    case CoreOp.ArrayLengthOp alo -> {
                        Op len = new SpirvOp.ArrayLengthOp(JavaType.INT, List.of(valueMap.get(alo.operands().get(0))));
                        spirvBlock.op(len);
                        valueMap.put(alo.result(), len.result());
                    }
                    case CoreOp.AndOp andop -> {
                        TypeElement type = andop.operands().get(0).type();
                        List<Value> operands = mapOperands(andop);
                        SpirvOp saop = new SpirvOp.BitwiseAndOp(type, operands);
                        spirvBlock.op(saop);
                        valueMap.put(andop.result(), saop.result());
                     }
                    case CoreOp.AddOp aop -> {
                        TypeElement type = aop.operands().get(0).type();
                        List<Value> operands = mapOperands(aop);
                        SpirvOp addOp;
                        if (isIntegerType(type)) addOp = new SpirvOp.IAddOp(type, operands);
                        else if (isFloatType(type)) addOp = new SpirvOp.FAddOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(addOp);
                        valueMap.put(aop.result(), addOp.result());
                     }
                    case CoreOp.SubOp sop -> {
                        TypeElement  type = sop.operands().get(0).type();
                        List<Value> operands = mapOperands(sop);
                        SpirvOp subOp;
                        if (isIntegerType(type)) subOp = new SpirvOp.ISubOp(type, operands);
                        else if (isFloatType(type)) subOp = new SpirvOp.FSubOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(subOp);
                        valueMap.put(sop.result(), subOp.result());
                     }
                    case CoreOp.MulOp mop -> {
                        TypeElement type = mop.operands().get(0).type();
                        List<Value> operands = mapOperands(mop);
                        SpirvOp mulOp;
                        if (isIntegerType(type)) mulOp = new SpirvOp.IMulOp(type, operands);
                        else if (isFloatType(type)) mulOp = new SpirvOp.FMulOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(mulOp);
                        valueMap.put(mop.result(), mulOp.result());
                    }
                    case CoreOp.DivOp dop -> {
                        TypeElement type = dop.operands().get(0).type();
                        List<Value> operands = mapOperands(dop);
                        SpirvOp divOp;
                        if (isIntegerType(type)) divOp = new SpirvOp.IDivOp(type, operands);
                        else if (isFloatType(type)) divOp = new SpirvOp.FDivOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(divOp);
                        valueMap.put(dop.result(), divOp.result());
                    }
                    case CoreOp.ModOp mop -> {
                        TypeElement type = mop.operands().get(0).type();
                        List<Value> operands = mapOperands(mop);
                        SpirvOp modOp = new SpirvOp.ModOp(type, operands);
                        spirvBlock.op(modOp);
                        valueMap.put(mop.result(), modOp.result());
                    }
                    case CoreOp.NegOp negop -> {
                        TypeElement type = negop.operands().get(0).type();
                        List<Value> operands = mapOperands(negop);
                        SpirvOp snegop;
                        if (isIntegerType(type)) snegop = new SpirvOp.SNegateOp(type, operands);
                        else if (isFloatType(type)) snegop = new SpirvOp.FNegateOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(snegop);
                        valueMap.put(negop.result(), snegop.result());
                    }
                    case CoreOp.EqOp eqop -> {
                        TypeElement type = eqop.operands().get(0).type();
                        List<Value> operands = mapOperands(eqop);
                        SpirvOp seqop;
                        if (isIntegerType(type)) seqop = new SpirvOp.IEqualOp(type, operands);
                        else if (isFloatType(type)) seqop = new SpirvOp.FEqualOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(seqop);
                        valueMap.put(eqop.result(), seqop.result());
                    }
                    case CoreOp.NeqOp neqop -> {
                        TypeElement type = neqop.operands().get(0).type();
                        List<Value> operands = mapOperands(neqop);
                        SpirvOp sneqop;
                        if (isIntegerType(type)) sneqop = new SpirvOp.INotEqualOp(type, operands);
                        else if (isFloatType(type)) sneqop = new SpirvOp.FNotEqualOp(type, operands);
                        else if (isObjectType(type)) sneqop = new SpirvOp.PtrNotEqualOp(type, operands);
                        else throw unsupported("type", type);
                        spirvBlock.op(sneqop);
                        valueMap.put(neqop.result(), sneqop.result());
                    }
                    case CoreOp.LtOp ltop -> {
                        TypeElement type = ltop.operands().get(0).type();
                        List<Value> operands = mapOperands(ltop);
                        SpirvOp sltop = new SpirvOp.LtOp(type, operands);
                        spirvBlock.op(sltop);
                        valueMap.put(ltop.result(), sltop.result());
                    }
                    case CoreOp.GtOp gtop -> {
                        TypeElement type = gtop.operands().get(0).type();
                        List<Value> operands = mapOperands(gtop);
                        SpirvOp sgtop = new SpirvOp.LtOp(type, operands);
                        spirvBlock.op(sgtop);
                        valueMap.put(gtop.result(), sgtop.result());
                    }
                    case CoreOp.GeOp geop -> {
                        TypeElement type = geop.operands().get(0).type();
                        List<Value> operands = mapOperands(geop);
                        SpirvOp sgeop = new SpirvOp.GeOp(type, operands);
                        spirvBlock.op(sgeop);
                        valueMap.put(geop.result(), sgeop.result());
                    }
                    case CoreOp.InvokeOp inv -> {
                        List<Value> operands = mapOperands(inv);
                        SpirvOp spirvCall = new SpirvOp.CallOp(inv.invokeDescriptor(), operands);
                        spirvBlock.op(spirvCall);
                        valueMap.put(inv.result(), spirvCall.result());
                    }
                    case CoreOp.ConstantOp cop -> {
                        SpirvOp scop = new SpirvOp.ConstantOp(cop.resultType(), cop.value());
                        spirvBlock.op(scop);
                        valueMap.put(cop.result(), scop.result());
                    }
                    case CoreOp.ConvOp cop -> {
                        List<Value> operands = mapOperands(cop);
                        SpirvOp scop = new SpirvOp.ConvertOp(cop.resultType(), operands);
                        spirvBlock.op(scop);
                        valueMap.put(cop.result(), scop.result());
                    }
                    case CoreOp.FieldAccessOp.FieldLoadOp flo -> {
                        SpirvOp load = new SpirvOp.FieldLoadOp(flo.resultType(), flo.fieldDescriptor(), mapOperands(flo));
                        spirvBlock.op(load);
                        valueMap.put(flo.result(), load.result());
                    }
                    case CoreOp.BranchOp bop -> {
                        Block branchBlock = bop.branch().targetBlock();
                        List<Value> sargs = new ArrayList<>();
                        for (Value arg : bop.branch().arguments()) {
                            sargs.add(valueMap.get(arg));
                        }
                        Block.Reference spvTargetBlock = blockMap.get(branchBlock).successor(sargs);
                        spirvBlock.op(new SpirvOp.BranchOp(spvTargetBlock));
                    }
                    case CoreOp.ConditionalBranchOp cbop -> {
                        Block trueBlock = cbop.trueBranch().targetBlock();
                        List<Value> targs = new ArrayList<>();
                        for (Value targ : cbop.trueBranch().arguments()) {
                            targs.add(valueMap.get(targ));
                        }
                        Block falseBlock = cbop.falseBranch().targetBlock();
                        List<Value> fargs = new ArrayList<>();
                        for (Value farg : cbop.falseBranch().arguments()) {
                            fargs.add(valueMap.get(farg));
                        }
                        Block.Reference spvTrueBlock = blockMap.get(trueBlock).successor(targs);
                        Block.Reference spvFalseBlock = blockMap.get(falseBlock).successor(fargs);
                        spirvBlock.op(new SpirvOp.ConditionalBranchOp(spvTrueBlock, spvFalseBlock, mapOperands(cbop)));
                    }
                    default -> throw unsupported("op", op.getClass());
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
            if (spirvValue == null) throw new RuntimeException("no value mapping from %s" + javaValue);
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

    private boolean isObjectType(TypeElement type) {
        // TODO: not correct
        return !isIntegerType(type) && !isFloatType(type);
    }

    private static class PhiMap {
        public static record Predecessor(Block block, Value value) {}

        private Map<Block, List<List<Predecessor>>> data;

        private PhiMap() {
            data = new HashMap<>();
        }

        public static PhiMap create() {
            return new PhiMap();
        }

        private void addBlock(Block block) {
            data.putIfAbsent(block, new ArrayList<>());
        }

        private void addParameter(Block block, int index) {
            addBlock(block);
            int paramCount = data.get(block).size();
            if (paramCount <= index) {
                data.get(block).add(new ArrayList<>());
            }
        }

        public void addPredecessor(Block block, int paramIndex, Predecessor predecessor) {
            addBlock(block);
            addParameter(block, paramIndex);
            data.get(block).get(paramIndex).add(predecessor);
        }

        public boolean containsBlock(Block block) {
            return data.containsKey(block);
        }

        public List<Predecessor> getPredecessors(Block block, int paramIndex) {
            return data.get(block).get(paramIndex);
        }

        public void removeBlock(Block block) {
            data.remove(block);
        }

        public String toString() {
            return data.toString();
        }
    }
}
