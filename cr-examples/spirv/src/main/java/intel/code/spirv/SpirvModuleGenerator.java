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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.function.Function;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.math.BigInteger;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.Vector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.FloatVector;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.JavaType;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVHeader;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVModule;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVFunction;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVBlock;
import uk.ac.manchester.beehivespirvtoolkit.lib.instructions.*;
import uk.ac.manchester.beehivespirvtoolkit.lib.instructions.operands.*;
import uk.ac.manchester.beehivespirvtoolkit.lib.disassembler.Disassembler;
import uk.ac.manchester.beehivespirvtoolkit.lib.disassembler.SPIRVDisassemblerOptions;
import uk.ac.manchester.beehivespirvtoolkit.lib.disassembler.SPVByteStreamReader;

public class SpirvModuleGenerator {
    public static MemorySegment generateModule(String moduleName, CoreOps.FuncOp func) {
        SpirvOps.FuncOp spirvFunc = TranslateToSpirvModel.translateFunction(func);
        MemorySegment module = SpirvModuleGenerator.generateModule(moduleName, spirvFunc);
        return module;
    }

    public static MemorySegment generateModule(String moduleName, SpirvOps.FuncOp func) {
        return new SpirvModuleGenerator().generateModuleInternal(moduleName, func);
    }

    public static void writeModuleToFile(MemorySegment module, String filepath)  {
        ByteBuffer buffer = module.asByteBuffer();
        File out = new File(filepath);
        try (FileChannel channel = new FileOutputStream(out, false).getChannel()) {
            channel.write(buffer);
        }
        catch (IOException e)  {
            throw new RuntimeException(e);
        }
    }

    public static String disassembleModule(MemorySegment module) {
        SPVByteStreamReader input = new SPVByteStreamReader(new ByteArrayInputStream(module.toArray(ValueLayout.JAVA_BYTE)));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (PrintStream ps = new PrintStream(out))  {
            SPIRVDisassemblerOptions options = new SPIRVDisassemblerOptions(false, false, false, false, true);
            Disassembler dis = new Disassembler(input, ps, options);
            dis.run();
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
        return new String(out.toByteArray());
    }

    private record SpirvResult(SPIRVId type, SPIRVId address, SPIRVId value) {}

    private final SPIRVModule module;
    private final Symbols symbols;

    private SpirvModuleGenerator() {
        this.module = new SPIRVModule(new SPIRVHeader(1, 2, 32, 0, 0));
        this.symbols = new Symbols();
    }

    private MemorySegment generateModuleInternal(String moduleName, SpirvOps.FuncOp func) {
        initModule();
        generateFunction(moduleName, moduleName, func);
        ByteBuffer buffer = ByteBuffer.allocateDirect(module.getByteCount());
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        module.close().write(buffer);
        buffer.flip();
        return MemorySegment.ofBuffer(buffer);
    }

    private void generateFunction(String moduleName, String fnName, SpirvOps.FuncOp func) {
        TypeElement returnType = func.invokableType().returnType();
        SPIRVId functionID = nextId(fnName);
        String signature = func.invokableType().returnType().toString();
        List<TypeElement> paramTypes = func.invokableType().parameterTypes();
        // build signature string
        for (int i = 0; i < paramTypes.size(); i++) {
            signature += "_" + paramTypes.get(i).toString();
        }
        // declare function type if not already present
        SPIRVId functionSig = getIdOrNull(signature);
        if (functionSig == null) {
            SPIRVId[] typeIdsArray = new SPIRVId[paramTypes.size()];
            for (int i = 0; i < paramTypes.size(); i++) {
                typeIdsArray[i] = spirvType(paramTypes.get(i).toString());
            }
            functionSig = nextId(fnName + "Signature");
            module.add(new SPIRVOpTypeFunction(functionSig, spirvType(returnType.toString()), new SPIRVMultipleOperands<>(typeIdsArray)));
            addId(signature, functionSig);
        }
        // declare function as modeule entry point
        SPIRVId spirvReturnType = spirvType(returnType.toString());
        SPIRVFunction function = (SPIRVFunction)module.add(new SPIRVOpFunction(spirvReturnType, functionID, SPIRVFunctionControl.DontInline(), functionSig));
        SPIRVOpLabel entryPoint = new SPIRVOpLabel(nextId());
        SPIRVBlock entryBlock = (SPIRVBlock)function.add(entryPoint);
        SPIRVMultipleOperands<SPIRVId> operands = new SPIRVMultipleOperands<>(getId("globalInvocationId"), getId("globalSize"), getId("subgroupSize"), getId("subgroupId"));
        module.add(new SPIRVOpEntryPoint(SPIRVExecutionModel.Kernel(), functionID, new SPIRVLiteralString(fnName), operands));

        translateBody(func.body(), function, entryBlock);
        function.add(new SPIRVOpFunctionEnd());
    }

    private void translateBody(Body body, SPIRVFunction function, SPIRVBlock entryBlock) {
        int labelNumber = 0;
        SPIRVBlock spirvBlock = entryBlock;
        for (int bi = 1; bi < body.blocks().size(); bi++)  {
            Block block = body.blocks().get(bi);
            String blockName = String.valueOf(block.hashCode());
            SPIRVOpLabel blockLabel = new SPIRVOpLabel(nextId());
            SPIRVBlock newBlock = (SPIRVBlock)function.add(blockLabel);
            symbols.putBlock(block, newBlock);
            symbols.putLabel(block, blockLabel);
        }
        for (Value param : body.entryBlock().parameters()) {
            SPIRVId paramId = nextId();
            addResult(param, new SpirvResult(spirvType(param.type().toString()), null, paramId));
        }
        for (int bi = 0; bi < body.blocks().size(); bi++)  {
            Block block = body.blocks().get(bi);
            if (bi > 0) {
                spirvBlock = symbols.getBlock(block);
            }
            List<Op> ops = block.ops();
            for (Op op : block.ops()) {
                // debug("---------- spirv op = %s", op.toText());
                switch (op)  {
                    case SpirvOps.VariableOp vop -> {
                        String typeName = vop.varType().toString();
                        SPIRVId type = spirvType(typeName);
                        SPIRVId varType = spirvVariableType(type);
                        SPIRVId var = nextId(vop.varName());
                        spirvBlock.add(new SPIRVOpVariable(varType, var, SPIRVStorageClass.Function(), new SPIRVOptionalOperand<>()));
                        addResult(vop.result(), new SpirvResult(varType, var, null));
                    }
                    case SpirvOps.FunctionParameterOp fpo -> {
                        SPIRVId result = nextId();
                        SPIRVId type = spirvType(fpo.resultType().toString());
                        function.add(new SPIRVOpFunctionParameter(type, result));
                        addResult(fpo.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOps.LoadOp lo -> {
                        if (((JavaType)lo.resultType()).equals(JavaType.type(VectorSpecies.class))) {
                            addResult(lo.result(), new SpirvResult(getType("int"), null, getConst("int_EIGHT")));
                        }
                        else {
                            SPIRVId type = spirvType(lo.resultType().toString());
                            SpirvResult toLoad = getResult(lo.operands().get(0));
                            SPIRVId varAddr = toLoad.address();
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(type, result, varAddr, align(type.getName())));
                            addResult(lo.result(), new SpirvResult(type, varAddr, result));
                        }
                    }
                    case SpirvOps.StoreOp so -> {
                        SpirvResult var = getResult(so.operands().get(0));
                        SPIRVId varAddr = var.address();
                        SPIRVId value = getResult(so.operands().get(1)).value();
                        spirvBlock.add(new SPIRVOpStore(varAddr, value, align(var.type().getName())));
                    }
                    case SpirvOps.IAddOp _, SpirvOps.FAddOp _ -> {
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId floatType = getType("float");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId ans = nextId();
                        if (lhsType == intType) spirvBlock.add(new SPIRVOpIAdd(intType, ans, lhs, rhs));
                        else if (lhsType == longType) spirvBlock.add(new SPIRVOpIAdd(longType, ans, lhs, rhs));
                        else if (lhsType == floatType) spirvBlock.add(new SPIRVOpFAdd(floatType, ans, lhs, rhs));
                        else unsupported("type", lhsType.getName());
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOps.IMulOp _, SpirvOps.FMulOp _, SpirvOps.IDivOp _, SpirvOps.FDivOp _ -> {
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId floatType = getType("float");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId rhsType = getResult(op.operands().get(1)).type();
                        SPIRVId ans = nextId();
                        if (lhsType == intType) {
                            if (op instanceof SpirvOps.IMulOp) spirvBlock.add(new SPIRVOpIMul(intType, ans, lhs, rhs));
                            else if (op instanceof SpirvOps.IDivOp) spirvBlock.add(new SPIRVOpSDiv(intType, ans, lhs, rhs));
                        }
                        else if (lhsType == longType) {
                            SPIRVId rhsId = rhsType == intType ? nextId() : rhs;
                            if (rhsType == intType) spirvBlock.add(new SPIRVOpSConvert(longType, rhsId, rhs));
                            if (op instanceof SpirvOps.IMulOp) spirvBlock.add(new SPIRVOpIMul(longType, ans, lhs, rhsId));
                            else if (op instanceof SpirvOps.IDivOp) spirvBlock.add(new SPIRVOpSDiv(longType, ans, lhs, rhs));
                        }
                        else if (lhsType == floatType) {
                            if (op instanceof SpirvOps.FMulOp) spirvBlock.add(new SPIRVOpFMul(floatType, ans, lhs, rhs));
                            else if (op instanceof SpirvOps.FDivOp) spirvBlock.add(new SPIRVOpFDiv(floatType, ans, lhs, rhs));
                        }
                        else unsupported("type", lhsType);
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOps.ModOp mop -> {
                        SPIRVId type = getType(mop.operands().get(0).type().toString());
                        SPIRVId lhs = getResult(mop.operands().get(0)).value();
                        SPIRVId rhs = getResult(mop.operands().get(1)).value();
                        SPIRVId result = nextId();
                        spirvBlock.add(new SPIRVOpUMod(type, result, lhs, rhs));
                        addResult(mop.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOps.IEqualOp eqop -> {
                        SPIRVId boolType = getType("bool");
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId floatType = getType("float");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId ans = nextId();
                        if (lhsType == intType) spirvBlock.add(new SPIRVOpIEqual(boolType, ans, lhs, rhs));
                        else if (lhsType == longType) spirvBlock.add(new SPIRVOpIEqual(boolType, ans, lhs, rhs));
                        else unsupported("type", lhsType.getName());
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOps.CallOp call -> {
                        if (call.callDescriptor().equals(MethodRef.ofString("spirvdemo.IntArray::get(long)int")) ||
                            call.callDescriptor().equals(MethodRef.ofString("spirvdemo.FloatArray::get(long)float"))) {
                            SPIRVId longType = getType("long");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 1;
                            SPIRVId index = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, resultAddr, array, index, new SPIRVMultipleOperands<>()));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(elementType, result, resultAddr, align(elementType.getName())));
                            addResult(call.result(), new SpirvResult(elementType, resultAddr, result));
                        }
                        else if (call.callDescriptor().equals(MethodRef.ofString("spirvdemo.IntArray::set(long, int)void")) ||
                                call.callDescriptor().equals(MethodRef.ofString("spirvdemo.FloatArray::set(long, float)void"))) {
                            SPIRVId longType = getType("long");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 2;
                            int valueIndex = nIndexes + 1;
                            SPIRVId index = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId dest = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, dest, array, index, new SPIRVMultipleOperands<>()));
                            SPIRVId value = getResult(call.operands().get(valueIndex)).value();
                            spirvBlock.add(new SPIRVOpStore(dest, value, align(elementType.getName())));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "fromArray", IntVector.class, VectorSpecies.class, int[].class, int.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "fromArray", FloatVector.class, VectorSpecies.class, float[].class, int.class))) {
                            SPIRVId oclExtension = getId("oclExtension");
                            SpirvResult speciesResult = getResult(call.operands().get(0));
                            SpirvResult arrayResult = getResult(call.operands().get(1));
                            String arrayType = arrayResult.type().getName();
                            int laneCount = 8;  //TODO: remove hard code, instruction below needs a literal
                            String vTypeName = ((JavaType)call.callDescriptor().refType()).toClassName();
                            SPIRVId vType = spirvVectorType(vTypeName, laneCount);
                            SPIRVId array = arrayResult.value();
                            SPIRVId index = getResult(call.operands().get(2)).value();
                            SPIRVId vectorIndex = nextId();
                            spirvBlock.add(new SPIRVOpSDiv(getType("int"), vectorIndex, index, speciesResult.value()));
                            SPIRVId longIndex = nextId();
                            spirvBlock.add(new SPIRVOpSConvert(getType("long"), longIndex, vectorIndex));
                            SPIRVId vector = nextId();
                            SPIRVMultipleOperands<SPIRVId> operands = new SPIRVMultipleOperands<>(longIndex, array, new SPIRVId(laneCount)); // TODO: lanes must be a literal
                            spirvBlock.add(new SPIRVOpExtInst(vType, vector, oclExtension, new SPIRVLiteralExtInstInteger(171, "vloadn"), operands));
                            addResult(call.result(), new SpirvResult(vType, null, vector));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "fromMemorySegment", IntVector.class, VectorSpecies.class, MemorySegment.class, long.class, ByteOrder.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "fromMemorySegment", FloatVector.class, VectorSpecies.class, MemorySegment.class, long.class, ByteOrder.class))) {
                            SPIRVId oclExtension = getId("oclExtension");
                            SPIRVId species = getResult(call.operands().get(0)).value();
                            SPIRVId lanesLong = nextId();
                            spirvBlock.add(new SPIRVOpSConvert(getType("long"), lanesLong, species));
                            int laneCount = 8; //TODO: remove hard code, vloadn instruction below needs a literal lane count, get value from env
                            SPIRVId segment = getResult(call.operands().get(1)).value();
                            String vTypeName = ((JavaType)call.callDescriptor().refType()).toClassName();
                            SPIRVId vType = spirvVectorType(vTypeName, laneCount);
                            SPIRVId temp = nextId();
                            spirvBlock.add(new SPIRVOpConvertPtrToU(getType("long"), temp, segment));
                            SPIRVId typedSegment = nextId();
                            SPIRVId pointerType = (SPIRVId)map(x -> x.equals(vTypeName), "jdk.incubator.vector.IntVector", "jdk.incubator.vector.FloatVector", getType("ptrInt"), getType("ptrFloat"));
                            spirvBlock.add(new SPIRVOpConvertUToPtr(pointerType, typedSegment, temp));
                            SPIRVId offset = getResult(call.operands().get(2)).value();
                            SPIRVId vectorIndex = nextId();
                            spirvBlock.add(new SPIRVOpSDiv(getType("long"), vectorIndex, offset, lanesLong)); // divide by lane count
                            SPIRVId finalIndex = nextId();
                            SPIRVId vector = nextId();
                            SPIRVMultipleOperands<SPIRVId> operands = new SPIRVMultipleOperands<>(vectorIndex, typedSegment, new SPIRVId(laneCount)); // TODO: lanes must be a literal
                            spirvBlock.add(new SPIRVOpExtInst(vType, vector, oclExtension, new SPIRVLiteralExtInstInteger(171, "vloadn"), operands));
                            addResult(call.result(), new SpirvResult(vType, null, vector));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "intoArray", void.class, int[].class, int.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "intoArray", void.class, float[].class, int.class))) {
                            SPIRVId oclExtension = getId("oclExtension");
                            SpirvResult vectorResult = getResult(call.operands().get(0));
                            SPIRVId vector = vectorResult.value();
                            SPIRVId vectorType = vectorResult.type();
                            SpirvResult arrayResult = getResult(call.operands().get(1));
                            SPIRVId array = arrayResult.value();
                            SPIRVId index = getResult(call.operands().get(2)).value();
                            SPIRVId vectorIndex = nextId();
                            spirvBlock.add(new SPIRVOpShiftRightArithmetic(getType("int"), vectorIndex, index, vectorExponent(vectorType.getName())));
                            SPIRVId longIndex = nextId();
                            spirvBlock.add(new SPIRVOpSConvert(getType("long"), longIndex, vectorIndex));
                            SPIRVMultipleOperands<SPIRVId> operandsR = new SPIRVMultipleOperands<>(vector, longIndex, array);
                            spirvBlock.add(new SPIRVOpExtInst(getType("void"), nextId(), oclExtension, new SPIRVLiteralExtInstInteger(172, "vstoren"), operandsR));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "intoMemorySegment", void.class, MemorySegment.class, long.class, ByteOrder.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "intoMemorySegment", void.class, MemorySegment.class, long.class, ByteOrder.class))) {
                            SPIRVId oclExtension = getId("oclExtension");
                            SpirvResult vectorResult = getResult(call.operands().get(0));
                            SPIRVId vector = vectorResult.value();
                            SPIRVId vectorType = vectorResult.type();
                            SpirvResult segmentResult = getResult(call.operands().get(1));;
                            SPIRVId segment = segmentResult.value();
                            SPIRVId temp = nextId();
                            spirvBlock.add(new SPIRVOpConvertPtrToU(getType("long"), temp, segment));
                            SPIRVId typedSegment = nextId();
                            String vectorElementType = vectorElementType(vectorType).getName();
                            SPIRVId pointerType = (SPIRVId)map(x -> x.equals(vectorElementType), "int", "float", getType("ptrInt"), getType("ptrFloat"));
                            spirvBlock.add(new SPIRVOpConvertUToPtr(pointerType, typedSegment, temp));
                            SPIRVId offset = getResult(call.operands().get(2)).value();
                            SPIRVId vectorIndex = nextId();
                            int laneCount = laneCount(vectorType.getName());
                            spirvBlock.add(new SPIRVOpShiftRightArithmetic(getType("long"), vectorIndex, offset, vectorExponent(vectorType.getName())));
                            SPIRVMultipleOperands<SPIRVId> operandsR = new SPIRVMultipleOperands<>(vector, vectorIndex, typedSegment);
                            spirvBlock.add(new SPIRVOpExtInst(getId("void"), nextId(), oclExtension, new SPIRVLiteralExtInstInteger(172, "vstoren"), operandsR));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "reduceLanes", int.class, VectorOperators.Associative.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "reduceLanes", float.class, VectorOperators.Associative.class))) {
                            SpirvResult vectorResult = getResult(call.operands().get(0));
                            SPIRVId vectorType = vectorResult.type();
                            SPIRVId vector = vectorResult.value();
                            String vTypeName = vectorType.getName();
                            SPIRVId elementType = vectorElementType(vectorType);
                            Op reduceOp = ((Op.Result)call.operands().get(1)).op();
                            if (reduceOp instanceof SpirvOps.FieldLoadOp flo) {
                                assert flo.fieldDescriptor().refType().equals(JavaType.type(VectorOperators.class));
                                assert flo.fieldDescriptor().name().equals("ADD");
                                String operation = flo.fieldDescriptor().name();
                            }
                            else unsupported("operation expression", reduceOp.toText());
                            String tempTag = nextTempTag();
                            SPIRVId temp_0 = nextId(tempTag + 0);
                            spirvBlock.add(new SPIRVOpCompositeExtract(elementType, temp_0, vector, new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(0))));
                            for (int lane = 1; lane < laneCount(vectorType.getName()); lane++) {
                                SPIRVId temp = nextId(tempTag + lane);
                                SPIRVId element = nextId();
                                spirvBlock.add(new SPIRVOpCompositeExtract(elementType, element, vector, new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(lane))));
                                if (elementType == getType("int")) {
                                    spirvBlock.add(new SPIRVOpIAdd(elementType, temp, getId(tempTag + (lane - 1)), element));
                                }
                                else if (elementType == getType("float")) {
                                    spirvBlock.add(new SPIRVOpFAdd(elementType, temp, getId(tempTag + (lane - 1)), element));
                                }
                                else unsupported("type", elementType.getName());
                            }
                            addResult(call.result(), new SpirvResult(elementType, null, getId(tempTag + (laneCount(vectorType.getName()) - 1))));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "add", IntVector.class, Vector.class))
                              || call.callDescriptor().equals(MethodRef.method(IntVector.class, "mul", IntVector.class, Vector.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "add", FloatVector.class, Vector.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "mul", FloatVector.class, Vector.class))) {
                            SPIRVId oclExtension = getId("oclExtension");
                            SpirvResult lhsResult = getResult(call.operands().get(0));
                            SPIRVId lhsType = lhsResult.type();
                            SPIRVId lhs = lhsResult.value();
                            SPIRVId rhs = getResult(call.operands().get(1)).value();
                            SPIRVId add = nextId();
                            if (call.callDescriptor().name().equals("add")) {
                                spirvBlock.add(lhsType.getName().endsWith("int") ? new SPIRVOpIAdd(lhsType, add, lhs, rhs) : new SPIRVOpFAdd(lhsType, add, lhs, rhs));
                            }
                            else if (call.callDescriptor().name().equals("mul")) {
                                spirvBlock.add(lhsType.getName().endsWith("int") ? new SPIRVOpIMul(lhsType, add, lhs, rhs) : new SPIRVOpFMul(lhsType, add, lhs, rhs));
                            }
                            addResult(call.result(), new SpirvResult(lhsType, null, add));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(FloatVector.class, "fma", FloatVector.class, Vector.class, Vector.class))) {
                            SPIRVId oclExtension = getId("oclExtension");
                            SpirvResult aResult = getResult(call.operands().get(0));
                            SPIRVId vType = aResult.type();
                            SPIRVId a = aResult.value();
                            SPIRVId b = getResult(call.operands().get(1)).value();
                            SPIRVId c = getResult(call.operands().get(2)).value();
                            String vTypeStr = vType.getName();
                            assert vTypeStr.endsWith("float");
                            SPIRVId result  = nextId();
                            SPIRVMultipleOperands<SPIRVId> operands = new SPIRVMultipleOperands<>(a, b, c);
                            spirvBlock.add(new SPIRVOpExtInst(vType, result, oclExtension, new SPIRVLiteralExtInstInteger(26, "fma"), operands));
                            addResult(call.result(), new SpirvResult(vType, null, result));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "zero", IntVector.class, VectorSpecies.class))
                             || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "zero", FloatVector.class, VectorSpecies.class))) {
                            SpirvResult speciesResult = getResult(call.operands().get(0));
                            SPIRVId vType = spirvType(((JavaType)call.callDescriptor().refType()).toClassName());
                            String elementType = vectorElementType(vType).getName();
                            SPIRVId value = getId(elementType + "_ZERO");
                            int laneCount = laneCount(vType.getName());
                            assert laneCount == 8 || laneCount == 16;
                            SPIRVId vector = nextId();
                            SPIRVMultipleOperands<SPIRVId> operands = spirvOperands(value, laneCount);
                            spirvBlock.add(new SPIRVOpCompositeConstruct(vType, vector, operands));
                            addResult(call.result(), new SpirvResult(vType, null, vector));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(IntVector.class, "lane", int.class, int.class))
                              || call.callDescriptor().equals(MethodRef.method(FloatVector.class, "lane", float.class, int.class)))  {
                            SpirvResult lhsResult = getResult(call.operands().get(0));
                            SPIRVId lhsType = lhsResult.type();
                            SPIRVId lhs = lhsResult.value();
                            String vTypeStr = lhsType.getName();
                            SPIRVId vType = lhsResult.type();
                            SPIRVId elementType = vectorElementType(vType);
                            SPIRVId result = nextId();
                            Op laneOp = ((Op.Result)call.operands().get(1)).op();
                            assert laneOp instanceof SpirvOps.ConstantOp;
                            int lane = (int)((SpirvOps.ConstantOp)laneOp).value();
                            spirvBlock.add(new SPIRVOpCompositeExtract(elementType, result, lhsResult.value(), new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(lane))));
                            addResult(call.result(), new SpirvResult(elementType, null, result));
                        }
                        else if (call.callDescriptor().equals(MethodRef.method(VectorSpecies.class, "length", int.class))) {
                            addResult(call.result(), new SpirvResult(getType("int"), null, getConst("int_EIGHT"))); // TODO: remove hardcode
                        }
                        else unsupported("method", call.callDescriptor());
                    }
                    case SpirvOps.ConstantOp cop -> {
                        SPIRVId type = spirvType(cop.resultType().toString());
                        SPIRVId result = nextId();
                        Object value = cop.value();
                        if (type == getType("int")) {
                            module.add(new SPIRVOpConstant(type, result, new SPIRVContextDependentInt(new BigInteger(String.valueOf(value)))));
                        }
                        else if (type == getType("long")) {
                            module.add(new SPIRVOpConstant(type, result, new SPIRVContextDependentLong(new BigInteger(String.valueOf(value)))));
                        }
                        else if (type == getType("float")) {
                            module.add(new SPIRVOpConstant(type, result, new SPIRVContextDependentFloat((float)value)));
                        }
                        else unsupported("type", cop.resultType());
                        addResult(cop.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOps.ConvertOp scop -> {
                        SPIRVId toType = spirvType(scop.resultType().toString());
                        SPIRVId to = nextId();
                        SpirvResult valueResult = getResult(scop.operands().get(0));
                        SPIRVId from = valueResult.value();
                        SPIRVId fromType = valueResult.type();
                        if (isIntegerType(fromType)) {
                            if (isIntegerType(toType)) {
                                spirvBlock.add(new SPIRVOpSConvert(toType, to, from));
                            }
                            else if (isFloatType(toType)) {
                                spirvBlock.add(new SPIRVOpConvertSToF(toType, to, from));
                            }
                            else unsupported("conversion type", scop.resultType());
                        }
                        else unsupported("conversion type", scop.operands().get(0));
                        addResult(scop.result(), new SpirvResult(toType, null, to));
                    }
                    case SpirvOps.InBoundAccessChainOp iacop -> {
                        SPIRVId type = spirvType(iacop.resultType().toString());
                        SPIRVId result = nextId();
                        SPIRVId object = getResult(iacop.operands().get(0)).value();
                        SPIRVId index = getResult(iacop.operands().get(1)).value();
                        spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(type, result, object, index, new SPIRVMultipleOperands<>()));
                        addResult(iacop.result(), new SpirvResult(type, result, null));
                    }
                    case SpirvOps.FieldLoadOp flo -> {
                        if (flo.operands().size() > 0 && flo.operands().get(0).type().equals(JavaType.ofString("spirvdemo.GPU$Index"))) {
                            SpirvResult result;
                            int group = -1;
                            int index = -1;
                            String fieldName = flo.fieldDescriptor().name();
                            switch(fieldName) {
                                case "x": group = 0; index = 0; break;
                                case "y": group = 0; index = 1; break;
                                case "z": group = 0; index = 2; break;
                                case "w": group = 1; index = 0; break;
                                case "h": group = 1; index = 1; break;
                                case "d": group = 1; index = 2; break;
                            }
                            switch (group) {
                                case 0: result = globalId(index, spirvBlock); break;
                                case 1: result = globalSize(index, spirvBlock); break;
                                default: throw new RuntimeException("Unknown Index field: " + fieldName);
                            }
                            addResult(flo.result(), result);
                        }
                        else if (((JavaType)flo.resultType()).equals(JavaType.type(VectorSpecies.class))) {
                            addResult(flo.result(), new SpirvResult(getType("int"), null, getConst("int_EIGHT")));
                        }
                        else if (flo.fieldDescriptor().refType().equals(JavaType.type(VectorOperators.class))) {
                            // currently ignored
                        }
                        else if (flo.fieldDescriptor().refType().equals(JavaType.type(ByteOrder.class))) {
                            // currently ignored
                        }
                        else unsupported("field load", ((JavaType)flo.fieldDescriptor().refType()).toClassName() + "." + flo.fieldDescriptor().name());
                    }
                    case SpirvOps.BranchOp bop -> {
                        SPIRVId trueLabel = symbols.getLabel(bop.branch()).getResultId();
                        spirvBlock.add(new SPIRVOpBranch(trueLabel));
                    }
                    case SpirvOps.ConditionalBranchOp cbop -> {
                        SPIRVId test = getResult(cbop.operands().get(0)).value();
                        SPIRVId trueLabel = symbols.getLabel(cbop.trueBranch()).getResultId();
                        SPIRVId falseLabel = symbols.getLabel(cbop.falseBranch()).getResultId();
                        spirvBlock.add(new SPIRVOpBranchConditional(test, trueLabel, falseLabel, new SPIRVMultipleOperands<SPIRVLiteralInteger>()));
                    }
                    case SpirvOps.LtOp ltop -> {
                        SPIRVId lhs = getResult(ltop.operands().get(0)).value();
                        SPIRVId rhs = getResult(ltop.operands().get(1)).value();
                        SPIRVId boolType = getType("bool");
                        SPIRVId result = nextId();
                        spirvBlock.add(new SPIRVOpSLessThan(boolType, result, lhs, rhs));
                        addResult(ltop.result(), new SpirvResult(boolType, null, result));
                    }
                    case SpirvOps.ReturnOp rop -> {
                        if (rop.operands().size() == 0) {
                            spirvBlock.add(new SPIRVOpReturn());
                        }
                        else {
                            SPIRVId returnValue = getResult(rop.operands().get(0)).value();
                            spirvBlock.add(new SPIRVOpReturnValue(returnValue));
                        }
                    }
                    default -> unsupported("op", op.getClass());
                }
            }
        }
    }

    private void initModule() {
        module.add(new SPIRVOpCapability(SPIRVCapability.Addresses()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Linkage()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Kernel()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Int8()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Int16()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Int64()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Vector16()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Float16()));
        module.add(new SPIRVOpMemoryModel(SPIRVAddressingModel.Physical64(), SPIRVMemoryModel.OpenCL()));

        // OpenCL extension provides built-in variables suitable for kernel programming
        // Import extention and declare fourn variables
        SPIRVId oclExtension = nextId("oclExtension");
        module.add(new SPIRVOpExtInstImport(oclExtension, new SPIRVLiteralString("OpenCL.std")));

        SPIRVId globalInvocationId = nextId("globalInvocationId");
        SPIRVId globalSize = nextId("globalSize");
        SPIRVId subgroupSize = nextId("subgroupSize");
        SPIRVId subgroupId = nextId("subgroupId");

        module.add(new SPIRVOpDecorate(globalInvocationId, SPIRVDecoration.BuiltIn(SPIRVBuiltIn.GlobalInvocationId())));
        module.add(new SPIRVOpDecorate(globalInvocationId, SPIRVDecoration.Constant()));
        module.add(new SPIRVOpDecorate(globalInvocationId, SPIRVDecoration.LinkageAttributes(new SPIRVLiteralString("spirv_BuiltInGlobalInvocationId"), SPIRVLinkageType.Import())));
        module.add(new SPIRVOpDecorate(globalSize, SPIRVDecoration.BuiltIn(SPIRVBuiltIn.GlobalSize())));
        module.add(new SPIRVOpDecorate(globalSize, SPIRVDecoration.Constant()));
        module.add(new SPIRVOpDecorate(globalSize, SPIRVDecoration.LinkageAttributes(new SPIRVLiteralString("spirv_BuiltInGlobalSize"), SPIRVLinkageType.Import())));
        module.add(new SPIRVOpDecorate(subgroupSize, SPIRVDecoration.BuiltIn(SPIRVBuiltIn.SubgroupSize())));
        module.add(new SPIRVOpDecorate(subgroupSize, SPIRVDecoration.Constant()));
        module.add(new SPIRVOpDecorate(subgroupSize, SPIRVDecoration.LinkageAttributes(new SPIRVLiteralString("spirv_BuiltInSubgroupSize"), SPIRVLinkageType.Import())));
        module.add(new SPIRVOpDecorate(subgroupId, SPIRVDecoration.BuiltIn(SPIRVBuiltIn.SubgroupId())));
        module.add(new SPIRVOpDecorate(subgroupId, SPIRVDecoration.Constant()));
        module.add(new SPIRVOpDecorate(subgroupId, SPIRVDecoration.LinkageAttributes(new SPIRVLiteralString("spirv_BuiltInSubgroupId"), SPIRVLinkageType.Import())));

        module.add(new SPIRVOpVariable(getType("ptrV3long"), globalInvocationId, SPIRVStorageClass.Input(), new SPIRVOptionalOperand<>()));
        module.add(new SPIRVOpVariable(getType("ptrV3long"), globalSize, SPIRVStorageClass.Input(), new SPIRVOptionalOperand<>()));
        module.add(new SPIRVOpVariable(getType("ptrV3long"), subgroupSize, SPIRVStorageClass.Input(), new SPIRVOptionalOperand<>()));
        module.add(new SPIRVOpVariable(getType("ptrV3long"), subgroupId, SPIRVStorageClass.Input(), new SPIRVOptionalOperand<>()));
    }

    private SPIRVId spirvType(String javaType) {
        SPIRVId ans = switch(javaType) {
            case "byte" -> getType("byte");
            case "short" -> getType("short");
            case "int" -> getType("int");
            case "long" -> getType("long");
            case "float" -> getType("float");
            case "double" -> getType("double");
            case "int[]" -> getType("int[]");
            case "float[]" -> getType("float[]");
            case "double[]" -> getType("double[]");
            case "long[]" -> getType("long[]");
            case "bool" -> getType("bool");
            case "spirvdemo.IntArray" -> getType("int[]");
            case "spirvdemo.FloatArray" -> getType("float[]");
            case "jdk.incubator.vector.IntVector" -> spirvVectorType("IntVector", 8);
            case "jdk.incubator.vector.FloatVector" -> spirvVectorType("FloatVector", 8);
            case "jdk.incubator.vector.VectorSpecies<java.lang.Integer>" -> getType("int");
            case "jdk.incubator.vector.VectorSpecies<java.lang.Long>" -> getType("long");
            case "jdk.incubator.vector.VectorSpecies<java.lang.Float>" -> getType("int");
            case "VectorSpecies" -> getType("int");
            case "void" -> getType("void");
            case "spirvdemo.GPU$Index" -> getType("ptrGPUIndex");
            case "java.lang.foreign.MemorySegment" -> getType("ptrByte");
            default -> null;
        };
        if (ans == null) unsupported("type", javaType);
        return ans;
    }

    private SPIRVId spirvElementType(String javaType) {
        SPIRVId ans = switch(javaType) {
            case "byte[]" -> getType("byte");
            case "short[]" -> getType("short");
            case "int[]" -> getType("int");
            case "long[]" -> getType("long");
            case "float[]" -> getType("float");
            case "double[]" -> getType("double");
            case "boolean[]" -> getType("bool");
            case "spirvdemo.IntArray" -> getType("int");
            case "spirvdemo.FloatArray" -> getType("float");
            case "jdk.incubator.vector.LongVector" -> getType("long");
            case "jdk.incubator.vector.FloatVector" -> getType("float");
            case "IntVector" -> getType("int");
            case "LongVector" -> getType("long");
            case "FloatVector" -> getType("float");
            case "java.lang.foreign.MemorySegment" -> getType("byte");
            default -> null;
        };
        if (ans == null) unsupported("type", javaType);
        return ans;
    }

    private SPIRVId vectorElementType(SPIRVId type) {
        SPIRVId ans = switch(type.getName()) {
            case "v8int" -> getType("int");
            case "v16int" -> getType("int");
            case "v8long" -> getType("long");
            case "v8float" -> getType("float");
            case "v16float" -> getType("float");
            default -> null;
        };
        if (ans == null) unsupported("type", type.getName());
        return ans;
    }

    private SPIRVId spirvVariableType(SPIRVId spirvType) {
        SPIRVId ans = switch(spirvType.getName()) {
            case "byte" -> getType("ptrByte");
            case "short" -> getType("ptrShort");
            case "int" -> getType("ptrInt");
            case "long" -> getType("ptrLong");
            case "float" -> getType("ptrFloat");
            case "double" -> getType("ptrDouble");
            case "boolean" -> getType("ptrBool");
            case "int[]" -> getType("ptrInt[]");
            case "long[]" -> getType("ptrLong[]");
            case "float[]" -> getType("ptrFloat[]");
            case "double[]" -> getType("ptrDouble[]");
            case "v8int" -> getType("ptrV8int");
            case "v16int" -> getType("ptrV16int");
            case "v8long" -> getType("ptrV8long");
            case "v8float" -> getType("ptrV8float");
            case "v16float" -> getType("ptrV16float");
            case "ptrGPUIndex" -> getType("ptrPtrGPUIndex");
            case "ptrByte" -> getType("ptrPtrByte");
            default -> null;
        };
        if (ans == null) unsupported("type", spirvType.getName());
        return ans;
    }

    private SPIRVId spirvVectorType(String javaVectorType, int vectorLength) {
        String prefix = "v" + vectorLength;
        String elementType = spirvElementType(javaVectorType).getName();
        return getType(prefix + elementType);
    }

    private int alignment(String spirvType) {
        int ans = switch(spirvType) {
            case "byte" -> 1;
            case "short" -> 2;
            case "int" -> 4;
            case "long" -> 8;
            case "float" -> 4;
            case "double" -> 8;
            case "boolean" -> 1;
            case "v8int" -> 32;
            case "v16int" -> 64;
            case "v8long" -> 64;
            case "v8float" -> 32;
            case "v16float" -> 64;
            case "ptrGPUIndex" -> 32;
            case "int[]" -> 8;
            case "long[]" -> 8;
            case "float[]" -> 8;
            case "double[]" -> 8;
            case "ptrByte" -> 8;
            case "ptrInt" -> 8;
            case "ptrInt[]" -> 8;
            case "ptrLong" -> 8;
            case "ptrLong[]" -> 8;
            case "ptrFloat" -> 8;
            case "ptrFloat[]" -> 8;
            case "ptrV8int" -> 8;
            case "ptrV8float" -> 8;
            case "ptrPtrGPUIndex" -> 8;
            default -> 0;
        };
        if (ans == 0) unsupported("type", spirvType);
        return ans;
    }

    private int laneCount(String vectorType) {
        int ans = switch(vectorType) {
            case "v8int" -> 8;
            case "v8long" -> 8;
            case "v8float" -> 8;
            case "v16int" -> 16;
            case "v16float" -> 16;
            default -> 0;
        };
        if (ans == 0) unsupported("type", vectorType);
        return ans;
    }

    private SPIRVId vectorExponent(String vectorType) {
        SPIRVId ans = null;
        switch(vectorType) {
            case "v8int" -> ans = getId("int_THREE");
            case "v8long" -> ans = getId("int_THREE");
            case "v8float" -> ans = getId("int_THREE");
            case "v16int" -> ans = getId("int_FOUR");
            case "v16float" -> ans = getId("int_FOUR");
            default -> unsupported("type", vectorType);
        };
        return ans;
    }

    private Set<String> moduleTypes = new HashSet<>();

    private SPIRVId getType(String name) {
        if (!moduleTypes.contains(name)) {
            switch (name) {
                case "void" -> module.add(new SPIRVOpTypeVoid(nextId(name)));
                case "bool" -> module.add(new SPIRVOpTypeBool(nextId(name)));
                case "byte" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(8), new SPIRVLiteralInteger(0)));
                case "short" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(16), new SPIRVLiteralInteger(0)));
                case "int" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(32), new SPIRVLiteralInteger(0)));
                case "long" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(64), new SPIRVLiteralInteger(0)));
                case "float" -> module.add(new SPIRVOpTypeFloat(nextId(name), new SPIRVLiteralInteger(32)));
                case "double" -> module.add(new SPIRVOpTypeFloat(nextId(name), new SPIRVLiteralInteger(64)));
                case "ptrByte" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("byte")));
                case "ptrInt" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("int")));
                case "ptrLong" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("long")));
                case "ptrFloat" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("float")));
                case "short[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("short")));
                case "int[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("int")));
                case "long[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("long")));
                case "float[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("float")));
                case "double[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("double")));
                case "boolean[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("boolean")));
                case "ptrInt[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("int[]")));
                case "ptrLong[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("long[]")));
                case "ptrFloat[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("float[]")));
                case "spirvdemo.GPUIndex" -> module.add(new SPIRVOpTypeStruct(nextId(name), new SPIRVMultipleOperands<>(getType("long"), getType("long"), getType("long"))));
                case "ptrGPUIndex" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("spirvdemo.GPUIndex")));
                case "ptrCrossGroupByte"-> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("byte")));
                case "ptrPtrGPUIndex" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrGPUIndex")));
                case "ptrPtrByte" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrByte")));
                case "v3long" -> module.add(new SPIRVOpTypeVector(nextId(name), getType("long"), new SPIRVLiteralInteger(3)));
                case "v8int" -> module.add(new SPIRVOpTypeVector(nextId(name), getType("int"), new SPIRVLiteralInteger(8)));
                case "v8long" -> module.add(new SPIRVOpTypeVector(nextId(name), getType("long"), new SPIRVLiteralInteger(8)));
                case "v16int" -> module.add(new SPIRVOpTypeVector(nextId(name), getType("int"), new SPIRVLiteralInteger(16)));
                case "v8float" -> module.add(new SPIRVOpTypeVector(nextId(name), getType("float"), new SPIRVLiteralInteger(8)));
                case "v16float" -> module.add(new SPIRVOpTypeVector(nextId(name), getType("float"), new SPIRVLiteralInteger(16)));
                case "ptrV3long" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Input(), getType("v3long")));
                case "ptrV8long" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("v8long")));
                case "ptrV8int" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("v8int")));
                case "ptrV16int" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("v16int")));
                case "ptrV8float" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("v8float")));
                case "ptrV16float" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("v16float")));
                case "ptrPtrV8int" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrV8int")));
                case "ptrPtrV16int" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrV16int")));
                case "ptrPtrV8float" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrV8float")));
                case "ptrPtrV16float" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrV16float")));
                default -> unsupported("type", name);
            }
            moduleTypes.add(name);
        }
        return getId(name);
    }

    private Set<String> moduleConstants = new HashSet<>();

    private SPIRVId getConst(String name) {
        if (!moduleConstants.contains(name)) {
            switch (name) {
                case "int_ZERO" -> module.add(new SPIRVOpConstant(getType("int"), nextId("int_ZERO"), new SPIRVContextDependentInt(new BigInteger("0"))));
                case "int_ONE" -> module.add(new SPIRVOpConstant(getType("int"), nextId("int_ONE"), new SPIRVContextDependentInt(new BigInteger("1"))));
                case "int_TWO" -> module.add(new SPIRVOpConstant(getType("int"), nextId("int_TWO"), new SPIRVContextDependentInt(new BigInteger("2"))));
                case "int_EIGHT" -> module.add(new SPIRVOpConstant(getType("int"), nextId("int_EIGHT"), new SPIRVContextDependentInt(new BigInteger("8"))));
                default -> unsupported("constant", name);
            }
            moduleConstants.add(name);
        }
        return getId(name);
    }

    private SPIRVOptionalOperand<SPIRVMemoryAccess> align(int align) {
        return new SPIRVOptionalOperand<>(SPIRVMemoryAccess.Aligned(new SPIRVLiteralInteger(align)));
    }

    private SPIRVOptionalOperand<SPIRVMemoryAccess> align(String type) {
        return align(alignment(type));
    }

    private SPIRVMultipleOperands<SPIRVId> spirvOperands(SPIRVId value, int count) {
        SPIRVId[] values = new SPIRVId[count];
        Arrays.fill(values, value);
        return new SPIRVMultipleOperands<>(values);
    }

    private SPIRVOptionalOperand<SPIRVMemoryAccess> none() {
        return new SPIRVOptionalOperand<>();
    }

    private SpirvResult globalSize(int index, SPIRVBlock spirvBlock) {
        SPIRVId longType = getType("long");
        SPIRVId v3long = getId("v3long");
        SPIRVId globalSizeId = getId("globalSize");
        SPIRVId globalSizes = nextId();
        spirvBlock.add(new SPIRVOpLoad(v3long, globalSizes, globalSizeId, align(32)));
        SPIRVId globalSize = nextId();
        spirvBlock.add(new SPIRVOpCompositeExtract(longType, globalSize, globalSizes, new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(index))));
        return new SpirvResult(longType, null, globalSize);
    }

    private SpirvResult globalId(int index, SPIRVBlock spirvBlock) {
        SPIRVId longType = getType("long");
        SPIRVId v3long = getId("v3long");
        SPIRVId globalInvocationId = getId("globalInvocationId");
        SPIRVId globalIds = nextId();
        spirvBlock.add(new SPIRVOpLoad(v3long, globalIds, globalInvocationId, align(32)));
        SPIRVId globalIndex = nextId();
        spirvBlock.add(new SPIRVOpCompositeExtract(longType, globalIndex, globalIds, new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(index))));
        return new SpirvResult(longType, null, globalIndex);
    }

    private SPIRVId nextId() {
        return module.getNextId();
    }

    private SPIRVId nextId(String name) {
        SPIRVId ans = nextId();
        ans.setName(name);
        symbols.putId(name, ans);
        module.add(new SPIRVOpName(ans, new SPIRVLiteralString(name)));
        return ans;
    }

    private static int counter = 0;

    private String nextTempTag() {
        counter++;
        return "temp_" + counter + "_";
    }

    private boolean isIntegerType(SPIRVId type) {
        String name = type.getName();
        return name.equals("short") || name.equals("int") || name.equals("long");
    }

    private boolean isFloatType(SPIRVId type) {
        String name = type.getName();
        return name.equals("float") || name.equals("double");
    }

    private boolean isVectorSpecies(String javaType) {
        return javaType.equals("VectorSpecies");
    }

    private boolean isVectorType(String javaType) {
        return javaType.equals("IntVector") || javaType.equals("FloatVector");
    }

    private void addId(String name, SPIRVId id) {
        symbols.putId(name, id);
    }

    private SPIRVId getId(String name) {
        SPIRVId ans = symbols.getId(name);
        assert ans != null : name + " not found";
        return ans;
    }

    private SPIRVId getIdOrNull(String name) {
        return symbols.getId(name);
    }

    private static Object map(Function<Object, Boolean> test, Object... args) {
        int len = args.length;
        assert len >= 2 && len % 2 == 0;
        int pairs = len / 2;
        for (int i = 0; i < pairs; i++) {
            if (test.apply(args[i])) return args[i + pairs];
        }
        throw new RuntimeException("No match: " + args[0]);
    }

    private void unsupported(String message, Object value) {
        throw new RuntimeException("Unsupported " + message + ": " + value);
    }

    private void addResult(Value value, SpirvResult result) {
        assert symbols.getResult(value) == null : "result already present";
        symbols.putResult(value, result);
    }

    private SpirvResult getResult(Value value) {
        return symbols.getResult(value);
    }

    private static class Symbols {
        private final HashMap<Value, SpirvResult> results;
        private final HashMap<String, SPIRVId> ids;
        private final HashMap<Block, SPIRVBlock> blocks;
        private final HashMap<Block, SPIRVOpLabel> labels;

        public Symbols() {
            this.results = new HashMap<>();
            this.ids = new HashMap<>();
            this.blocks = new HashMap<>();
            this.labels = new HashMap<>();
        }

        public void putId(String name, SPIRVId id) {
            ids.put(name, id);
        }

        public SPIRVId getId(String name) {
            return ids.get(name);
        }

        public void putBlock(Block block, SPIRVBlock spirvBlock) {
            blocks.put(block, spirvBlock);
        }

        public SPIRVBlock getBlock(Block block) {
            return blocks.get(block);
        }

        public void putLabel(Block block, SPIRVOpLabel spirvLabel) {
            labels.put(block, spirvLabel);
        }

        public SPIRVOpLabel getLabel(Block block) {
            return labels.get(block);
        }

        public void putResult(Value value, SpirvResult result) {
            results.put(value, result);
        }

        public SpirvResult getResult(Value value) {
            return results.get(value);
        }

        public String toString() {
            return String.format("results %s\n\nids %s\n\nblocks %s\nlabels %s\n", results.keySet(), ids.keySet(), blocks.keySet(), labels.keySet());
        }
    }
}