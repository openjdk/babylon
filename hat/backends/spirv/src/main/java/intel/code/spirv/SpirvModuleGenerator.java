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
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Optional;
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
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.type.ClassType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.FunctionType;
import hat.util.StreamCounter;
import hat.buffer.Buffer;
import hat.callgraph.CallGraph;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpWrapper;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVHeader;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVModule;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVFunction;
import uk.ac.manchester.beehivespirvtoolkit.lib.SPIRVBlock;
import uk.ac.manchester.beehivespirvtoolkit.lib.instructions.*;
import uk.ac.manchester.beehivespirvtoolkit.lib.instructions.operands.*;
import uk.ac.manchester.beehivespirvtoolkit.lib.disassembler.Disassembler;
import uk.ac.manchester.beehivespirvtoolkit.lib.disassembler.SPIRVDisassemblerOptions;
import uk.ac.manchester.beehivespirvtoolkit.lib.disassembler.SPVByteStreamReader;
import intel.code.spirv.SpirvOp.PhiOp;

public class SpirvModuleGenerator {
    private final String moduleName;
    private final SPIRVModule module;
    private final Symbols symbols;
    // map of class name to map of field name to field index
    private final HashMap<String, HashMap<String, Integer>> classMap = new HashMap<>();
    // map of class name to size of the class
    private final HashMap<String, Integer> sizeMap = new HashMap<>();

    public static SpirvModuleGenerator create(String moduleName) {
        return new SpirvModuleGenerator(moduleName);
    }

    public static MemorySegment generateModule(String moduleName, KernelCallGraph callGraph, Object... args) {
        SpirvModuleGenerator generator = SpirvModuleGenerator.create(moduleName);

        generator.generateTypeDeclaration(args);
        generator.generateDependentFunctions(callGraph);
        KernelEntrypoint kernelEntrypoint = callGraph.entrypoint;
        CoreOp.FuncOp funcOp = kernelEntrypoint.funcOpWrapper().op();
        String kernelName = funcOp.funcName();
        SpirvOp.FuncOp spirvFunc = TranslateToSpirvModel.translateFunction(funcOp);
        generator.generateFunction(funcOp.funcName(), spirvFunc, true);
        return generator.finalizeModule();
    }

    public static MemorySegment generateModule(String moduleName, CoreOp.FuncOp func) {
        SpirvOp.FuncOp spirvFunc = TranslateToSpirvModel.translateFunction(func);
        MemorySegment moduleSegment = SpirvModuleGenerator.generateModule(moduleName, spirvFunc);
        return moduleSegment;
    }

    public static MemorySegment generateModule(String moduleName, SpirvOp.FuncOp func) {
        SpirvModuleGenerator generator = new SpirvModuleGenerator(moduleName);
        MemorySegment moduleSegment = generator.generateModuleInternal(func);
        return moduleSegment;
    }

    public static void writeModuleToFile(MemorySegment module, String filepath)  {
        ByteBuffer buffer = module.asByteBuffer();
        File out = new File(filepath);
        try (FileChannel channel = new FileOutputStream(out, false).getChannel()) {
            channel.write(buffer);
            channel.close();
        }
        catch (IOException e)  {
            throw new RuntimeException(e);
        }
    }

    private static void writeModuleToFile(SPIRVModule module, String filepath)
    {
        ByteBuffer buffer = ByteBuffer.allocate(module.getByteCount());
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        module.close().write(buffer);
        buffer.flip();
        File out = new File(filepath);
        try (FileChannel channel = new FileOutputStream(out, false).getChannel())
        {
            channel.write(buffer);
        }
        catch (IOException e)
        {
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

    public MemorySegment finalizeModule() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(module.getByteCount());
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        module.close().write(buffer);
        buffer.flip();
        return MemorySegment.ofBuffer(buffer);
    }

    private record SpirvResult(SPIRVId type, SPIRVId address, SPIRVId value) {}

    private SpirvModuleGenerator(String moduleName) {
        this.moduleName = moduleName;
        this.module = new SPIRVModule(new SPIRVHeader(1, 2, 32, 0, 0));
        this.symbols = new Symbols();
        initModule();
    }

    private MemorySegment generateModuleInternal(SpirvOp.FuncOp func) {
        generateFunction(moduleName, func, true);
        return finalizeModule();
    }

    private int getTypeSize(String typeName) {
        if (sizeMap.containsKey(typeName)) {
            return sizeMap.get(typeName);
        }
        switch (typeName) {
            case "byte" -> {
                return 1;
            }
            case "boolean" -> {
                return 1;
            }
            case "short" -> {
                return 2;
            }
            case "int" -> {
                return 4;
            }
            case "long" -> {
                return 8;
            }
            case "float" -> {
                return 4;
            }
            case "double" -> {
                return 8;
            }
            default -> {
                throw new IllegalStateException("unknown type " + typeName);
            }
        }
    }

    private void addTypeToModule(String name, SPIRVId typeIdsArray[]) {
        String upperName = name.substring(0, 1).toUpperCase() + name.substring(1);
        String ptrUpperName = "ptr" + upperName;
        String ptrPtrUpperName = "ptrPtr" + upperName;
        module.add(new SPIRVOpTypeStruct(nextId(name), new SPIRVMultipleOperands<>(typeIdsArray)));
        moduleTypes.add(name);
        module.add(new SPIRVOpTypePointer(nextId(ptrUpperName), SPIRVStorageClass.CrossWorkgroup(), getType(name)));
        moduleTypes.add(ptrUpperName);
        module.add(new SPIRVOpTypePointer(nextId(ptrPtrUpperName), SPIRVStorageClass.CrossWorkgroup(), getType(ptrUpperName)));
        moduleTypes.add(ptrPtrUpperName);
    }

    private void addArrayToModule(String name, String typeName, int len) {
        SPIRVOpConstant constant = new SPIRVOpConstant(getType("int"), nextId(), new SPIRVContextDependentInt(new BigInteger(String.valueOf(len))));
        module.add(constant);
        SPIRVOpTypeArray typeArray = new SPIRVOpTypeArray(nextId(name + "Array"), getType(typeName), constant.getResultId());
        module.add(typeArray);
        moduleTypes.add(name + "Array");
        SPIRVOpTypePointer ptrTypeArray = new SPIRVOpTypePointer(nextId("ptr" + name + "Array"), SPIRVStorageClass.CrossWorkgroup(), getType(name + "Array"));
        module.add(ptrTypeArray);
        moduleTypes.add("ptr" + name + "Array");
    }

    private void generateTypeDeclaration(Object... args) {
        Arrays.stream(args)
            .filter(arg -> arg instanceof Buffer)
            .map(arg -> (Buffer) arg)
            .forEach(ifaceBuffer -> {
                BoundSchema<?> boundSchema = Buffer.getBoundSchema(ifaceBuffer);
                boundSchema.schema().rootIfaceType.visitTypes(0, t -> {
                    int fieldCount = t.fields.size();
                    List<Object[]> typesNames = new ArrayList<>();
                    int[] count = new int[]{0};
                    classMap.put(t.iface.getCanonicalName(), new HashMap<String, Integer>());
                    StreamCounter.of(t.fields, (c, field) -> {
                        boolean isLast = c.value() == fieldCount - 1;
                        if (field instanceof Schema.FieldNode.AbstractPrimitiveField primitiveField) {
                            if (primitiveField instanceof Schema.FieldNode.PrimitiveArray array) {
                                int arrayLen;
                                if (array instanceof Schema.FieldNode.PrimitiveFieldControlledArray fieldControlledArray) {
                                    int[] len = new int[]{0};
                                    if (isLast && t.parent == null) {
                                        len[0] = 1;
                                    } else {
                                        boolean[] done = new boolean[]{false};
                                        boundSchema.boundArrayFields().forEach(a -> {
                                            if (a.field.equals(array)) {
                                                len[0] = a.len;
                                                done[0] = true;
                                            }
                                        });
                                        if (!done[0]) {
                                            throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                        }
                                    }
                                    arrayLen = len[0];
                                } else if (array instanceof Schema.FieldNode.PrimitiveFixedArray fixed) {
                                    arrayLen = fixed.len;
                                } else {
                                    throw new IllegalStateException("what kind of array ");
                                }
                                addArrayToModule(primitiveField.name, primitiveField.type.getCanonicalName(), arrayLen);
                                int fieldSize = getTypeSize(primitiveField.type.getCanonicalName()) * arrayLen;
                                typesNames.add(new Object[]{primitiveField.name + "Array", fieldSize});
                                classMap.get(t.iface.getCanonicalName()).put(primitiveField.name, count[0]);
                                sizeMap.put(t.iface.getCanonicalName(), fieldSize);
                                count[0]++;
                            } else {
                                int fieldSize = getTypeSize(primitiveField.type.getCanonicalName());
                                typesNames.add(new Object[]{primitiveField.type.getCanonicalName(), fieldSize});
                                classMap.get(t.iface.getCanonicalName()).put(primitiveField.name, count[0]);
                                sizeMap.put(t.iface.getCanonicalName(), fieldSize);
                                count[0]++;
                            }
                        } else if (field instanceof Schema.FieldNode.AbstractIfaceField ifaceField) {
                            if (ifaceField instanceof Schema.FieldNode.IfaceArray array) {
                                int arrayLen;
                                if (array instanceof Schema.FieldNode.IfaceFieldControlledArray fieldControlledArray) {          
                                    int[] len = new int[]{0};
                                    if (isLast && t.parent == null) {
                                        len[0] = 1;
                                    } else {
                                        boolean[] done = new boolean[]{false};
                                        boundSchema.boundArrayFields().forEach(a -> {
                                            if (a.field.equals(ifaceField)) {
                                                len[0] = a.len;
                                                done[0] = true;
                                            }
                                        });
                                        if (!done[0]) {
                                            throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                        }
                                    }
                                    arrayLen = len[0];
                                } else if (array instanceof Schema.FieldNode.IfaceFixedArray fixed) {
                                    arrayLen = fixed.len;
                                } else {
                                    throw new IllegalStateException("what kind of array ");
                                }
                                addArrayToModule(ifaceField.ifaceType.iface.getSimpleName(), ifaceField.ifaceType.iface.getCanonicalName(), arrayLen);
                                int fieldSize = getTypeSize(ifaceField.ifaceType.iface.getCanonicalName()) * arrayLen;
                                typesNames.add(new Object[]{ ifaceField.ifaceType.iface.getSimpleName() + "Array", fieldSize});
                                classMap.get(t.iface.getCanonicalName()).put(ifaceField.name, count[0]);
                                sizeMap.put(t.iface.getCanonicalName(), fieldSize);
                                count[0]++;
                            } else {
                                int fieldSize = getTypeSize(ifaceField.ifaceType.iface.getCanonicalName());
                                typesNames.add(new Object[]{ifaceField.ifaceType.iface.getCanonicalName(), fieldSize});
                                classMap.get(t.iface.getCanonicalName()).put(ifaceField.name, count[0]);
                                sizeMap.put(ifaceField.ifaceType.iface.getCanonicalName(), fieldSize);
                                count[0]++;
                            }
                        } else if (field instanceof Schema.SchemaNode.Padding) {
                            // SKIP
                            System.out.println("Padding ");
                        } else {
                            throw new IllegalStateException("hmm");
                        }
                    }
                );
                String name = t.iface.getCanonicalName();
                SPIRVId[] typeIdsArray;
                if (Buffer.Struct.class.isAssignableFrom(t.iface) || Buffer.class.isAssignableFrom(t.iface)) {
                    // struct
                    typeIdsArray = new SPIRVId[count[0]];
                    for (int i = 0; i < count[0]; i++) {
                        SPIRVId typeId = getType((String)typesNames.get(i)[0]);
                        typeIdsArray[i] = typeId;
                    }
                    addTypeToModule(name, typeIdsArray);
                } else {
                    // union
                    typeIdsArray = new SPIRVId[1];
                    SPIRVId typeId = getType("int");
                    int maxTypeSize = 0;
                    for (int i = 0; i < count[0]; i++) {
                        SPIRVId currentTypeId = getType((String)typesNames.get(i)[0]);
                        int currentTypeSize = (int)typesNames.get(i)[1];
                        if (currentTypeSize > maxTypeSize) {
                            typeId = currentTypeId;
                        }
                    }
                    typeIdsArray[0] = typeId;
                    addTypeToModule(name, typeIdsArray);
                    HashMap<String, Integer> map = classMap.get(t.iface.getCanonicalName());
                    // only one field in union to make sure size is correct
                    for (Map.Entry<String, Integer> entry : map.entrySet()) {
                        map.put(entry.getKey(), 0);
                    }
                }
            });
        });
    }

    private void generateDependentFunctions(KernelCallGraph callGraph) {
        for (KernelCallGraph.KernelReachableResolvedMethodCall call : callGraph.kernelReachableResolvedStream().sorted((lhs, rhs) -> rhs.rank - lhs.rank).toList()) {
            if (call.targetMethodRef != null) {
                try {
                    FuncOpWrapper calledFunc = call.funcOpWrapper();
                    FuncOpWrapper loweredFunc = calledFunc.lower();
                    CoreOp.FuncOp fo = loweredFunc.op();
                    SpirvOp.FuncOp spirvFunc = TranslateToSpirvModel.translateFunction(fo);
                    SPIRVId fnId = generateFunction(fo.funcName(), spirvFunc, false);
                    symbols.putId(call.targetMethodRef.toString(), fnId);
                } catch (Exception e) {
                    Throwable cause = e;
                    while (cause.getCause() != null) {
                        cause = cause.getCause();
                    }
                    cause.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        }
    }

    private SPIRVId generateFunction(String fnName, SpirvOp.FuncOp func, boolean isEntryPoint) {
        TypeElement returnType = func.invokableType().returnType();
        SPIRVId functionId = nextId(fnName);
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
        SPIRVId spirvReturnType = spirvType(returnType.toString());
        SPIRVFunction function = (SPIRVFunction)module.add(new SPIRVOpFunction(spirvReturnType, functionId, SPIRVFunctionControl.DontInline(), functionSig));
        SPIRVOpLabel entryLabel = new SPIRVOpLabel(nextId());
        symbols.putLabel(func.body().entryBlock(), entryLabel);
        SPIRVBlock entryBlock = (SPIRVBlock)function.add(entryLabel);
        SPIRVMultipleOperands<SPIRVId> operands = new SPIRVMultipleOperands<>(getId("globalInvocationId"), getId("globalSize"), getId("subgroupSize"), getId("subgroupId"));
        if (isEntryPoint) {
            module.add(new SPIRVOpEntryPoint(SPIRVExecutionModel.Kernel(), functionId, new SPIRVLiteralString(fnName), operands));
        }
        translateBody(func.body(), function, entryBlock);
        function.add(new SPIRVOpFunctionEnd());
        return functionId;
    }

    private void translateBody(Body body, SPIRVFunction function, SPIRVBlock entryBlock) {
        int labelNumber = 0;
        SPIRVBlock spirvBlock = entryBlock;
        for (int bi = 1; bi < body.blocks().size(); bi++)  {
            Block block = body.blocks().get(bi);
            SPIRVOpLabel blockLabel = new SPIRVOpLabel(nextId());
            SPIRVBlock newBlock = (SPIRVBlock)function.add(blockLabel);
            symbols.putBlock(block, newBlock);
            symbols.putLabel(block, blockLabel);
            for (int i = 0; i < block.parameters().size(); i++) {
                Value param = block.parameters().get(i);
                SPIRVId paramId = nextId();
                addResult(param, new SpirvResult(spirvType(param.type().toString()), null, paramId));
            }
        }
        for (int bi = 0; bi < body.blocks().size(); bi++)  {
            Block block = body.blocks().get(bi);
            if (bi > 0) {
                spirvBlock = symbols.getBlock(block);
            }
            for (Op op : block.ops()) {
                switch (op)  {
                    case SpirvOp.PhiOp phop -> {
                        List<PhiOp.Predecessor> inPredecessors = phop.predecessors();
                        SPIRVPairIdRefIdRef[] outPredecessors = new SPIRVPairIdRefIdRef[inPredecessors.size()];
                        for (int i = 0; i < inPredecessors.size(); i++) {
                            PhiOp.Predecessor predecessor = inPredecessors.get(i);
                            SPIRVId label;
                            if (predecessor.block() == null) {
                                // This is the entry block
                                label = symbols.getLabel(body.entryBlock()).getResultId();
                            } else {
                                label = symbols.getLabel(predecessor.block().targetBlock()).getResultId();
                            }
                            SPIRVId value = getResult(predecessor.value()).value();
                            outPredecessors[i] = new SPIRVPairIdRefIdRef(value, label);
                        }
                        SPIRVId result = nextId();
                        SPIRVId type = spirvType(phop.resultType().toString());
                        SPIRVOpPhi phiOp = new SPIRVOpPhi(spirvType(phop.resultType().toString()), result, new SPIRVMultipleOperands<>(outPredecessors));
                        spirvBlock.add(phiOp);
                        addResult(phop.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOp.VariableOp vop -> {
                        String typeName = vop.varType().toString();
                        SPIRVId type = spirvType(typeName);
                        SPIRVId varType = spirvVariableType(type);
                        SPIRVId var = nextId(vop.varName());
                        spirvBlock.add(new SPIRVOpVariable(varType, var, SPIRVStorageClass.Function(), new SPIRVOptionalOperand<>()));
                        addResult(vop.result(), new SpirvResult(varType, var, null));
                    }
                    case SpirvOp.FunctionParameterOp fpo -> {
                        SPIRVId result = nextId();
                        SPIRVId type = spirvType(fpo.resultType().toString());
                        function.add(new SPIRVOpFunctionParameter(type, result));
                        module.add(new SPIRVOpDecorate(result, SPIRVDecoration.Alignment(new SPIRVLiteralInteger(8))));
                        addResult(fpo.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOp.LoadOp lo -> {
                        SPIRVId type = spirvType(lo.resultType().toString());
                        SpirvResult toLoad = getResult(lo.operands().get(0));
                        SPIRVId varAddr = toLoad.address();
                        SPIRVId result = nextId();
                        spirvBlock.add(new SPIRVOpLoad(type, result, varAddr, align(type.getName())));
                        addResult(lo.result(), new SpirvResult(type, varAddr, result));
                    }
                    case SpirvOp.StoreOp so -> {
                        SpirvResult var = getResult(so.operands().get(0));
                        SPIRVId varAddr = var.address();
                        SPIRVId value = getResult(so.operands().get(1)).value();
                        spirvBlock.add(new SPIRVOpStore(varAddr, value, align(var.type().getName())));
                    }
                    case SpirvOp.IAddOp _, SpirvOp.FAddOp _ -> {
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId floatType = getType("float");
                        SPIRVId doubleType = getType("double");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId ans = nextId();
                        if (lhsType == intType) spirvBlock.add(new SPIRVOpIAdd(intType, ans, lhs, rhs));
                        else if (lhsType == longType) spirvBlock.add(new SPIRVOpIAdd(longType, ans, lhs, rhs));
                        else if (lhsType == floatType) spirvBlock.add(new SPIRVOpFAdd(floatType, ans, lhs, rhs));
                        else if (lhsType == doubleType) spirvBlock.add(new SPIRVOpFAdd(doubleType, ans, lhs, rhs));
                        else unsupported("type", lhsType.getName());
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOp.ISubOp _, SpirvOp.FSubOp _ -> {
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId floatType = getType("float");
                        SPIRVId doubleType = getType("double");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId ans = nextId();
                        if (lhsType == intType) spirvBlock.add(new SPIRVOpISub(intType, ans, lhs, rhs));
                        else if (lhsType == longType) spirvBlock.add(new SPIRVOpISub(longType, ans, lhs, rhs));
                        else if (lhsType == floatType) spirvBlock.add(new SPIRVOpFSub(floatType, ans, lhs, rhs));
                        else if (lhsType == doubleType) spirvBlock.add(new SPIRVOpFSub(doubleType, ans, lhs, rhs));
                        else unsupported("type", lhsType.getName());
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOp.IMulOp _, SpirvOp.FMulOp _, SpirvOp.IDivOp _, SpirvOp.FDivOp _ -> {
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId floatType = getType("float");
                        SPIRVId doubleType = getType("double");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId rhsType = getResult(op.operands().get(1)).type();
                        SPIRVId ans = nextId();
                        if (lhsType == intType) {
                            if (op instanceof SpirvOp.IMulOp) spirvBlock.add(new SPIRVOpIMul(intType, ans, lhs, rhs));
                            else if (op instanceof SpirvOp.IDivOp) spirvBlock.add(new SPIRVOpSDiv(intType, ans, lhs, rhs));
                        }
                        else if (lhsType == longType) {
                            SPIRVId rhsId = rhsType == intType ? nextId() : rhs;
                            if (rhsType == intType) spirvBlock.add(new SPIRVOpSConvert(longType, rhsId, rhs));
                            if (op instanceof SpirvOp.IMulOp) spirvBlock.add(new SPIRVOpIMul(longType, ans, lhs, rhsId));
                            else if (op instanceof SpirvOp.IDivOp) spirvBlock.add(new SPIRVOpSDiv(longType, ans, lhs, rhs));
                        }
                        else if (lhsType == floatType) {
                            if (op instanceof SpirvOp.FMulOp) spirvBlock.add(new SPIRVOpFMul(floatType, ans, lhs, rhs));
                            else if (op instanceof SpirvOp.FDivOp) spirvBlock.add(new SPIRVOpFDiv(floatType, ans, lhs, rhs));
                        }
                        else if (lhsType == doubleType) {
                            if (op instanceof SpirvOp.FMulOp) spirvBlock.add(new SPIRVOpFMul(doubleType, ans, lhs, rhs));
                            else if (op instanceof SpirvOp.FDivOp) spirvBlock.add(new SPIRVOpFDiv(doubleType, ans, lhs, rhs));
                        }
                        else unsupported("type", lhsType.getName());
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOp.ModOp mop -> {
                        SPIRVId type = getType(mop.operands().get(0).type().toString());
                        SPIRVId lhs = getResult(mop.operands().get(0)).value();
                        SPIRVId rhs = getResult(mop.operands().get(1)).value();
                        SPIRVId result = nextId();
                        spirvBlock.add(new SPIRVOpUMod(type, result, lhs, rhs));
                        addResult(mop.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOp.IEqualOp eqop -> {
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
                    case SpirvOp.AshrOp ashop -> {
                        SPIRVId intType = getType("int");
                        SPIRVId longType = getType("long");
                        SPIRVId lhs = getResult(ashop.operands().get(0)).value();
                        SPIRVId rhs = getResult(ashop.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(ashop.resultType().toString());
                        SPIRVId ans = nextId();
                        if (lhsType == intType) spirvBlock.add(new SPIRVOpShiftRightArithmetic(intType, ans, lhs, rhs));
                        else if (lhsType == longType) spirvBlock.add(new SPIRVOpShiftRightArithmetic(longType, ans, lhs, rhs));
                        else unsupported("type", lhsType.getName());
                        addResult(ashop.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOp.GeOp eqop -> {
                        SPIRVId boolType = getType("bool");
                        SPIRVId lhs = getResult(op.operands().get(0)).value();
                        SPIRVId rhs = getResult(op.operands().get(1)).value();
                        SPIRVId lhsType = spirvType(op.resultType().toString());
                        SPIRVId ans = nextId();
                        spirvBlock.add(new SPIRVOpSGreaterThanEqual(boolType, ans, lhs, rhs));
                        addResult(op.result(), new SpirvResult(lhsType, null, ans));
                    }
                    case SpirvOp.PtrNotEqualOp neqop -> {
                        SPIRVId boolType = getType("bool");
                        SPIRVId longType = getType("long");
                        SPIRVId lhs = getResult(neqop.operands().get(0)).value();
                        SPIRVId rhs = getResult(neqop.operands().get(1)).value();
                        SPIRVId ans = nextId();
                        SPIRVId lhsLong = nextId();
                        SPIRVId rhsLong = nextId();
                        spirvBlock.add(new SPIRVOpConvertPtrToU(longType, lhsLong, lhs));
                        spirvBlock.add(new SPIRVOpConvertPtrToU(longType, rhsLong, rhs));
                        spirvBlock.add(new SPIRVOpINotEqual(boolType, ans, lhsLong, rhsLong));
                        addResult(op.result(), new SpirvResult(boolType, null, ans));
                    }
                    case SpirvOp.CallOp call -> {
                        MethodRef methodRef = call.callDescriptor();
                        if (methodRef.equals(MethodRef.ofString("hat.buffer.S32Array::array(long)int")) ||
                            methodRef.equals(MethodRef.ofString("hat.buffer.F32Array::array(long)float")))
                        {
                            SPIRVId longType = getType("long");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 1;
                            SPIRVId indexX = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId temp1 = nextId();
                            SPIRVId temp2 = nextId();
                            spirvBlock.add(new SPIRVOpConvertPtrToU(longType, temp1, array));
                            spirvBlock.add(new SPIRVOpIAdd(longType, temp2, temp1, getConst("long", 8)));
                            SPIRVId elementBase = nextId();
                            spirvBlock.add(new SPIRVOpConvertUToPtr(arrayType, elementBase, temp2));
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, resultAddr, elementBase, indexX, new SPIRVMultipleOperands<>()));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(elementType, result, resultAddr, align(elementType.getName())));
                            addResult(call.result(), new SpirvResult(elementType, resultAddr, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("hat.buffer.S32Array2D::array(long)int")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.F32Array2D::array(long)float")))
                        {
                            SPIRVId longType = getType("long");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 1;
                            SPIRVId indexX = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            SPIRVId temp1 = nextId();
                            SPIRVId temp2 = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            spirvBlock.add(new SPIRVOpConvertPtrToU(longType, temp1, array));
                            spirvBlock.add(new SPIRVOpIAdd(longType, temp2, temp1, getConst("long", 8)));
                            SPIRVId elementBase = nextId();
                            spirvBlock.add(new SPIRVOpConvertUToPtr(arrayType, elementBase, temp2));
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, resultAddr, elementBase, indexX, new SPIRVMultipleOperands<>()));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(elementType, result, resultAddr, align(elementType.getName())));
                            addResult(call.result(), new SpirvResult(elementType, resultAddr, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("hat.buffer.S32Array::array(long, int)void")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.S32Array::array(long, float)void")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.F32Array::array(long, int)void")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.F32Array::array(long, float)void"))) {
                            SPIRVId longType = getType("long");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 2;
                            int valueIndex = nIndexes + 1;
                            SPIRVId indexX = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId temp1 = nextId();
                            SPIRVId temp2 = nextId();
                            spirvBlock.add(new SPIRVOpConvertPtrToU(longType, temp1, array));
                            spirvBlock.add(new SPIRVOpIAdd(longType, temp2, temp1, getConst("long", 8)));
                            SPIRVId elementBase = nextId();
                            spirvBlock.add(new SPIRVOpConvertUToPtr(arrayType, elementBase, temp2));
                            SPIRVId dest = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, dest, elementBase, indexX, new SPIRVMultipleOperands<>()));
                            SPIRVId value = getResult(call.operands().get(valueIndex)).value();
                            spirvBlock.add(new SPIRVOpStore(dest, value, align(elementType.getName())));
                        }
                        else if (methodRef.equals(MethodRef.ofString("hat.buffer.S32Array2D::array(long, int)void")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.F32Array2D::array(long, float)void"))) {
                            SPIRVId longType = getType("long");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 2;
                            int valueIndex = nIndexes + 1;
                            SPIRVId indexX = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId temp1 = nextId();
                            SPIRVId temp2 = nextId();
                            spirvBlock.add(new SPIRVOpConvertPtrToU(longType, temp1, array));
                            spirvBlock.add(new SPIRVOpIAdd(longType, temp2, temp1, getConst("long", 8)));
                            SPIRVId elementBase = nextId();
                            spirvBlock.add(new SPIRVOpConvertUToPtr(arrayType, elementBase, temp2));
                            SPIRVId dest = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, dest, elementBase, indexX, new SPIRVMultipleOperands<>()));
                            SPIRVId value = getResult(call.operands().get(valueIndex)).value();
                            spirvBlock.add(new SPIRVOpStore(dest, value, align(elementType.getName())));
                        }
                        else if (methodRef.equals(MethodRef.ofString("hat.buffer.S32Array::length()int")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.S32Array2D::width()int"))||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.F32Array2D::width()int"))) {
                            SPIRVId intType = getType("int");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, resultAddr, array, getConst("int", 0), new SPIRVMultipleOperands<>()));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(intType, result, resultAddr, align(arrayType.getName())));
                            addResult(call.result(), new SpirvResult(intType, resultAddr, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("hat.buffer.S32Array2D::height()int")) ||
                                 methodRef.equals(MethodRef.ofString("hat.buffer.F32Array2D::height()int"))) {
                            SPIRVId intType = getType("int");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, resultAddr, array, getConst("int", 1), new SPIRVMultipleOperands<>()));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(intType, result, resultAddr, align(arrayType.getName())));
                            addResult(call.result(), new SpirvResult(intType, resultAddr, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("hat.buffer.S08x3RGBImage::data(long)byte"))) {
                            SPIRVId longType = getType("long");
                            SPIRVId byteType = getType("byte");
                            String arrayTypeName = call.operands().get(0).type().toString();
                            SpirvResult arrayResult = getResult(call.operands().get(0));
                            SPIRVId arrayAddr = arrayResult.address();
                            SPIRVId arrayType = spirvType(arrayTypeName);
                            SPIRVId elementType = spirvElementType(arrayTypeName);
                            int nIndexes = call.operands().size() - 1;
                            SPIRVId indexX = getResult(call.operands().get(1)).value();
                            SPIRVId array = nextId();
                            spirvBlock.add(new SPIRVOpLoad(arrayType, array, arrayAddr, align(arrayType.getName())));
                            SPIRVId temp1 = nextId();
                            SPIRVId temp2 = nextId();
                            spirvBlock.add(new SPIRVOpConvertPtrToU(longType, temp1, array));
                            spirvBlock.add(new SPIRVOpIAdd(longType, temp2, temp1, getConst("long", 8)));
                            SPIRVId elementBase = nextId();
                            spirvBlock.add(new SPIRVOpConvertUToPtr(arrayType, elementBase, temp2));
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(arrayType, resultAddr, elementBase, indexX, new SPIRVMultipleOperands<>()));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(elementType, result, resultAddr, align(elementType.getName())));
                            addResult(call.result(), new SpirvResult(elementType, resultAddr, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("java.lang.Math::sqrt(double)double"))) {
                            SPIRVId floatType = getType("double");
                            SPIRVId result = nextId();
                            SPIRVId operand = getResult(call.operands().get(0)).value();
                            spirvBlock.add(new SPIRVOpExtInst(floatType, result, getId("oclExtension"), new SPIRVLiteralExtInstInteger(61, "sqrt"), new SPIRVMultipleOperands<>(operand)));
                            addResult(call.result(), new SpirvResult(floatType, null, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("java.lang.Math::exp(double)double"))) {
                            SPIRVId floatType = getType("double");
                            SPIRVId result = nextId();
                            SPIRVId operand = getResult(call.operands().get(0)).value();
                            spirvBlock.add(new SPIRVOpExtInst(floatType, result, getId("oclExtension"), new SPIRVLiteralExtInstInteger(19, "exp"), new SPIRVMultipleOperands<>(operand)));
                            addResult(call.result(), new SpirvResult(floatType, null, result));
                        }
                        else if (methodRef.equals(MethodRef.ofString("java.lang.Math::log(double)double"))) {
                            SPIRVId floatType = getType("double");
                            SPIRVId result = nextId();
                            SPIRVId operand = getResult(call.operands().get(0)).value();
                            spirvBlock.add(new SPIRVOpExtInst(floatType, result, getId("oclExtension"), new SPIRVLiteralExtInstInteger(37, "log"), new SPIRVMultipleOperands<>(operand)));
                            addResult(call.result(), new SpirvResult(floatType, null, result));
                        }
                        else {
                            SPIRVId fnId = getFunctionId(methodRef);
                            if (fnId == null) {
                                if (!isBufferType((JavaType) methodRef.refType()))
                                    unsupported("method", methodRef);
                                FunctionType fnType = methodRef.type();
                                String returnTypeName = fnType.returnType().toString();
                                SPIRVId accessReturnType;
                                if (isPrimitiveType(fnType.returnType().toString())) {
                                    accessReturnType = spirvVariableType(spirvType(returnTypeName));
                                } else {
                                    accessReturnType = spirvType(returnTypeName);
                                }    
                                String typeName = call.operands().get(0).type().toString().replaceAll("\\$", ".");
                                SPIRVId returnType = spirvType(fnType.returnType().toString());
                                SPIRVId accessResult = nextId();
                                SPIRVId result = nextId();
                                SPIRVId operand = getResult(call.operands().get(0)).value();
                                String methodName = methodRef.name();
                                boolean atomic_op = false;
                                boolean setter = false;
                                SPIRVMultipleOperands accessOperands;

                                if (methodName.startsWith("atomic") && methodName.endsWith("Inc")) {
                                    atomic_op = true;
                                    methodName = methodName.substring(0, methodName.length() - 3);
                                }

                                int offset = classMap.get(typeName).get(methodName);
                                if (fnType.returnType().toString().equals("void")) {
                                    // field setter
                                    setter = true;
                                    accessOperands = new SPIRVMultipleOperands<>(getConst("int", offset));
                                    accessReturnType = spirvVariableType(spirvType(call.operands().get(1).type().toString()));
                                } else if (call.operands().size() > 1) {
                                    // array access
                                    SPIRVId arrayIdx = getResult(call.operands().get(1)).value();
                                    accessOperands = new SPIRVMultipleOperands<>(getConst("int", offset), arrayIdx);
                                } else {
                                    // field access
                                    accessOperands = new SPIRVMultipleOperands<>(getConst("int", offset));
                                }
                                spirvBlock.add(new SPIRVOpAccessChain(accessReturnType, accessResult, operand, accessOperands));
                                if (atomic_op) {
                                    // only support atomic increment for now
                                    spirvBlock.add(new SPIRVOpAtomicIIncrement(returnType, result, accessResult, getConst("int", 0), getConst("int", 0x8)));
                                    addResult(call.result(), new SpirvResult(returnType, null, result));
                                } else if (isPrimitiveType(fnType.returnType().toString())) {    
                                    spirvBlock.add(new SPIRVOpLoad(returnType, result, accessResult, align(returnType.getName())));
                                    addResult(call.result(), new SpirvResult(returnType, null, result));
                                } else if (setter) {
                                    spirvBlock.add(new SPIRVOpStore(accessResult, getResult(call.operands().get(1)).value(), align(returnType.getName())));
                                } else {
                                    addResult(call.result(), new SpirvResult(accessReturnType, null, accessResult));
                                }
                            }
                            else {
                                FunctionType fnType = methodRef.type();
                                SPIRVId[] args = new SPIRVId[call.operands().size()];
                                for (int i = 0; i < args.length; i++) {
                                    SPIRVId argId = getResult(call.operands().get(i)).value();
                                    args[i] = argId;
                                }
                                SPIRVId returnType = spirvType(fnType.returnType().toString());
                                SPIRVId callResult = nextId();
                                spirvBlock.add(new SPIRVOpFunctionCall(returnType, callResult, fnId, new SPIRVMultipleOperands<>(args)));
                                addResult(call.result(), new SpirvResult(returnType, null, callResult));
                            }
                        }
                    }
                    case SpirvOp.ConstantOp cop -> {
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
                        else if (type == getType("bool")) {
                            module.add(((boolean)value) ? new SPIRVOpConstantTrue(type, result) : new SPIRVOpConstantFalse(type, result));
                        }
                        else {
                            module.add(new SPIRVOpConstantNull(type, result));
                        }
                        addResult(cop.result(), new SpirvResult(type, null, result));
                    }
                    case SpirvOp.ConvertOp scop -> {
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
                        else if (isFloatType(fromType)) {
                            if (isIntegerType(toType)) {
                                spirvBlock.add(new SPIRVOpConvertFToS(toType, to, from));
                            }
                            else if (isFloatType(toType)) {
                                spirvBlock.add(new SPIRVOpFConvert(toType, to, from));
                            }
                            else unsupported("conversion type", scop.resultType());
                        }
                        else unsupported("conversion type", scop.operands().get(0));
                        addResult(scop.result(), new SpirvResult(toType, null, to));
                    }
                    case SpirvOp.CastOp cop -> {
                        SPIRVId toType = spirvType(cop.resultType().toString());
                        SPIRVId to = nextId();
                        SpirvResult valueResult = getResult(cop.operands().get(0));
                        SPIRVId from = valueResult.value();
                        SPIRVId fromType = valueResult.type();
                        spirvBlock.add(new SPIRVOpBitcast(toType, to, from));
                        addResult(cop.result(), new SpirvResult(toType, null, to));
                    }
                    case SpirvOp.InBoundsAccessChainOp iacop -> {
                        SPIRVId type = spirvType(iacop.resultType().toString());
                        SPIRVId result = nextId();
                        SPIRVId object = getResult(iacop.operands().get(0)).value();
                        SPIRVId index = getResult(iacop.operands().get(1)).value();
                        spirvBlock.add(new SPIRVOpInBoundsPtrAccessChain(type, result, object, index, new SPIRVMultipleOperands<>()));
                        addResult(iacop.result(), new SpirvResult(type, result, null));
                    }
                    case SpirvOp.FieldLoadOp flo -> {
                        if (flo.operands().size() > 0 && (flo.operands().get(0).type().equals(JavaType.ofString("hat.KernelContext")))) {
                            SpirvResult result;
                            int group = -1;
                            int index = -1;
                            String fieldName = flo.fieldDescriptor().name();
                            switch (fieldName) {
                                case "x": group = 0; index = 0; break;
                                case "y": group = 0; index = 1; break;
                                case "z": group = 0; index = 2; break;
                                case "maxX": group = 1; index = 0; break;
                                case "maxY": group = 1; index = 1; break;
                                case "maxZ": group = 1; index = 2; break;
                            }
                            switch (group) {
                                case 0: result = globalId(index, spirvBlock); break;
                                case 1: result = globalSize(index, spirvBlock); break;
                                default: throw new RuntimeException("Unknown Index field: " + fieldName);
                            }
                            addResult(flo.result(), result);
                        }
                        else if (flo.operands().get(0).type().equals(JavaType.ofString("hat.KernelContext"))) {
                            String fieldName = flo.fieldDescriptor().name();
                            SPIRVId fieldIndex = switch (fieldName) {
                                case "x" -> getConst("long", 0);
                                case "maxX" -> getConst("long", 1);
                                default -> throw new RuntimeException("Unknown field: " + fieldName);
                            };
                            SPIRVId intType = getType("int");
                            String contextTypeName = flo.operands().get(0).type().toString();
                            SpirvResult kernalContext = getResult(flo.operands().get(0));
                            SPIRVId contextAddr = kernalContext.address();
                            SPIRVId contextType = spirvType(contextTypeName);
                            SPIRVId context = nextId();
                            spirvBlock.add(new SPIRVOpLoad(contextType, context, contextAddr, align(contextType.getName())));
                            SPIRVId fieldType = intType;
                            SPIRVId resultAddr = nextId();
                            spirvBlock.add(new SPIRVOpInBoundsAccessChain(getType("ptrInt"), resultAddr, context, new SPIRVMultipleOperands<>(fieldIndex)));
                            SPIRVId result = nextId();
                            spirvBlock.add(new SPIRVOpLoad(intType, result, resultAddr, align("int")));
                            addResult(flo.result(), new SpirvResult(intType, resultAddr, result));
                        }
                        else if (flo.fieldDescriptor().refType().equals(JavaType.type(ByteOrder.class))) {
                            // currently ignored
                        }
                        else unsupported("field load", ((ClassType)flo.fieldDescriptor().refType()).toClassName() + "." + flo.fieldDescriptor().name());
                    }
                    case SpirvOp.BranchOp bop -> {
                        SPIRVId label = symbols.getLabel(bop.branch().targetBlock()).getResultId();
                        Block.Reference target = bop.branch();
                        spirvBlock.add(new SPIRVOpBranch(label));
                    }
                    case SpirvOp.ConditionalBranchOp cbop -> {
                        SPIRVId test = getResult(cbop.operands().get(0)).value();
                        SPIRVId trueLabel = symbols.getLabel(cbop.trueBranch().targetBlock()).getResultId();
                        SPIRVId falseLabel = symbols.getLabel(cbop.falseBranch().targetBlock()).getResultId();
                        spirvBlock.add(new SPIRVOpBranchConditional(test, trueLabel, falseLabel, new SPIRVMultipleOperands<SPIRVLiteralInteger>()));
                    }
                    case SpirvOp.LtOp ltop -> {
                        SpirvResult lhs = getResult(ltop.operands().get(0));
                        SpirvResult rhs = getResult(ltop.operands().get(1));
                        SPIRVId boolType = getType("bool");
                        SPIRVId result = nextId();
                        String operandType = lhs.type().getName();
                        SPIRVInstruction sop = switch (operandType) {
                            case "float" -> new SPIRVOpFUnordLessThan(boolType, result, lhs.value(), rhs.value());
                            case "int" -> new SPIRVOpSLessThan(boolType, result, lhs.value(), rhs.value());
                            case "long" -> new SPIRVOpSLessThan(boolType, result, lhs.value(), rhs.value());
                            default -> throw new RuntimeException("Unsupported type: " + lhs.type().getName());
                        };
                        spirvBlock.add(sop);
                        addResult(ltop.result(), new SpirvResult(boolType, null, result));
                    }
                    case SpirvOp.GtOp gtop -> {
                        SpirvResult lhs = getResult(gtop.operands().get(0));
                        SpirvResult rhs = getResult(gtop.operands().get(1));
                        SPIRVId boolType = getType("bool");
                        SPIRVId result = nextId();
                        String operandType = lhs.type().getName();
                        SPIRVInstruction sop = switch (operandType) {
                            case "float" -> new SPIRVOpFUnordGreaterThan(boolType, result, lhs.value(), rhs.value());
                            case "int" -> new SPIRVOpSGreaterThan(boolType, result, lhs.value(), rhs.value());
                            case "long" -> new SPIRVOpSGreaterThan(boolType, result, lhs.value(), rhs.value());
                            default -> throw new RuntimeException("Unsupported type: " + lhs.type().getName());
                        };
                        spirvBlock.add(sop);
                        addResult(gtop.result(), new SpirvResult(boolType, null, result));
                    }
                    case SpirvOp.FNegateOp fnop -> {
                        SPIRVId floatType = getType("float");
                        SPIRVId result = nextId();
                        SPIRVId operand = getResult(fnop.operands().get(0)).value();
                        spirvBlock.add(new SPIRVOpFNegate(floatType, result, operand));
                        addResult(fnop.result(), new SpirvResult(floatType, null, result));
                    }
                    case SpirvOp.BitwiseAndOp baop -> {
                        SpirvResult lhs = getResult(baop.operands().get(0));
                        SpirvResult rhs = getResult(baop.operands().get(1));
                        SPIRVId resultType = spirvType(baop.resultType().toString());
                        SPIRVId result = nextId();
                        spirvBlock.add(new SPIRVOpBitwiseAnd(resultType, result, lhs.value(), rhs.value()));
                        addResult(baop.result(), new SpirvResult(resultType, null, result));
                    }
                    case SpirvOp.ReturnOp rop -> {
                        spirvBlock.add(new SPIRVOpReturn());
                    }
                    case SpirvOp.ReturnValueOp rop -> {
                        SPIRVId returnValue = getResult(rop.operands().get(0)).value();
                        spirvBlock.add(new SPIRVOpReturnValue(returnValue));
                    }
                    default -> unsupported("op", op.getClass());
                }
            }
        } // end bi
    }

    private SPIRVId getFunctionId(MethodRef methodRef) {
        SPIRVId fnId = symbols.getId(methodRef.toString());
        return fnId;
    }

    private boolean isBufferType(JavaType javaType) {
        boolean bufferMethod = false;
        Class<?>[] classes = new Class<?>[] {Buffer.class, Buffer.Struct.class, Buffer.Union.class};
        if (javaType instanceof ClassType classType) {
            try {
                Class<?> javaTypeClass = Class.forName(classType.toString());
                for (Class<?> clazz : classes) {
                    if (clazz.isAssignableFrom(javaTypeClass)) {
                        bufferMethod = true;
                        break;
                    }
                }
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
        }
        return bufferMethod;
    }

    private void initModule() {
        module.add(new SPIRVOpCapability(SPIRVCapability.Addresses()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Linkage()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Kernel()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Int8()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Int16()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Int64()));
        module.add(new SPIRVOpCapability(SPIRVCapability.Float64()));
        module.add(new SPIRVOpMemoryModel(SPIRVAddressingModel.Physical64(), SPIRVMemoryModel.OpenCL()));

        // OpenCL extension provides built-in variables suitable for kernel programming
        // Import extension and declare four variables
        SPIRVId oclExtension = nextId("oclExtension");
        module.add(new SPIRVOpExtInstImport(oclExtension, new SPIRVLiteralString("OpenCL.std")));
        symbols.putId("oclExtension", oclExtension);

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

    private SPIRVId spirvType(String inputType) {
        String javaType = inputType.replaceAll("\\$", ".");
        SPIRVId ans = switch(javaType) {
            case "byte" -> getType("byte");
            case "short" -> getType("short");
            case "int" -> getType("int");
            case "long" -> getType("long");
            case "float" -> getType("float");
            case "double" -> getType("double");
            case "byte[]" -> getType("byte[]");
            case "int[]" -> getType("int[]");
            case "float[]" -> getType("float[]");
            case "double[]" -> getType("double[]");
            case "long[]" -> getType("long[]");
            case "bool" -> getType("bool");
            case "boolean" -> getType("bool");
            case "java.lang.Object" -> getType("java.lang.Object");
            case "hat.buffer.S32Array" -> getType("int[]");
            case "hat.buffer.S32Array2D" -> getType("int[]");
            case "hat.buffer.F32Array" -> getType("float[]");
            case "hat.buffer.F32Array2D" -> getType("float[]");
            case "hat.buffer.S08x3RGBImage" -> getType("byte[]");
            case "void" -> getType("void");
            case "hat.KernelContext" -> getType("ptrKernelContext");
            case "java.lang.foreign.MemorySegment" -> getType("ptrByte");
            default -> getType("ptr" + javaType.substring(0, 1).toUpperCase() + javaType.substring(1));
        };
        if (ans == null) unsupported("type", javaType);
        return ans;
    }

    private SPIRVId spirvElementType(String inputType) {
        String javaType = inputType.replaceAll("\\$", ".");
        SPIRVId ans = switch(javaType) {
            case "byte[]" -> getType("byte");
            case "short[]" -> getType("short");
            case "int[]" -> getType("int");
            case "long[]" -> getType("long");
            case "float[]" -> getType("float");
            case "double[]" -> getType("double");
            case "boolean[]" -> getType("bool");
            case "hat.buffer.S32Array" -> getType("int");
            case "hat.buffer.F32Array" -> getType("float");
            case "hat.buffer.S32Array2D" -> getType("int");
            case "hat.buffer.F32Array2D" -> getType("float");
            case "hat.buffer.S08x3RGBImage" -> getType("byte");
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
            case "bool" -> getType("ptrBool");
            case "byte" -> getType("ptrByte");
            case "short" -> getType("ptrShort");
            case "int" -> getType("ptrInt");
            case "long" -> getType("ptrLong");
            case "float" -> getType("ptrFloat");
            case "double" -> getType("ptrDouble");
            case "boolean" -> getType("ptrBool");
            case "byte[]" -> getType("ptrByte[]");
            case "int[]" -> getType("ptrInt[]");
            case "long[]" -> getType("ptrLong[]");
            case "float[]" -> getType("ptrFloat[]");
            case "double[]" -> getType("ptrDouble[]");
            case "v8int" -> getType("ptrV8int");
            case "v16int" -> getType("ptrV16int");
            case "v8long" -> getType("ptrV8long");
            case "v8float" -> getType("ptrV8float");
            case "v16float" -> getType("ptrV16float");
            case "ptrKernelContext" -> getType("ptrPtrKernelContext");
            case "hat.KernelContext" -> getType("ptrKernelContext");
            case "ptrByte" -> getType("ptrPtrByte");
            default -> getType("ptr" + spirvType.getName().substring(0, 1).toUpperCase() + spirvType.getName().substring(1));
        };
        if (ans == null) unsupported("type", spirvType.getName());
        return ans;
    }

    private SPIRVId spirvVectorType(String javaVectorType, int vectorLength) {
        String prefix = "v" + vectorLength;
        String elementType = spirvElementType(javaVectorType).getName();
        return getType(prefix + elementType);
    }

    private int alignment(String inputType) {
        String spirvType = inputType.replaceAll("\\$", ".");
        if (inputType.startsWith("ptr")) return 32;
        int ans = switch(spirvType) {
            case "void" -> 1;
            case "bool" -> 1;
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
            case "hat.KernelContext" -> 32;
            case "ptrKernelContext" -> 32;
            case "byte[]" -> 8;
            case "int[]" -> 8;
            case "long[]" -> 8;
            case "float[]" -> 8;
            case "double[]" -> 8;
            case "ptrBool" -> 8;
            case "ptrByte" -> 8;
            case "ptrInt" -> 8;
            case "ptrByte[]" -> 8;
            case "ptrInt[]" -> 8;
            case "ptrLong" -> 8;
            case "ptrLong[]" -> 8;
            case "ptrFloat" -> 8;
            case "ptrFloat[]" -> 8;
            case "ptrV8int" -> 8;
            case "ptrV8float" -> 8;
            case "ptrPtrKernelContext" -> 8;
            default -> 0;
        };
        if (ans == 0) unsupported("type", spirvType);
        return ans;
    }

    private Set<String> moduleTypes = new HashSet<>();

    private SPIRVId getType(String inputName) {
        String name = inputName.replaceAll("\\$", ".");
        if (!moduleTypes.contains(name)) {
            switch (name) {
                case "void" -> module.add(new SPIRVOpTypeVoid(nextId(name)));
                case "bool" -> module.add(new SPIRVOpTypeBool(nextId(name)));
                case "boolean" -> module.add(new SPIRVOpTypeBool(nextId(name)));
                case "byte" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(8), new SPIRVLiteralInteger(0)));
                case "short" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(16), new SPIRVLiteralInteger(0)));
                case "int" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(32), new SPIRVLiteralInteger(0)));
                case "long" -> module.add(new SPIRVOpTypeInt(nextId(name), new SPIRVLiteralInteger(64), new SPIRVLiteralInteger(0)));
                case "float" -> module.add(new SPIRVOpTypeFloat(nextId(name), new SPIRVLiteralInteger(32)));
                case "double" -> module.add(new SPIRVOpTypeFloat(nextId(name), new SPIRVLiteralInteger(64)));
                case "ptrBool" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("bool")));
                case "ptrByte" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("byte")));
                case "ptrShort" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("short")));
                case "ptrInt" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("int")));
                case "ptrLong" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("long")));
                case "ptrFloat" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("float")));
                case "byte[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("byte")));
                case "short[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("short")));
                case "int[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("int")));
                case "long[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("long")));
                case "float[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("float")));
                case "double[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("double")));
                case "boolean[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("boolean")));
                case "ptrByte[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("byte[]")));
                case "ptrInt[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("int[]")));
                case "ptrLong[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("long[]")));
                case "ptrFloat[]" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("float[]")));
                case "java.lang.Object" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("void")));
                case "hat.KernelContext" -> module.add(new SPIRVOpTypeStruct(nextId(name), new SPIRVMultipleOperands<>(getType("int"), getType("int"))));
                case "ptrKernelContext" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("hat.KernelContext")));
                case "ptrCrossGroupByte"-> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.CrossWorkgroup(), getType("byte")));
                case "ptrPtrKernelContext" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrKernelContext")));
                case "ptrPtrByte" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrByte")));
                case "ptrPtrInt" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrInt")));
                case "ptrPtrFloat" -> module.add(new SPIRVOpTypePointer(nextId(name), SPIRVStorageClass.Function(), getType("ptrFloat")));
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

    private SPIRVId getConst(String typeName, long value) {
        String name = typeName + "_" + value;
        if (!moduleConstants.contains(name)) {
            String valueStr = String.valueOf(value);
            switch (typeName) {
                case "int" -> module.add(new SPIRVOpConstant(getType(typeName), nextId(name), new SPIRVContextDependentInt(new BigInteger(valueStr))));
                case "long" -> module.add(new SPIRVOpConstant(getType(typeName), nextId(name), new SPIRVContextDependentLong(new BigInteger(valueStr))));
                case "boolean" -> module.add(value == 0 ? new SPIRVOpConstantFalse(getType(typeName), nextId(name)) : new SPIRVOpConstantTrue(getType(typeName), nextId(name)));
                default -> unsupported("constant", name);
            };
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
        SPIRVId intType = getType("int");
        SPIRVId longType = getType("long");
        SPIRVId v3long = getId("v3long");
        SPIRVId globalSizeId = getId("globalSize");
        SPIRVId globalSizes = nextId();
        spirvBlock.add(new SPIRVOpLoad(v3long, globalSizes, globalSizeId, align(32)));
        SPIRVId longSize = nextId();
        SPIRVId globalSize = nextId();
        spirvBlock.add(new SPIRVOpCompositeExtract(longType, longSize, globalSizes, new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(index))));
        spirvBlock.add(new SPIRVOpSConvert(intType, globalSize, longSize));
        return new SpirvResult(intType, null, globalSize);
    }

    private SpirvResult globalId(int index, SPIRVBlock spirvBlock) {
        SPIRVId intType = getType("int");
        SPIRVId longType = getType("long");
        SPIRVId v3long = getId("v3long");
        SPIRVId globalInvocationId = getId("globalInvocationId");
        SPIRVId globalIds = nextId();
        spirvBlock.add(new SPIRVOpLoad(v3long, globalIds, globalInvocationId, align(32)));
        SPIRVId longIndex = nextId();
        SPIRVId globalIndex = nextId();
        spirvBlock.add(new SPIRVOpCompositeExtract(longType, longIndex, globalIds, new SPIRVMultipleOperands<>(new SPIRVLiteralInteger(index))));
        spirvBlock.add(new SPIRVOpSConvert(intType, globalIndex, longIndex));
        return new SpirvResult(intType, null, globalIndex);
    }

    private SpirvResult flatIndex(SPIRVId sizeX, SPIRVId sizeY, SPIRVId sizeZ, SPIRVId indexX, SPIRVId indexY, SPIRVId indexZ, SPIRVBlock spirvBlock)
    {
        SPIRVId longType = getType("long");
        SPIRVId xTerm0 = nextId();
        SPIRVId xTerm1 = nextId();
        SPIRVId yTerm = nextId();
        SPIRVId flat0 = nextId();
        SPIRVId flat1 = nextId();
        spirvBlock.add(new SPIRVOpIMul(longType, xTerm0, sizeY, sizeZ));
        spirvBlock.add(new SPIRVOpIMul(longType, xTerm1, xTerm0, indexX));
        spirvBlock.add(new SPIRVOpIMul(longType, yTerm, sizeZ, indexY));
        spirvBlock.add(new SPIRVOpIAdd(longType, flat0, xTerm1, yTerm));
        spirvBlock.add(new SPIRVOpIAdd(longType, flat1, flat0, indexZ));
        return new SpirvResult(longType, null, flat1);
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

    private boolean isPrimitiveType(String javaType) {
        return javaType.equals("byte") || javaType.equals("boolean") || javaType.equals("short") || javaType.equals("int") || javaType.equals("long") || javaType.equals("float") || javaType.equals("double");
    }

    private static int counter = 0;

    private String nextTempTag() {
        counter++;
        return "temp_" + counter + "_";
    }

    private boolean isIntegerType(SPIRVId type) {
        String name = type.getName();
        return name.equals("byte") || name.equals("short") || name.equals("int") || name.equals("long");
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

    public static void debug(String message, Object... args) {
        System.out.println(String.format(message, args));
    }
}