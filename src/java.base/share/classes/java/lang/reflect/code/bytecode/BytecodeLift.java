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

package java.lang.reflect.code.bytecode;

import java.lang.classfile.Attributes;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeModel;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.PseudoInstruction;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.reflect.AccessFlag;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.constant.MethodTypeDesc;


public final class BytecodeLift {

    record VarID(String name, ClassDesc type) {};

    private final Block.Builder entryBlock;
    private final CodeModel codeModel;
    private final Map<Label, Block.Builder> blockMap;
    private final CodeTracker codeTracker;
    private final Map<VarID, Op.Result> varMap;
    private final List<CodeElement> elements;
    private final Deque<Value> stack;
    private Block.Builder currentBlock;

    private static TypeElement toTypeElement(StackMapFrameInfo.VerificationTypeInfo vti) {
        return switch (vti) {
            case ITEM_INTEGER -> JavaType.INT;
            case ITEM_FLOAT -> JavaType.FLOAT;
            case ITEM_DOUBLE -> JavaType.DOUBLE;
            case ITEM_LONG -> JavaType.LONG;
            case ITEM_NULL -> JavaType.J_L_OBJECT;
            case StackMapFrameInfo.ObjectVerificationTypeInfo ovti ->
                    JavaType.type(ovti.classSymbol());
            case StackMapFrameInfo.UninitializedVerificationTypeInfo _ ->
                    JavaType.J_L_OBJECT;
            default ->
                throw new IllegalArgumentException("Unexpected VTI: " + vti);

        };
    }

    private TypeElement toTypeElement(ClassEntry ce) {
        return JavaType.type(ce.asSymbol());
    }

    private BytecodeLift(Block.Builder entryBlock, MethodModel methodModel) {
        if (!methodModel.flags().has(AccessFlag.STATIC)) {
            throw new IllegalArgumentException("Unsuported lift of non-static method: " + methodModel);
        }
        this.entryBlock = entryBlock;
        this.currentBlock = entryBlock;
        this.codeModel = methodModel.code().orElseThrow();
        this.elements = codeModel.elementList();
        this.varMap = new HashMap<>();
        this.stack = new ArrayDeque<>();
        var smta = codeModel.findAttribute(Attributes.STACK_MAP_TABLE);
        this.blockMap = smta.map(sma ->
                sma.entries().stream().collect(Collectors.toUnmodifiableMap(
                        StackMapFrameInfo::target,
                        smfi -> entryBlock.block(smfi.stack().stream().map(BytecodeLift::toTypeElement).toList())))).orElse(Map.of());
        MethodTypeDesc mtd = methodModel.methodTypeSymbol();
        this.codeTracker = new CodeTracker(methodModel, smta);
        elements.forEach(codeTracker);

        List<Block.Parameter> bps = entryBlock.parameters();
        for (int i = 0, slot = 0; i < bps.size(); i++) {
            CodeTracker.Local local = codeTracker.initLocals.get(slot);
            if (local != null) {
                VarID varID = getVarID(local.type, slot);
                varMap.put(varID, op(CoreOp.var(varID.name(), JavaType.type(local.type), bps.get(i))));
            }
            slot += TypeKind.from(mtd.parameterType(i)).slotSize();
        }
    }

    private VarID getVarID(ClassDesc varType, int slot) {
        if (varType.isPrimitive()) switch (varType.descriptorString().charAt(0)) {
            case 'B', 'S', 'C', 'Z' -> varType = ConstantDescs.CD_int;
        }
        return new VarID(varType.displayName() + "-" + slot, varType);
    }
    private VarID findVarID(Instruction i, int slot) {
        ClassDesc varType = codeTracker.insMap.get(i).type;
        return getVarID(varType, slot);
    }

    private void varStore(Instruction i, int slot, Value value) {
        varMap.compute(findVarID(i, slot), (varID, varOp) -> {
            if (varOp == null) {
                return op(CoreOp.var(varID.name(), JavaType.type(varID.type()), value));
            } else {
                op(CoreOp.varStore(varOp, value));
                return varOp;
            }
        });
    }

    private Op.Result var(Instruction i, int slot) {
        return varMap.computeIfAbsent(findVarID(i, slot), name -> {
            throw new IllegalArgumentException("Undeclared variable: " + name);
        });
    }

    private Op.Result op(Op op) {
        return currentBlock.op(op);
    }

    // Lift to core dialect
    public static CoreOp.FuncOp lift(byte[] classdata, String methodName) {
        return lift(classdata, methodName, null);
    }

    public static CoreOp.FuncOp lift(byte[] classdata, String methodName, MethodTypeDesc methodType) {
        return lift(ClassFile.of(
                ClassFile.DebugElementsOption.DROP_DEBUG,
                ClassFile.LineNumbersOption.DROP_LINE_NUMBERS).parse(classdata).methods().stream()
                        .filter(mm -> mm.methodName().equalsString(methodName) && (methodType == null || mm.methodTypeSymbol().equals(methodType)))
                        .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown method: " + methodName)));
    }

    public static CoreOp.FuncOp lift(MethodModel methodModel) {
        return CoreOp.func(
                methodModel.methodName().stringValue(),
                MethodRef.ofNominalDescriptor(methodModel.methodTypeSymbol())).body(entryBlock ->
                        new BytecodeLift(entryBlock, methodModel).lift());
    }

    private Block.Builder getBlock(Label l) {
        Block.Builder bb = blockMap.get(l);
        if (bb == null) {
            if (currentBlock == null) {
                throw new IllegalArgumentException("Block without an stack frame detected.");
            } else {
                return newBlock();
            }
        }
        return bb;
    }

    private Block.Builder newBlock() {
        return entryBlock.block(stack.stream().map(Value::type).toList());
    }

    private void moveTo(Block.Builder next) {
        currentBlock = next;
        // Stack is reconstructed from block parameters
        stack.clear();
        if (currentBlock != null) {
            currentBlock.parameters().forEach(stack::add);
        }
    }

    private void endOfFlow() {
        currentBlock = null;
        // Flow discontinued, stack cleared to be ready for the next label target
        stack.clear();
    }

    private void lift() {
        final Map<ExceptionCatch, Op.Result> exceptionRegionsMap = new HashMap<>();
        for (int i = 0; i < elements.size(); i++) {
            switch (elements.get(i)) {
                case ExceptionCatch _ -> {
                    // Exception blocks are inserted by label target (below)
                }
                case LabelTarget lt -> {
                    Block.Builder next = blockMap.get(lt.label());
                    // Start of a new block if defined by stack maps
                    if (next != null) {
                        if (currentBlock != null) {
                            // Implicit goto next block, add explicitly
                            // Use stack content as next block arguments
                            op(CoreOp.branch(next.successor(List.copyOf(stack))));
                        }
                        moveTo(next);
                        // Insert relevant tryStart and construct handler blocks, all in reversed order
                        for (ExceptionCatch ec : codeModel.exceptionHandlers().reversed()) {
                            if (lt.label() == ec.tryStart()) {
                                Block.Builder handler = getBlock(ec.handler());
                                // Create start block
                                next = newBlock();
                                Op ere = CoreOp.exceptionRegionEnter(next.successor(List.copyOf(stack)), handler.successor());
                                op(ere);
                                // Store ERE into map for exit
                                exceptionRegionsMap.put(ec, ere.result());
                                moveTo(next);
                            }
                        }
                        // Insert relevant tryEnd blocks in normal order
                        for (ExceptionCatch ec : codeModel.exceptionHandlers()) {
                            if (lt.label() == ec.tryEnd()) {
                                // Create exit block with parameters constructed from the stack
                                next = newBlock();
                                op(CoreOp.exceptionRegionExit(exceptionRegionsMap.get(ec), next.successor()));
                                moveTo(next);
                            }
                        }
                    }
                }
                case BranchInstruction inst when inst.opcode().isUnconditionalBranch() -> {
                    op(CoreOp.branch(getBlock(inst.target()).successor(List.copyOf(stack))));
                    endOfFlow();
                }
                case BranchInstruction inst -> {
                    // Conditional branch
                    Value operand = stack.pop();
                    Op cop = switch (inst.opcode()) {
                        case IFNE -> CoreOp.eq(operand, op(CoreOp.constant(JavaType.INT, 0)));
                        case IFEQ -> CoreOp.neq(operand, op(CoreOp.constant(JavaType.INT, 0)));
                        case IFGE -> CoreOp.lt(operand, op(CoreOp.constant(JavaType.INT, 0)));
                        case IFLE -> CoreOp.gt(operand, op(CoreOp.constant(JavaType.INT, 0)));
                        case IFGT -> CoreOp.le(operand, op(CoreOp.constant(JavaType.INT, 0)));
                        case IFLT -> CoreOp.ge(operand, op(CoreOp.constant(JavaType.INT, 0)));
                        case IFNULL -> CoreOp.neq(operand, op(CoreOp.constant(JavaType.J_L_OBJECT, null)));
                        case IFNONNULL -> CoreOp.eq(operand, op(CoreOp.constant(JavaType.J_L_OBJECT, null)));
                        case IF_ICMPNE -> CoreOp.eq(stack.pop(), operand);
                        case IF_ICMPEQ -> CoreOp.neq(stack.pop(), operand);
                        case IF_ICMPGE -> CoreOp.lt(stack.pop(), operand);
                        case IF_ICMPLE -> CoreOp.gt(stack.pop(), operand);
                        case IF_ICMPGT -> CoreOp.le(stack.pop(), operand);
                        case IF_ICMPLT -> CoreOp.ge(stack.pop(), operand);
                        case IF_ACMPEQ -> CoreOp.neq(stack.pop(), operand);
                        case IF_ACMPNE -> CoreOp.eq(stack.pop(), operand);
                        default -> throw new UnsupportedOperationException("Unsupported branch instruction: " + inst);
                    };
                    Block.Builder next = newBlock();
                    op(CoreOp.conditionalBranch(op(cop),
                            next.successor(List.copyOf(stack)),
                            getBlock(inst.target()).successor(List.copyOf(stack))));
                    moveTo(next);
                }
    //                case LookupSwitchInstruction si -> {
    //                    // Default label is first successor
    //                    b.addSuccessor(blockMap.get(si.defaultTarget()));
    //                    addSuccessors(si.cases(), blockMap, b);
    //                }
    //                case TableSwitchInstruction si -> {
    //                    // Default label is first successor
    //                    b.addSuccessor(blockMap.get(si.defaultTarget()));
    //                    addSuccessors(si.cases(), blockMap, b);
    //                }
                case ReturnInstruction inst when inst.typeKind() == TypeKind.VoidType -> {
                    op(CoreOp._return());
                    endOfFlow();
                }
                case ReturnInstruction _ -> {
                    op(CoreOp._return(stack.pop()));
                    endOfFlow();
                }
                case ThrowInstruction _ -> {
                    op(CoreOp._throw(stack.pop()));
                    endOfFlow();
                }
                case LoadInstruction inst -> {
                    stack.push(op(CoreOp.varLoad(var(inst, inst.slot()))));
                }
                case StoreInstruction inst -> {
                    varStore(inst, inst.slot(), stack.pop());
                }
                case IncrementInstruction inst -> {
                    Op.Result v = var(inst, inst.slot());
                    op(CoreOp.varStore(v, op(CoreOp.add(
                            op(CoreOp.varLoad(v)),
                            op(CoreOp.constant(JavaType.INT, inst.constant()))))));
                }
                case ConstantInstruction inst -> {
                    stack.push(op(switch (inst.constantValue()) {
                        case ClassDesc v -> CoreOp.constant(JavaType.J_L_CLASS, JavaType.type(v));
                        case Double v -> CoreOp.constant(JavaType.DOUBLE, v);
                        case Float v -> CoreOp.constant(JavaType.FLOAT, v);
                        case Integer v -> CoreOp.constant(JavaType.INT, v);
                        case Long v -> CoreOp.constant(JavaType.LONG, v);
                        case String v -> CoreOp.constant(JavaType.J_L_STRING, v);
                        default ->
                            // @@@ MethodType, MethodHandle, ConstantDynamic
                            throw new IllegalArgumentException("Unsupported constant value: " + inst.constantValue());
                    }));
                }
                case ConvertInstruction inst -> {
                    stack.push(op(CoreOp.conv(switch (inst.toType()) {
                        case ByteType -> JavaType.BYTE;
                        case ShortType -> JavaType.SHORT;
                        case IntType -> JavaType.INT;
                        case FloatType -> JavaType.FLOAT;
                        case LongType -> JavaType.LONG;
                        case DoubleType -> JavaType.DOUBLE;
                        case CharType -> JavaType.CHAR;
                        case BooleanType -> JavaType.BOOLEAN;
                        default ->
                            throw new IllegalArgumentException("Unsupported conversion target: " + inst.toType());
                    }, stack.pop())));
                }
                case OperatorInstruction inst -> {
                    Value operand = stack.pop();
                    stack.push(op(switch (inst.opcode()) {
                        case IADD, LADD, FADD, DADD ->
                                CoreOp.add(stack.pop(), operand);
                        case ISUB, LSUB, FSUB, DSUB ->
                                CoreOp.sub(stack.pop(), operand);
                        case IMUL, LMUL, FMUL, DMUL ->
                                CoreOp.mul(stack.pop(), operand);
                        case IDIV, LDIV, FDIV, DDIV ->
                                CoreOp.div(stack.pop(), operand);
                        case IREM, LREM, FREM, DREM ->
                                CoreOp.mod(stack.pop(), operand);
                        case INEG, LNEG, FNEG, DNEG ->
                                CoreOp.neg(operand);
                        case ARRAYLENGTH ->
                                CoreOp.arrayLength(operand);
                        case IAND, LAND ->
                                CoreOp.and(stack.pop(), operand);
                        case IOR, LOR ->
                                CoreOp.or(stack.pop(), operand);
                        case IXOR, LXOR ->
                                CoreOp.xor(stack.pop(), operand);
                        case ISHL, LSHL ->
                                CoreOp.lshl(stack.pop(), operand);
                        case ISHR, LSHR ->
                                CoreOp.ashr(stack.pop(), operand);
                        case IUSHR, LUSHR ->
                                CoreOp.lshr(stack.pop(), operand);
                        default ->
                            throw new IllegalArgumentException("Unsupported operator opcode: " + inst.opcode());
                    }));
                }
                case FieldInstruction inst -> {
                        FieldRef fd = FieldRef.field(
                                JavaType.type(inst.owner().asSymbol()),
                                inst.name().stringValue(),
                                JavaType.type(inst.typeSymbol()));
                        switch (inst.opcode()) {
                            case GETFIELD ->
                                stack.push(op(CoreOp.fieldLoad(fd, stack.pop())));
                            case GETSTATIC ->
                                stack.push(op(CoreOp.fieldLoad(fd)));
                            case PUTFIELD -> {
                                Value value = stack.pop();
                                stack.push(op(CoreOp.fieldStore(fd, stack.pop(), value)));
                            }
                            case PUTSTATIC ->
                                stack.push(op(CoreOp.fieldStore(fd, stack.pop())));
                            default ->
                                throw new IllegalArgumentException("Unsupported field opcode: " + inst.opcode());
                        }
                }
                case ArrayStoreInstruction _ -> {
                    Value value = stack.pop();
                    Value index = stack.pop();
                    op(CoreOp.arrayStoreOp(stack.pop(), index, value));
                }
                case ArrayLoadInstruction _ -> {
                    Value index = stack.pop();
                    stack.push(op(CoreOp.arrayLoadOp(stack.pop(), index)));
                }
                case InvokeInstruction inst -> {
                    FunctionType mType = MethodRef.ofNominalDescriptor(inst.typeSymbol());
                    List<Value> operands = new ArrayList<>();
                    for (var _ : mType.parameterTypes()) {
                        operands.add(stack.pop());
                    }
                    MethodRef mDesc = MethodRef.method(
                            JavaType.type(inst.owner().asSymbol()),
                            inst.name().stringValue(),
                            mType);
                    Op.Result result = switch (inst.opcode()) {
                        case INVOKEVIRTUAL, INVOKEINTERFACE -> {
                            operands.add(stack.pop());
                            yield op(CoreOp.invoke(mDesc, operands.reversed()));
                        }
                        case INVOKESTATIC ->
                            op(CoreOp.invoke(mDesc, operands.reversed()));
                        case INVOKESPECIAL -> {
                            if (inst.name().equalsString(ConstantDescs.INIT_NAME)) {
                                yield op(CoreOp._new(
                                        FunctionType.functionType(
                                                mDesc.refType(),
                                                mType.parameterTypes()),
                                        operands.reversed()));
                            } else {
                                operands.add(stack.pop());
                                yield op(CoreOp.invoke(mDesc, operands.reversed()));
                            }
                        }
                        default ->
                            throw new IllegalArgumentException("Unsupported invocation opcode: " + inst.opcode());
                    };
                    if (!result.type().equals(JavaType.VOID)) {
                        stack.push(result);
                    }
                }
                case NewObjectInstruction _ -> {
                    // Skip over this and the dup to process the invoke special
                    if (i + 2 < elements.size() - 1
                            && elements.get(i + 1) instanceof StackInstruction dup
                            && dup.opcode() == Opcode.DUP) {
                        i++;
                    } else {
                        throw new UnsupportedOperationException("New must be followed by dup");
                    }
                }
                case NewPrimitiveArrayInstruction inst -> {
                    stack.push(op(CoreOp.newArray(
                            switch (inst.typeKind()) {
                                case BooleanType -> JavaType.BOOLEAN_ARRAY;
                                case ByteType -> JavaType.BYTE_ARRAY;
                                case CharType -> JavaType.CHAR_ARRAY;
                                case DoubleType -> JavaType.DOUBLE_ARRAY;
                                case FloatType -> JavaType.FLOAT_ARRAY;
                                case IntType -> JavaType.INT_ARRAY;
                                case LongType -> JavaType.LONG_ARRAY;
                                case ShortType -> JavaType.SHORT_ARRAY;
                                default ->
                                        throw new UnsupportedOperationException("Unsupported new primitive array type: " + inst.typeKind());
                            },
                            stack.pop())));
                }
                case NewReferenceArrayInstruction inst -> {
                    stack.push(op(CoreOp.newArray(
                            JavaType.type(inst.componentType().asSymbol().arrayType()),
                            stack.pop())));
                }
                case NewMultiArrayInstruction inst -> {
                    stack.push(op(CoreOp._new(
                            FunctionType.functionType(
                                    JavaType.type(inst.arrayType().asSymbol()),
                                    Collections.nCopies(inst.dimensions(), JavaType.INT)),
                            IntStream.range(0, inst.dimensions()).mapToObj(_ -> stack.pop()).toList().reversed())));
                }
                case TypeCheckInstruction inst when inst.opcode() == Opcode.CHECKCAST -> {
                    stack.push(op(CoreOp.cast(JavaType.type(inst.type().asSymbol()), stack.pop())));
                }
                case TypeCheckInstruction inst -> {
                    stack.push(op(CoreOp.instanceOf(JavaType.type(inst.type().asSymbol()), stack.pop())));
                }
                case StackInstruction inst -> {
                    switch (inst.opcode()) {
                        case POP -> {
                            stack.pop();
                        }
                        case POP2 -> {
                            if (isCategory1(stack.pop())) {
                                stack.pop();
                            }
                        }
                        case DUP -> {
                            stack.push(stack.peek());
                        }
                        case DUP_X1 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            stack.push(value1);
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case DUP_X2 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            if (isCategory1(value2)) {
                                var value3 = stack.pop();
                                stack.push(value1);
                                stack.push(value3);
                            } else {
                                stack.push(value1);
                            }
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case DUP2 -> {
                            var value1 = stack.peek();
                            if (isCategory1(value1)) {
                                stack.pop();
                                var value2 = stack.peek();
                                stack.push(value1);
                                stack.push(value2);
                            }
                            stack.push(value1);
                        }
                        case DUP2_X1 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            if (isCategory1(value1)) {
                                var value3 = stack.pop();
                                stack.push(value2);
                                stack.push(value1);
                                stack.push(value3);
                            } else {
                                stack.push(value1);
                            }
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case DUP2_X2 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            if (isCategory1(value1)) {
                                var value3 = stack.pop();
                                if (isCategory1(value3)) {
                                    var value4 = stack.pop();
                                    stack.push(value2);
                                    stack.push(value1);
                                    stack.push(value4);
                                } else {
                                    stack.push(value2);
                                    stack.push(value1);
                                }
                                stack.push(value3);
                            } else {
                                if (isCategory1(value2)) {
                                    var value3 = stack.pop();
                                    stack.push(value1);
                                    stack.push(value3);
                                } else {
                                    stack.push(value1);
                                }
                            }
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case SWAP -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            stack.push(value1);
                            stack.push(value2);
                        }
                        default ->
                            throw new UnsupportedOperationException("Unsupported stack instruction: " + inst);
                    }
                }
                case PseudoInstruction _ -> {}
                case Instruction inst ->
                    throw new UnsupportedOperationException("Unsupported instruction: " + inst.opcode().name());
                default ->
                    throw new UnsupportedOperationException("Unsupported code element: " + elements.get(i));
            }
        }
    }

    private static boolean isCategory1(Value v) {
        return BytecodeGenerator.toTypeKind(v.type()).slotSize() == 1;
    }
}
