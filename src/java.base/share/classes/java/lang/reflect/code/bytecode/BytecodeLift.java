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
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapFrameInfo.*;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Op.Result;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.FieldDesc;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.lang.reflect.code.op.CoreOps.ExceptionRegionEnter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class BytecodeLift {

    private BytecodeLift() {
    }

    // Lift to core dialect
    public static CoreOps.FuncOp lift(byte[] classdata, String methodName) {
        return lift(ClassFile.of(
                ClassFile.DebugElementsOption.DROP_DEBUG,
                ClassFile.LineNumbersOption.DROP_LINE_NUMBERS).parse(classdata).methods().stream()
                        .filter(mm -> mm.methodName().equalsString(methodName))
                        .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown method: " + methodName)));
    }

    public static CoreOps.FuncOp lift(MethodModel methodModel) {
        return CoreOps.func(
                methodModel.methodName().stringValue(),
                MethodTypeDesc.ofNominalDescriptor(methodModel.methodTypeSymbol())).body(entryBlock -> {

            final CodeModel codeModel = methodModel.code().orElseThrow();

            // Fill block map
            final Map<Label, Block.Builder> blockMap = codeModel.findAttribute(Attributes.STACK_MAP_TABLE).map(smta ->
                smta.entries().stream().collect(Collectors.toMap(
                        StackMapFrameInfo::target,
                        frameInfo -> entryBlock.block(frameInfo.stack().stream().map(vti ->
                                switch (vti) {
                                    case SimpleVerificationTypeInfo.ITEM_INTEGER ->
                                        TypeDesc.INT;
                                    case SimpleVerificationTypeInfo.ITEM_FLOAT ->
                                        TypeDesc.FLOAT;
                                    case SimpleVerificationTypeInfo.ITEM_DOUBLE ->
                                        TypeDesc.DOUBLE;
                                    case SimpleVerificationTypeInfo.ITEM_LONG ->
                                        TypeDesc.LONG;
                                    case SimpleVerificationTypeInfo.ITEM_NULL -> // @@@
                                        TypeDesc.J_L_OBJECT;
                                    case SimpleVerificationTypeInfo.ITEM_UNINITIALIZED_THIS ->
                                        TypeDesc.ofNominalDescriptor(methodModel.parent().get().thisClass().asSymbol());
                                    case ObjectVerificationTypeInfo i ->
                                        TypeDesc.ofNominalDescriptor(i.classSymbol());
                                    case UninitializedVerificationTypeInfo i ->
                                        throw new IllegalArgumentException("Unexpected item at frame stack: " + i);
                                    case SimpleVerificationTypeInfo.ITEM_TOP ->
                                        throw new IllegalArgumentException("Unexpected item at frame stack: TOP");
                                }).toList())))).orElseGet(HashMap::new);

            final Deque<Value> stack = new ArrayDeque<>();
            final Map<Integer, Op.Result> locals = new HashMap<>();
            final Map<ExceptionCatch, Result> exceptionRegionsMap = new HashMap<>();

            // Map Block arguments to local variables
            int lvm = 0;
            for (Block.Parameter bp : entryBlock.parameters()) {
                // @@@ Reference type
                Op.Result local = entryBlock.op(CoreOps.var(Integer.toString(lvm), bp));
                locals.put(lvm++, local);
            }

            final List<CodeElement> elements = codeModel.elementList();
            Block.Builder currentBlock = entryBlock;
            for (int i = 0; i < elements.size(); i++) {
                switch (elements.get(i)) {
                    case ExceptionCatch ec -> {
                        // Exception blocks are inserted by label target (below)
                    }
                    case LabelTarget lt -> {
                        // Start of a new block
                        Block.Builder nextBlock = blockMap.computeIfAbsent(lt.label(), _ ->
                                // New block parameter types are calculated from the actual stack
                                entryBlock.block(stack.stream().map(Value::type).toList()));
                        if (currentBlock != null) {
                            // Implicit goto next block, add explicitly
                            // Use stack content as next block arguments
                            currentBlock.op(CoreOps.branch(nextBlock.successor(List.copyOf(stack))));
                            stack.clear();
                        }
                        // Stack is reconstructed from block parameters
                        nextBlock.parameters().forEach(stack::add);
                        currentBlock = nextBlock;
                        // Insert relevant tryStart and tryEnd blocks
                        for (ExceptionCatch ec : codeModel.exceptionHandlers().reversed()) {
                            if (lt.label() == ec.tryStart()) {
                                nextBlock = entryBlock.block(stack.stream().map(Value::type).toList());
                                ExceptionRegionEnter ere = CoreOps.exceptionRegionEnter(nextBlock.successor(List.copyOf(stack)), blockMap.get(ec.handler()).successor());
                                currentBlock.op(ere);
                                exceptionRegionsMap.put(ec, ere.result());
                                stack.clear();
                                // Stack is reconstructed from block parameters
                                nextBlock.parameters().forEach(stack::add);
                                currentBlock = nextBlock;
                            }
                        }
                        for (ExceptionCatch ec : codeModel.exceptionHandlers()) {
                            if (lt.label() == ec.tryEnd()) {
                                nextBlock = entryBlock.block(stack.stream().map(Value::type).toList());
                                currentBlock.op(CoreOps.exceptionRegionExit(exceptionRegionsMap.get(ec), nextBlock.successor()));
                                stack.clear();
                                // Stack is reconstructed from block parameters
                                nextBlock.parameters().forEach(stack::add);
                                currentBlock = nextBlock;
                            }
                        }
                    }
                    case BranchInstruction inst when inst.opcode().isUnconditionalBranch() -> {
                        // Use stack content as target block arguments
                        currentBlock.op(CoreOps.branch(blockMap.get(inst.target()).successor(List.copyOf(stack))));
                        // Flow discontinued, stack cleared to be ready for the next label target
                        stack.clear();
                        currentBlock = null;
                    }
                    case BranchInstruction inst -> {
                        // Conditional branch
                        Value operand = stack.pop();
                        Op cop = switch (inst.opcode()) {
                            case IFNE -> CoreOps.eq(operand, currentBlock.op(CoreOps.constant(TypeDesc.INT, 0)));
                            case IFEQ -> CoreOps.neq(operand, currentBlock.op(CoreOps.constant(TypeDesc.INT, 0)));
                            case IFGE -> CoreOps.lt(operand, currentBlock.op(CoreOps.constant(TypeDesc.INT, 0)));
                            case IFLE -> CoreOps.gt(operand, currentBlock.op(CoreOps.constant(TypeDesc.INT, 0)));
                            case IFGT -> CoreOps.le(operand, currentBlock.op(CoreOps.constant(TypeDesc.INT, 0)));
                            case IFLT -> CoreOps.ge(operand, currentBlock.op(CoreOps.constant(TypeDesc.INT, 0)));
                            case IF_ICMPNE -> CoreOps.eq(stack.pop(), operand);
                            case IF_ICMPEQ -> CoreOps.neq(stack.pop(), operand);
                            case IF_ICMPGE -> CoreOps.lt(stack.pop(), operand);
                            case IF_ICMPLE -> CoreOps.gt(stack.pop(), operand);
                            case IF_ICMPGT -> CoreOps.le(stack.pop(), operand);
                            case IF_ICMPLT -> CoreOps.ge(stack.pop(), operand);
                            default -> throw new UnsupportedOperationException("Unsupported branch instruction: " + inst);
                        };
                        if (!stack.isEmpty()) {
                            throw new UnsupportedOperationException("Operands on stack for branch not supported");
                        }
                        Block.Builder nextBlock = currentBlock.block();
                        currentBlock.op(CoreOps.conditionalBranch(
                                currentBlock.op(cop),
                                nextBlock.successor(),
                                blockMap.get(inst.target()).successor()));
                        currentBlock = nextBlock;
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
                        currentBlock.op(CoreOps._return());
                        // Flow discontinued, stack cleared to be ready for the next label target
                        stack.clear();
                        currentBlock = null;
                    }
                    case ReturnInstruction _ -> {
                        currentBlock.op(CoreOps._return(stack.pop()));
                        // Flow discontinued, stack cleared to be ready for the next label target
                        stack.clear();
                        currentBlock = null;
                    }
                    case ThrowInstruction _ -> {
                        currentBlock.op(CoreOps._throw(stack.pop()));
                        // Flow discontinued, stack cleared to be ready for the next label target
                        stack.clear();
                        currentBlock = null;
                    }
                    case LoadInstruction inst -> {
                        stack.push(currentBlock.op(CoreOps.varLoad(locals.get(inst.slot()))));
                    }
                    case StoreInstruction inst -> {
                        Value operand = stack.pop();
                        Op.Result local = locals.get(inst.slot());
                        if (local == null) {
                            local = currentBlock.op(CoreOps.var(Integer.toString(lvm), operand));
                            locals.put(lvm++, local);
                        } else {
                            TypeDesc varType = ((CoreOps.VarOp) local.op()).varType();
                            if (!operand.type().equals(varType)) {
                                local = currentBlock.op(CoreOps.var(Integer.toString(lvm), operand));
                                locals.put(lvm++, local);
                                // @@@  The slot is reused with a different type
                                // so we need to update the existing entry in the map.
                                // This likely always connects to how to manage the map with conditional branching.
                            } else {
                                currentBlock.op(CoreOps.varStore(local, operand));
                            }
                        }
                    }
                    case IncrementInstruction inst -> {
                        Op.Result local = locals.get(inst.slot());
                        currentBlock.op(CoreOps.varStore(local, currentBlock.op(CoreOps.add(
                                currentBlock.op(CoreOps.varLoad(local)),
                                currentBlock.op(CoreOps.constant(TypeDesc.INT, inst.constant()))))));
                    }
                    case ConstantInstruction inst -> {
                        stack.push(currentBlock.op(switch (inst.constantValue()) {
                            case ClassDesc v -> CoreOps.constant(TypeDesc.J_L_CLASS, TypeDesc.ofNominalDescriptor(v));
                            case Double v -> CoreOps.constant(TypeDesc.DOUBLE, v);
                            case Float v -> CoreOps.constant(TypeDesc.FLOAT, v);
                            case Integer v -> CoreOps.constant(TypeDesc.INT, v);
                            case Long v -> CoreOps.constant(TypeDesc.LONG, v);
                            case String v -> CoreOps.constant(TypeDesc.J_L_STRING, v);
                            default ->
                                // @@@ MethodType, MethodHandle, ConstantDynamic
                                throw new IllegalArgumentException("Unsupported constant value: " + inst.constantValue());
                        }));
                    }
                    case OperatorInstruction inst -> {
                        Value operand = stack.pop();
                        stack.push(currentBlock.op(switch (inst.opcode()) {
                            case IADD, LADD, FADD, DADD ->
                                    CoreOps.add(stack.pop(), operand);
                            case ISUB, LSUB, FSUB, DSUB ->
                                    CoreOps.sub(stack.pop(), operand);
                            case IMUL, LMUL, FMUL, DMUL ->
                                    CoreOps.mul(stack.pop(), operand);
                            case IDIV, LDIV, FDIV, DDIV ->
                                    CoreOps.div(stack.pop(), operand);
                            case IREM, LREM, FREM, DREM ->
                                    CoreOps.mod(stack.pop(), operand);
                            case INEG, LNEG, FNEG, DNEG ->
                                    CoreOps.neg(operand);
                            case ARRAYLENGTH ->
                                    CoreOps.arrayLength(operand);
                            default ->
                                throw new IllegalArgumentException("Unsupported operator opcode: " + inst.opcode());
                        }));
                    }
                    case FieldInstruction inst -> {
                            FieldDesc fd = FieldDesc.field(
                                    TypeDesc.ofNominalDescriptor(inst.owner().asSymbol()),
                                    inst.name().stringValue(),
                                    TypeDesc.ofNominalDescriptor(inst.typeSymbol()));
                            switch (inst.opcode()) {
                                case GETFIELD ->
                                    stack.push(currentBlock.op(CoreOps.fieldLoad(fd, stack.pop())));
                                case GETSTATIC ->
                                    stack.push(currentBlock.op(CoreOps.fieldLoad(fd)));
                                case PUTFIELD -> {
                                    Value value = stack.pop();
                                    stack.push(currentBlock.op(CoreOps.fieldStore(fd, stack.pop(), value)));
                                }
                                case PUTSTATIC ->
                                    stack.push(currentBlock.op(CoreOps.fieldStore(fd, stack.pop())));
                                default ->
                                    throw new IllegalArgumentException("Unsupported field opcode: " + inst.opcode());
                            }
                    }
                    case ArrayStoreInstruction _ -> {
                        Value value = stack.pop();
                        Value index = stack.pop();
                        currentBlock.op(CoreOps.arrayStoreOp(stack.pop(), index, value));
                    }
                    case ArrayLoadInstruction _ -> {
                        Value index = stack.pop();
                        stack.push(currentBlock.op(CoreOps.arrayLoadOp(stack.pop(), index)));
                    }
                    case InvokeInstruction inst -> {
                        MethodTypeDesc mType = MethodTypeDesc.ofNominalDescriptor(inst.typeSymbol());
                        List<Value> operands = new ArrayList<>();
                        for (var _ : mType.parameters()) {
                            operands.add(stack.pop());
                        }
                        MethodDesc mDesc = MethodDesc.method(TypeDesc.ofNominalDescriptor(inst.owner().asSymbol()), inst.name().stringValue(), mType);
                        Op.Result result = switch (inst.opcode()) {
                            case INVOKEVIRTUAL, INVOKEINTERFACE -> {
                                operands.add(stack.pop());
                                yield currentBlock.op(CoreOps.invoke(mDesc, operands.reversed()));
                            }
                            case INVOKESTATIC ->
                                currentBlock.op(CoreOps.invoke(mDesc, operands.reversed()));
                            case INVOKESPECIAL -> {
                                if (inst.name().equalsString(ConstantDescs.INIT_NAME)) {
                                    yield currentBlock.op(CoreOps._new(
                                            MethodTypeDesc.methodType(
                                                    mType.parameters().get(0),
                                                    mType.parameters().subList(1, mType.parameters().size())),
                                            operands.reversed()));
                                } else {
                                    operands.add(stack.pop());
                                    yield currentBlock.op(CoreOps.invoke(mDesc, operands.reversed()));
                                }
                            }
                            default ->
                                throw new IllegalArgumentException("Unsupported invocation opcode: " + inst.opcode());
                        };
                        if (!result.type().equals(TypeDesc.VOID)) {
                            stack.push(result);
                        }
                    }
                    case NewObjectInstruction _ -> {
                        // Skip over this and the dup to process the invoke special
                        if (i + 2 < elements.size() - 1
                                && elements.get(i + 1) instanceof StackInstruction dup
                                && dup.opcode() == Opcode.DUP
                                && elements.get(i + 2) instanceof InvokeInstruction init
                                && init.name().equalsString(ConstantDescs.INIT_NAME)) {
                            i++;
                        } else {
                            throw new UnsupportedOperationException("New must be followed by dup and invokespecial for <init>");
                        }
                    }
                    case NewPrimitiveArrayInstruction inst -> {
                        stack.push(currentBlock.op(CoreOps.newArray(
                                switch (inst.typeKind()) {
                                    case BooleanType -> TypeDesc.BOOLEAN_ARRAY;
                                    case ByteType -> TypeDesc.BYTE_ARRAY;
                                    case CharType -> TypeDesc.CHAR_ARRAY;
                                    case DoubleType -> TypeDesc.DOUBLE_ARRAY;
                                    case FloatType -> TypeDesc.FLOAT_ARRAY;
                                    case IntType -> TypeDesc.INT_ARRAY;
                                    case LongType -> TypeDesc.LONG_ARRAY;
                                    case ShortType -> TypeDesc.SHORT_ARRAY;
                                    default ->
                                            throw new UnsupportedOperationException("Unsupported new primitive array type: " + inst.typeKind());
                                },
                                stack.pop())));
                    }
                    case NewReferenceArrayInstruction inst -> {
                        stack.push(currentBlock.op(CoreOps.newArray(
                                TypeDesc.type(TypeDesc.ofNominalDescriptor(inst.componentType().asSymbol()), 1),
                                stack.pop())));
                    }
                    case NewMultiArrayInstruction inst -> {
                        stack.push(currentBlock.op(CoreOps._new(
                                MethodTypeDesc.methodType(
                                        TypeDesc.ofNominalDescriptor(inst.arrayType().asSymbol()),
                                        Collections.nCopies(inst.dimensions(), TypeDesc.INT)),
                                IntStream.range(0, inst.dimensions()).mapToObj(_ -> stack.pop()).toList().reversed())));
                    }
                    case TypeCheckInstruction inst when inst.opcode() == Opcode.CHECKCAST -> {
                        stack.push(currentBlock.op(CoreOps.cast(TypeDesc.ofNominalDescriptor(inst.type().asSymbol()), stack.pop())));
                    }
                    case StackInstruction inst -> {
                        switch (inst.opcode()) {
                            case POP, POP2 -> stack.pop(); // @@@ check the type width
                            case DUP, DUP2 -> stack.push(stack.peek());
                            //@@@ implement all other stack ops
                            default ->
                                throw new UnsupportedOperationException("Unsupported stack instruction: " + inst);
                        }
                    }
                    default ->
                        throw new UnsupportedOperationException("Unsupported code element: " + elements.get(i));
                }
            }
        });
    }
}
