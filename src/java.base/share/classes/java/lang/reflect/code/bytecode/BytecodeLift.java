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

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeModel;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.reflect.AccessFlag;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.FieldDesc;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.op.CoreOps.ExceptionRegionEnter;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
        if (!methodModel.flags().has(AccessFlag.STATIC)) {
            throw new IllegalArgumentException("Unsuported lift of non-static method: " + methodModel);
        }
        MethodTypeDesc mtd = MethodTypeDesc.ofNominalDescriptor(methodModel.methodTypeSymbol());
        return CoreOps.func(
                methodModel.methodName().stringValue(),
                mtd.toFunctionType()).body(entryBlock -> {

            final CodeModel codeModel = methodModel.code().orElseThrow();
            final HashMap<Label, Block.Builder> blockMap = new HashMap<>();
            final HashMap<Label, Map<Integer, Op.Result>> localsMap = new HashMap<>();
            final Map<ExceptionCatch, Op.Result> exceptionRegionsMap = new HashMap<>();

            Block.Builder currentBlock = entryBlock;
            final Deque<Value> stack = new ArrayDeque<>();
            Map<Integer, Op.Result> locals = new HashMap<>();

            int varIndex = 0;int slot = 0;
            // Initialize local variables from entry block parameters
            for (Block.Parameter bp : entryBlock.parameters()) {
                // @@@ Reference type
                locals.put(slot, entryBlock.op(CoreOps.var(Integer.toString(varIndex++), bp)));
                TypeElement te = bp.type();
                slot += te.equals(JavaType.DOUBLE) || te.equals(JavaType.LONG) ? 2 : 1;
            }

            final List<CodeElement> elements = codeModel.elementList();
            final BitSet visited = new BitSet();
            int initiallyResolved; // This is counter helping to determine if the remaining code is not accessible ("dead")
            while ((initiallyResolved = visited.cardinality()) < elements.size()) {
                for (int i = visited.nextClearBit(0); i < elements.size();) {
                    // We start from the first unvisited instruction and mark it as visited
                    visited.set(i);
                    switch (elements.get(i)) {
                        case ExceptionCatch ec -> {
                            // Exception blocks are inserted by label target (below)
                        }
                        case LabelTarget lt -> {
                            // Start of a new block
                            Block.Builder next = blockMap.get(lt.label());
                            if (currentBlock != null) {
                                // Flow has not been interrupted and we can build next block based on the actual stack and locals
                                if (next == null) {
                                    // New block parameter types are calculated from the actual stack
                                    next = entryBlock.block(stack.stream().map(Value::type).toList());
                                    blockMap.put(lt.label(), next);
                                    localsMap.put(lt.label(), locals);
                                }
                                // Implicit goto next block, add explicitly
                                // Use stack content as next block arguments
                                currentBlock.op(CoreOps.branch(next.successor(List.copyOf(stack))));
                            }
                            if (next != null) {
                                // We know the next block so we can continue
                                currentBlock = next;
                                // Stack is reconstructed from block parameters
                                stack.clear();
                                locals = localsMap.get(lt.label());
                                currentBlock.parameters().forEach(stack::add);
                                // Insert relevant tryStart and construct handler blocks, all in reversed order
                                for (ExceptionCatch ec : codeModel.exceptionHandlers().reversed()) {
                                    if (lt.label() == ec.tryStart()) {
                                        // Get or create handler with the exception as parameter
                                        Block.Builder handler = blockMap.computeIfAbsent(ec.handler(), _ ->
                                                entryBlock.block(List.of(JavaType.ofNominalDescriptor(
                                                        ec.catchType().map(ClassEntry::asSymbol).orElse(ConstantDescs.CD_Throwable)))));
                                        localsMap.putIfAbsent(ec.handler(), locals);
                                        // Create start block
                                        next = entryBlock.block(stack.stream().map(Value::type).toList());
                                        ExceptionRegionEnter ere = CoreOps.exceptionRegionEnter(next.successor(List.copyOf(stack)), handler.successor());
                                        currentBlock.op(ere);
                                        // Store ERE into map for exit
                                        exceptionRegionsMap.put(ec, ere.result());
                                        currentBlock = next;
                                        // Stack is reconstructed from block parameters
                                        stack.clear();
                                        currentBlock.parameters().forEach(stack::add);
                                    }
                                }
                                // Insert relevant tryEnd blocks in normal order
                                for (ExceptionCatch ec : codeModel.exceptionHandlers()) {
                                    if (lt.label() == ec.tryEnd()) {
                                        // Create exit block with parameters constructed from the stack
                                        next = entryBlock.block(stack.stream().map(Value::type).toList());
                                        currentBlock.op(CoreOps.exceptionRegionExit(exceptionRegionsMap.get(ec), next.successor()));
                                        currentBlock = next;
                                        // Stack is reconstructed from block parameters
                                        stack.clear();
                                        currentBlock.parameters().forEach(stack::add);
                                    }
                                }
                            } else {
                                // Here we do not know the next block parameters, stack and locals
                                // so we make it unvisited
                                visited.clear(i);
                                // interrupt the flow
                                currentBlock = null;
                                stack.clear();
                                // and skip to a next block
                                while (i < elements.size() - 1 && !(elements.get(i + 1) instanceof LabelTarget)) i++;
                            }
                        }
                        case BranchInstruction inst when inst.opcode().isUnconditionalBranch() -> {
                            // Get or create target block with parameters constructed from the stack
                            currentBlock.op(CoreOps.branch(blockMap.computeIfAbsent(inst.target(), _ ->
                                    entryBlock.block(stack.stream().map(Value::type).toList())).successor(List.copyOf(stack))));
                            localsMap.putIfAbsent(inst.target(), locals);
                            // Flow discontinued, stack cleared to be ready for the next label target
                            stack.clear();
                            currentBlock = null;
                        }
                        case BranchInstruction inst -> {
                            // Conditional branch
                            Value operand = stack.pop();
                            Op cop = switch (inst.opcode()) {
                                case IFNE -> CoreOps.eq(operand, currentBlock.op(CoreOps.constant(JavaType.INT, 0)));
                                case IFEQ -> CoreOps.neq(operand, currentBlock.op(CoreOps.constant(JavaType.INT, 0)));
                                case IFGE -> CoreOps.lt(operand, currentBlock.op(CoreOps.constant(JavaType.INT, 0)));
                                case IFLE -> CoreOps.gt(operand, currentBlock.op(CoreOps.constant(JavaType.INT, 0)));
                                case IFGT -> CoreOps.le(operand, currentBlock.op(CoreOps.constant(JavaType.INT, 0)));
                                case IFLT -> CoreOps.ge(operand, currentBlock.op(CoreOps.constant(JavaType.INT, 0)));
                                case IFNULL -> CoreOps.neq(operand, currentBlock.op(CoreOps.constant(JavaType.J_L_OBJECT, Op.NULL_ATTRIBUTE_VALUE)));
                                case IFNONNULL -> CoreOps.eq(operand, currentBlock.op(CoreOps.constant(JavaType.J_L_OBJECT, Op.NULL_ATTRIBUTE_VALUE)));
                                case IF_ICMPNE -> CoreOps.eq(stack.pop(), operand);
                                case IF_ICMPEQ -> CoreOps.neq(stack.pop(), operand);
                                case IF_ICMPGE -> CoreOps.lt(stack.pop(), operand);
                                case IF_ICMPLE -> CoreOps.gt(stack.pop(), operand);
                                case IF_ICMPGT -> CoreOps.le(stack.pop(), operand);
                                case IF_ICMPLT -> CoreOps.ge(stack.pop(), operand);
                                case IF_ACMPEQ -> CoreOps.neq(stack.pop(), operand);
                                case IF_ACMPNE -> CoreOps.eq(stack.pop(), operand);
                                default -> throw new UnsupportedOperationException("Unsupported branch instruction: " + inst);
                            };
                            if (!stack.isEmpty()) {
                                throw new UnsupportedOperationException("Operands on stack for branch not supported");
                            }
                            Block.Builder nextBlock = currentBlock.block();
                            currentBlock.op(CoreOps.conditionalBranch(currentBlock.op(cop),
                                    nextBlock.successor(),
                                    // Get or create target block
                                    blockMap.computeIfAbsent(inst.target(), _ ->
                                            entryBlock.block(stack.stream().map(Value::type).toList())).successor()));
                            localsMap.putIfAbsent(inst.target(), locals);
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
                                locals.put(inst.slot(), currentBlock.op(CoreOps.var(Integer.toString(varIndex++), operand)));
                            } else {
                                TypeElement varType = ((CoreOps.VarOp) local.op()).varType();
                                if (!operand.type().equals(varType)) {
                                    // @@@ How to override local slots?
                                    locals = new HashMap<>(locals);
                                    locals.put(inst.slot(), currentBlock.op(CoreOps.var(Integer.toString(varIndex++), operand)));
                                } else {
                                    currentBlock.op(CoreOps.varStore(local, operand));
                                }
                            }
                        }
                        case IncrementInstruction inst -> {
                            Op.Result local = locals.get(inst.slot());
                            currentBlock.op(CoreOps.varStore(local, currentBlock.op(CoreOps.add(
                                    currentBlock.op(CoreOps.varLoad(local)),
                                    currentBlock.op(CoreOps.constant(JavaType.INT, inst.constant()))))));
                        }
                        case ConstantInstruction inst -> {
                            stack.push(currentBlock.op(switch (inst.constantValue()) {
                                case ClassDesc v -> CoreOps.constant(JavaType.J_L_CLASS, JavaType.ofNominalDescriptor(v));
                                case Double v -> CoreOps.constant(JavaType.DOUBLE, v);
                                case Float v -> CoreOps.constant(JavaType.FLOAT, v);
                                case Integer v -> CoreOps.constant(JavaType.INT, v);
                                case Long v -> CoreOps.constant(JavaType.LONG, v);
                                case String v -> CoreOps.constant(JavaType.J_L_STRING, v);
                                default ->
                                    // @@@ MethodType, MethodHandle, ConstantDynamic
                                    throw new IllegalArgumentException("Unsupported constant value: " + inst.constantValue());
                            }));
                        }
                        case ConvertInstruction inst -> {
                            stack.push(currentBlock.op(CoreOps.conv(switch (inst.toType()) {
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
                                        JavaType.ofNominalDescriptor(inst.owner().asSymbol()),
                                        inst.name().stringValue(),
                                        JavaType.ofNominalDescriptor(inst.typeSymbol()));
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
                            MethodDesc mDesc = MethodDesc.method(
                                    JavaType.ofNominalDescriptor(inst.owner().asSymbol()),
                                    inst.name().stringValue(),
                                    mType);
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
                            if (!result.type().equals(JavaType.VOID)) {
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
                            stack.push(currentBlock.op(CoreOps.newArray(
                                    JavaType.type(JavaType.ofNominalDescriptor(inst.componentType().asSymbol()), 1),
                                    stack.pop())));
                        }
                        case NewMultiArrayInstruction inst -> {
                            stack.push(currentBlock.op(CoreOps._new(
                                    MethodTypeDesc.methodType(
                                            JavaType.ofNominalDescriptor(inst.arrayType().asSymbol()),
                                            Collections.nCopies(inst.dimensions(), JavaType.INT)),
                                    IntStream.range(0, inst.dimensions()).mapToObj(_ -> stack.pop()).toList().reversed())));
                        }
                        case TypeCheckInstruction inst when inst.opcode() == Opcode.CHECKCAST -> {
                            stack.push(currentBlock.op(CoreOps.cast(JavaType.ofNominalDescriptor(inst.type().asSymbol()), stack.pop())));
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
                        case Instruction inst ->
                            throw new UnsupportedOperationException("Unsupported instruction: " + inst.opcode().name());
                        default ->
                            throw new UnsupportedOperationException("Unsupported code element: " + elements.get(i));
                    }
                    if (visited.get(++i)) {
                        // Interrupt the flow if the following instruction has been already visited
                        currentBlock = null;
                        stack.clear();
                        // and continue with the next unvisited instruction
                        i = visited.nextClearBit(i);
                    }
                }
                if (visited.cardinality() == initiallyResolved) {
                    // If there is no progress, all remaining blocks are dead code
                    // we may alternatively just exit and ignore the dead code
                    throw new IllegalArgumentException("Dead code detected.");
                }
            }
        });
    }
}
