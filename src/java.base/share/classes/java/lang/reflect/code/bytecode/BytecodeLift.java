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
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapTableAttribute;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.ExceptionCatch;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LineNumber;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.LocalVariable;
import java.lang.classfile.instruction.LocalVariableType;
import java.lang.classfile.instruction.LookupSwitchInstruction;
import java.lang.classfile.instruction.NewMultiArrayInstruction;
import java.lang.classfile.instruction.NewObjectInstruction;
import java.lang.classfile.instruction.NewPrimitiveArrayInstruction;
import java.lang.classfile.instruction.NewReferenceArrayInstruction;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.classfile.instruction.ReturnInstruction;
import java.lang.classfile.instruction.StackInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.classfile.instruction.SwitchCase;
import java.lang.classfile.instruction.TableSwitchInstruction;
import java.lang.classfile.instruction.ThrowInstruction;
import java.lang.classfile.instruction.TypeCheckInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.FieldDesc;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import sun.nio.ch.Streams;

public class BytecodeLift {

    BytecodeLift() {
    }

//    //
//    // Lift to core dialect
//
//    public static CoreOps.FuncOp liftToCoreDialect(CoreOps.FuncOp lf) {
//        Body.Builder body = Body.Builder.of(null, lf.funcDescriptor());
//        liftToCoreDialect(lf.body(), body);
//        return CoreOps.func(lf.funcName(), body);
//    }
//
//    // @@@ boolean, byte, short, and char are erased to int on the stack
//
//    public static void liftToCoreDialect(Body lbody, Body.Builder c) {
//        Block.Builder eb = c.entryBlock();
//
//        // Create blocks
//        Map<Block, Block.Builder> blockMap = new HashMap<>();
//        for (Block lb : lbody.blocks()) {
//            Block.Builder b = lb.isEntryBlock() ? eb : eb.block();
//            if (!lb.isEntryBlock()) {
//                for (Block.Parameter lbp : lb.parameters()) {
//                    b.parameter(lbp.type());
//                }
//            }
//            blockMap.put(lb, b);
//        }
//
//
//        // @@@ catch/finally handlers are disconnected block
//        // treat as effectively separate bodies
//
//        Set<Block> visited = new HashSet<>();
//        Deque<Block> wl = new ArrayDeque<>();
//        wl.push(lbody.entryBlock());
//        while (!wl.isEmpty()) {
//            Block lb = wl.pop();
//            if (!visited.add(lb)) {
//                continue;
//            }
//
//            Block.Builder b = blockMap.get(lb);
//
//            // If a non-entry block has parameters then they correspond to stack arguments
//            // Push back block parameters on the stack, in reverse order to preserve order on the stack
//            if (b.parameters().size() > 0 && !b.isEntryBlock()) {
//                for (int i = b.parameters().size() - 1; i >= 0; i--) {
//                    stack.push(b.parameters().get(i));
//                }
//            }
//        }
//    }


    //
    // Lift to bytecode dialect

    static final class LiftContext {
        final Map<BytecodeBasicBlock, Block.Builder> blockMap = new HashMap<>();
    }

    public static CoreOps.FuncOp lift(byte[] classdata, String methodName) {
        BytecodeMethodBody bcr = createBodyForMethod(classdata, methodName);

        MethodTypeDesc methodTypeDescriptor = MethodTypeDesc.ofNominalDescriptor(bcr.methodModel.methodTypeSymbol());

        CoreOps.FuncOp f = CoreOps.func(
                bcr.methodModel.methodName().stringValue(),
                methodTypeDescriptor).body(entryBlock -> {
            LiftContext c = new LiftContext();

            // Create blocks
            int count = 0;
            for (BytecodeBasicBlock bcb : bcr.blocks) {
                Block.Builder b = count > 0 ? entryBlock.block() : entryBlock;

                count++;
                c.blockMap.put(bcb, b);
            }

            // @@@ Needs to be cloned when there are two or more successors
            Map<Integer, Op.Result> locals = new HashMap<>();
            Deque<Value> stack = new ArrayDeque<>();

            // Map Block arguments to local variables
            int lvm = 0;
            for (Block.Parameter bp : entryBlock.parameters()) {
                // @@@ Reference type
                Op.Result local = entryBlock.op(CoreOps.var(Integer.toString(lvm), bp));
                locals.put(lvm++, local);
            }

            // Process blocks
            for (BytecodeBasicBlock bcb : bcr.blocks) {
                Block.Builder b = c.blockMap.get(bcb);

                // Add exception parameter to catch handler blocks
                for (ExceptionCatch tryCatchBlock : bcr.codeModel.exceptionHandlers()) {
                    BytecodeBasicBlock handler = bcr.blockMap.get(tryCatchBlock.handler());
                    if (handler == bcb) {
                        if (b.parameters().isEmpty()) {
                            TypeDesc throwableType = tryCatchBlock.catchType()
                                    .map(ClassEntry::asSymbol)
                                    .map(TypeDesc::ofNominalDescriptor)
                                    .orElse(null);
                            if (throwableType != null) {
                                b.parameter(throwableType);
                            }
                        }
                        break;
                    }
                }

                // If the frame has operand stack elements then represent as block arguments
                if (bcb.frame != null) {
//                    BytecodeInstructionOps.Frame frame = BytecodeInstructionOps.frame(bcb.frame);
//                    if (frame.hasOperandStackElements()) {
//                        for (TypeDesc t : frame.operandStackTypes()) {
//                            b.parameter(t);
//                        }
//                    }
//                    b.op(frame);
                }

                int ni = bcb.instructions.size();
                for (int i = 0; i < ni; i++) {
                    switch (bcb.instructions.get(i)) {
                        case LabelTarget labelTarget -> {
                            // Insert control instructions for exception start/end bodies
        //                    for (ExceptionCatch tryCatchBlock : bcr.codeModel.exceptionHandlers()) {
        //                        if (labelTarget.label() == tryCatchBlock.tryStart()) {
        //                            BytecodeBasicBlock handler = bcr.blockMap.get(tryCatchBlock.handler());
        //                            b.op(BytecodeInstructionOps.
        //                                    exceptionTableStart(c.blockMap.get(handler).successor()));
        //                        } else if (labelTarget.label() == tryCatchBlock.tryEnd()) {
        //                            b.op(BytecodeInstructionOps.exceptionTableEnd());
        //                        }
        //                    }
                        }
                        case LineNumber ln -> {
                            // @@@ Add special line number instructions
                        }
                        case LocalVariable lv -> {
                            // @@@
                        }
                        case LocalVariableType lvt -> {
                            // @@@
                        }
                        case LoadInstruction inst -> {
                            stack.push(b.op(CoreOps.varLoad(locals.get(inst.slot()))));
                        }
                        case StoreInstruction inst -> {
                            Value operand = stack.pop();
                            Op.Result local = locals.get(inst.slot());
                            if (local == null) {
                                local = b.op(CoreOps.var(Integer.toString(lvm), operand));
                                locals.put(lvm++, local);
                            } else {
                                TypeDesc varType = ((CoreOps.VarOp) local.op()).varType();
                                if (!operand.type().equals(varType)) {
                                    local = b.op(CoreOps.var(Integer.toString(lvm), operand));
                                    locals.put(lvm++, local);
                                } else {
                                    b.op(CoreOps.varStore(local, operand));
                                }
                            }
                        }
                        case IncrementInstruction inst -> {
                            Op.Result local = locals.get(inst.slot());
                            b.op(CoreOps.varStore(local, b.op(CoreOps.add(
                                    b.op(CoreOps.varLoad(local)),
                                    b.op(CoreOps.constant(TypeDesc.INT, inst.constant()))))));
                        }
                        case ConstantInstruction.LoadConstantInstruction inst -> {
                            stack.push(b.op(switch (inst.constantValue()) {
                                case ClassDesc v -> CoreOps.constant(TypeDesc.J_L_CLASS, v);
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
                        case ConstantInstruction inst -> {
                            Op.Result result = b.op(CoreOps.constant(TypeDesc.INT, inst.constantValue()));
                            stack.push(result);
                        }
                        case OperatorInstruction inst -> {
                            Value operand = stack.pop();
                            stack.push(b.op(switch (inst.opcode()) {
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
                                        stack.push(b.op(CoreOps.fieldLoad(fd, stack.pop())));
                                    case GETSTATIC ->
                                        stack.push(b.op(CoreOps.fieldLoad(fd)));
                                    case PUTFIELD -> {
                                        Value value = stack.pop();
                                        stack.push(b.op(CoreOps.fieldStore(fd, stack.pop(), value)));
                                    }
                                    case PUTSTATIC ->
                                        stack.push(b.op(CoreOps.fieldStore(fd, stack.pop())));
                                    default ->
                                        throw new IllegalArgumentException("Unsupported field opcode: " + inst.opcode());
                                }
                        }
                        case ArrayStoreInstruction _ -> {
                            Value value = stack.pop();
                            Value index = stack.pop();
                            b.op(CoreOps.arrayStoreOp(stack.pop(), index, value));
                        }
                        case ArrayLoadInstruction _ -> {
                            Value index = stack.pop();
                            stack.push(b.op(CoreOps.arrayLoadOp(stack.pop(), index)));
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
                                    yield b.op(CoreOps.invoke(mDesc, operands.reversed()));
                                }
                                case INVOKESTATIC ->
                                    b.op(CoreOps.invoke(mDesc, operands.reversed()));
                                case INVOKESPECIAL -> {
                                    if (inst.name().equalsString(ConstantDescs.INIT_NAME)) {
                                        yield b.op(CoreOps._new(
                                                MethodTypeDesc.methodType(
                                                        mType.parameters().get(0),
                                                        mType.parameters().subList(1, mType.parameters().size())),
                                                operands.reversed()));
                                    } else {
                                        operands.add(stack.pop());
                                        yield b.op(CoreOps.invoke(mDesc, operands.reversed()));
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
                            if (i + 2 < ni - 1
                                    && bcb.instructions.get(i + 1) instanceof StackInstruction dup
                                    && dup.opcode() == Opcode.DUP
                                    && bcb.instructions.get(i + 2) instanceof InvokeInstruction init
                                    && init.name().equalsString(ConstantDescs.INIT_NAME)) {
                                i++;
                            } else {
                                throw new UnsupportedOperationException("New must be followed by dup and invokespecial for <init>");
                            }
                        }
                        case NewPrimitiveArrayInstruction inst -> {
                            stack.push(b.op(CoreOps.newArray(
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
                            stack.push(b.op(CoreOps.newArray(
                                    TypeDesc.type(TypeDesc.ofNominalDescriptor(inst.componentType().asSymbol()), 1),
                                    stack.pop())));
                        }
                        case NewMultiArrayInstruction inst -> {
                            stack.push(b.op(CoreOps._new(
                                    MethodTypeDesc.methodType(
                                            TypeDesc.ofNominalDescriptor(inst.arrayType().asSymbol()),
                                            Collections.nCopies(inst.dimensions(), TypeDesc.INT)),
                                    IntStream.range(0, inst.dimensions()).mapToObj(_ -> stack.pop()).toList().reversed())));
                        }
                        case TypeCheckInstruction inst when inst.opcode() == Opcode.CHECKCAST-> {
                            stack.push(b.op(CoreOps.cast(TypeDesc.ofNominalDescriptor(inst.type().asSymbol()), stack.pop())));
                        }
                        case StackInstruction inst -> {
                            switch (inst.opcode()) {
                                case POP, POP2 -> stack.pop(); //check the type width
                                case DUP, DUP2 -> stack.push(stack.peek());
                                //implement all other stack ops
                                default ->
                                    throw new UnsupportedOperationException("Unsupported stack instruction: " + inst);
                            }
                        }
                        case BranchInstruction inst when inst.opcode().isUnconditionalBranch() -> {
                            BytecodeBasicBlock succ = bcb.successors.get(0);
                            Block.Reference sb;
                            // If the block has block parameters for stack operands then
                            // pop arguments off the stack and use as successor arguments
                            if (!b.parameters().isEmpty()) {
                                List<Value> args = new ArrayList<>();
                                for (int x = 0; x < b.parameters().size(); x++) {
                                    args.add(stack.pop());
                                }
                                sb = c.blockMap.get(succ).successor(args);
                            } else {
                                sb = c.blockMap.get(succ).successor();
                            }
                            stack.push(b.op(CoreOps.branch(sb)));
                        }
                        case BranchInstruction inst -> {
                            Value operand = stack.pop();
                            Op cop = switch (inst.opcode()) {
                                case IFNE -> CoreOps.eq(operand, b.op(CoreOps.constant(TypeDesc.INT, 0)));
                                case IFEQ -> CoreOps.neq(operand, b.op(CoreOps.constant(TypeDesc.INT, 0)));
                                case IFGE -> CoreOps.lt(operand, b.op(CoreOps.constant(TypeDesc.INT, 0)));
                                case IFLE -> CoreOps.gt(operand, b.op(CoreOps.constant(TypeDesc.INT, 0)));
                                case IFGT -> CoreOps.le(operand, b.op(CoreOps.constant(TypeDesc.INT, 0)));
                                case IFLT -> CoreOps.ge(operand, b.op(CoreOps.constant(TypeDesc.INT, 0)));
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
                            BytecodeBasicBlock fslb = bcb.successors.get(0);
                            BytecodeBasicBlock tslb = bcb.successors.get(1);
                            stack.push(b.op(CoreOps.conditionalBranch(b.op(cop), c.blockMap.get(fslb).successor(), c.blockMap.get(tslb).successor())));
                        }
                        case ReturnInstruction inst when inst.typeKind() == TypeKind.VoidType -> {
                            b.op(CoreOps._return());
                        }
                        case ReturnInstruction _ -> {
                            b.op(CoreOps._return(stack.pop()));
                        }
                        default ->
                            throw new UnsupportedOperationException("Unsupported code element: " + bcb.instructions.get(i));
                    }
                }
                // @@@ cast, select last Instruction, and adjust prior loop
                if (bcb.isImplicitTermination) {
                    BytecodeBasicBlock succ = bcb.successors.get(0);
                    Block.Reference sb;
                    // If the block has block parameters for stack operands then
                    // pop arguments off the stack and use as successor arguments
                    if (!b.parameters().isEmpty()) {
                        List<Value> args = new ArrayList<>();
                        for (int x = 0; x < b.parameters().size(); x++) {
                            args.add(stack.pop());
                        }
                        sb = c.blockMap.get(succ).successor(args);
                    } else {
                        sb = c.blockMap.get(succ).successor();
                    }
                    stack.push(b.op(CoreOps.branch(sb)));
                }
            }
        });

        return f;
    }

    //
    // Lift to basic blocks of code elements

    record BytecodeMethodBody(MethodModel methodModel,
                              CodeModel codeModel,
                              List<BytecodeBasicBlock> blocks,
                              Map<Label, BytecodeBasicBlock> blockMap) {
    }

    static final class BytecodeBasicBlock {
        final List<CodeElement> instructions;

        final List<BytecodeBasicBlock> successors;

        StackMapFrameInfo frame;

        boolean isImplicitTermination;

        public BytecodeBasicBlock() {
            this.instructions = new ArrayList<>();
            this.successors = new ArrayList<>();
        }

        void setFrame(StackMapFrameInfo frame) {
            this.frame = frame;
        }

        void setImplicitTermination() {
            isImplicitTermination = true;
        }

        void addInstruction(CodeElement i) {
            instructions.add(i);
        }

        CodeElement firstInstruction() {
            return instructions.get(0);
        }

        CodeElement lastInstruction() {
            return instructions.get(instructions.size() - 1);
        }

        void addSuccessor(BytecodeBasicBlock s) {
            successors.add(s);
        }
    }

    static BytecodeMethodBody createBodyForMethod(byte[] classdata, String methodName) {
        MethodModel methodModel = ClassFile.of().parse(classdata).methods().stream()
                .filter(mm -> mm.methodName().equalsString(methodName))
                .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown method: " + methodName));

        return createBlocks(methodModel);
    }

    static BytecodeMethodBody createBlocks(MethodModel methodModel) {
        CodeModel codeModel = methodModel.code().orElseThrow();

        // Obtain stack map frames
        Map<Label, StackMapFrameInfo> labelToFrameMap = codeModel.attributes().stream()
                .<StackMapFrameInfo>mapMulti((a, consumer) -> {
                    if (a instanceof StackMapTableAttribute sa) {
                        sa.entries().forEach(consumer::accept);
                    }
                })
                .collect(Collectors.toMap(StackMapFrameInfo::target, sa -> sa));

        // Construct list of basic blocks
        Map<Label, BytecodeBasicBlock> blockMap = new HashMap<>();
        List<BytecodeBasicBlock> blocks = new ArrayList<>();
        BytecodeBasicBlock currentBlock = new BytecodeBasicBlock();
        for (CodeElement ce : codeModel) {
            if (ce instanceof LabelTarget labelTarget) {
                StackMapFrameInfo frame = labelToFrameMap.get(labelTarget.label());
                if (frame != null) {
                    // Not first block, nor prior block with non-terminating instruction
                    if (!currentBlock.instructions.isEmpty()) {
                        blocks.add(currentBlock);
                        currentBlock = new BytecodeBasicBlock();
                    }

                    currentBlock.setFrame(frame);
                }

                blockMap.put(labelTarget.label(), currentBlock);
                currentBlock.addInstruction(ce);
            } else if (ce instanceof BranchInstruction ||
                    ce instanceof TableSwitchInstruction ||
                    ce instanceof LookupSwitchInstruction) {
                // End of block, branch
                currentBlock.addInstruction(ce);

                blocks.add(currentBlock);
                currentBlock = new BytecodeBasicBlock();
            } else if (ce instanceof ReturnInstruction || ce instanceof ThrowInstruction) {
                // End of block, method terminating instruction,
                currentBlock.addInstruction(ce);

                blocks.add(currentBlock);
                currentBlock = new BytecodeBasicBlock();
            } else {
                currentBlock.addInstruction(ce);
            }
        }

        // Update successors
        for (int i = 0; i < blocks.size(); i++) {
            BytecodeBasicBlock b = blocks.get(i);
            CodeElement lastElement = b.lastInstruction();
            switch (lastElement) {
                case BranchInstruction bi -> {
                    switch (bi.opcode()) {
                        case GOTO, GOTO_W -> {
                            BytecodeBasicBlock branch = blockMap.get(bi.target());
                            b.addSuccessor(branch);
                        }
                        // Conditional branch
                        default -> {
                            assert !bi.opcode().isUnconditionalBranch();

                            BytecodeBasicBlock tBranch = blockMap.get(bi.target());
                            BytecodeBasicBlock fBranch = blocks.get(i + 1);
                            // True branch is first
                            b.addSuccessor(tBranch);
                            // False (or continuation) branch is second
                            b.addSuccessor(fBranch);
                        }
                    }
                }
                case LookupSwitchInstruction si -> {
                    // Default label is first successor
                    b.addSuccessor(blockMap.get(si.defaultTarget()));
                    addSuccessors(si.cases(), blockMap, b);
                }
                case TableSwitchInstruction si -> {
                    // Default label is first successor
                    b.addSuccessor(blockMap.get(si.defaultTarget()));
                    addSuccessors(si.cases(), blockMap, b);
                }
                // @@@ Merge cases and use _, after merge with master
                case ReturnInstruction ri -> {
                    // Ignore, method terminating
                }
                case ThrowInstruction ti -> {
                    // Ignore, method terminating
                }
                default -> {
                    // Implicit goto next block, add explicitly
                    b.setImplicitTermination();
                    BytecodeBasicBlock branch = blocks.get(i + 1);
                    b.addSuccessor(branch);
                }
            }
        }

        return new BytecodeMethodBody(methodModel, codeModel, blocks, blockMap);
    }

    static void addSuccessors(List<SwitchCase> cases,
                              Map<Label, BytecodeBasicBlock> blockMap,
                              BytecodeBasicBlock b) {
        cases.stream().map(SwitchCase::target)
                .map(blockMap::get)
                .forEach(b::addSuccessor);
    }
}
