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
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapTableAttribute;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.constantpool.FloatEntry;
import java.lang.classfile.constantpool.IntegerEntry;
import java.lang.classfile.constantpool.LongEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.ExceptionCatch;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LineNumber;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.LocalVariable;
import java.lang.classfile.instruction.LocalVariableType;
import java.lang.classfile.instruction.LookupSwitchInstruction;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.classfile.instruction.ReturnInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.classfile.instruction.SwitchCase;
import java.lang.classfile.instruction.TableSwitchInstruction;
import java.lang.classfile.instruction.ThrowInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.FieldDesc;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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
                        if (b.parameters().size() == 0) {
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
                for (int i = 0; i < ni - 1; i++) {
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
        //                } else if (lop instanceof BytecodeInstructionOps.InvokeInstructionOp inst) {
        //                    MethodTypeDesc descriptor = inst.callOpDescriptor();
        //
        //                    List<Value> operands = new ArrayList<>();
        //                    for (int p = 0; p < inst.desc().type().parameters().size(); p++) {
        //                        operands.add(stack.pop());
        //                    }
        //
        //                    switch (inst.kind()) {
        //                        case VIRTUAL:
        //                        case INTERFACE:
        //                            operands.add(stack.pop());
        //                            // Fallthrough
        //                        case STATIC: {
        //                            Collections.reverse(operands);
        //                            Op.Result result = b.op(CoreOps.invoke(descriptor.returnType(), inst.desc(), operands.toArray(Value[]::new)));
        //                            if (!result.type().equals(TypeDesc.VOID)) {
        //                                stack.push(result);
        //                            }
        //                            break;
        //                        }
        //                        case SPECIAL: {
        //                            if (inst.desc().name().equals("<init>")) {
        //                                Collections.reverse(operands);
        //
        //                                TypeDesc ref = descriptor.parameters().get(0);
        //                                List<TypeDesc> params = descriptor.parameters().subList(1, descriptor.parameters().size());
        //                                MethodTypeDesc constructorDescriptor = MethodTypeDesc.methodType(ref, params);
        //                                Op.Result result = b.op(CoreOps._new(constructorDescriptor, operands.toArray(Value[]::new)));
        //                                stack.push(result);
        //                            } else {
        //                                operands.add(stack.pop());
        //                                Collections.reverse(operands);
        //                                Op.Result result = b.op(CoreOps.invoke(descriptor.returnType(), inst.desc(), operands.toArray(Value[]::new)));
        //                                if (!result.type().equals(TypeDesc.VOID)) {
        //                                    stack.push(result);
        //                                }
        //                                break;
        //                            }
        //                        }
        //
        //                    }
        //                } else if (lop instanceof BytecodeInstructionOps.NewInstructionOp inst) {
        //                    // Skip over this and the dup to process the invoke special
        //                    if (i + 2 >= nops - 1) {
        //                        throw new UnsupportedOperationException("new must be followed by dup and invokespecial");
        //                    }
        //                    Op dup = lb.ops().get(i + 1);
        //                    if (!(dup instanceof BytecodeInstructionOps.DupInstructionOp)) {
        //                        throw new UnsupportedOperationException("new must be followed by dup and invokespecial");
        //                    }
        //                    Op special = lb.ops().get(i + 2);
        //                    if (special instanceof BytecodeInstructionOps.InvokeInstructionOp invoke) {
        //                        if (!invoke.desc().name().equals("<init>")) {
        //                            throw new UnsupportedOperationException("new must be followed by dup and invokespecial for <init>");
        //                        }
        //                    } else {
        //                        throw new UnsupportedOperationException("new must be followed by dup and invokespecial");
        //                    }
        //
        //                    i++;
        //                } else if (lop instanceof BytecodeInstructionOps.NewArrayInstructionOp inst) {
        //                    Value length = stack.pop();
        //                    Op.Result result = b.op(CoreOps.newArray(TypeDesc.type(inst.desc(), 1), length));
        //                    stack.push(result);
        //                } else if (lop instanceof BytecodeInstructionOps.MultiNewArrayInstructionOp inst) {
        //                    int dims = inst.dims();
        //                    Value[] counts = new Value[dims];
        //                    for (int d = dims - 1; d >= 0; d--) {
        //                        counts[d] = stack.pop();
        //                    }
        //                    MethodTypeDesc m = MethodTypeDesc.methodType(inst.desc(), Collections.nCopies(dims, TypeDesc.INT));
        //                    Op.Result result = b.op(CoreOps._new(m, counts));
        //                    stack.push(result);
        //                } else if (lop instanceof BytecodeInstructionOps.CheckCastInstructionOp inst) {
        //                    Value instance = stack.pop();
        //                    Op.Result result = b.op(CoreOps.cast(inst.desc(), instance));
        //                    stack.push(result);
        //                } else if (lop instanceof BytecodeInstructionOps.PopInstructionOp inst) {
        //                    stack.pop();
        //                } else if (lop instanceof BytecodeInstructionOps.DupInstructionOp inst) {
        //                    stack.push(stack.peek());
        //                } else if (lop instanceof BytecodeInstructionOps.Frame inst) {
        //                    // Ignore
        //                } else {

                        default ->
                            throw new UnsupportedOperationException("Unsupported code element: " + bcb.instructions.get(i));
                    }
                }

        //        // @@@ cast, select last Instruction, and adjust prior loop
        //        Instruction top = (Instruction) bcb.instructions.get(ni - 1);
        //        if (bcb.isImplicitTermination) {
        //            b.op(new InstructionOp(top));
        //
        //            BytecodeBasicBlock succ = bcb.successors.get(0);
        //            b.op(new TerminatingInstructionOp(null, List.of(c.blockMap.get(succ).successor())));
        //        } else {
        //            List<Block.Reference> successors = bcb.successors.stream().map(s -> c.blockMap.get(s).successor()).toList();
        //            b.op(new TerminatingInstructionOp(top, successors));
        //        }


        //            if (ltop instanceof BytecodeInstructionOps.GotoInstructionOp inst) {
        //                Block slb = inst.successors().get(0).targetBlock();
        //                Block.Reference sb;
        //                // If the block has block parameters for stack operands then
        //                // pop arguments off the stack and use as successor arguments
        //                if (!slb.parameters().isEmpty()) {
        //                    List<Value> args = new ArrayList<>();
        //                    for (int x = 0; x < slb.parameters().size(); x++) {
        //                        args.add(stack.pop());
        //                    }
        //                    sb = blockMap.get(slb).successor(args);
        //                } else {
        //                    sb = blockMap.get(slb).successor();
        //                }
        //                b.op(CoreOps.branch(sb));
        //
        //                wl.push(slb);
        //            } else if (ltop instanceof BytecodeInstructionOps.IfInstructionOp inst) {
        //                Value operand = stack.pop();
        //                Value zero = b.op(CoreOps.constant(TypeDesc.INT, 0));
        //
        //                if (!stack.isEmpty()) {
        //                    throw new UnsupportedOperationException("Operands on stack for branch not supported");
        //                }
        //
        //                BytecodeInstructionOps.Comparison cond = inst.cond().inverse();
        //                Op cop = switch (cond) {
        //                    case EQ -> CoreOps.eq(operand, zero);
        //                    case NE -> CoreOps.neq(operand, zero);
        //                    case LT -> CoreOps.lt(operand, zero);
        //                    case GT -> CoreOps.gt(operand, zero);
        //                    default -> throw new UnsupportedOperationException("Unsupported condition " + cond);
        //                };
        //
        //                Block fslb = inst.successors().get(0).targetBlock();
        //                Block tslb = inst.successors().get(1).targetBlock();
        //                b.op(CoreOps.conditionalBranch(b.op(cop), blockMap.get(fslb).successor(), blockMap.get(tslb).successor()));
        //
        //                wl.push(tslb);
        //                wl.push(fslb);
        //            } else if (ltop instanceof BytecodeInstructionOps.IfcmpInstructionOp inst) {
        //                Value operand2 = stack.pop();
        //                Value operand1 = stack.pop();
        //
        //                if (!stack.isEmpty()) {
        //                    throw new UnsupportedOperationException("Operands on stack for branch not supported");
        //                }
        //
        //                BytecodeInstructionOps.Comparison cond = inst.cond().inverse();
        //                Op cop = switch (cond) {
        //                    case EQ -> CoreOps.eq(operand1, operand2);
        //                    case NE -> CoreOps.neq(operand1, operand2);
        //                    case LT -> CoreOps.lt(operand1, operand2);
        //                    case GT -> CoreOps.gt(operand1, operand2);
        //                    default -> throw new UnsupportedOperationException("Unsupported condition " + cond);
        //                };
        //
        //                Block tslb = inst.trueBranch();
        //                Block fslb = inst.falseBranch();
        //                b.op(CoreOps.conditionalBranch(b.op(cop), blockMap.get(fslb).successor(), blockMap.get(tslb).successor()));
        //
        //                wl.push(tslb);
        //                wl.push(fslb);
        //            } else if (ltop instanceof BytecodeInstructionOps.ReturnInstructionOp inst) {
        //                Value operand = stack.pop();
        //                b.op(CoreOps._return(operand));
        //            } else if (ltop instanceof BytecodeInstructionOps.VoidReturnInstructionOp inst) {
        //                b.op(CoreOps._return());
        //            } else {
        //                throw new UnsupportedOperationException("Unsupported terminating operation: " + ltop.opName());
        //            }

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
