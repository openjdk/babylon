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
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapTableAttribute;
import java.lang.classfile.constantpool.ClassEntry;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ExceptionCatch;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LineNumber;
import java.lang.classfile.instruction.LocalVariable;
import java.lang.classfile.instruction.LocalVariableType;
import java.lang.classfile.instruction.LookupSwitchInstruction;
import java.lang.classfile.instruction.ReturnInstruction;
import java.lang.classfile.instruction.SwitchCase;
import java.lang.classfile.instruction.TableSwitchInstruction;
import java.lang.classfile.instruction.ThrowInstruction;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
//        // @@@ Needs to be cloned when there are two or more successors
//        Map<Integer, Op.Result> locals = new HashMap<>();
//        Deque<Value> stack = new ArrayDeque<>();
//
//        // Map Block arguments to local variables
//        int lvm = 0;
//        for (Block.Parameter bp : eb.parameters()) {
//            // @@@ Reference type
//            Op.Result local = eb.op(CoreOps.var(Integer.toString(lvm), bp));
//            locals.put(lvm++, local);
//        }
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

                liftBytecodeBlock(bcr, bcb, b, c);
            }
        });

        return f;
    }

    static void liftBytecodeBlock(BytecodeMethodBody bcr, BytecodeBasicBlock bcb, Block.Builder b, LiftContext c) {
        int ni = bcb.instructions.size();
        for (int i = 0; i < ni - 1; i++) {
            switch (bcb.instructions.get(i)) {
                case LabelTarget labelTarget -> {
                    // Insert control instructions for exception start/end bodies
                    for (ExceptionCatch tryCatchBlock : bcr.codeModel.exceptionHandlers()) {
                        if (labelTarget.label() == tryCatchBlock.tryStart()) {
                            BytecodeBasicBlock handler = bcr.blockMap.get(tryCatchBlock.handler());
//                            b.op(BytecodeInstructionOps.
//                                    exceptionTableStart(c.blockMap.get(handler).successor()));
                        } else if (labelTarget.label() == tryCatchBlock.tryEnd()) {
//                            b.op(BytecodeInstructionOps.exceptionTableEnd());
                        }
                    }
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
//                if (lop instanceof BytecodeInstructionOps.LoadInstructionOp inst) {
//                    Op.Result result = b.op(CoreOps.varLoad(locals.get(inst.slot())));
//                    stack.push(result);
//                } else if (lop instanceof BytecodeInstructionOps.StoreInstructionOp inst) {
//                    Value operand = stack.pop();
//
//                    Op.Result local = locals.get(inst.slot());
//                    if (local == null) {
//                        local = b.op(CoreOps.var(Integer.toString(lvm), operand));
//                        locals.put(lvm++, local);
//                    } else {
//                        TypeDesc varType = ((CoreOps.VarOp) local.op()).varType();
//                        if (!operand.type().equals(varType)) {
//                            local = b.op(CoreOps.var(Integer.toString(lvm), operand));
//                            locals.put(lvm++, local);
//                        } else {
//                            b.op(CoreOps.varStore(local, operand));
//                        }
//                    }
//                } else if (lop instanceof BytecodeInstructionOps.IIncInstructionOp inst) {
//                    Op.Result local = locals.get(inst.index());
//                    Op.Result v1 = b.op(CoreOps.varLoad(local));
//                    Op.Result v2 = b.op(CoreOps.constant(TypeDesc.INT, inst.incr()));
//                    Op.Result result = b.op(CoreOps.add(v1, v2));
//                    b.op(CoreOps.varStore(local, result));
//                } else if (lop instanceof BytecodeInstructionOps.LdcInstructionOp inst) {
//                    Op.Result result = b.op(CoreOps.constant(inst.type(), inst.value()));
//                    stack.push(result);
//                } else if (lop instanceof BytecodeInstructionOps.ConstInstructionOp inst) {
//                    Op.Result result = b.op(CoreOps.constant(inst.typeDesc(), inst.value()));
//                    stack.push(result);
//                } else if (lop instanceof BytecodeInstructionOps.BipushInstructionOp inst) {
//                    Op.Result result = b.op(CoreOps.constant(TypeDesc.INT, inst.value()));
//                    stack.push(result);
//                } else if (lop instanceof BytecodeInstructionOps.AddInstructionOp inst) {
//                    Value operand2 = stack.pop();
//                    Value operand1 = stack.pop();
//                    Op.Result result = b.op(CoreOps.add(operand1, operand2));
//                    stack.push(result);
//                } else if (lop instanceof BytecodeInstructionOps.GetFieldInstructionOp inst) {
//                    if (inst.kind() == BytecodeInstructionOps.FieldKind.INSTANCE) {
//                        Value operand = stack.pop();
//                        Op.Result result = b.op(CoreOps.fieldLoad(inst.desc(), operand));
//                        stack.push(result);
//                    } else {
//                        Op.Result result = b.op(CoreOps.fieldLoad(inst.desc()));
//                        stack.push(result);
//                    }
//                } else if (lop instanceof BytecodeInstructionOps.PutFieldInstructionOp inst) {
//                    Value value = stack.pop();
//                    if (inst.kind() == BytecodeInstructionOps.FieldKind.INSTANCE) {
//                        Value receiver = stack.pop();
//                        Op.Result result = b.op(CoreOps.fieldStore(inst.desc(), receiver, value));
//                        stack.push(result);
//                    } else {
//                        Op.Result result = b.op(CoreOps.fieldStore(inst.desc(), value));
//                        stack.push(result);
//                    }
//                } else if (lop instanceof BytecodeInstructionOps.ArrayStoreInstructionOp inst) {
//                    Value value = stack.pop();
//                    Value index = stack.pop();
//                    Value array = stack.pop();
//                    b.op(CoreOps.arrayStoreOp(array, index, value));
//                } else if (lop instanceof BytecodeInstructionOps.ArrayLoadInstructionOp inst) {
//                    Value index = stack.pop();
//                    Value array = stack.pop();
//                    Op.Result result = b.op(CoreOps.arrayLoadOp(array, index));
//                    stack.push(result);
//                } else if (lop instanceof BytecodeInstructionOps.ArrayLengthInstructionOp inst) {
//                    Value array = stack.pop();
//                    Op.Result result = b.op(CoreOps.arrayLength(array));
//                    stack.push(result);
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

//                case Instruction instruction -> {
//                    b.op(new InstructionOp(instruction));
//                }
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
