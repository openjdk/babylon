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

import java.lang.classfile.ClassModel;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.components.ClassPrinter;
import java.lang.reflect.code.op.CoreOps.*;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.classfile.CodeBuilder.BlockCodeBuilder;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import static java.lang.classfile.TypeKind.DoubleType;
import static java.lang.classfile.TypeKind.FloatType;
import static java.lang.classfile.TypeKind.LongType;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.code.Block.Reference;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.Liveness;
import java.lang.reflect.code.descriptor.FieldDesc;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

public final class BytecodeGenerator {
    private BytecodeGenerator() {
    }

    public static MethodHandle generate(MethodHandles.Lookup l, CoreOps.FuncOp fop) {
        byte[] classBytes = generateClassData(l, fop);

        {
            print(classBytes);
            try {
                File f = new File("f.class");
                try (FileOutputStream fos = new FileOutputStream(f)) {
                    fos.write(classBytes);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        MethodHandles.Lookup hcl;
        try {
            hcl = l.defineHiddenClass(classBytes, true);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }

        try {
            MethodType mt = fop.funcDescriptor().resolve(hcl);
            return hcl.findStatic(hcl.lookupClass(), fop.funcName(), mt);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static void print(byte[] classBytes) {
        ClassModel cm = ClassFile.of().parse(classBytes);
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES, System.out::print);
    }

    private static class DebugTransform implements CodeTransform {
        private int bci = 0;

        @Override
        public void accept(CodeBuilder cob, CodeElement coe) {
            cob.accept(coe);
            switch (coe) {
                case LabelTarget lt -> System.out.println(bci + ": " + lt + " (" + lt.label().hashCode() + ")");
                case BranchInstruction bi -> System.out.println(bci + ": " + bi + " (" + bi.target().hashCode() + ")");
                default -> System.out.println(bci + ": " + coe);
            }
            if (coe instanceof Instruction ins) bci += ins.sizeInBytes();
        }

    }

    public static byte[] generateClassData(MethodHandles.Lookup lookup, CoreOps.FuncOp fop) {
        String packageName = lookup.lookupClass().getPackageName();
        String className = packageName.isEmpty()
                ? fop.funcName()
                : packageName + "." + fop.funcName();
        Liveness liveness = new Liveness(fop);
        byte[] classBytes = ClassFile.of().build(ClassDesc.of(className), clb ->
                clb.withMethodBody(
                        fop.funcName(),
                        fop.funcDescriptor().toNominalDescriptor(),
                        ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                        cb -> cb.transforming(new DebugTransform(), cob -> {
                            ConversionContext c = new ConversionContext(lookup, liveness, cob);
                            generateBody(fop.body(), cob, c);
                        })));
        return classBytes;
    }

    /*
        Live list of slot, value, v, and value, r, after which no usage of v dominates r
        i.e. liveness range.
        Free list, once slot goes dead it is added to the free list, so it can be reused.

        Block args need to have a fixed mapping to locals, unless the stack is used.
     */

    static final class ConversionContext implements BytecodeInstructionOps.MethodVisitorContext {
        final MethodHandles.Lookup lookup;
        final Liveness liveness;
        final CodeBuilder cb;
        final Deque<BytecodeInstructionOps.ExceptionTableStart> labelStack;
        final Map<Object, Label> labels;
        final Map<Block, LiveSlotSet> liveSet;
        Block current;
        final Set<Block> catchingBlocks;
        final Map<Block, ExceptionRegionNode> coveredBlocks;

        public ConversionContext(MethodHandles.Lookup lookup, Liveness liveness, CodeBuilder cb) {
            this.lookup = lookup;
            this.liveness = liveness;
            this.cb = cb;
            this.labelStack = new ArrayDeque<>();
            this.labels = new HashMap<>();
            this.liveSet = new HashMap<>();
            this.catchingBlocks = new HashSet<>();
            this.coveredBlocks = new HashMap<>();
        }

        @Override
        public Deque<BytecodeInstructionOps.ExceptionTableStart> exceptionRegionStack() {
            return labelStack;
        }

        @Override
        public Label getLabel(Object b) {
            return labels.computeIfAbsent(b, _b -> cb.newLabel());
        }

        void setCurrentBlock(Block current) {
            this.current = current;
            liveSet.computeIfAbsent(current, b -> new LiveSlotSet());
        }

        LiveSlotSet liveSlotSet(Block b) {
            return liveSet.computeIfAbsent(b, _b -> new LiveSlotSet());
        }

        LiveSlotSet liveSlotSet() {
            return liveSet.get(current);
        }

        int getSlot(Value v) {
            return liveSlotSet().getSlot(v);
        }


        int getOrAssignSlot(Value v, boolean assignIfUnused) {
            return liveSlotSet().getOrAssignSlot(v, assignIfUnused);
        }

        int getOrAssignSlot(Value v) {
            return getOrAssignSlot(v, false);
        }

        int assignSlot(Value v) {
            return liveSlotSet().assignSlot(v);
        }

        void freeSlot(Value v) {
            liveSlotSet().freeSlot(v);
        }

        boolean isLastUse(Value v, Op op) {
            return liveness.isLastUse(v, op);
        }

        void freeSlotsOfOp(Op op) {
            for (Value v : op.operands()) {
                if (isLastUse(v, op)) {
                    freeSlot(v);
                }
            }

            for (Block.Reference s : op.successors()) {
                for (Value v : s.arguments()) {
                    if (isLastUse(v, op)) {
                        freeSlot(v);
                    }
                }
            }
        }

        void transitionLiveSlotSetTo(Block successor) {
            liveSlotSet(successor).transitionLiveSlotSetFrom(liveSlotSet());
        }
    }

    private static void processOperands(CodeBuilder cob,
                                        ConversionContext c,
                                        Op op,
                                        boolean isLastOpResultOnStack) {
        for (int i = isLastOpResultOnStack ? 1 : 0; i < op.operands().size(); i++) {
            Value operand = op.operands().get(i);
            if (operand instanceof Op.Result or &&
                    or.op() instanceof CoreOps.ConstantOp constantOp &&
                    !constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                cob.constantInstruction(fromValue(constantOp.value()));
            } else {
                int slot = c.getSlot(operand);
                cob.loadInstruction(toTypeKind(operand.type()), slot);
            }
        }
    }

    private static ConstantDesc fromValue(Object value) {
        return switch (value) {
            case ConstantDesc cd -> cd;
            case TypeDesc td -> td.toNominalDescriptor();
            default -> throw new IllegalArgumentException("Unsupported constant value: " + value);
        };
    }

    // Determines if the operation result used only by the next operation as the first operand
    private static boolean isResultOnlyUse(Op.Result opr) {
        Set<Op.Result> uses = opr.uses();
        if (uses.size() != 1) {
            return false;
        }

        // Pass over constant operations
        Op.Result use = uses.iterator().next();
        Op nextOp = opr.op();
        do {
            nextOp = opr.declaringBlock().nextOp(nextOp);
        } while (nextOp instanceof CoreOps.ConstantOp);

        if (nextOp == null || use != nextOp.result()) {
            return false;
        }

        // Check if used in successor
        for (Block.Reference s : nextOp.successors()) {
            if (s.arguments().contains(opr)) {
                return false;
            }
        }

        List<Value> operands = nextOp.operands();
        return !operands.isEmpty() && opr == operands.get(0);
    }

    private static boolean isConditionForCondBrOp(CoreOps.BinaryTestOp op) {
        // Result of op has one use as the operand of a CondBrOp op,
        // and both ops are in the same block

        Set<Op.Result> uses = op.result().uses();
        if (uses.size() != 1) {
            return false;
        }
        Op.Result use = uses.iterator().next();

        if (use.declaringBlock() != op.parentBlock()) {
            return false;
        }

        // Check if used in successor
        for (Block.Reference s : use.op().successors()) {
            if (s.arguments().contains(op.result())) {
                return false;
            }
        }

        return use.op() instanceof CoreOps.ConditionalBranchOp;
    }

    private static TypeKind toTypeKind(TypeDesc t) {
        TypeDesc rbt = t.toBasicType();

        if (rbt.equals(TypeDesc.INT)) {
            return TypeKind.IntType;
        } else if (rbt.equals(TypeDesc.LONG)) {
            return TypeKind.LongType;
        } else if (rbt.equals(TypeDesc.FLOAT)) {
            return TypeKind.FloatType;
        } else if (rbt.equals(TypeDesc.DOUBLE)) {
            return TypeKind.DoubleType;
        } else if (rbt.equals(TypeDesc.J_L_OBJECT)) {
            return TypeKind.ReferenceType;
        } else {
            throw new IllegalArgumentException("Bad type: " + t);
        }
    }

    record ExceptionRegionNode(CoreOps.ExceptionRegionEnter ere, int size, ExceptionRegionNode next) {
    }

    static final ExceptionRegionNode NIL = new ExceptionRegionNode(null, 0, null);

    private static void computeExceptionRegionMembership(Body r, ConversionContext c) {
        Set<Block> visited = new HashSet<>();
        Deque<Block> stack = new ArrayDeque<>();
        stack.push(r.entryBlock());

        // Set of catching blocks
        Set<Block> catchingBlocks = c.catchingBlocks;
        // Map of block to stack of covered exception regions
        Map<Block, ExceptionRegionNode> coveredBlocks = c.coveredBlocks;
        // Compute exception region membership
        while (!stack.isEmpty()) {
            Block b = stack.pop();
            if (!visited.add(b)) {
                continue;
            }

            Op top = b.terminatingOp();
            ExceptionRegionNode bRegions = coveredBlocks.get(b);
            if (top instanceof CoreOps.BranchOp bop) {
                if (bRegions != null) {
                    coveredBlocks.put(bop.branch().targetBlock(), bRegions);
                }

                stack.push(bop.branch().targetBlock());
            } else if (top instanceof CoreOps.ConditionalBranchOp cop) {
                if (bRegions != null) {
                    coveredBlocks.put(cop.falseBranch().targetBlock(), bRegions);
                    coveredBlocks.put(cop.trueBranch().targetBlock(), bRegions);
                }

                stack.push(cop.falseBranch().targetBlock());
                stack.push(cop.trueBranch().targetBlock());
            } else if (top instanceof CoreOps.ExceptionRegionEnter er) {
                ArrayList<Block.Reference> catchBlocks = new ArrayList<>(er.catchBlocks());
                Collections.reverse(catchBlocks);
                for (Block.Reference catchBlock : catchBlocks) {
                    catchingBlocks.add(catchBlock.targetBlock());
                    if (bRegions != null) {
                        coveredBlocks.put(catchBlock.targetBlock(), bRegions);
                    }

                    stack.push(catchBlock.targetBlock());
                }

                ExceptionRegionNode n;
                if (bRegions != null) {
                    n = new ExceptionRegionNode(er, bRegions.size + 1, bRegions);
                } else {
                    n = new ExceptionRegionNode(er, 1, NIL);
                }
                coveredBlocks.put(er.start().targetBlock(), n);

                stack.push(er.start().targetBlock());
            } else if (top instanceof CoreOps.ExceptionRegionExit er) {
                assert bRegions != null;

                if (bRegions.size() > 1) {
                    coveredBlocks.put(er.end().targetBlock(), bRegions.next());
                }

                stack.push(er.end().targetBlock());
            }
        }
    }

    private static void branch(CodeBuilder cob, ConversionContext c, List<Block> blocks, Block source, Block target) {
        int bi = blocks.indexOf(source);
        int si = blocks.indexOf(target);
        // If successor occurs immediately after this block,
        // then no need for goto instruction
        if (bi != si - 1) {
            cob.goto_(c.getLabel(target));
        }
    }

    private static void generateBody(Body body, CodeBuilder cob, ConversionContext c) {
        computeExceptionRegionMembership(body, c);

        // Process blocks in topological order
        // A jump instruction assumes the false successor block is
        // immediately after, in sequence, to the predecessor
        // since the jump instructions branch on a true condition
        // Conditions are inverted when lowered to bytecode
        List<Block> blocks = body.blocks();
        for (Block b : blocks) {
            // Ignore any non-entry blocks that have no predecessors
            if (body.entryBlock() != b && b.predecessors().isEmpty()) {
                continue;
            }

            c.setCurrentBlock(b);
            Label blockLabel = c.getLabel(b);
            cob.labelBinding(blockLabel);

            // If b is the entry block then all its parameters conservatively require slots
            // Some unused parameters might be declared before others that are used
            b.parameters().forEach(p -> c.getOrAssignSlot(p, b.isEntryBlock()));

            // If b is a catch block then the exception argument will be represented on the stack
            if (c.catchingBlocks.contains(b)) {
                // Retain block argument for exception table generation
                Block.Parameter ex = b.parameters().get(0);
//                clb.parameter(ex.type());

                // Store in slot if used, otherwise pop
                if (!ex.uses().isEmpty()) {
                    int slot = c.getSlot(ex);
                    cob.storeInstruction(toTypeKind(ex.type()), slot);
                } else {
                    cob.pop();
                }
            }

            List<Op> ops = b.ops();
            // True if the last result is retained on the stack for use as first operand of current operation
            boolean isLastOpResultOnStack = false;
            Op.Result oprOnStack = null;
            for (int i = 0; i < ops.size() - 1; i++) {
                Op o = ops.get(i);
                TypeDesc oprType = o.resultType();
                TypeKind rvt = oprType.equals(TypeDesc.VOID) ? null : toTypeKind(oprType);
                switch (o) {
                    case ConstantOp op -> {
                        if (op.resultType().equals(TypeDesc.J_L_CLASS)) {
                            // Loading a class constant may throw an exception so it cannot be deferred
                            cob.ldc(fromValue(op.value()));
                        } else {
                          // Defer process to use, where constants are inlined
                          // This applies to both operands and successor arguments
                          rvt = null;
                        }
                    }
                    case VarOp op -> {
                        //     %1 : Var<int> = var %0 @"i";
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        isLastOpResultOnStack = false;
                        // Use slot of variable result
                        int slot = c.assignSlot(op.result());
                        cob.storeInstruction(toTypeKind(op.varType()), slot);
                        // Ignore result
                        rvt = null;
                    }
                    case VarAccessOp.VarLoadOp op -> {
                        // Use slot of variable result
                        int slot = c.getSlot(op.operands().get(0));
                        CoreOps.VarOp vop = op.varOp();
                        cob.loadInstruction(toTypeKind(vop.varType()), slot);
                    }
                    case VarAccessOp.VarStoreOp op -> {
                        if (!isLastOpResultOnStack) {
                            Value operand = op.operands().get(1);
                            if (operand instanceof Op.Result or &&
                                    or.op() instanceof CoreOps.ConstantOp constantOp &&
                                    !constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                                cob.constantInstruction(fromValue(constantOp.value()));
                            } else {
                                int slot = c.getSlot(operand);
                                cob.loadInstruction(toTypeKind(operand.type()), slot);
                            }
                            isLastOpResultOnStack = false;
                        }
                        // Use slot of variable result
                        int slot = c.getSlot(op.operands().get(0));
                        CoreOps.VarOp vop = op.varOp();
                        cob.storeInstruction(toTypeKind(vop.varType()), slot);
                    }
                    case NegOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::neg(TypeKind)
                            case IntType -> cob.ineg();
                            case LongType -> cob.lneg();
                            case FloatType -> cob.fneg();
                            case DoubleType -> cob.dneg();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case NotOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.ifThenElse(CodeBuilder::iconst_0, CodeBuilder::iconst_1);
                    }
                    case AddOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::add(TypeKind)
                            case IntType -> cob.iadd();
                            case LongType -> cob.ladd();
                            case FloatType -> cob.fadd();
                            case DoubleType -> cob.dadd();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case SubOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::sub(TypeKind)
                            case IntType -> cob.isub();
                            case LongType -> cob.lsub();
                            case FloatType -> cob.fsub();
                            case DoubleType -> cob.dsub();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case MulOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::mul(TypeKind)
                            case IntType -> cob.imul();
                            case LongType -> cob.lmul();
                            case FloatType -> cob.fmul();
                            case DoubleType -> cob.dmul();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case DivOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::div(TypeKind)
                            case IntType -> cob.idiv();
                            case LongType -> cob.ldiv();
                            case FloatType -> cob.fdiv();
                            case DoubleType -> cob.ddiv();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case ModOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::rem(TypeKind)
                            case IntType -> cob.irem();
                            case LongType -> cob.lrem();
                            case FloatType -> cob.frem();
                            case DoubleType -> cob.drem();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case ArrayAccessOp.ArrayLoadOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.arrayLoadInstruction(rvt);
                    }
                    case ArrayAccessOp.ArrayStoreOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        TypeKind evt = toTypeKind(op.operands().get(2).type());
                        cob.arrayStoreInstruction(evt);
                    }
                    case ArrayLengthOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.arraylength();
                    }
                    case BinaryTestOp op -> {
                        if (!isConditionForCondBrOp(op)) {
                            processOperands(cob, c, op, isLastOpResultOnStack);
                            conditionalBranch(cob, op, CodeBuilder::iconst_1, CodeBuilder::iconst_0);
                        } else {
                            // Processing is deferred to the CondBrOp, do not process the op result
                            rvt = null;
                        }
                    }
                    case NewOp op -> {
                        TypeDesc t = op.constructorDescriptor().returnType();
                        switch (t.dimensions()) {
                            case 0 -> {
                                if (isLastOpResultOnStack) {
                                    int slot = c.assignSlot(oprOnStack);
                                    cob.storeInstruction(rvt, slot);
                                    isLastOpResultOnStack = false;
                                    oprOnStack = null;
                                }
                                cob.new_(t.toNominalDescriptor())
                                   .dup();
                                processOperands(cob, c, op, false);
                                cob.invokespecial(
                                        op.resultType().toNominalDescriptor(),
                                        ConstantDescs.INIT_NAME,
                                        op.constructorDescriptor().toNominalDescriptor().changeReturnType(ConstantDescs.CD_void));
                            }
                            case 1 -> {
                                processOperands(cob, c, op, isLastOpResultOnStack);
                                ClassDesc ctd = t.componentType().toNominalDescriptor();
                                if (ctd.isPrimitive()) {
                                    cob.newarray(TypeKind.from(ctd));
                                } else {
                                    cob.anewarray(ctd);
                                }
                            }
                            default -> {
                                processOperands(cob, c, op, isLastOpResultOnStack);
                                cob.multianewarray(t.toNominalDescriptor(), op.operands().size());
                            }
                        }
                    }
                    case InvokeOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        // @@@ Enhance method descriptor to include how the method is to be invoked
                        // Example result of DirectMethodHandleDesc.toString()
                        //   INTERFACE_VIRTUAL/IntBinaryOperator::applyAsInt(IntBinaryOperator,int,int)int
                        // This will avoid the need to reflectively operate on the descriptor
                        // which may be insufficient in certain cases.
                        DirectMethodHandleDesc.Kind descKind;
                        try {
                            descKind = resolveToMethodHandleDesc(c.lookup, op.invokeDescriptor()).kind();
                        } catch (ReflectiveOperationException e) {
                            // @@@ Approximate fallback
                            if (op.hasReceiver()) {
                                descKind = DirectMethodHandleDesc.Kind.VIRTUAL;
                            } else {
                                descKind = DirectMethodHandleDesc.Kind.STATIC;
                            }
                        }
                        MethodDesc md = op.invokeDescriptor();
                        cob.invokeInstruction(
                                switch (descKind) {
                                    case STATIC, INTERFACE_STATIC   -> Opcode.INVOKESTATIC;
                                    case VIRTUAL                    -> Opcode.INVOKEVIRTUAL;
                                    case INTERFACE_VIRTUAL          -> Opcode.INVOKEINTERFACE;
                                    case SPECIAL, INTERFACE_SPECIAL -> Opcode.INVOKESPECIAL;
                                    default ->
                                        throw new IllegalStateException("Bad method descriptor resolution: " + op.descriptor() + " > " + op.invokeDescriptor());
                                },
                                md.refType().toNominalDescriptor(),
                                md.name(),
                                md.type().toNominalDescriptor(),
                                switch (descKind) {
                                    case INTERFACE_STATIC, INTERFACE_VIRTUAL, INTERFACE_SPECIAL -> true;
                                    default -> false;
                                });

                        if (op.resultType().equals(TypeDesc.VOID) && !op.operands().isEmpty()) {
                            isLastOpResultOnStack = false;
                        }
                    }
                    case FieldAccessOp.FieldLoadOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        FieldDesc fd = op.fieldDescriptor();
                        if (op.operands().isEmpty()) {
                            cob.getstatic(fd.refType().toNominalDescriptor(), fd.name(), fd.type().toNominalDescriptor());
                        } else {
                            cob.getfield(fd.refType().toNominalDescriptor(), fd.name(), fd.type().toNominalDescriptor());
                        }
                    }
                    case FieldAccessOp.FieldStoreOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        isLastOpResultOnStack = false;
                        FieldDesc fd = op.fieldDescriptor();
                        if (op.operands().size() == 1) {
                            cob.putstatic(fd.refType().toNominalDescriptor(), fd.name(), fd.type().toNominalDescriptor());
                        } else {
                            cob.putfield(fd.refType().toNominalDescriptor(), fd.name(), fd.type().toNominalDescriptor());
                        }
                    }
                    case InstanceOfOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.instanceof_(op.type().toNominalDescriptor());
                    }
                    case CastOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.checkcast(op.type().toNominalDescriptor());
                    }
                    default ->
                        throw new UnsupportedOperationException("Unsupported operation: " + ops.get(i));
                }
                // Free up slots for values that are no longer live
                c.freeSlotsOfOp(o);
                // Assign slot to operation result
                if (rvt != null) {
                    if (!isResultOnlyUse(o.result())) {
                        isLastOpResultOnStack = false;
                        oprOnStack = null;
                        int slot = c.assignSlot(o.result());
                        cob.storeInstruction(rvt, slot);
                    } else {
                        isLastOpResultOnStack = true;
                        oprOnStack = o.result();
                    }
                }
            }
            Op top = b.terminatingOp();
            c.freeSlotsOfOp(top);
            switch (top) {
                case CoreOps.ReturnOp op -> {
                    Value a = op.returnValue();
                    if (a == null) {
                        cob.return_();
                    } else {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.returnInstruction(toTypeKind(a.type()));
                    }
                }
                case ThrowOp op -> {
                    processOperands(cob, c, op, isLastOpResultOnStack);
                    cob.athrow();
                }
                case BranchOp op -> {
                    assignBlockArguments(op, op.branch(), cob, c);
                    branch(cob, c, blocks, b, op.branch().targetBlock());
                }
                case ConditionalBranchOp op -> {
                    if (getConditionForCondBrOp(op) instanceof CoreOps.BinaryTestOp btop) {
                        // Processing of the BinaryTestOp was deferred, so it can be merged with CondBrOp
                        processOperands(cob, c, btop, isLastOpResultOnStack);
                        conditionalBranch(cob, btop,
                                trueBuilder -> {
                                    assignBlockArguments(btop, op.trueBranch(), trueBuilder, c);
                                    branch(trueBuilder, c, blocks, b, op.trueBranch().targetBlock());
                                },
                                falseBuilder -> {
                                    assignBlockArguments(btop, op.falseBranch(), falseBuilder, c);
                                    branch(falseBuilder, c, blocks, b, op.falseBranch().targetBlock());
                                });
                    } else {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.ifThenElse(
                                trueBuilder -> {
                                    assignBlockArguments(op, op.trueBranch(), trueBuilder, c);
                                    branch(trueBuilder, c, blocks, b, op.trueBranch().targetBlock());
                                },
                                falseBuilder -> {
                                    assignBlockArguments(op, op.falseBranch(), falseBuilder, c);
                                    branch(falseBuilder, c, blocks, b, op.falseBranch().targetBlock());
                                });
                    }
                }
                case ExceptionRegionEnter op -> {
                    assignBlockArguments(op, op.start(), cob, c);
                    for (Block.Reference catchBlock : op.catchBlocks()) {
                        c.transitionLiveSlotSetTo(catchBlock.targetBlock());
                    }
                }
                case ExceptionRegionExit op -> {
                    assignBlockArguments(op, op.end(), cob, c);
                    ExceptionRegionEnter enterOp = op.regionStart();
                    Label start = c.getLabel(enterOp.start().targetBlock());
                    Label end = cob.newBoundLabel();
                    for (Reference cbr : enterOp.catchBlocks()) {
                        Block cb = cbr.targetBlock();
                        if (!cb.parameters().isEmpty()) {
                            ClassDesc type = cb.parameters().get(0).type().toNominalDescriptor();
                            cob.exceptionCatch(start, end, c.getLabel(cb), type);
                        } else {
                            cob.exceptionCatchAll(start, end, c.getLabel(cb));
                        }
                    }
                    branch(cob, c, blocks, b, op.end().targetBlock());
                }
                default ->
                    throw new UnsupportedOperationException("Terminating operation not supported: " + top);
            }
        }
    }

    static final class LiveSlotSet {
        final Map<Value, Integer> liveSet;
        final BitSet freeSlots;

        public LiveSlotSet() {
            this.liveSet = new HashMap<>();
            this.freeSlots = new BitSet();
        }

        void transitionLiveSlotSetFrom(LiveSlotSet that) {
            freeSlots.or(that.freeSlots);

            // Filter dead values, those whose slots have been freed
            Iterator<Map.Entry<Value, Integer>> slots = that.liveSet.entrySet().iterator();
            while (slots.hasNext()) {
                var slot = slots.next();
                if (!freeSlots.get(slot.getValue())) {
                    liveSet.put(slot.getKey(), slot.getValue());
                }
            }
        }

        int getSlot(Value v) {
            Integer slot = liveSet.get(v);
            if (slot == null) {
                throw new IllegalArgumentException("Value is not assigned a slot");
            }

            return slot;
        }

        int assignSlot(Value v) {
            if (liveSet.containsKey(v)) {
                throw new IllegalArgumentException("Value is assigned a slot");
            }

            // If no uses then no slot is assigned
            Set<Op.Result> uses = v.uses();
            if (uses.isEmpty()) {
                // @@@
                return -1;
            }

            // Find a free slot
            int slot = findSlot(slotsPerValue(v));

            liveSet.put(v, slot);
            return slot;
        }

        int getOrAssignSlot(Value v) {
            return getOrAssignSlot(v, false);
        }

        int getOrAssignSlot(Value v, boolean assignIfUnused) {
            // If value is already active return slot
            Integer slotBox = liveSet.get(v);
            if (slotBox != null) {
                // Remove any free slot if present for reassignment
                freeSlots.clear(slotBox);
                if (slotsPerValue(v) == 2) {
                    freeSlots.clear(slotBox + 1);
                }
                return slotBox;
            }

            // If no users then no slot is assigned
            Set<Op.Result> users = v.uses();
            if (!assignIfUnused && users.isEmpty()) {
                // @@@
                return -1;
            }

            // Find a free slot
            int slot = findSlot(slotsPerValue(v));

            liveSet.put(v, slot);
            return slot;
        }

        private int findSlot(int nSlots) {
            if (freeSlots.isEmpty()) {
                return createNewSlot();
            } else if (nSlots == 1) {
                int slot = freeSlots.nextSetBit(0);
                freeSlots.clear(slot);
                return slot;
            } else {
                assert nSlots == 2;
                // Find first 2 contiguous slots
                int slot = 0;
                slot = freeSlots.nextSetBit(slot);
                while (slot != -1) {
                    int next = freeSlots.nextSetBit(slot + 1);
                    if (next - slot == 1) {
                        freeSlots.clear(slot);
                        freeSlots.clear(next);
                        return slot;
                    }

                    slot = next;
                }
                return createNewSlot();
            }
        }

        private int createNewSlot() {
            int slot = 0;
            if (!liveSet.isEmpty()) {
                // @@@ this is inefficient, track mox slot value
                Map.Entry<Value, Integer> e = liveSet.entrySet().stream().reduce((e1, e2) -> {
                    return e1.getValue() >= e2.getValue()
                            ? e1 : e2;
                }).get();
                slot = e.getValue() + slotsPerValue(e.getKey());
            }
            return slot;
        }

        void freeSlot(Value v) {
            // Add the value's slot to the free list, if present
            // The value and slot are still preserved in the live set,
            // so slots can still be queried, but no slots should be assigned
            // to new values until it is safe to do so
            Integer slot = liveSet.get(v);
            if (slot != null) {
                freeSlots.set(slot);
                if (slotsPerValue(v) == 2) {
                    freeSlots.set(slot + 1);
                }
            }
        }

        static int slotsPerValue(Value x) {
            return x.type().equals(TypeDesc.DOUBLE) || x.type().equals(TypeDesc.LONG)
                    ? 2
                    : 1;
        }
    }

    private static Op getConditionForCondBrOp(CoreOps.ConditionalBranchOp op) {
        Value p = op.predicate();
        if (p.uses().size() != 1) {
            return null;
        }

        if (p.declaringBlock() != op.parentBlock()) {
            return null;
        }

        // Check if used in successor
        for (Block.Reference s : op.successors()) {
            if (s.arguments().contains(p)) {
                return null;
            }
        }

        if (p instanceof Op.Result or) {
            return or.op();
        } else {
            return null;
        }
    }

    private static void conditionalBranch(CodeBuilder cob, BinaryTestOp op,
                                          Consumer<BlockCodeBuilder> trueBlock, Consumer<BlockCodeBuilder> falseBlock) {
        TypeKind vt = toTypeKind(op.operands().get(0).type());
        if (vt == TypeKind.IntType) {
            cob.ifThenElse(switch (op) {
                case EqOp _ -> Opcode.IF_ICMPEQ;
                case NeqOp _ -> Opcode.IF_ICMPNE;
                case GtOp _ -> Opcode.IF_ICMPGT;
                case GeOp _ -> Opcode.IF_ICMPGE;
                case LtOp _ -> Opcode.IF_ICMPLT;
                case LeOp _ -> Opcode.IF_ICMPLE;
                default ->
                    throw new UnsupportedOperationException(op.opName());
            }, trueBlock, falseBlock);
        } else {
            switch (vt) {
                case FloatType -> cob.fcmpg(); // FCMPL?
                case LongType -> cob.lcmp();
                case DoubleType -> cob.dcmpg(); //CMPL?
                default ->
                    throw new UnsupportedOperationException(op.opName() + " on " + vt);
            }
            cob.ifThenElse(switch (op) {
                case EqOp _ -> Opcode.IFEQ;
                case NeqOp _ -> Opcode.IFNE;
                case GtOp _ -> Opcode.IFGT;
                case GeOp _ -> Opcode.IFGE;
                case LtOp _ -> Opcode.IFLT;
                case LeOp _ -> Opcode.IFLE;
                default ->
                    throw new UnsupportedOperationException(op.opName());
            }, trueBlock, falseBlock);
        }
    }

    private static void assignBlockArguments(Op op, Block.Reference s, CodeBuilder cob, ConversionContext c) {
        List<Value> sargs = s.arguments();
        List<Block.Parameter> bargs = s.targetBlock().parameters();

        // Transition over live-out to successor block
        // All predecessors of successor will have the same live-out set so it does not
        // matter which predecessor performs this action
        c.transitionLiveSlotSetTo(s.targetBlock());

        // First push successor arguments on the stack, then pop and assign
        // so as not to overwrite slots that are reused slots at different argument positions

        // Push successor values on the stack
        for (int i = 0; i < bargs.size(); i++) {
            Block.Parameter barg = bargs.get(i);
            int bslot = c.liveSlotSet(s.targetBlock()).getOrAssignSlot(barg);

            Value value = sargs.get(i);
            if (value instanceof Op.Result or &&
                    or.op() instanceof CoreOps.ConstantOp constantOp &&
                    !constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                cob.constantInstruction(fromValue(constantOp.value()));
            } else {
                int sslot = c.getSlot(value);

                // Assignment only required if slots differ
                if (sslot != bslot) {
                    TypeKind vt = toTypeKind(barg.type());
                    cob.loadInstruction(vt, sslot);
                }
            }
        }

        // Pop successor arguments on the stack assigning to block argument slots if necessary
        for (int i = bargs.size() - 1; i >= 0; i--) {
            Block.Parameter barg = bargs.get(i);
            int bslot = c.liveSlotSet(s.targetBlock()).getOrAssignSlot(barg);

            Value value = sargs.get(i);
            if (value instanceof Op.Result or &&
                    or.op() instanceof CoreOps.ConstantOp constantOp &&
                    !constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                TypeKind vt = toTypeKind(barg.type());
                cob.storeInstruction(vt, bslot);
            } else {
                int sslot = c.getSlot(value);

                // Assignment only required if slots differ
                if (sslot != bslot) {
                    TypeKind vt = toTypeKind(barg.type());
                    cob.storeInstruction(vt, bslot);
                }
            }
        }
    }

    static DirectMethodHandleDesc resolveToMethodHandleDesc(MethodHandles.Lookup l, MethodDesc d) throws ReflectiveOperationException {
        MethodHandle mh = d.resolve(l);

        if (mh.describeConstable().isEmpty()) {
            throw new NoSuchMethodException();
        }

        MethodHandleDesc mhd = mh.describeConstable().get();
        if (!(mhd instanceof DirectMethodHandleDesc dmhd)) {
            throw new NoSuchMethodException();
        }

        return dmhd;
    }
}
