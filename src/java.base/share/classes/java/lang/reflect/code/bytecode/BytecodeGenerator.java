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
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.ArrayDeque;
import java.util.BitSet;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

    public static byte[] generateClassData(MethodHandles.Lookup l, CoreOps.FuncOp fop) {
        String packageName = l.lookupClass().getPackageName();
        String className = packageName.isEmpty()
                ? fop.funcName()
                : packageName + "." + fop.funcName();
        byte[] classBytes = ClassFile.of().build(ClassDesc.of(className),
                clb -> {
                    clb.withMethodBody(
                            fop.funcName(),
                            fop.funcDescriptor().toNominalDescriptor(),
                            ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                            cob -> {
                                ConversionContext c = new ConversionContext(cob);
                                generateBody(fop.body(), cob, c);
                            });
                });
        return classBytes;
    }

    /*
        Live list of slot, value, v, and value, r, after which no usage of v dominates r
        i.e. liveness range.
        Free list, once slot goes dead it is added to the free list, so it can be reused.

        Block args need to have a fixed mapping to locals, unless the stack is used.
     */

    static final class ConversionContext implements BytecodeInstructionOps.MethodVisitorContext {
        final CodeBuilder cb;
        final Deque<BytecodeInstructionOps.ExceptionTableStart> labelStack;
        final Map<Object, Label> labels;
        final Map<Block, LiveSlotSet> liveSet;
        Block current;

        public ConversionContext(CodeBuilder cb) {
            this.cb = cb;
            this.labelStack = new ArrayDeque<>();
            this.labels = new HashMap<>();
            this.liveSet = new HashMap<>();
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

        LiveSlotSet liveSlotSet() {
            return liveSet.get(current);
        }

        int getSlot(Value v) {
            return liveSlotSet().getSlot(v);
        }

        int assignSlot(Value v) {
            return liveSlotSet().assignSlot(v);
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
                cob.ldc(fromValue(constantOp.value()));
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

    private static void generateBody(Body body, CodeBuilder cob, ConversionContext c) {
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

            List<Op> ops = b.ops();
            // True if the last result is retained on the stack for use as first operand of current operation
            boolean isLastOpResultOnStack = false;
            for (int i = 0; i < ops.size() - 1; i++) {
                switch (ops.get(i)) {
                    case ConstantOp op -> {
                        if (op.resultType().equals(TypeDesc.J_L_CLASS)) {
                            // Loading a class constant may throw an exception so it cannot be deferred
                            cob.ldc(fromValue(op.value()));
                        } // else {
                          // Defer process to use, where constants are inlined
                          // This applies to both operands and successor arguments
                    }
                    case VarOp op -> {
                        //     %1 : Var<int> = var %0 @"i";
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        isLastOpResultOnStack = false;
                        // Use slot of variable result
                        int slot = c.assignSlot(op.result());
                        cob.storeInstruction(toTypeKind(op.varType()), slot);
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
                                cob.ldc(fromValue(constantOp.value()));
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
                        switch (toTypeKind(op.resultType())) { //this can be moved to CodeBuilder::neg(TypeKind)
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
                        switch (toTypeKind(op.resultType())) { //this can be moved to CodeBuilder::add(TypeKind)
                            case IntType -> cob.iadd();
                            case LongType -> cob.ladd();
                            case FloatType -> cob.fadd();
                            case DoubleType -> cob.dadd();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case SubOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (toTypeKind(op.resultType())) { //this can be moved to CodeBuilder::sub(TypeKind)
                            case IntType -> cob.isub();
                            case LongType -> cob.lsub();
                            case FloatType -> cob.fsub();
                            case DoubleType -> cob.dsub();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case MulOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (toTypeKind(op.resultType())) { //this can be moved to CodeBuilder::mul(TypeKind)
                            case IntType -> cob.imul();
                            case LongType -> cob.lmul();
                            case FloatType -> cob.fmul();
                            case DoubleType -> cob.dmul();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case DivOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (toTypeKind(op.resultType())) { //this can be moved to CodeBuilder::div(TypeKind)
                            case IntType -> cob.idiv();
                            case LongType -> cob.ldiv();
                            case FloatType -> cob.fdiv();
                            case DoubleType -> cob.ddiv();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case ModOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        switch (toTypeKind(op.resultType())) { //this can be moved to CodeBuilder::rem(TypeKind)
                            case IntType -> cob.irem();
                            case LongType -> cob.lrem();
                            case FloatType -> cob.frem();
                            case DoubleType -> cob.drem();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case ArrayAccessOp.ArrayLoadOp op -> {
                        processOperands(cob, c, op, isLastOpResultOnStack);
                        cob.arrayLoadInstruction(toTypeKind(op.resultType()));
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
                            TypeKind vt = toTypeKind(op.operands().get(0).type());
                            if (vt == TypeKind.IntType) {
                                // Inverse condition and ensure true block is the immediate successor
                                cob.ifThenElse(switch (op) {
                                    case EqOp o -> Opcode.IF_ICMPNE;
                                    case NeqOp o -> Opcode.IF_ICMPEQ;
                                    case GtOp o -> Opcode.IF_ICMPLE;
                                    case GeOp o -> Opcode.IF_ICMPLT;
                                    case LtOp o -> Opcode.IF_ICMPGE;
                                    case LeOp o -> Opcode.IF_ICMPGT;
                                    default ->
                                        throw new UnsupportedOperationException(op.opName());
                                }, CodeBuilder::iconst_1, CodeBuilder::iconst_0);
                            } else {
                                switch (vt) {
                                    case FloatType -> cob.fcmpg(); // FCMPL?
                                    case LongType -> cob.lcmp();
                                    case DoubleType -> cob.dcmpg(); //CMPL?
                                    default ->
                                        throw new UnsupportedOperationException(op.opName() + " on " + vt);
                                }
                                // Inverse condition and ensure true block is the immediate successor
                                cob.ifThenElse(switch (op) {
                                    case EqOp o -> Opcode.IFNE;
                                    case NeqOp o -> Opcode.IFEQ;
                                    case GtOp o -> Opcode.IFLE;
                                    case GeOp o -> Opcode.IFLT;
                                    case LtOp o -> Opcode.IFGE;
                                    case LeOp o -> Opcode.IFGT;
                                    default ->
                                        throw new UnsupportedOperationException(op.opName());
                                }, CodeBuilder::iconst_1, CodeBuilder::iconst_0);
                            }
                        }//else {
                          // Processing is deferred to the CondBrOp, do not process the op result
                    }

//                    case FuncCallOp op -> {}
//                    case ModuleOp op -> {}
//                    case QuotedOp op -> {}
//                    case LambdaOp op -> {}
//                    case ClosureOp op -> {}
//                    case ClosureCallOp op -> {}
//                    case ReturnOp op -> {}
//                    case ThrowOp op -> {}
//                    case UnreachableOp op -> {}
//                    case YieldOp op -> {}
//                    case BranchOp op -> {}
//                    case ConditionalBranchOp op -> {}
//                    case InvokeOp op -> {}
//                    case ConvOp op -> {}
//                    case NewOp op -> {}
//                    case FieldAccessOp.FieldLoadOp op -> {}
//                    case FieldAccessOp.FieldStoreOp op -> {}
//                    case InstanceOfOp op -> {}
//                    case CastOp op -> {}
//                    case VarAccessOp.VarLoadOp op -> {}
//                    case VarAccessOp.VarStoreOp op -> {}
//                    case TupleOp op -> {}
//                    case TupleLoadOp op -> {}
//                    case TupleWithOp op -> {}
                    default ->
                        throw new UnsupportedOperationException("Unsupported operation: " + ops.get(i));
                }
            }

//            Op top = b.terminatingOp();
//            if (top instanceof BytecodeInstructionOps.GotoInstructionOp inst) {
//                Block s = inst.successors().get(0).targetBlock();
//                int bi = blocks.indexOf(b);
//                int si = blocks.indexOf(s);
//                // If successor occurs immediately after this block,
//                // then no need for goto instruction
//                if (bi != si - 1) {
//                    inst.apply(mv, c);
//                }
//            } else if (top instanceof BytecodeInstructionOps.TerminatingInstructionOp inst) {
//                inst.apply(mv, c);
//            } else if (top instanceof BytecodeInstructionOps.ControlInstructionOp inst) {
//                inst.apply(mv, c);
//            } else {
//                throw new UnsupportedOperationException("Unsupported operation: " + top.opName());
//            }
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
            if (users.isEmpty()) {
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
}
