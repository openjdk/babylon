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

import java.lang.classfile.TypeKind;

import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.analysis.Liveness;
import java.lang.reflect.code.bytecode.BytecodeLower.ConditionalBranchConsumer;
import java.lang.reflect.code.bytecode.BytecodeLower.ExceptionRegionNode;
import java.lang.reflect.code.bytecode.BytecodeLower.LiveSlotSet;
import java.lang.reflect.code.bytecode.BytecodeLower.LoweringContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
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

public class BytecodeLower {
    private BytecodeLower() {
    }

    public static CoreOps.FuncOp lowerToBytecodeDialect(CoreOps.FuncOp f) {
        return lowerToBytecodeDialect(MethodHandles.publicLookup(), f);
    }

    public static CoreOps.FuncOp lowerToBytecodeDialect(MethodHandles.Lookup l, CoreOps.FuncOp f) {
        return lowerToBytecodeDialect(l, f, false);
    }

    public static CoreOps.FuncOp lowerToBytecodeDialect(MethodHandles.Lookup lookup, CoreOps.FuncOp f, boolean neverFreeSlots) {
        CoreOps.FuncOp lf = CoreOps.func(f.funcName(), f.funcDescriptor()).body(block -> {
            Liveness l = new Liveness(f);
            LoweringContext c = new LoweringContext(lookup, l, neverFreeSlots);
            lowerBodyToBytecodeDialect(f.body(), block, c);
        });
        return lf;
    }

    static final class LiveSlotSet {
        final boolean neverFreeSlots;
        final Map<Value, Integer> liveSet;
        final BitSet freeSlots;

        public LiveSlotSet(boolean neverFreeSlots) {
            this.neverFreeSlots = neverFreeSlots;
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
            if (neverFreeSlots) {
                return;
            }

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

    static final class LoweringContext {
        final MethodHandles.Lookup lookup;
        final boolean neverFreeSlots;
        final Liveness liveness;
        final Map<Block, LiveSlotSet> liveSet;
        final Map<Block, Block.Builder> blockMap;
        final Set<Block> catchingBlocks;
        final Map<Block, ExceptionRegionNode> coveredBlocks;

        Block current;

        LoweringContext(MethodHandles.Lookup lookup, Liveness liveness, boolean neverFreeSlots) {
            this.lookup = lookup;
            this.neverFreeSlots = neverFreeSlots;
            this.liveness = liveness;
            this.liveSet = new HashMap<>();
            this.blockMap = new HashMap<>();
            this.catchingBlocks = new HashSet<>();
            this.coveredBlocks = new HashMap<>();
        }

        void setCurrentBlock(Block current) {
            this.current = current;
            liveSet.computeIfAbsent(current, b -> new LiveSlotSet(neverFreeSlots));
        }

        LiveSlotSet liveSlotSet(Block b) {
            return liveSet.computeIfAbsent(b, _b -> new LiveSlotSet(neverFreeSlots));
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

        int getOrAssignSlot(Value v, boolean assignIfUnused) {
            return liveSlotSet().getOrAssignSlot(v, assignIfUnused);
        }

        int getOrAssignSlot(Value v) {
            return getOrAssignSlot(v, false);
        }

        void freeSlot(Value v) {
            liveSlotSet().freeSlot(v);
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

        boolean isLastUse(Value v, Op op) {
            return liveness.isLastUse(v, op);
        }

        public void mapBlock(Block b, Block.Builder lb) {
            blockMap.put(b, lb);
        }

        public Block.Builder getLoweredBlock(Block b) {
            return blockMap.get(b);
        }
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

    private static BytecodeInstructionOps.Comparison toComparisonType(CoreOps.BinaryTestOp op) {
        if (op instanceof CoreOps.EqOp) {
            return BytecodeInstructionOps.Comparison.EQ;
        } else if (op instanceof CoreOps.GtOp) {
            return BytecodeInstructionOps.Comparison.GT;
        } else if (op instanceof CoreOps.LtOp) {
            return BytecodeInstructionOps.Comparison.LT;
        } else {
            throw new UnsupportedOperationException(op.opName());
        }
    }

    record ExceptionRegionNode(CoreOps.ExceptionRegionEnter ere, int size, ExceptionRegionNode next) {
    }

    static final ExceptionRegionNode NIL = new ExceptionRegionNode(null, 0, null);

    private static void computeExceptionRegionMembership(Body r, LoweringContext c) {
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

    private static void lowerBodyToBytecodeDialect(Body r, Block.Builder entryBlock, LoweringContext c) {
        computeExceptionRegionMembership(r, c);

        // Copy blocks, preserving topological order
        // Lowered blocks have no arguments
        c.mapBlock(r.entryBlock(), entryBlock);
        List<Block> blocks = r.blocks();
        for (Block b : blocks.subList(1, blocks.size())) {
            Block.Builder lb;
            lb = entryBlock.block();
            c.mapBlock(b, lb);
        }

        // Process blocks in topological order
        for (int bi = 0; bi < blocks.size(); bi++) {
            // Previous block in topological order
            Block pb = bi > 0 ? blocks.get(bi - 1) : null;

            Block b = blocks.get(bi);
            c.setCurrentBlock(b);
            Block.Builder lb = c.getLoweredBlock(b);
            Block.Builder clb = lb;

            // @@@ Generate linear exception ranges when generating bytecode?

            // If disjoint adjacent blocks, then may need to insert linear exception region enter and exit operations
            if (pb != null && !b.predecessors().contains(pb)) {
                ExceptionRegionNode pbRegions = c.coveredBlocks.getOrDefault(pb, NIL);
                ExceptionRegionNode bRegions = c.coveredBlocks.getOrDefault(b, NIL);
                if (pbRegions.size() < bRegions.size()) {
                    // 1. pb < b
                    //    A. enter regions in b up to that covered by pb
                    // @@@ is pbRegions always empty ?

                    // Enter regions in reverse order
                    Deque<CoreOps.ExceptionRegionEnter> regions = new ArrayDeque<>();
                    while (pbRegions != bRegions && bRegions != NIL) {
                        regions.push(bRegions.ere);
                        bRegions = bRegions.next();
                    }

                    for (CoreOps.ExceptionRegionEnter region : regions) {
                        Block.Builder exRegionEnter = lb.block();
                        lb.op(BytecodeInstructionOps.exceptionTableStart(
                                exRegionEnter.successor(),
                                region.catchBlocks().stream().map(b1 -> c.getLoweredBlock(b1.targetBlock()).successor()).toList()));
                        lb = exRegionEnter;
                    }
                } else if (pbRegions.size() > bRegions.size()) {
                    // 2. pb > b
                    //    2.1 pb.exit
                    //      2.1.1 pb.exit.target == b
                    //        A. Nothing
                    //      2.1.2 Otherwise,
                    //        A. ??? Can this occur ???
                    //    2.2 Otherwise,
                    //      A. exit regions in pb up to that covered by b
                    if (!(pb.terminatingOp() instanceof CoreOps.ExceptionRegionExit ere)) {
                        lb.op(BytecodeInstructionOps.exceptionTableEnd());
                    } else {
                        ExceptionRegionNode tRegions = c.coveredBlocks.getOrDefault(ere.end().targetBlock(), NIL);
                        if (tRegions == bRegions) {
                        } else {
                            // @@@ Can this case occur?
                            throw new UnsupportedOperationException();
                        }
                    }
                } else if (pb.terminatingOp() instanceof CoreOps.ExceptionRegionExit) {
                    // 3. pb == b
                    //    3.1 pb.exit
                    //      A. enter pb.exit region in b
                    //         or replace pb.exit.target with pb.branch.target
                    //    3.2 Otherwise,
                    //      A. Nothing
                    Block.Builder exRegionEnter = lb.block();
                    lb.op(BytecodeInstructionOps.exceptionTableStart(
                            exRegionEnter.successor(),
                            bRegions.ere().catchBlocks().stream().map(b1 -> c.getLoweredBlock(b1.targetBlock()).successor()).toList()));
                    lb = exRegionEnter;
                }
            }

            // If b is the entry block then all its parameters conservatively require slots
            // Some unused parameters might be declared before others that are used
            b.parameters().forEach(p -> c.getOrAssignSlot(p, pb == null));

            // If b is a catch block then the exception argument will be represented on the stack
            if (c.catchingBlocks.contains(b)) {
                // Retain block argument for exception table generation
                Block.Parameter ex = b.parameters().get(0);
                clb.parameter(ex.type());

                // Store in slot if used, otherwise pop
                if (!ex.uses().isEmpty()) {
                    int slot = c.getSlot(ex);
                    lb.op(BytecodeInstructionOps.store(toTypeKind(ex.type()), slot));
                } else {
                    lb.op(BytecodeInstructionOps.pop());
                }
            }

            List<Op> ops = b.ops();
            // True if the last result is retained on the stack for use as first operand of current operation
            boolean isLastOpResultOnStack = false;
            Op.Result oprOnStack = null;
            for (int i = 0; i < ops.size() - 1; i++) {
                Op op = ops.get(i);
                Op.Result opr = op.result();
                TypeDesc oprType = opr.type();
                TypeKind rvt = oprType.equals(TypeDesc.VOID) ? null : toTypeKind(oprType);

                if (op instanceof CoreOps.ConstantOp constantOp) {
                    if (constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                        // Loading a class constant may throw an exception so it cannot be deferred
                        lb.op(BytecodeInstructionOps.ldc(constantOp.resultType(), constantOp.value()));
                    } else {
                        // Defer process to use, where constants are inlined
                        // This applies to both operands and successor arguments
                        rvt = null;
                    }
                } else if (op instanceof CoreOps.VarOp vop) {
                    //     %1 : Var<int> = var %0 @"i";
                    processOperands(lb, op, c, isLastOpResultOnStack);
                    isLastOpResultOnStack = false;

                    // Use slot of variable result
                    int slot = c.assignSlot(opr);
                    lb.op(BytecodeInstructionOps.store(toTypeKind(vop.varType()), slot));

                    // Ignore result
                    rvt = null;
                } else if (op instanceof CoreOps.VarAccessOp.VarLoadOp vaop) {
                    // Use slot of variable result
                    int slot = c.getSlot(op.operands().get(0));

                    CoreOps.VarOp vop = vaop.varOp();
                    lb.op(BytecodeInstructionOps.load(toTypeKind(vop.varType()), slot));
                } else if (op instanceof CoreOps.VarAccessOp.VarStoreOp vaop) {
                    if (!isLastOpResultOnStack) {
                        Value operand = op.operands().get(1);
                        if (operand instanceof Op.Result or &&
                                or.op() instanceof CoreOps.ConstantOp constantOp &&
                                !constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                            lb.op(BytecodeInstructionOps.ldc(constantOp.resultType(), constantOp.value()));
                        } else {
                            int slot = c.getSlot(operand);
                            lb.op(BytecodeInstructionOps.load(toTypeKind(operand.type()), slot));
                        }
                        isLastOpResultOnStack = false;
                    }

                    // Use slot of variable result
                    int slot = c.getSlot(op.operands().get(0));

                    CoreOps.VarOp vop = vaop.varOp();
                    lb.op(BytecodeInstructionOps.store(toTypeKind(vop.varType()), slot));
                } else if (op instanceof CoreOps.NegOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.neg(rvt));
                } else if (op instanceof CoreOps.NotOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    // True block
                    Block.Builder ctBlock = lb.block();
                    // False block
                    Block.Builder cfBlock = lb.block();
                    // Merge block
                    Block.Builder mergeBlock = lb.block();

                    lb.op(BytecodeInstructionOps._if(BytecodeInstructionOps.Comparison.NE, ctBlock.successor(), cfBlock.successor()));

                    ctBlock.op(BytecodeInstructionOps._const(TypeKind.IntType, 0));
                    ctBlock.op(BytecodeInstructionOps._goto(mergeBlock.successor()));

                    cfBlock.op(BytecodeInstructionOps._const(TypeKind.IntType, 1));
                    cfBlock.op(BytecodeInstructionOps._goto(mergeBlock.successor()));

                    lb = mergeBlock;
                } else if (op instanceof CoreOps.AddOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.add(rvt));
                } else if (op instanceof CoreOps.SubOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.sub(rvt));
                } else if (op instanceof CoreOps.MulOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.mul(rvt));
                } else if (op instanceof CoreOps.DivOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.div(rvt));
                } else if (op instanceof CoreOps.ModOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.rem(rvt));
                } else if (op instanceof CoreOps.ArrayAccessOp.ArrayLoadOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.aload(rvt));
                } else if (op instanceof CoreOps.ArrayAccessOp.ArrayStoreOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    TypeKind evt = toTypeKind(op.operands().get(2).type());
                    lb.op(BytecodeInstructionOps.astore(evt));
                } else if (op instanceof CoreOps.ArrayLengthOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.arraylength());
                } else if (op instanceof CoreOps.BinaryTestOp btop) {
                    if (!isConditionForCondBrOp(btop)) {
                        lb = comparison(lb, op, toComparisonType(btop),
                                c, isLastOpResultOnStack);
                    } else {
                        // Processing is deferred to the CondBrOp, do not process the op result
                        rvt = null;
                    }
                } else if (op instanceof CoreOps.NewOp newOp) {
                    TypeDesc t = newOp.constructorDescriptor().returnType();
                    if (t.dimensions() > 0) {
                        processOperands(lb, op, c, isLastOpResultOnStack);
                        if (t.dimensions() == 1) {
                            lb.op(BytecodeInstructionOps.newarray(t.componentType()));
                        } else {
                            lb.op(BytecodeInstructionOps.multinewarray(t, op.operands().size()));
                        }
                    } else {
                        if (isLastOpResultOnStack) {
                            int slot = c.assignSlot(oprOnStack);
                            lb.op(BytecodeInstructionOps.store(rvt, slot));
                            isLastOpResultOnStack = false;
                            oprOnStack = null;
                        }

                        lb.op(BytecodeInstructionOps._new(t));
                        lb.op(BytecodeInstructionOps.dup());

                        processOperands(lb, op, c, false);
                        lb.op(BytecodeInstructionOps.invoke(BytecodeInstructionOps.InvokeKind.SPECIAL, MethodDesc.initMethod(newOp.constructorDescriptor())));
                    }
                } else if (op instanceof CoreOps.InvokeOp invokeOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    // @@@ Enhance method descriptor to include how the method is to be invoked
                    // Example result of DirectMethodHandleDesc.toString()
                    //   INTERFACE_VIRTUAL/IntBinaryOperator::applyAsInt(IntBinaryOperator,int,int)int
                    // This will avoid the need to reflectively operate on the descriptor
                    // which may be insufficient in certain cases.
                    DirectMethodHandleDesc.Kind descKind;
                    try {
                        descKind = resolveToMethodHandleDesc(c.lookup, invokeOp.invokeDescriptor()).kind();
                    } catch (ReflectiveOperationException e) {
                        // @@@ Approximate fallback
                        if (invokeOp.hasReceiver()) {
                            descKind = DirectMethodHandleDesc.Kind.VIRTUAL;
                        } else {
                            descKind = DirectMethodHandleDesc.Kind.STATIC;
                        }
                    }

                    BytecodeInstructionOps.InvokeKind ik = switch (descKind) {
                        case STATIC, INTERFACE_STATIC -> BytecodeInstructionOps.InvokeKind.STATIC;
                        case VIRTUAL -> BytecodeInstructionOps.InvokeKind.VIRTUAL;
                        case INTERFACE_VIRTUAL -> BytecodeInstructionOps.InvokeKind.INTERFACE;
                        case SPECIAL, INTERFACE_SPECIAL -> BytecodeInstructionOps.InvokeKind.SPECIAL;
                        default -> throw new IllegalStateException("Bad method descriptor resolution: " +
                                invokeOp.descriptor() + " > " + invokeOp.invokeDescriptor());
                    };
                    boolean isInterface = switch (descKind) {
                        case INTERFACE_STATIC, INTERFACE_VIRTUAL, INTERFACE_SPECIAL -> true;
                        default -> false;
                    };
                    lb.op(BytecodeInstructionOps.invoke(ik, invokeOp.invokeDescriptor(), isInterface));

                    if (rvt == null && !op.operands().isEmpty()) {
                        isLastOpResultOnStack = false;
                    }
                } else if (op instanceof CoreOps.FieldAccessOp.FieldLoadOp fieldOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    if (fieldOp.operands().isEmpty()) {
                        lb.op(BytecodeInstructionOps.getField(BytecodeInstructionOps.FieldKind.STATIC, fieldOp.fieldDescriptor()));
                    } else {
                        lb.op(BytecodeInstructionOps.getField(BytecodeInstructionOps.FieldKind.INSTANCE, fieldOp.fieldDescriptor()));
                    }
                } else if (op instanceof CoreOps.FieldAccessOp.FieldStoreOp fieldOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);
                    isLastOpResultOnStack = false;

                    if (fieldOp.operands().size() == 1) {
                        lb.op(BytecodeInstructionOps.putField(BytecodeInstructionOps.FieldKind.STATIC, fieldOp.fieldDescriptor()));
                    } else {
                        lb.op(BytecodeInstructionOps.putField(BytecodeInstructionOps.FieldKind.INSTANCE, fieldOp.fieldDescriptor()));
                    }
                } else if (op instanceof CoreOps.InstanceOfOp instOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.instanceOf(instOp.type()));
                } else if (op instanceof CoreOps.CastOp castOp) {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    lb.op(BytecodeInstructionOps.checkCast(castOp.type()));
                } else {
                    throw new UnsupportedOperationException("Operation not supported: " + op);
                }

                // Free up slots for values that are no longer live
                c.freeSlotsOfOp(op);

                // Assign slot to operation result
                if (rvt != null) {
                    if (!isResultOnlyUse(opr)) {
                        isLastOpResultOnStack = false;
                        oprOnStack = null;
                        int slot = c.assignSlot(opr);
                        lb.op(BytecodeInstructionOps.store(rvt, slot));
                    } else {
                        isLastOpResultOnStack = true;
                        oprOnStack = opr;
                    }
                }
            }

            Op top = b.terminatingOp();
            c.freeSlotsOfOp(top);
            if (top instanceof CoreOps.ReturnOp op) {
                Value a = op.returnValue();
                if (a == null) {
                    lb.op(BytecodeInstructionOps._return());
                } else {
                    processOperands(lb, op, c, isLastOpResultOnStack);

                    TypeKind vt = toTypeKind(a.type());
                    lb.op(BytecodeInstructionOps._return(vt));
                }
            } else if (top instanceof CoreOps.YieldOp op) {
                processOperands(lb, op, c, isLastOpResultOnStack);
                lb.op(CoreOps._yield());
            } else if (top instanceof CoreOps.ThrowOp _throw) {
                processOperands(lb, _throw, c, isLastOpResultOnStack);

                lb.op(BytecodeInstructionOps.athrow());
            } else if (top instanceof CoreOps.BranchOp op) {
                assignBlockArguments(op, op.branch(), lb, c);
                lb.op(BytecodeInstructionOps._goto(c.getLoweredBlock(op.branch().targetBlock()).successor()));
            } else if (top instanceof CoreOps.ConditionalBranchOp cop) {
                if (getConditionForCondBrOp(cop) instanceof CoreOps.BinaryTestOp btop) {
                    // Processing of the BinaryTestOp was deferred, so it can be merged with CondBrOp
                    conditionalBranch(lb, btop, cop, c, isLastOpResultOnStack,
                            (_lb, tBlock, fBlock) -> {
                                // Inverse condition and ensure true block is the immediate successor, in sequence, of lb
                                var vt = toTypeKind(btop.operands().get(0).type());
                                var cond = toComparisonType(btop).inverse();
                                if (vt == TypeKind.IntType) {
                                    _lb.op(BytecodeInstructionOps.if_cmp(TypeKind.IntType, cond, fBlock.successor(), tBlock.successor()));
                                } else {
                                    _lb.op(BytecodeInstructionOps.cmp(vt));
                                    _lb.op(BytecodeInstructionOps._if(cond, fBlock.successor(), tBlock.successor()));
                                }
                            });

                } else {
                    conditionalBranch(lb, cop, cop, c, isLastOpResultOnStack,
                            (_lb, tBlock, fBlock) -> {
                                _lb.op(BytecodeInstructionOps._if(BytecodeInstructionOps.Comparison.EQ, fBlock.successor(), tBlock.successor()));
                            });
                }
            } else if (top instanceof CoreOps.ExceptionRegionEnter er) {
                assignBlockArguments(er, er.start(), lb, c);
                lb.op(BytecodeInstructionOps.exceptionTableStart(c.getLoweredBlock(er.start().targetBlock()).successor(),
                        er.catchBlocks().stream().map(b1 -> c.getLoweredBlock(b1.targetBlock()).successor()).toList()));

                for (Block.Reference catchBlock : er.catchBlocks()) {
                    c.transitionLiveSlotSetTo(catchBlock.targetBlock());
                }
            } else if (top instanceof CoreOps.ExceptionRegionExit er) {
                assignBlockArguments(er, er.end(), lb, c);
                lb.op(BytecodeInstructionOps.exceptionTableEnd());
                lb.op(BytecodeInstructionOps._goto(c.getLoweredBlock(er.end().targetBlock()).successor()));
            } else {
                throw new UnsupportedOperationException("Terminating operation not supported: " + top);
            }
        }
    }

    private static void processOperands(Block.Builder lb,
                                        Op op,
                                        LoweringContext c,
                                        boolean isLastOpResultOnStack) {
        for (int i = isLastOpResultOnStack ? 1 : 0; i < op.operands().size(); i++) {
            Value operand = op.operands().get(i);
            if (operand instanceof Op.Result or &&
                    or.op() instanceof CoreOps.ConstantOp constantOp &&
                    !constantOp.resultType().equals(TypeDesc.J_L_CLASS)) {
                lb.op(BytecodeInstructionOps.ldc(constantOp.resultType(), constantOp.value()));
            } else {
                int slot = c.getSlot(operand);
                lb.op(BytecodeInstructionOps.load(toTypeKind(operand.type()), slot));
            }
        }
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
        return operands.size() > 0 && opr == operands.get(0);
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

    private static Block.Builder comparison(Block.Builder lb,
                                            Op btop,
                                            BytecodeInstructionOps.Comparison cond, LoweringContext c,
                                            boolean isLastOpResultOnStack) {
        processOperands(lb, btop, c, isLastOpResultOnStack);

        TypeKind vt = toTypeKind(btop.operands().get(0).type());

        // Inverse condition and ensure true block is the immediate successor, in sequence, of lb
        cond = cond.inverse();
        // True block
        Block.Builder ctBlock = lb.block();
        // False block
        Block.Builder cfBlock = lb.block();
        // Merge block
        Block.Builder mergeBlock = lb.block();
        if (vt == TypeKind.IntType) {
            lb.op(BytecodeInstructionOps.if_cmp(TypeKind.IntType, cond, cfBlock.successor(), ctBlock.successor()));
        } else {
            lb.op(BytecodeInstructionOps.cmp(vt));
            lb.op(BytecodeInstructionOps._if(cond, cfBlock.successor(), ctBlock.successor()));
        }

        ctBlock.op(BytecodeInstructionOps._const(TypeKind.IntType, 1));
        ctBlock.op(BytecodeInstructionOps._goto(mergeBlock.successor()));

        cfBlock.op(BytecodeInstructionOps._const(TypeKind.IntType, 0));
        cfBlock.op(BytecodeInstructionOps._goto(mergeBlock.successor()));

        return mergeBlock;
    }

    interface ConditionalBranchConsumer {
        void accept(Block.Builder lb, Block.Builder tBlock, Block.Builder fBlock);
    }

    private static void conditionalBranch(Block.Builder lb,
                                          Op operandOp, CoreOps.ConditionalBranchOp cop,
                                          LoweringContext c,
                                          boolean isLastOpResultOnStack,
                                          ConditionalBranchConsumer cbc) {
        processOperands(lb, operandOp, c, isLastOpResultOnStack);

        Block.Builder tBlock = lb.block();
        Block.Builder fBlock = lb.block();

        cbc.accept(lb, tBlock, fBlock);

        assignBlockArguments(cop, cop.trueBranch(), tBlock, c);
        tBlock.op(BytecodeInstructionOps._goto(c.getLoweredBlock(cop.trueBranch().targetBlock()).successor()));

        assignBlockArguments(cop, cop.falseBranch(), fBlock, c);
        fBlock.op(BytecodeInstructionOps._goto(c.getLoweredBlock(cop.falseBranch().targetBlock()).successor()));
    }

    private static void assignBlockArguments(Op op, Block.Reference s, Block.Builder lb, LoweringContext c) {
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
                lb.op(BytecodeInstructionOps.ldc(constantOp.resultType(), constantOp.value()));
            } else {
                int sslot = c.getSlot(value);

                // Assignment only required if slots differ
                if (sslot != bslot) {
                    TypeKind vt = toTypeKind(barg.type());
                    lb.op(BytecodeInstructionOps.load(vt, sslot));
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
                lb.op(BytecodeInstructionOps.store(vt, bslot));
            } else {
                int sslot = c.getSlot(value);

                // Assignment only required if slots differ
                if (sslot != bslot) {
                    TypeKind vt = toTypeKind(barg.type());
                    lb.op(BytecodeInstructionOps.store(vt, bslot));
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
