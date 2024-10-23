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
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CodeElement;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 *
 */
final class SlotToVarTransformer {

    record ExcStackMap(List<Block> catchBlocks, Map<Block, BitSet> map) implements Function<Block, ExcStackMap> {
        @Override
        public ExcStackMap apply(Block b) {
            BitSet excStack = map.computeIfAbsent(b, _ -> new BitSet());
            switch (b.terminatingOp()) {
                case CoreOp.ExceptionRegionEnter ere -> {
                    BitSet entries = new BitSet();
                    for (Block.Reference cbr : ere.catchBlocks()) {
                        Block cb = cbr.targetBlock();
                        int i = catchBlocks.indexOf(cb);
                        if (i < 0) {
                            i = catchBlocks.size();
                            catchBlocks.add(cb);
                            map.put(cb, excStack);
                        }
                        entries.set(i);
                    }
                    entries.or(excStack);
                    map.put(ere.start().targetBlock(), entries);
                }
                case CoreOp.ExceptionRegionExit ere -> {
                    excStack = (BitSet) excStack.clone();
                    for (Block.Reference cbr : ere.catchBlocks()) {
                        excStack.clear(catchBlocks.indexOf(cbr.targetBlock()));
                    }
                    map.put(ere.end().targetBlock(), excStack);
                }
                case Op op -> {
                    for (Block.Reference tbr : op.successors()) {
                        map.put(tbr.targetBlock(), excStack);
                    }
                }
            }
            return this;
        }

        void forEachHandler(Block b, Consumer<Block> hbc) {
            map.get(b).stream().mapToObj(catchBlocks::get).forEach(hbc);
        }

        void forEachTryBlock(Block hb, Consumer<Block> bc) {
            int i = catchBlocks.indexOf(hb);
            if (i >= 0) {
                for (var me : map.entrySet()) {
                    if (me.getValue().get(i)) bc.accept(me.getKey());
                }
            }
        }
    }

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {

        // Composing exception stack map to be able to follow slot ops from try to the handler
        ExcStackMap excMap = func.traverse(new ExcStackMap(new ArrayList<>(), new HashMap<>()),
                CodeElement.blockVisitor((map, b) -> map.apply(b)));

        List<SlotOp.Var> toInitialize = func.body().traverse(new ArrayList<>(), CodeElement.opVisitor((toInit, op) -> {
            if (op instanceof SlotOp slotOp && slotOp.var == null) {

                // Assign variable to segments, calculate var slotType
                SlotOp.Var var = new SlotOp.Var(); // New variable
                var q = new ArrayDeque<SlotOp>();
                var stores = new ArrayList<SlotOp.SlotStoreOp>();
                q.add(slotOp);
                while (!q.isEmpty()) {
                    SlotOp se = q.pop();
                    if (se.var == null) {
                        se.var = var; // Assign variable to the segment
                        var.typeKind = se.typeKind(); // TypeKind is identical for all SlotOps of the same variable
                        for (SlotOp to : slotImmediateSuccessors(se, excMap)) {
                            // All following SlotLoadOp belong to the same variable
                            if (to instanceof SlotOp.SlotLoadOp) {
                                if (to.var == null) {
                                    q.add(to);
                                }
                            }
                        };
                        if (se instanceof SlotOp.SlotLoadOp) {
                            // Segments preceeding SlotLoadOp also belong to the same variable
                            for (SlotOp from : slotImmediatePredecessors(se, excMap)) {
                                if (from.var == null) {
                                    q.add(from);
                                }
                            };
                        }
                    }
                    if (se.var == var && se instanceof SlotOp.SlotStoreOp store) {
                        stores.add(store); // Collection of all SlotStoreOps of the variable
                    }
                }

                // Single-assigned variable has only one SlotStoreOp
                var.single = stores.size() < 2;

                // Identification of initial SlotStoreOp
                for (var it = stores.iterator(); it.hasNext();) {
                    SlotOp s = it.next();
                    if (isDominatedByTheSameVar(s, excMap)) {
                        // A store preceeding dominantly with segments of the same variable is not initial
                        it.remove();
                    }
                }

                // Remaining stores are all initial.
                if (stores.size() > 1) {
                    // A synthetic default-initialized dominant segment must be inserted to the variable, if there is more than one initial store segment.
                    // It is not necessary to link it with other variable segments, the analysys ends here.
                    toInit.add(stores.getFirst().var);
                }


            }
            return toInit;
        }));

        return func.transform((block, op) -> {
            if (!toInitialize.isEmpty()) {
                for (SlotOp.Var var : toInitialize) {
                    var.value = block.op(CoreOp.var(block.op(liftDefaultValue(var.typeKind))));
                }
                toInitialize.clear();
            }
            CopyContext cc = block.context();
            switch (op) {
                case SlotOp.SlotLoadOp slo -> {
                    assert slo.var.value != null;
                    cc.mapValue(op.result(), slo.var.single ? slo.var.value : block.op(CoreOp.varLoad(slo.var.value)));
                }
                case SlotOp.SlotStoreOp sso -> {
                    Value val = sso.operands().getFirst();
                    val = cc.getValueOrDefault(val, val);
                    if (sso.var.single) {
                        sso.var.value = val;
                    } else if (sso.var.value == null) {
                        sso.var.value = block.op(CoreOp.var(val));
                    } else {
                        block.op(CoreOp.varStore(sso.var.value, val));
                    }
                }
                default ->
                    block.op(op);
            }
            return block;
        });
    }

    // @@@ can be replaced with unitialized VarOp
    private static CoreOp.ConstantOp liftDefaultValue(TypeKind tk) {
        return switch (tk) {
            case INT -> CoreOp.constant(UnresolvedType.unresolvedInt(), 0);
            case REFERENCE -> CoreOp.constant(UnresolvedType.unresolvedRef(), null);
            case LONG -> CoreOp.constant(JavaType.LONG, 0l);
            case DOUBLE -> CoreOp.constant(JavaType.DOUBLE, 0d);
            case FLOAT -> CoreOp.constant(JavaType.FLOAT, 0f);
            case BOOLEAN -> CoreOp.constant(JavaType.BOOLEAN, false);
            case BYTE -> CoreOp.constant(JavaType.BYTE, (byte)0);
            case SHORT -> CoreOp.constant(JavaType.SHORT, (short)0);
            case CHAR -> CoreOp.constant(JavaType.CHAR, (char)0);
            case VOID -> throw new IllegalStateException("Unexpected void type.");
        };
    }

    // Traverse immediate same-slot successors of a SlotOp
    private static Iterable<SlotOp> slotImmediateSuccessors(SlotOp slotOp, ExcStackMap excMap) {
        return () -> new SlotOpIterator(slotOp, excMap, true);
    }

    // Traverse immediate same-slot predecessors of a SlotOp
    private static Iterable<SlotOp> slotImmediatePredecessors(SlotOp slotOp, ExcStackMap excMap) {
        return () -> new SlotOpIterator(slotOp, excMap, false);
    }

    private static boolean isDominatedByTheSameVar(SlotOp slotOp, ExcStackMap excMap) {
        boolean any = false;
        for (SlotOp pred : slotImmediatePredecessors(slotOp, excMap)) {
            if (pred.var != slotOp.var) {
                return false;
            }
            any = true;
        }
        return any;
    }


    static final class SlotOpIterator implements Iterator<SlotOp> {

        SlotOp op;
        final int slot;
        final ExcStackMap map;
        final TypeKind tk;
        final boolean fwd;
        Block b;
        List<Op> ops;
        int i;
        BitSet visited;
        ArrayDeque<Block> toVisit;


        public SlotOpIterator(SlotOp slotOp, ExcStackMap excMap, boolean forward) {
            slot = slotOp.slot;
            tk = slotOp.typeKind();
            map = excMap;
            fwd = forward;
            b = slotOp.parentBlock();
            ops = fwd ? b.ops() : b.ops().reversed();
            i = ops.indexOf(slotOp) + 1;
        }

        @Override
        public boolean hasNext() {
            while (hasNextSlot()) {
                // filter loads and stores of the same TypeKind
                if (op.typeKind() == tk) {
                    return true;
                }
                op = null;
            }
            return false;
        }

        private boolean hasNextSlot() {
            if (op != null) {
                return true;
            } else {
                while (b != null || toVisit != null && !toVisit.isEmpty()) {
                    if (b == null) {
                        b = toVisit.pop();
                        ops = fwd ? b.ops() : b.ops().reversed();
                        i = 0;
                    }
                    while (i < ops.size()) {
                        if (ops.get(i++) instanceof SlotOp sop && sop.slot == slot) {
                            op = sop;
                            b = null;
                            return true;
                        }
                    }
                    if (toVisit == null) {
                        toVisit = new ArrayDeque<>();
                        visited = new BitSet();
                    }
                    if (fwd) {
                        for (Block.Reference sr : b.successors()) {
                            Block sb = sr.targetBlock();
                            if (!visited.get(sb.index())) {
                                toVisit.add(sb);
                                visited.set(sb.index());
                            }
                        }
                        // Visit also relevant exception handlers
                        map.forEachHandler(b, sb -> {
                            if (!visited.get(sb.index())) {
                                toVisit.add(sb);
                                visited.set(sb.index());
                            }
                        });
                    } else {
                        for (Block pb : b.predecessors()) {
                            if (!visited.get(pb.index())) {
                                toVisit.add(pb);
                                visited.set(pb.index());
                            }
                        }
                        // Visit also relevant try blocks from handler
                        map.forEachTryBlock(b, sb -> {
                            if (!visited.get(sb.index())) {
                                toVisit.add(sb);
                                visited.set(sb.index());
                            }
                        });
                    }
                    b = null;
                }
                return false;
            }
        }

        @Override
        public SlotOp next() {
            if (!hasNext()) throw new NoSuchElementException();
            SlotOp ret = op;
            op = null;
            return ret;
        }
    }
}
