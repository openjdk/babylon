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
import java.lang.reflect.code.Body;
import java.lang.reflect.code.CodeElement;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Deque;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 *
 */
final class SlotToVarTransformer {

    static final class Var {
        boolean single;
        TypeKind typeKind;
        Value value;
        Body parentBody;
    }

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
        try {
            return new SlotToVarTransformer().convert(func);
        } catch (Throwable t) {
            System.out.println(func.toText());
            throw t;
        }
    }

    private final Map<SlotOp, Var> varMap;

    private SlotToVarTransformer() {
        varMap = new IdentityHashMap<>();
    }

    private CoreOp.FuncOp convert(CoreOp.FuncOp func) {
        // Composing exception stack map to be able to follow slot ops from try to the handler
        ExcStackMap excMap = func.traverse(new ExcStackMap(new ArrayList<>(), new IdentityHashMap<>()),
                CodeElement.blockVisitor((map, b) -> map.apply(b)));

        List<Var> toInitialize = func.body().traverse(new ArrayList<>(), CodeElement.opVisitor((toInit, op) -> {
            if (op instanceof SlotOp slotOp && !varMap.containsKey(slotOp)) {

                // Assign variable to segments, calculate var slotType
                Var var = new Var(); // New variable
                var.parentBody = slotOp.ancestorBody();
                var q = new ArrayDeque<SlotOp>();
                var stores = new ArrayList<SlotOp.SlotStoreOp>();
                q.add(slotOp);
                while (!q.isEmpty()) {
                    SlotOp se = q.pop();
                    if (!varMap.containsKey(se)) {
                        varMap.put(se, var); // Assign variable to the segment
                        if (var.typeKind == null) var.typeKind = se.typeKind(); // TypeKind is identical for all SlotOps of the same variable
                        for (SlotOp to : slotImmediateSuccessors(se, excMap)) {
                            // All following SlotLoadOp belong to the same variable
                            if (to instanceof SlotOp.SlotLoadOp) {
                                if (!varMap.containsKey(to)) {
                                    q.add(to);
                                }
                            }
                        }
                        if (se instanceof SlotOp.SlotLoadOp) {
                            // Segments preceeding SlotLoadOp also belong to the same variable
                            for (SlotOp from : slotImmediatePredecessors(se, excMap)) {
                                if (!varMap.containsKey(from)) {
                                    q.add(from);
                                }
                            }
                        } else if (se instanceof SlotOp.SlotStoreOp store) {
                            stores.add(store); // Collection of all SlotStoreOps of the variable
                        }
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
                    toInit.add(varMap.get(stores.getFirst()));
                }


            }
            return toInit;
        }));

        return func.transform((block, op) -> {
            if (!toInitialize.isEmpty()) {
                for (var it = toInitialize.iterator(); it.hasNext();) {
                    Var var = it.next();
                    if (var.parentBody == op.ancestorBody()) {
                        var.value = block.op(CoreOp.var(toTypeElement(var.typeKind)));
                        it.remove();
                    }
                }
            }
            CopyContext cc = block.context();
            switch (op) {
                case SlotOp.SlotLoadOp slo -> {
                    Var var = varMap.get(slo);
                    if (var.value == null) {
                        System.out.println(slo);
                        throw new AssertionError();
                    }
                    cc.mapValue(op.result(), var.single ? var.value : block.op(CoreOp.varLoad(var.value)));
                }
                case SlotOp.SlotStoreOp sso -> {
                    Var var = varMap.get(sso);
                    Value val = sso.operands().getFirst();
                    val = cc.getValueOrDefault(val, val);
                    if (var.single) {
                        var.value = val;
                    } else if (var.value == null) {
                        TypeElement varType = switch (val.type()) {
                            case UnresolvedType.Ref _ -> UnresolvedType.unresolvedRef();
                            case UnresolvedType.Int _ -> UnresolvedType.unresolvedInt();
                            default -> val.type();
                        };
                        var.value = block.op(CoreOp.var(null, varType, val));
                    } else {
                        block.op(CoreOp.varStore(var.value, val));
                    }
                }
                default ->
                    block.op(op);
            }
            return block;
        });
    }

    private static TypeElement toTypeElement(TypeKind tk) {
        return switch (tk) {
            case INT -> UnresolvedType.unresolvedInt();
            case REFERENCE -> UnresolvedType.unresolvedRef();
            case LONG -> JavaType.LONG;
            case DOUBLE -> JavaType.DOUBLE;
            case FLOAT -> JavaType.FLOAT;
            case BOOLEAN -> JavaType.BOOLEAN;
            case BYTE -> JavaType.BYTE;
            case SHORT -> JavaType.SHORT;
            case CHAR -> JavaType.CHAR;
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

    private boolean isDominatedByTheSameVar(SlotOp slotOp, ExcStackMap excMap) {
        Var var = varMap.get(slotOp);
        Set<Op.Result> predecessors = new HashSet<>();
        for (SlotOp pred : slotImmediatePredecessors(slotOp, excMap)) {
            if (varMap.get(pred) != var) {
                return false;
            }
            if (pred != slotOp) predecessors.add(pred.result());
        }
        return isDominatedBy(slotOp.result(), predecessors);
    }

    /**
     * Returns {@code true} if this value is dominated by the given set of values {@code doms}.
     * <p>
     * The set dominates if every path from the entry node go through any member of the set.
     * <p>
     * First part checks individual dominance of every member of the set.
     * <p>
     * If no member of the set is individually dominant, the second part tries to find path
     * to the entry block bypassing all blocks from the tested set.
     * <p>
     * Implementation searches for the paths by traversing the value declaring block predecessors,
     * stopping at blocks where values from the tested set are declared or at blocks already processed.
     * Negative test result is returned when the entry block is reached.
     * Positive test result is returned when no path to the entry block is found.
     *
     * @param value the value
     * @param doms the dominating set of values
     * @return {@code true} if this value is dominated by the given set of values {@code dom}.
     * @throws IllegalStateException if the declaring block is partially built
     */
    public static boolean isDominatedBy(Value value, Set<? extends Value> doms) {
        if (doms.isEmpty()) {
            return false;
        }

        for (Value dom : doms) {
            if (value.isDominatedBy(dom)) {
                return true;
            }
        }

        Set<Block> stopBlocks = new HashSet<>();
        for (Value dom : doms) {
            stopBlocks.add(dom.declaringBlock());
        }

        Deque<Block> toProcess = new ArrayDeque<>();
        toProcess.add(value.declaringBlock());
        stopBlocks.add(value.declaringBlock());
        while (!toProcess.isEmpty()) {
            for (Block b : toProcess.pop().predecessors()) {
                if (b.isEntryBlock()) {
                    return false;
                }
                if (stopBlocks.add(b)) {
                    toProcess.add(b);
                }
            }
        }
        return true;
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
                // filter loads and stores of the same TypeKind (if known)
                if (op.typeKind() == tk || op.typeKind() == null || tk == null) {
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
