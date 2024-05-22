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

import java.lang.reflect.code.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Functionality to transform a code model into pure SSA form, replacing operations that declare variables and
 * access them with the use of values they depend on or additional block parameters.
 */
public final class SlotSSA {
    private SlotSSA() {
    }

    /**
     * Applies an SSA transformation to an invokable operation, replacing operations that declare variables and
     * access them with the use of values they depend on or additional block parameters.
     * <p>
     * The operation should first be in lowered form before applying this transformation.
     * <p>
     * Note: this implementation does not currently work correctly when a variable is stored to within an exception
     * region and read from outside as a result of catching an exception. In such cases a complete transformation may be
     * not possible and such variables will need to be retained.
     *
     * @param iop the invokable operation
     * @return the transformed operation
     * @param <T> the invokable type
     */
    public static <T extends Op & Op.Invokable> T transform(T iop) {
        Map<Block, Set<Integer>> joinPoints = new HashMap<>();
        Map<SlotOp.SlotLoadOp, Object> loadValues = new HashMap<>();
        Map<Block.Reference, List<SlotValue>> joinSuccessorValues = new HashMap<>();

        Map<Body, Boolean> visited = new HashMap<>();
        Map<Block, Map<Integer, Block.Parameter>> joinBlockArguments = new HashMap<>();
        @SuppressWarnings("unchecked")
        T liop = (T) iop.transform(CopyContext.create(), (block, op) -> {
            // Compute join points and value mappings for body
            visited.computeIfAbsent(op.ancestorBody(), b -> {
                findJoinPoints(b, joinPoints);
                variableToValue(b, joinPoints, loadValues, joinSuccessorValues);
                return true;
            });

            if (op instanceof SlotOp) {
                // Drop slot operations
                if (op instanceof SlotOp.SlotLoadOp vl) {
                    // Replace result of load
                    Object loadValue = loadValues.get(vl);
                    CopyContext cc = block.context();
                    Value v = loadValue instanceof SlotBlockArgument vba
                            ? joinBlockArguments.get(vba.b()).get(vba.slot())
                            : cc.getValue((Value) loadValue);
                    cc.mapValue(op.result(), v);
                }
            } else if (op instanceof Op.Terminating) {
                for (Block.Reference s : op.successors()) {
                    List<SlotValue> joinValues = joinSuccessorValues.get(s);
                    // Successor has join values
                    if (joinValues != null) {
                        CopyContext cc = block.context();

                        // Lazily append target block arguments
                        joinBlockArguments.computeIfAbsent(s.targetBlock(), b -> {
                            Block.Builder bb = cc.getBlock(b);
                            return joinPoints.get(b).stream().collect(Collectors.toMap(
                                    slot -> slot,
                                    // @@@
                                    slot -> bb.parameter(joinValues.stream().filter(sv -> sv.slot == slot).findAny().map(sv ->
                                            (sv.value instanceof SlotBlockArgument vba
                                                ? joinBlockArguments.get(vba.b()).get(vba.slot())
                                                : cc.getValue((Value) sv.value)).type()).orElseThrow())));
                        });

                        // Append successor arguments
                        List<Value> values = new ArrayList<>();
                        for (SlotValue sv : joinValues) {
                            Value v = sv.value instanceof SlotBlockArgument vba
                                    ? joinBlockArguments.get(vba.b()).get(vba.slot())
                                    : cc.getValue((Value) sv.value);
                            values.add(v);
                        }

                        // Map successor with append arguments
                        List<Value> toArgs = cc.getValues(s.arguments());
                        toArgs.addAll(values);
                        Block.Reference toS = cc.getBlock(s.targetBlock()).successor(toArgs);
                        cc.mapSuccessor(s, toS);
                    }
                }

                block.apply(op);
            } else {
                block.apply(op);
            }

            return block;
        });
        return liop;
    }

    record SlotBlockArgument(Block b, int slot) {
    }

    record SlotValue(int slot, Object value) {
    }

    // @@@ Check for var uses in exception regions
    //     A variable cannot be converted to SAA form if the variable is stored
    //     to in an exception region and accessed from an associated catch region

    static void variableToValue(Body body,
                                Map<Block, Set<Integer>> joinPoints,
                                Map<SlotOp.SlotLoadOp, Object> loadValues,
                                Map<Block.Reference, List<SlotValue>> joinSuccessorValues) {
        Map<Integer, Deque<Object>> variableStack = new HashMap<>();
        Node top = buildDomTree(body.entryBlock(), body.immediateDominators());
        variableToValue(top, variableStack, joinPoints, loadValues, joinSuccessorValues);
    }

    /**
     * Replaces usages of a variable with the corresponding value, from a given block node in the dominator tree.
     * <p>
     * The result of a {@code VarLoadOp} for variable, {@code V} say the result of a {@code VarOp} operation,
     * is replaced with the value passed as an operand to the immediately dominating {@code VarStoreOp} that operates
     * on {@code V}, or a block argument representing the equivalent of a phi-value of {@code V}.
     * After which, any related {@code VarOp}, {@code VarLoadOp}, or {@code VarStoreOp} operations are removed.
     *
     * @param n             the node in the dominator tree
     * @param variableStack the variable stack
     * @param joinPoints    the join points
     * @implNote See "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph" by Ron Cytron et. al.
     * Section 5.2 and Figure 12.
     */
    static void variableToValue(Node n,
                                Map<Integer, Deque<Object>> variableStack,
                                Map<Block, Set<Integer>> joinPoints,
                                Map<SlotOp.SlotLoadOp, Object> loadValues,
                                Map<Block.Reference, List<SlotValue>> joinSuccessorValues) {

        int size = n.b().ops().size();

        // Check if slot is associated with block argument (phi)
        // Push argument onto slot's stack
        {
            Set<Integer> slots = joinPoints.get(n.b());
            if (slots != null) {
                slots.forEach(slot -> {
                    assert variableStack.get(slot) != null;
                    variableStack.get(slot).push(new SlotBlockArgument(n.b(), slot));
                });
            }
        }

        {
            for (int i = 0; i < size - 1; i++) {
                Op op = n.b().ops().get(i);

                if (op instanceof SlotOp.SlotStoreOp storeOp) {
                    // Value assigned to slot
                    Value current = op.operands().get(0);
                    variableStack.computeIfAbsent(storeOp.slot(), _ -> new ArrayDeque<>())
                            .push(current);
                } else if (op instanceof SlotOp.SlotLoadOp loadOp) {
                    Object to = variableStack.get(loadOp.slot()).peek();
                    loadValues.put(loadOp, to);
                }
            }

            // Add successor args for joint points
            for (Block.Reference succ : n.b().successors()) {
                Set<Integer> slots = joinPoints.get(succ.targetBlock());
                if (slots != null) {
                    List<SlotValue> joinValues = slots.stream()
                            .map(vop -> new SlotValue(vop, variableStack.get(vop).peek())).toList();
                    joinSuccessorValues.put(succ, joinValues);
                }
            }
        }

        // Traverse children of dom tree
        for (Node y : n.children()) {
            variableToValue(y, variableStack, joinPoints, loadValues, joinSuccessorValues);
        }

        // Pop off values for slots
        {
            Set<Integer> slots = joinPoints.get(n.b());
            if (slots != null) {
                slots.forEach(slot -> {
                    variableStack.get(slot).pop();
                });
            }

            for (int i = 0; i < size - 1; i++) {
                Op op = n.b().ops().get(i);

                if (op instanceof SlotOp.SlotStoreOp storeOp) {
                    variableStack.get(storeOp.slot()).pop();
                }
            }
        }
    }

    /**
     * Finds the join points of a body.
     * <p>
     * A join point is a block that is in the dominance frontier of one or more predecessors, that make one or more
     * stores to variables (using the {@code VarStoreOp} operation on the result of a {@code VarOp} operation).
     * The join point contains the set variables ({@code VarOp} operations) that are stored to.
     * <p>
     * A variable of a joint point indicates that a block argument may need to be added to the join point's block
     * when converting variables to SSA form. Different values of a variable may occur at different control flow
     * paths at the join point. The block argument represents the convergence of multiple values for the same
     * variable, where a predecessor assigns to the block argument.
     * (Block arguments are equivalent to phi-values, or phi-nodes, used in other representations.)
     *
     * @param body the body.
     * @param joinPoints the returned join points.
     * @implNote See "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph" by Ron Cytron et. al.
     * Section 5.1 and Figure 11.
     */
    public static void findJoinPoints(Body body, Map<Block, Set<Integer>> joinPoints) {
        Map<Block, Set<Block>> df = body.dominanceFrontier();
        Map<Integer, Set<Block>> a = findSlotStores(body);

        int iterCount = 0;
        int[] hasAlready = new int[body.blocks().size()];
        int[] work = new int[body.blocks().size()];

        Deque<Block> w = new ArrayDeque<>();

        for (int slot : a.keySet()) {
            iterCount++;

            for (Block x : a.get(slot)) {
                work[x.index()] = iterCount;
                w.push(x);
            }

            while (!w.isEmpty()) {
                Block x = w.pop();

                for (Block y : df.getOrDefault(x, Set.of())) {
                    if (hasAlready[y.index()] < iterCount) {
                        if (isSlotLoadedFirst(y, slot, new boolean[body.blocks().size()])) {
                            joinPoints.computeIfAbsent(y, _ -> new LinkedHashSet<>()).add(slot);
                        }
                        hasAlready[y.index()] = iterCount;

                        if (work[y.index()] < iterCount) {
                            work[y.index()] = iterCount;
                            w.push(y);
                        }
                    }
                }
            }
        }
    }

    // Returns map of variable to blocks that contain stores to the variables declared in the body
    // Throws ISE if a descendant store operation is encountered
    // @@@ Compute map for whole tree, then traverse keys with filter
    static Map<Integer, Set<Block>> findSlotStores(Body r) {
        return r.traverse(new LinkedHashMap<>(), CodeElement.opVisitor((stores, op) -> {
            if (op instanceof SlotOp.SlotStoreOp storeOp) {
                stores.computeIfAbsent(storeOp.slot(), _ -> new LinkedHashSet<>()).add(storeOp.parentBlock());
            }
            return stores;
        }));
    }

    static boolean isSlotLoadedFirst(Block b, int slot, boolean[] visited) {
        if (visited[b.index()]) return false;
        visited[b.index()] = true;
        for (Op op : b.ops()) {
            if (op instanceof SlotOp.SlotStoreOp storeOp) {
                if (storeOp.slot == slot) return false;
            } else if (op instanceof SlotOp.SlotLoadOp loadOp) {
                if (loadOp.slot == slot) return true;
            }
        }
        for (var s : b.successors()) {
            if (isSlotLoadedFirst(s.targetBlock(), slot, visited)) return true;
        }
        return false;
    }

    record Node(Block b, Set<Node> children) {
    }

    static Node buildDomTree(Block entryBlock, Map<Block, Block> idoms) {
        Map<Block, Node> tree = new HashMap<>();
        for (Map.Entry<Block, Block> e : idoms.entrySet()) {
            Block id = e.getValue();
            Block b = e.getKey();

            Node parent = tree.computeIfAbsent(id, _k -> new Node(_k, new HashSet<>()));
            if (b == entryBlock) {
                continue;
            }

            Node child = tree.computeIfAbsent(b, _k -> new Node(_k, new HashSet<>()));
            parent.children.add(child);
        }
        return tree.get(entryBlock);
    }
}
