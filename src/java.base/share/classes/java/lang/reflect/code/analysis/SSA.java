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

package java.lang.reflect.code.analysis;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Functionality to transform a code model into pure SSA form, replacing operations that declare variables and
 * access them with the use of values they depend on or additional block parameters.
 */
public final class SSA {
    private SSA() {
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
        Map<Block, Set<CoreOp.VarOp>> joinPoints = new HashMap<>();
        Map<CoreOp.VarAccessOp.VarLoadOp, Object> loadValues = new HashMap<>();
        Map<Block.Reference, List<Object>> joinSuccessorValues = new HashMap<>();

        Map<Body, Boolean> visited = new HashMap<>();
        Map<Block, Map<CoreOp.VarOp, Block.Parameter>> joinBlockArguments = new HashMap<>();
        @SuppressWarnings("unchecked")
        T liop = (T) iop.transform(CopyContext.create(), (block, op) -> {
            // Compute join points and value mappings for body
            visited.computeIfAbsent(op.ancestorBody(), b -> {
                findJoinPoints(b, joinPoints);
                variableToValue(b, joinPoints, loadValues, joinSuccessorValues);
                return true;
            });

            if (op instanceof CoreOp.VarOp || op instanceof CoreOp.VarAccessOp) {
                // Drop var operations
                if (op instanceof CoreOp.VarAccessOp.VarLoadOp vl) {
                    // Replace result of load
                    Object loadValue = loadValues.get(vl);
                    CopyContext cc = block.context();
                    Value v = loadValue instanceof VarOpBlockArgument vba
                            ? joinBlockArguments.get(vba.b()).get(vba.vop())
                            : cc.getValue((Value) loadValue);
                    cc.mapValue(op.result(), v);
                }
            } else if (op instanceof Op.Terminating) {
                for (Block.Reference s : op.successors()) {
                    List<Object> joinValues = joinSuccessorValues.get(s);
                    // Successor has join values
                    if (joinValues != null) {
                        CopyContext cc = block.context();

                        // Lazily append target block arguments
                        joinBlockArguments.computeIfAbsent(s.targetBlock(), b -> {
                            Block.Builder bb = cc.getBlock(b);
                            return joinPoints.get(b).stream().collect(Collectors.toMap(
                                    varOp -> varOp,
                                    varOp -> bb.parameter(varOp.varType())));
                        });

                        // Append successor arguments
                        List<Value> values = new ArrayList<>();
                        for (Object o : joinValues) {
                            Value v = o instanceof VarOpBlockArgument vba
                                    ? joinBlockArguments.get(vba.b()).get(vba.vop())
                                    : cc.getValue((Value) o);
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

    record VarOpBlockArgument(Block b, CoreOp.VarOp vop) {
    }

    // @@@ Check for var uses in exception regions
    //     A variable cannot be converted to SAA form if the variable is stored
    //     to in an exception region and accessed from an associated catch region

    static void variableToValue(Body body,
                                Map<Block, Set<CoreOp.VarOp>> joinPoints,
                                Map<CoreOp.VarAccessOp.VarLoadOp, Object> loadValues,
                                Map<Block.Reference, List<Object>> joinSuccessorValues) {
        Map<CoreOp.VarOp, Deque<Object>> variableStack = new HashMap<>();
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
                                Map<CoreOp.VarOp, Deque<Object>> variableStack,
                                Map<Block, Set<CoreOp.VarOp>> joinPoints,
                                Map<CoreOp.VarAccessOp.VarLoadOp, Object> loadValues,
                                Map<Block.Reference, List<Object>> joinSuccessorValues) {
        int size = n.b().ops().size();

        // Check if V is associated with block argument (phi)
        // Push argument onto V's stack
        {
            Set<CoreOp.VarOp> varOps = joinPoints.get(n.b());
            if (varOps != null) {
                varOps.forEach(v -> {
                    assert variableStack.get(v) != null;
                    variableStack.get(v).push(new VarOpBlockArgument(n.b(), v));
                });
            }
        }

        {
            for (int i = 0; i < size - 1; i++) {
                Op op = n.b().ops().get(i);

                if (op instanceof CoreOp.VarOp varOp) {
                    // Initial value assigned to variable
                    Value current = op.operands().get(0);
                    variableStack.computeIfAbsent(varOp, _k -> new ArrayDeque<>())
                            .push(current);
                } else if (op instanceof CoreOp.VarAccessOp.VarStoreOp storeOp) {
                    // Value assigned to variable
                    Value current = op.operands().get(1);
                    variableStack.computeIfAbsent(storeOp.varOp(), _k -> new ArrayDeque<>())
                            .push(current);
                } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp loadOp) {
                    Object to = variableStack.get(loadOp.varOp()).peek();
                    loadValues.put(loadOp, to);
                } else if (op instanceof Op.Nested) {
                    // Traverse descendant variable loads for variables
                    // declared in the block's parent body
                    op.traverse(null, (o, codeElement) -> {
                        if (o instanceof CoreOp.VarAccessOp.VarLoadOp loadOp &&
                                loadOp.varOp().ancestorBody() == op.ancestorBody()) {
                            Object to = variableStack.get(loadOp.varOp()).peek();
                            loadValues.put(loadOp, to);
                        }
                        return null;
                    });
                }
            }

            // Add successor args for joint points
            for (Block.Reference succ : n.b().successors()) {
                Set<CoreOp.VarOp> varOps = joinPoints.get(succ.targetBlock());
                if (varOps != null) {
                    List<Object> joinValues = varOps.stream()
                            .map(vop -> variableStack.get(vop).peek()).toList();
                    joinSuccessorValues.put(succ, joinValues);
                }
            }

            // The result of a VarOp, a variable value, can only be used in VarStoreOp and VarLoadOp
            // therefore there is no need to check existing successor arguments
        }

        // Traverse children of dom tree
        for (Node y : n.children()) {
            variableToValue(y, variableStack, joinPoints, loadValues, joinSuccessorValues);
        }

        // Pop off values for variables
        {
            Set<CoreOp.VarOp> varOps = joinPoints.get(n.b());
            if (varOps != null) {
                varOps.forEach(v -> {
                    variableStack.get(v).pop();
                });
            }

            for (int i = 0; i < size - 1; i++) {
                Op op = n.b().ops().get(i);

                if (op instanceof CoreOp.VarOp varOp) {
                    variableStack.get(varOp).pop();
                } else if (op instanceof CoreOp.VarAccessOp.VarStoreOp storeOp) {
                    variableStack.get(storeOp.varOp()).pop();
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
    public static void findJoinPoints(Body body, Map<Block, Set<CoreOp.VarOp>> joinPoints) {
        Map<Block, Set<Block>> df = body.dominanceFrontier();
        Map<CoreOp.VarOp, Set<Block>> a = findVarStores(body);

        int iterCount = 0;
        int[] hasAlready = new int[body.blocks().size()];
        int[] work = new int[body.blocks().size()];

        Deque<Block> w = new ArrayDeque<>();

        for (CoreOp.VarOp v : a.keySet()) {
            iterCount++;

            for (Block x : a.get(v)) {
                work[x.index()] = iterCount;
                w.push(x);
            }

            while (!w.isEmpty()) {
                Block x = w.pop();

                for (Block y : df.getOrDefault(x, Set.of())) {
                    if (hasAlready[y.index()] < iterCount) {
                        // Only add to the join points if y is dominated by the var's block
                        if (y.isDominatedBy(v.parentBlock())) {
                            joinPoints.computeIfAbsent(y, _k -> new LinkedHashSet<>()).add(v);
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
    static Map<CoreOp.VarOp, Set<Block>> findVarStores(Body r) {
        return r.traverse(new LinkedHashMap<>(), CodeElement.opVisitor((stores, op) -> {
            if (op instanceof CoreOp.VarAccessOp.VarStoreOp storeOp) {
                if (storeOp.varOp().ancestorBody() != storeOp.ancestorBody()) {
                    throw new IllegalStateException("Descendant variable store operation");
                }
                if (storeOp.varOp().ancestorBody() == r) {
                    stores.computeIfAbsent(storeOp.varOp(), _v -> new LinkedHashSet<>()).add(storeOp.parentBlock());
                }
            }
            return stores;
        }));
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
