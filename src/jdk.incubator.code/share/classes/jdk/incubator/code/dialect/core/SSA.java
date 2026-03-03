/*
 * Copyright (c) 2024, 2025, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code.dialect.core;

import jdk.incubator.code.*;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Functionality to transform a code model into pure SSA form, replacing operations that declare variables and
 * access them with the use of values they depend on or additional block parameters.
 */
public final class SSA {
    private SSA() {
    }

    /**
     * Applies an SSA transformation to an operation with bodies, replacing operations that declare variables and
     * access them with the use of values they depend on or additional block parameters.
     * <p>
     * The operation should first be in lowered form before applying this transformation.
     * <p>
     * Note: this implementation does not currently work correctly when a variable is stored to within an exception
     * region and read from outside as a result of catching an exception. In such cases a complete transformation may be
     * not possible and such variables will need to be retained.
     *
     * @param nestedOp the operation with bodies
     * @return the transformed operation
     * @param <T> the operation type
     */
    public static <T extends Op & Op.Nested> T transform(T nestedOp) {
        // @@@ property is used to test both impls
        if (!"cytron".equalsIgnoreCase(System.getProperty("babylon.ssa"))) {
            return SSABraun.transform(nestedOp);
        } else {
            return SSACytron.transform(nestedOp);
        }
    }

    /**
     * An implementation of SSA construction based on
     * <a href="https://doi.org/10.1007/978-3-642-37051-9">
     * Simple end Efficient Construction of Static Single Assignment Form (pp 102-122)
     * </a>.
     * <p>
     * This implementation contains some adaptions, notably:
     * <ul>
     *     <li>Adapt to block parameters rather than phi functions.</li>
     *     <li>Adapt to work with multiple bodies.</li>
     * </ul>
     */
    static final class SSABraun implements CodeTransformer {
        private final Map<CoreOp.VarOp, Map<Block, Val>> currentDef = new HashMap<>();
        private final Set<Block> sealedBlocks = new HashSet<>();
        private final Map<Block, Map<CoreOp.VarOp, Phi>> incompletePhis = new HashMap<>();

        // according to the algorithm:
        // "As only filled blocks may have successors, predecessors are always filled."
        // In consequence, this means that only filled predecessors should be considered
        // when recursively searching for a definition
        private final Map<Block, SequencedSet<Block>> predecessors = new HashMap<>();
        // as we can't modify the graph at the same time as we analyze it,
        // we need to store which load op needs to remapped to which value
        private final Map<CoreOp.VarAccessOp.VarLoadOp, Val> loads = new HashMap<>();
        private final Map<Block, List<Phi>> additionalParameters = new HashMap<>();
        // as we look up definitions during the actual transformation again,
        // we might encounter deleted phis.
        // we use this set to be able to correct that during transformation
        private final Set<Phi> deletedPhis = new HashSet<>();

        static <O extends Op & Op.Nested> O transform(O nestedOp) {
            SSABraun construction = new SSABraun();
            construction.prepare(nestedOp);
            @SuppressWarnings("unchecked")
            O ssaOp = (O) nestedOp.transform(CodeContext.create(), construction);
            return ssaOp;
        }

        private SSABraun() {
        }

        private void prepare(Op nestedOp) {
            nestedOp.elements().forEach(e -> {
                switch (e) {
                    case CoreOp.VarAccessOp.VarLoadOp load -> {
                        Val val = readVariable(load.varOp(), load.ancestorBlock());
                        registerLoad(load, val);
                    }
                    case CoreOp.VarAccessOp.VarStoreOp store ->
                            writeVariable(store.varOp(), store.ancestorBlock(), new Holder(store.storeOperand()));
                    case CoreOp.VarOp initialStore -> {
                        Val val = initialStore.isUninitialized()
                                ? Uninitialized.VALUE
                                : new Holder(initialStore.initOperand());
                        writeVariable(initialStore, initialStore.ancestorBlock(), val);
                    }
                    case Op op when op instanceof Op.Terminating -> {
                        Block block = op.ancestorBlock();
                        // handle the sealing, i.e. only now make this block a predecessor of its successors
                        for (Block.Reference successor : block.successors()) {
                            Block successorBlock = successor.targetBlock();
                            Set<Block> blocks = this.predecessors.computeIfAbsent(successorBlock, _ -> new LinkedHashSet<>());
                            blocks.add(block);
                            // if this was the last predecessor added to successorBlock, seal it
                            if (blocks.size() == successorBlock.predecessors().size()) {
                                sealBlock(successorBlock);
                            }
                        }
                    }
                    default -> {
                    }
                }
            });
        }

        private void registerLoad(CoreOp.VarAccessOp.VarLoadOp load, Val val) {
            this.loads.put(load, val);
            if (val instanceof Phi phi) {
                phi.users.add(load);
            }
        }

        private void writeVariable(CoreOp.VarOp variable, Block block, Val value) {
            this.currentDef.computeIfAbsent(variable, _ -> new HashMap<>()).put(block, value);
        }

        private Val readVariable(CoreOp.VarOp variable, Block block) {
            Val value = this.currentDef.getOrDefault(variable, Map.of()).get(block);
            if (value == null
                // deleted Phi, this is an old reference
                // due to our 2-step variant of the original algorithm, we might encounter outdated definitions
                // when we read to prepare block arguments
                || value instanceof Phi phi && this.deletedPhis.contains(phi)) {
                return readVariableRecursive(variable, block);
            }
            return value;
        }

        private Val readVariableRecursive(CoreOp.VarOp variable, Block block) {
            Val value;
            if (!block.isEntryBlock() && !this.sealedBlocks.contains(block)) {
                Phi phi = new Phi(variable, block);
                value = phi;
                this.incompletePhis.computeIfAbsent(block, _ -> new HashMap<>()).put(variable, phi);
                this.additionalParameters.computeIfAbsent(block, _ -> new ArrayList<>()).add(phi);
            } else if (block.isEntryBlock() && variable.ancestorBody() != block.ancestorBody()) {
                // we are in an entry block but didn't find a definition yet
                Block enclosingBlock = block.parent().parent().parent();
                assert enclosingBlock != null : "def not found in entry block, with no enclosing block";
                value = readVariable(variable, enclosingBlock);
            } else if (this.predecessors.get(block).size() == 1) {
                value = readVariable(variable, this.predecessors.get(block).getFirst());
            } else {
                Phi param = new Phi(variable, block);
                writeVariable(variable, block, param);
                value = addPhiOperands(variable, param);
                // To go from Phis to block parameters, we remember that we produced a Phi here.
                // This means that edges to this block need to pass a value via parameter
                if (value == param) {
                    this.additionalParameters.computeIfAbsent(block, _ -> new ArrayList<>()).add(param);
                }
            }
            writeVariable(variable, block, value); // cache value for this variable + block
            return value;
        }

        private Val addPhiOperands(CoreOp.VarOp variable, Phi value) {
            for (Block pred : this.predecessors.getOrDefault(value.block(), Collections.emptySortedSet())) {
                value.appendOperand(readVariable(variable, pred));
            }
            return tryRemoveTrivialPhi(value);
        }

        private Val tryRemoveTrivialPhi(Phi phi) {
            Val same = null;
            for (Val op : phi.operands()) {
                if (op == same || op == phi) {
                    continue;
                }
                if (same != null) {
                    return phi;
                }
                same = op;
            }
            // we shouldn't have phis without operands (other than itself)
            assert same != null : "phi without different operands";
            List<Phi> phiUsers = phi.replaceBy(same, this);
            List<Phi> phis = this.additionalParameters.get(phi.block());
            if (phis != null) {
                phis.remove(phi);
            }
            for (Phi user : phiUsers) {
                Val res = tryRemoveTrivialPhi(user);
                if (same == user) {
                    same = res;
                }
            }
            return same;
        }

        private void sealBlock(Block block) {
            this.incompletePhis.getOrDefault(block, Map.of()).forEach(this::addPhiOperands);
            this.sealedBlocks.add(block);
        }

        // only used during transformation

        private Value resolveValue(CodeContext context, Val val) {
            return switch (val) {
                case Uninitialized _ -> throw new IllegalStateException("Uninitialized variable");
                case Holder holder -> context.getValueOrDefault(holder.value(), holder.value());
                case Phi phi -> {
                    List<Phi> phis = this.additionalParameters.get(phi.block());
                    int additionalParameterIndex = phis.indexOf(phi);
                    assert additionalParameterIndex >= 0 : "phi not in parameters " + phi;
                    int index = additionalParameterIndex + phi.block().parameters().size();
                    Block.Builder b = context.getBlock(phi.block());
                    yield b.parameters().get(index);
                }
            };
        }

        @Override
        public Block.Builder acceptOp(Block.Builder block, Op op) {
            Block originalBlock = op.ancestorBlock();
            CodeContext context = block.context();
            switch (op) {
                case CoreOp.VarAccessOp.VarLoadOp load -> {
                    Val val = this.loads.get(load);
                    context.mapValue(load.result(), resolveValue(context, val));
                }
                case CoreOp.VarOp _, CoreOp.VarAccessOp.VarStoreOp _ -> {
                }
                case Op.Terminating _ -> {
                    // make sure outgoing branches are corrected
                    for (Block.Reference successor : originalBlock.successors()) {
                        Block successorBlock = successor.targetBlock();
                        List<Phi> successorParams = this.additionalParameters.getOrDefault(successorBlock, List.of());
                        List<Value> additionalParams = successorParams.stream()
                                .map(phi -> readVariable(phi.variable, originalBlock))
                                .map(val -> resolveValue(context, val))
                                .toList();
                        List<Value> values = context.getValues(successor.arguments());
                        values.addAll(additionalParams);
                        Block.Builder successorBlockBuilder = context.getBlock(successorBlock);
                        context.mapSuccessor(successor, successorBlockBuilder.successor(values));
                    }
                    block.op(op);
                }
                default -> block.op(op);
            }
            return block;
        }

        @Override
        public void acceptBlock(Block.Builder block, Block b) {
            // add the required additional parameters to this block
            boolean isEntry = b.isEntryBlock();
            for (Phi phi : this.additionalParameters.getOrDefault(b, List.of())) {
                if (isEntry) {
                    // Phis in entry blocks denote captured values. Do not add as param but make sure
                    // the original value is used
                    assert phi.operands().size() == 1 : "entry block phi with multiple operands";
                    CodeContext context = block.context();
                    context.mapValue(resolveValue(context, phi), resolveValue(context, phi.operands().getFirst()));
                } else {
                    block.parameter(phi.variable.varValueType());
                }
            }

            // actually visit ops in this block
            CodeTransformer.super.acceptBlock(block, b);
        }

        sealed interface Val {
        }

        record Holder(Value value) implements Val {
        }

        enum Uninitialized implements Val {
            VALUE;
        }

        record Phi(CoreOp.VarOp variable, Block block, List<Val> operands, Set<Object> users) implements Val {
            Phi(CoreOp.VarOp variable, Block block) {
                this(variable, block, new ArrayList<>(), new HashSet<>());
            }

            void appendOperand(Val val) {
                this.operands.add(val);
                if (val instanceof Phi phi) { // load op uses are added separately
                    phi.users.add(this);
                }
            }

            @Override
            public boolean equals(Object obj) {
                return this == obj;
            }

            @Override
            public int hashCode() {
                return Objects.hash(this.variable, this.block);
            }

            public List<Phi> replaceBy(Val same, SSABraun construction) {
                List<Phi> usingPhis = new ArrayList<>();
                for (Object user : this.users) {
                    if (user == this) {
                        continue;
                    }
                    if (same instanceof Phi samePhi) {
                        samePhi.users.add(user);
                    }
                    switch (user) {
                        case Phi phi -> {
                            int i = phi.operands.indexOf(this);
                            assert i >= 0 : "use does not have this as operand";
                            phi.operands.set(i, same);
                            usingPhis.add(phi);
                        }
                        case CoreOp.VarAccessOp.VarLoadOp load -> construction.loads.put(load, same);
                        default -> throw new UnsupportedOperationException(user + ":" + user.getClass());
                    }
                }
                if (same instanceof Phi samePhi) {
                    samePhi.users.remove(this);
                }
                construction.currentDef.get(this.variable).put(this.block, same);
                construction.deletedPhis.add(this); // we might not replace all current defs, so mark this phi as deleted
                this.users.clear();
                return usingPhis;
            }

            @Override
            public String toString() {
                return "Phi[" + variable.varName() + "(" + block.index() + ")," + "operands: " + operands.size() + "}";
            }
        }
    }

    /**
     * An implementation of SSA construction based on
     * "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph" by Ron Cytron et. al.
     * <p>
     * This implementation contains some adaptions, notably:
     * <ul>
     *     <li>Adapt to block parameters rather than phi functions.</li>
     *     <li>Adapt to work with multiple bodies.</li>
     * </ul>
     */
    static final class SSACytron {
        private SSACytron() {
        }

        /**
         * Applies an SSA transformation to an operation with bodies, replacing operations that declare variables and
         * access them with the use of values they depend on or additional block parameters.
         * <p>
         * The operation should first be in lowered form before applying this transformation.
         * <p>
         * Note: this implementation does not currently work correctly when a variable is stored to within an exception
         * region and read from outside as a result of catching an exception. In such cases a complete transformation may be
         * not possible and such variables will need to be retained.
         *
         * @param nestedOp the operation with bodies
         * @return the transformed operation
         * @param <T> the operation type
         */
        static <T extends Op & Op.Nested> T transform(T nestedOp) {
            Map<Block, Set<CoreOp.VarOp>> joinPoints = new HashMap<>();
            Map<CoreOp.VarAccessOp.VarLoadOp, Object> loadValues = new HashMap<>();
            Map<Block.Reference, List<Object>> joinSuccessorValues = new HashMap<>();

            Map<Body, Boolean> visited = new HashMap<>();
            Map<Block, Map<CoreOp.VarOp, Block.Parameter>> joinBlockArguments = new HashMap<>();
            @SuppressWarnings("unchecked")
            T ssaOp = (T) nestedOp.transform(CodeContext.create(), (block, op) -> {
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
                        CodeContext cc = block.context();
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
                            CodeContext cc = block.context();

                            // Lazily append target block arguments
                            joinBlockArguments.computeIfAbsent(s.targetBlock(), b -> {
                                Block.Builder bb = cc.getBlock(b);
                                return joinPoints.get(b).stream().collect(Collectors.toMap(
                                        varOp -> varOp,
                                        varOp -> bb.parameter(varOp.varValueType())));
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

                    block.op(op);
                } else {
                    block.op(op);
                }

                return block;
            });
            return ssaOp;
        }

        record VarOpBlockArgument(Block b, CoreOp.VarOp vop) {
        }

        enum Uninitialized {
            UNINITIALIZED;
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
                        assert variableStack.containsKey(v);
                        variableStack.get(v).push(new VarOpBlockArgument(n.b(), v));
                    });
                }
            }

            {
                for (int i = 0; i < size - 1; i++) {
                    Op op = n.b().ops().get(i);

                    if (op instanceof CoreOp.VarOp varOp) {
                        // Initial value assigned to variable
                        Object current = varOp.isUninitialized()
                                ? Uninitialized.UNINITIALIZED
                                : op.operands().get(0);
                        assert !variableStack.containsKey(varOp);
                        variableStack.computeIfAbsent(varOp, _ -> new ArrayDeque<>())
                                .push(current);
                    } else if (op instanceof CoreOp.VarAccessOp.VarStoreOp storeOp) {
                        // Value assigned to variable
                        Value current = op.operands().get(1);
                        variableStack.get(storeOp.varOp()).push(current);
                    } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp loadOp &&
                               loadOp.varOp().ancestorBody() == op.ancestorBody()) {
                        Object to = peekAtCurrentVariable(variableStack, loadOp.varOp());
                        loadValues.put(loadOp, to);
                    } else if (op instanceof Op.Nested) {
                        // Traverse descendant variable loads for variables
                        // declared in the block's parent body
                        op.elements().forEach(ce -> {
                            if (ce instanceof CoreOp.VarAccessOp.VarLoadOp loadOp &&
                                    loadOp.varOp().ancestorBody() == op.ancestorBody()) {
                                Object to = peekAtCurrentVariable(variableStack, loadOp.varOp());
                                loadValues.put(loadOp, to);
                            }
                        });
                    }
                }

                // Add successor args for joint points
                for (Block.Reference succ : n.b().successors()) {
                    Set<CoreOp.VarOp> varOps = joinPoints.get(succ.targetBlock());
                    if (varOps != null) {
                        List<Object> joinValues = varOps.stream()
                                .map(vop -> peekAtCurrentVariable(variableStack, vop)).toList();
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

        static Object peekAtCurrentVariable(Map<CoreOp.VarOp, Deque<Object>> variableStack, CoreOp.VarOp vop) {
            Object to = variableStack.get(vop).peek();
            return throwIfUninitialized(vop, to);
        }

        static Object throwIfUninitialized(CoreOp.VarOp vop, Object to) {
            if (to instanceof Uninitialized) {
                throw new IllegalStateException("Loading from uninitialized variable: " + vop);
            }
            return to;
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
        static void findJoinPoints(Body body, Map<Block, Set<CoreOp.VarOp>> joinPoints) {
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
                            if (y.isDominatedBy(v.ancestorBlock())) {
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
            LinkedHashMap<CoreOp.VarOp, Set<Block>> stores = new LinkedHashMap<>();
            r.elements().forEach(e -> {
                if (e instanceof CoreOp.VarAccessOp.VarStoreOp storeOp) {
                    if (storeOp.varOp().ancestorBody() != storeOp.ancestorBody()) {
                        throw new IllegalStateException("Descendant variable store operation");
                    }
                    if (storeOp.varOp().ancestorBody() == r) {
                        stores.computeIfAbsent(storeOp.varOp(), _v -> new LinkedHashSet<>()).add(storeOp.ancestorBlock());
                    }
                }
            });
            return stores;
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
}
