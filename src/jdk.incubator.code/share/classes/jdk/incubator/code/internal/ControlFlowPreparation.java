/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code.internal;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

/**
 * Prepares a fixed point control-flow and builds the model once
 */
public final class ControlFlowPreparation  implements CodeTransformer {

    @SuppressWarnings("unchecked")
    public static <O extends Op> O transform(O operation) {
        Objects.requireNonNull(operation);
        List<Op> branches = operation.elements().filter(e -> e instanceof CoreOp.BranchOp
                || e instanceof CoreOp.ConditionalBranchOp).map(Op.class::cast).toList();
        if (branches.stream().noneMatch(CoreOp.ConditionalBranchOp.class::isInstance)) {
            return operation;
        }
        ControlFlowPreparation cfp = new ControlFlowPreparation(operation, branches);
        cfp.prepare();
        return cfp.rewrites.isEmpty() ? operation : (O) operation.transform(CodeContext.create(), cfp);
    }

    private record Target(Block block, List<Value> arguments) {
        Target(Block.Reference reference) {
            this(reference.targetBlock(), reference.arguments());
        }

        Block.Reference map(CodeContext context) {
            return context.getBlock(block).reference(context.getValues(arguments));
        }
    }

    final Op root;
    final List<Op> branches;
    final Map<Op, List<Target>> rewrites = new HashMap<>();
    final Map<Op, List<Target>> originalTargets = new HashMap<>();
    final Map<Body, BitSet> reachable = new HashMap<>();
    final BitSet routeVisited = new BitSet();
    final Map<Block, Boolean> routable = new HashMap<>();
    final Map<Block, List<List<Value>>> incoming = new HashMap<>();
    final ConstantValueAnalysis constants = new ConstantValueAnalysis(block -> incoming.getOrDefault(block, List.of()));

    private ControlFlowPreparation(Op root, List<Op> branches) {
        this.root = root;
        this.branches = branches;
    }

    private void prepare() {
        boolean changed;
        do {
            reachable.values().forEach(BitSet::clear);
            incoming.values().forEach(List::clear);
            Deque<Block> work = new ArrayDeque<>();
            for (var body : root.bodies()) work.add(body.entryBlock());
            while (!work.isEmpty()) {
                Block block = work.removeFirst();
                BitSet live = reachable.computeIfAbsent(block.parent(), _ -> new BitSet());
                if (live.get(block.index())) continue;
                live.set(block.index());
                for (Op operation : block.ops())
                    for (var body : operation.bodies()) work.add(body.entryBlock());
                for (Target target : targets(block.terminatingOp())) {
                    incoming.computeIfAbsent(target.block, _ -> new ArrayList<>()).add(target.arguments);
                    work.add(target.block);
                }
            }
            // stop when every reachable conditional is resolved
            if (branches.stream().noneMatch(op -> op instanceof CoreOp.ConditionalBranchOp
                    && isReachable(op.parent()) && targets(op).size() == 2)) return;
            // known constants remain valid
            constants.clearUnknowns();
            changed = false;
            for (Op branch : branches) {
                if (isReachable(branch.parent())) changed |= discover(branch);
            }
        } while (changed);
    }

    private List<Target> targets(Op operation) {
        List<Target> changed = rewrites.get(operation);
        return changed != null ? changed : originalTargets.computeIfAbsent(operation,
                op -> op.successors().stream().map(Target::new).toList());
    }

    private boolean isReachable(Block block) {
        BitSet live = reachable.get(block.parent());
        return live != null && live.get(block.index());
    }

    private boolean discover(Op operation) {
        List<Target> before = targets(operation);
        List<Target> after;
        if (operation instanceof CoreOp.ConditionalBranchOp branch && before.size() == 2
                && constants.booleanValue(branch.predicateOperand()) instanceof Boolean value) {
            after = List.of(route(before.get(value ? 0 : 1)));
        } else {
            // allocate replacement successors when routing changes
            after = null;
            for (int i = 0; i < before.size(); i++) {
                Target next = route(before.get(i));
                if (!next.equals(before.get(i))) {
                    if (after == null) after = new ArrayList<>(before);
                    after.set(i, next);
                }
            }
            if (after == null) return false;
        }
        if (before.equals(after)) return false;
        rewrites.put(operation, after);
        return true;
    }

    private boolean canRoute(Block block) {
        return routable.computeIfAbsent(block, b ->
                (b.terminatingOp() instanceof CoreOp.BranchOp
                        || b.terminatingOp() instanceof CoreOp.ConditionalBranchOp)
                && (b.ops().size() == 1 || b.parameters().size() == 1
                        && b.parameters().getFirst().type().equals(JavaType.BOOLEAN)
                        && b.ops().stream().limit(b.ops().size() - 1)
                                .allMatch(JavaOp.NotOp.class::isInstance)));
    }

    private Target route(Target start) {
        if (!canRoute(start.block)) return start;
        Target current = start, selected = start;
        Boolean argument = booleanArgument(current);
        // routing stays within one immutable body
        routeVisited.clear();
        while (!current.block.isEntryBlock() && !routeVisited.get(current.block.index())) {
            Block block = current.block;
            routeVisited.set(block.index());
            Op terminator = block.terminatingOp();
            List<Op> operations = block.ops().subList(0, block.ops().size() - 1);
            List<Target> outgoing = targets(terminator);
            // empty forwarding block with no side-used parameters
            if (operations.isEmpty() && outgoing.size() == 1
                    && (terminator instanceof CoreOp.BranchOp || terminator instanceof CoreOp.ConditionalBranchOp)
                    && block.parameters().stream().allMatch(p -> onlyUsedBy(p, terminator))) {
                // compose edge arguments through empty forwarding blocks without merging blocks or deleting parameters
                if (!canRoute(outgoing.getFirst().block)) break;
                List<Value> arguments = current.arguments;
                Boolean previous = argument;
                List<Value> forwarded = outgoing.getFirst().arguments.stream()
                        .map(v -> v instanceof Block.Parameter p && p.declaringBlock() == block
                                ? arguments.get(p.index()) : v).toList();
                argument = outgoing.getFirst().arguments.size() == 1
                        && block.parameters().size() == 1
                        && outgoing.getFirst().arguments.getFirst() == block.parameters().getFirst()
                        ? previous : booleanArgument(new Target(outgoing.getFirst().block, forwarded));
                current = new Target(outgoing.getFirst().block, forwarded);
                continue;
            }
            // remaining blocks with single boolean
            if (argument == null || block.parameters().size() != 1
                    || !block.parameters().getFirst().type().equals(JavaType.BOOLEAN)) break;
            Value value = block.parameters().getFirst();
            boolean valid = true;
            // successive negations with no side uses
            for (Op operation : operations) {
                if (!(operation instanceof JavaOp.NotOp not)
                        || not.operand() != value || !onlyUsedBy(value, operation)) {
                    valid = false;
                    break;
                }
                value = operation.result();
                argument = !argument;
            }
            if (!valid || !onlyUsedBy(value, terminator)) break;
            // shortcut for boolean dispatch with no-args exits
            if (terminator instanceof CoreOp.ConditionalBranchOp branch
                    && branch.predicateOperand() == value
                    && outgoing.stream().allMatch(target -> target.arguments.isEmpty())) {
                current = outgoing.get(outgoing.size() == 1 ? 0 : argument ? 0 : 1);
                selected = current;
                argument = booleanArgument(current);
            } else if (terminator instanceof CoreOp.BranchOp
                    && outgoing.getFirst().arguments.size() == 1
                    && outgoing.getFirst().arguments.getFirst() == value) {
                // carry boolean through a chain, no synthetic literal
                current = new Target(outgoing.getFirst().block, current.arguments);
            } else {
                break;
            }
        }
        // retain the last target
        return selected;
    }

    private Boolean booleanArgument(Target target) {
        return target.arguments.size() == 1
                ? constants.booleanValue(target.arguments.getFirst()) : null;
    }

    private boolean onlyUsedBy(Value value, Op selected) {
        // routing commits only zero-argument dispatch exits
        // conditional selection only removes edges
        for (Op.Result use : value.uses()) {
            Op operation = use.op();
            if (operation == selected || !isReachable(operation.parent())) continue;
            boolean collapsed = operation instanceof CoreOp.ConditionalBranchOp
                    && targets(operation).size() == 1;
            if (!collapsed && operation.operands().contains(value)
                    || targets(operation).stream().anyMatch(t -> t.arguments.contains(value))) return false;
        }
        return true;
    }

    @Override
    public void acceptBlock(Block.Builder builder, Block block) {
        if (isReachable(block)) CodeTransformer.super.acceptBlock(builder, block);
    }

    @Override
    public Block.Builder acceptOp(Block.Builder builder, Op operation) {
        List<Target> targets = rewrites.get(operation);
        CodeContext context = builder.context();
        Op replacement;
        if (targets == null) {
            Optional<Object> constant;
            if (operation.ancestorOp() instanceof JavaOp.LabeledOp labeled && labeled.label() == operation
                    || !operation.bodies().isEmpty()
                    || (constant = constants.evaluate(operation.result())).isEmpty()) {
                builder.add(operation);
                return builder;
            }
            // drop constant expressions consumed only by folded expressions
            if (onlyUsedBy(operation.result(), null)) {
                return builder;
            }
            replacement = CoreOp.constant(operation.resultType(), constant.orElseThrow());
        } else {
            replacement = targets.size() == 1
                    ? CoreOp.branch(targets.getFirst().map(context))
                    : CoreOp.conditionalBranch(context.getValue(operation.operands().getFirst()),
                            targets.get(0).map(context), targets.get(1).map(context));
        }
        Op.Result result = builder.add(replacement);
        result.op().setLocation(operation.location());
        context.mapValue(operation.result(), result);
        return builder;
    }
}
