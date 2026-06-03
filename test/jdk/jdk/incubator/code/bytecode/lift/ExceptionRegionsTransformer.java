/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

/// Normalizes consecutive lifted fragments of the same original exception region
///
/// Lifted:
///           entry
///             |
///     enter R#1 handler H
///        /          \
/// protected A    protected B
///      |              |
///   exit R#1       exit R#1
///      |              |
/// branch adapter branch adapter
///        \          /
///     enter R#2 handler H
///             |
///        protected C
///             |
///          exit R#2
///
/// Normalized:
///           entry
///             |
///     enter R handler H
///        /         \
/// protected A   protected B
///        \         /
///        protected C
///             |
///           exit R
final class ExceptionRegionsTransformer implements CodeTransformer {

    private boolean modified;

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {
        var t = new ExceptionRegionsTransformer();
        do {
            // Nested regions are exposed after the top are collapsed
            t.modified = false;
            func = func.transform(t);
        } while (t.modified);
        return func;
    }

    @Override
    public void acceptBlock(Block.Builder builder, Block b) {
        // Skip blocks whose terminating op was mapped by a previous collapse
        if (!builder.context().queryValue(b.terminatingOp().result()).isPresent()) {
            CodeTransformer.super.acceptBlock(builder, b);
        }
    }

    @Override
    public Block.Builder acceptOp(Block.Builder builder, Op op) {
        // Collapse matching exit/re-enter boundaries
        if (!(op instanceof JavaOp.ExceptionRegionExit exit) || !collapseExitToReenter(builder, exit)) {
            builder.add(op);
        }
        return builder;
    }

    private boolean collapseExitToReenter(Block.Builder builder, JavaOp.ExceptionRegionExit exit) {
        // Match exit -> optional branch adapter -> re-enter
        JavaOp.ExceptionRegionEnter exitedEnter = exit.enterOp();
        Block exitTarget = exit.endReference().targetBlock();
        CoreOp.BranchOp adapterBranch = asBranchAdapter(exitTarget);
        Block enterBlock = adapterBranch == null ? exitTarget : adapterBranch.branch().targetBlock();

        if (enterBlock.ops().size() != 1
                || !(enterBlock.terminatingOp() instanceof JavaOp.ExceptionRegionEnter reenter)
                || !isExitTargetOf(exitTarget, exitedEnter)
                || (adapterBranch != null && !isOnlyReachedByExitAdapters(enterBlock, exitedEnter))
                || !sameCatchTargets(exitedEnter, reenter)
                || !reenter.result().isDominatedBy(exitedEnter.result())) {
            return false;
        }

        // Reuse the old enter result and skip the adapter blocks
        var cc = builder.context();
        cc.mapValue(reenter.result(), cc.getValue(exitedEnter.result()));
        cc.mapValues(exitTarget.parameters(), cc.getValues(exit.endReference().arguments()));
        if (adapterBranch != null) {
            cc.mapValues(enterBlock.parameters(), cc.getValues(adapterBranch.branch().arguments()));
        }
        Op.Result result = builder.add(CoreOp.branch(cc.getReferenceOrCreate(reenter.startReference())));
        if (adapterBranch != null) {
            cc.mapValue(adapterBranch.result(), result);
        }
        modified = true;
        return true;
    }

    // Return a pure branch adapter, if this block is one
    private static CoreOp.BranchOp asBranchAdapter(Block block) {
        return block.ops().size() == 1 && block.terminatingOp() instanceof CoreOp.BranchOp branch
                ? branch
                : null;
    }

    // All predecessors must exit the same enter
    private static boolean isExitTargetOf(Block block, JavaOp.ExceptionRegionEnter enter) {
        return !block.predecessors().isEmpty()
                && block.predecessors().stream().allMatch(predecessor ->
                        predecessor.terminatingOp() instanceof JavaOp.ExceptionRegionExit e
                                && e.endReference().targetBlock() == block
                                && e.enterOp() == enter);
    }

    // Several branch adapters may feed the same re-enter block
    private static boolean isOnlyReachedByExitAdapters(Block block, JavaOp.ExceptionRegionEnter enter) {
        return block.predecessors().stream().allMatch(predecessor ->
                predecessor.terminatingOp() instanceof CoreOp.BranchOp branch
                        && branch.branch().targetBlock() == block
                        && isExitTargetOf(predecessor, enter));
    }

    // Compare real handler targets, not adapter blocks
    private static boolean sameCatchTargets(JavaOp.ExceptionRegionEnter first, JavaOp.ExceptionRegionEnter second) {
        return first.catchReferences().stream().map(ExceptionRegionsTransformer::skipBranchAdapter).toList()
                .equals(second.catchReferences().stream().map(ExceptionRegionsTransformer::skipBranchAdapter).toList());
    }

    // Peel one branch adapter
    private static Block skipBranchAdapter(Block.Reference reference) {
        Block target = reference.targetBlock();
        CoreOp.BranchOp adapterBranch = asBranchAdapter(target);
        return adapterBranch != null
                ? adapterBranch.branch().targetBlock()
                : target;
    }
}
