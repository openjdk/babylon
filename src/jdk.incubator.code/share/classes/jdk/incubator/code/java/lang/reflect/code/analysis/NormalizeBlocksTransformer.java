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

package jdk.incubator.code.java.lang.reflect.code.analysis;

import jdk.incubator.code.java.lang.reflect.code.*;
import jdk.incubator.code.java.lang.reflect.code.op.CoreOp;
import java.util.*;

/**
 * A model transformer that normalizes blocks.
 * <p>
 * Merges redundant blocks with their predecessors, those which are unconditionally
 * branched to and have only one predecessor.
 * <p>
 * Removes unused block parameters.
 */
public final class NormalizeBlocksTransformer implements OpTransformer {
    final Set<Block> mergedBlocks = new HashSet<>();
    final Map<Block, BitSet> adjustedBlocks = new HashMap<>();

    private NormalizeBlocksTransformer() {
    }

    /**
     * Transforms an operation, merging redundant blocks.
     *
     * @param op  the operation to transform
     * @param <O> the type of operation
     * @return the transformed operation
     */
    @SuppressWarnings("unchecked")
    public static <O extends Op> O transform(O op) {
        return (O) op.transform(CopyContext.create(), new NormalizeBlocksTransformer());
    }

    ;

    @Override
    public void apply(Block.Builder block, Block b) {
        // Ignore merged block
        if (!mergedBlocks.contains(b)) {
            OpTransformer.super.apply(block, b);
        }
    }

    @Override
    public Block.Builder apply(Block.Builder b, Op op) {
        if (op instanceof CoreOp.BranchOp bop &&
                bop.branch().targetBlock().predecessors().size() == 1) {
            // Merge the successor's target block with this block, and so on
            // The terminal branch operation is replaced with the operations in the
            // successor's target block
            mergeBlock(b, bop);
            return b;
        } else if (op instanceof CoreOp.ExceptionRegionEnter ere) {
            // Cannot remove block parameters from exception handlers
            removeUnusedBlockParameters(b, ere.start());
        } else if (op instanceof CoreOp.ExceptionRegionExit ere) {
            // Cannot remove block parameters from exception handlers
            removeUnusedBlockParameters(b, ere.end());
        } else if (op instanceof Op.BlockTerminating) {
            for (Block.Reference successor : op.successors()) {
                removeUnusedBlockParameters(b, successor);
            }
        }
        b.op(op);
        return b;
    }

    // Remove any unused block parameters and successor arguments
    private void removeUnusedBlockParameters(Block.Builder b, Block.Reference successor) {
        Block target = successor.targetBlock();
        BitSet unusedParameterIndexes = adjustedBlocks.computeIfAbsent(target,
                k -> adjustBlock(b, k));
        if (!unusedParameterIndexes.isEmpty()) {
            adjustSuccessor(unusedParameterIndexes, b, successor);
        }

    }

    // Remove any unused block parameters
    BitSet adjustBlock(Block.Builder b, Block target) {
        // Determine the indexes of unused block parameters
        List<Block.Parameter> parameters = target.parameters();
        BitSet unusedParameterIndexes = parameters.stream()
                .filter(p -> p.uses().isEmpty())
                .mapToInt(Block.Parameter::index)
                .collect(BitSet::new, BitSet::set, BitSet::or);

        if (!unusedParameterIndexes.isEmpty()) {
            // Create a new output block and remap it to the target block,
            // overriding any previous mapping
            Block.Builder adjustedBlock = b.block();
            b.context().mapBlock(target, adjustedBlock);

            // Update and remap the output block parameters
            for (int i = 0; i < parameters.size(); i++) {
                if (!unusedParameterIndexes.get(i)) {
                    Block.Parameter parameter = parameters.get(i);
                    b.context().mapValue(
                            parameter,
                            adjustedBlock.parameter(parameter.type()));
                }
            }
        }

        return unusedParameterIndexes;
    }

    // Remove any unused successor arguments
    void adjustSuccessor(BitSet unusedParameterIndexes, Block.Builder b, Block.Reference successor) {
        // Create a new output successor and remap it
        List<Value> arguments = new ArrayList<>();
        for (int i = 0; i < successor.arguments().size(); i++) {
            if (!unusedParameterIndexes.get(i)) {
                arguments.add(b.context().getValue(successor.arguments().get(i)));
            }
        }
        Block.Reference adjustedSuccessor = b.context().getBlock(successor.targetBlock())
                .successor(arguments);
        b.context().mapSuccessor(successor, adjustedSuccessor);
    }

    void mergeBlock(Block.Builder b, CoreOp.BranchOp bop) {
        Block.Reference reference = bop.branch();
        Block successor = reference.targetBlock();
        // Replace use of the successor's parameters with the reference's arguments
        b.context().mapValues(successor.parameters(),
                b.context().getValues(reference.arguments()));
        mergeBlock(b, successor);
    }

    void mergeBlock(Block.Builder b, Block successor) {
        mergedBlocks.add(successor);

        // Merge non-terminal operations
        for (int i = 0; i < successor.ops().size() - 1; i++) {
            b.op(successor.ops().get(i));
        }

        // Check if subsequent successor block can be normalized
        apply(b, successor.terminatingOp());
    }
}
