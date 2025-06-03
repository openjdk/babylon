import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;

import java.util.*;
import java.util.function.Predicate;

public class LoopAnalyzer {
    public record Loop(Block header, Block back, Set<Block> body, List<LoopExit> exits) {
    }

    public record LoopExit(Block exit, Block target) {
    }

    static List<Loop> findLoops(Body body) {
        List<Loop> loops = new ArrayList<>();
        for (Block b : body.blocks()) {
            List<Block> backBranchTargets = b.predecessors()
                    .stream().filter(p -> p.isDominatedBy(b))
                    .toList();
            if (backBranchTargets.size() > 1) {
                throw new UnsupportedOperationException();
            } else if (backBranchTargets.size() == 1) {
                Block loopHeader = b;
                Block loopBackBranch = backBranchTargets.getFirst();

                // Find loop body
                // Depth first search from loopBackBranch terminating at loopHeader
                Deque<Block> stack = new ArrayDeque<>();
                stack.push(loopBackBranch);
                SequencedSet<Block> loopBody = new LinkedHashSet<>();
                Block node;
                while ((node = stack.pop()) != loopHeader) {
                    if (!loopBody.add(node)) {
                        continue;
                    }

                    stack.addAll(node.predecessors().reversed());
                }
                loopBody.add(loopHeader);

                // Find loop exits
                List<LoopExit> loopExits = new ArrayList<>();
                for (Block block : loopBody) {
                    if (block.terminatingOp() instanceof CoreOp.ConditionalBranchOp) {
                        List<LoopExit> list = block.successors().stream()
                                .map(Block.Reference::targetBlock)
                                .filter(o -> !loopBody.contains(o))
                                .map(o -> new LoopExit(block, o))
                                .toList();
                        loopExits.addAll(list);
                    }
                }
                Loop l = new Loop(loopHeader, loopBackBranch, loopBody, loopExits);
                loops.add(l);
            }
        }
        return loops;
    }

    static Set<Value> analyzeConstants(Map<Block, Loop> loops,
                                       Predicate<Op> opConstant,
                                       CoreOp.FuncOp f) {
        Set<Value> constants = new LinkedHashSet<>();
        analyzeConstants(f.body().entryBlock(), p -> false, null, loops, constants, opConstant);
        return constants;
    }

    static void analyzeConstants(Block inEntryBlock,
                                 Predicate<Block> p, Deque<Block> outside,
                                 Map<Block, Loop> loops,
                                 Set<Value> constants, Predicate<Op> opConstant) {
        // Ensure blocks are visited in reverse post order
        Queue<Block> stack = new PriorityQueue<>(Comparator.comparingInt(Block::index));

        // The first block cannot have any successors so the queue will have at least one entry
        stack.add(inEntryBlock);
        BitSet visited = new BitSet();
        while (!stack.isEmpty()) {
            final Block inBlock = stack.poll();
            if (visited.get(inBlock.index())) {
                continue;
            }
            visited.set(inBlock.index());
            if (p.test(inBlock)) {
                outside.push(inBlock);
                continue;
            }

            if (loops.containsKey(inBlock)) {
                // Loop header
                // Speculate
                Loop loop = loops.get(inBlock);
                Deque<Block> exits = new ArrayDeque<>();
                Map<Block, Loop> otherLoops = new HashMap<>(loops);
                otherLoops.remove(loop.header);
                Set<Value> speculatedConstants = new LinkedHashSet<>(constants);
                if (analyzeConstantLoop(loop, exits, otherLoops, speculatedConstants, opConstant)) {
                    stack.addAll(exits);
                    constants.addAll(speculatedConstants);
                    // Continue processing exits
                    continue;
                } else {
                    // Re-process without speculation
                }
            }

            // Process all but the terminating operation
            int nops = inBlock.ops().size();
            for (int i = 0; i < nops - 1; i++) {
                Op op = inBlock.ops().get(i);
                isConstant(op, constants, opConstant);
            }

            // Process the terminating operation
            Op to = inBlock.terminatingOp();
            switch (to) {
                case CoreOp.ConditionalBranchOp cb -> {
                    stack.add(cb.falseBranch().targetBlock());
                    stack.add(cb.trueBranch().targetBlock());

                    if (constants.containsAll(to.operands())) {
                        constants.add(to.result());
                    }
                }
                case CoreOp.BranchOp b -> {
                    stack.add(b.branch().targetBlock());
                }
                default -> {
                }
            }
        }
    }

    static boolean analyzeConstantLoop(
            Loop loop,
            Deque<Block> outside,
            Map<Block, Loop> loops,
            Set<Value> constants, Predicate<Op> opConstant) {
        for (Block pred : loop.header.predecessors()) {
            if (loop.header.isDominatedBy(pred)) {
                // Find reference to loop header
                boolean constantArguments = pred.successors().stream()
                        .filter(r -> r.targetBlock() == loop.header)
                        .flatMap(r -> r.arguments().stream())
                        .allMatch(constants::contains);
                if (!constantArguments) {
                    return false;
                }
            }
        }

        // Speculate
        constants.addAll(loop.header.parameters());

        analyzeConstants(loop.header, o -> !loop.body.contains(o), outside, loops, constants, opConstant);

        // Check back arguments are constant
        boolean constantArguments = loop.back.successors().stream()
                .filter(r -> r.targetBlock() == loop.header)
                .flatMap(r -> r.arguments().stream())
                .allMatch(constants::contains);
        if (!constantArguments) {
            return false;
        }

        // Check exits are constant
        for (LoopExit exit : loop.exits) {
            // Check branch is constant
            if (!constants.contains(exit.exit().terminatingOp().result())) {
                return false;
            }

            // Check exit arguments are constant
            boolean constantArguments2 = exit.exit().successors().stream()
                    .filter(r -> r.targetBlock() == exit.target())
                    .flatMap(r -> r.arguments().stream())
                    .allMatch(constants::contains);
            if (!constantArguments2) {
                return false;
            }
        }

        return true;
    }

    static boolean isConstant(Op op, Set<Value> constants, Predicate<Op> opConstant) {
        if (constants.contains(op.result())) {
            return true;
        } else if (constants.containsAll(op.operands()) && opConstant.test(op)) {
            constants.add(op.result());
            return true;
        } else {
            return false;
        }
    }

}
