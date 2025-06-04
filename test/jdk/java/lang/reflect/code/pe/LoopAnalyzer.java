import jdk.incubator.code.Block;
import jdk.incubator.code.Body;

import java.util.*;

public class LoopAnalyzer {
    public record Loop(Block header, Block back, Set<Block> body, List<LoopExit> exits) {
    }

    public record LoopExit(Block exit, Block target) {
    }

    public static Optional<Loop> isLoop(Block header) {
        // @@@ Only works for natural loops, and not those with explicit break/continue
        List<Block> backBranchTargets = header.predecessors()
                .stream().filter(p -> p.isDominatedBy(header))
                .toList();
        if (backBranchTargets.size() == 1) {
            Block back = backBranchTargets.getFirst();
            SequencedSet<Block> body = loopBody(header, back);
            List<LoopExit> exits = loopExits(header, body);
            return Optional.of(new Loop(header, back, body, exits));
        } else {
            return Optional.empty();
        }
    }

    static SequencedSet<Block> loopBody(Block header, Block back) {
        Deque<Block> stack = new ArrayDeque<>();
        stack.push(back);
        SequencedSet<Block> loopBody = new LinkedHashSet<>();
        Block node;
        // Backward depth first search from back to header
        while ((node = stack.pop()) != header) {
            if (!loopBody.add(node)) {
                continue;
            }

            stack.addAll(node.predecessors().reversed());
        }
        loopBody.add(header);

        return loopBody;
    }

    static List<LoopExit> loopExits(Block header, Set<Block> loopBody) {
        List<LoopExit> loopExits = new ArrayList<>();
        for (Block block : loopBody) {
            for (Block t : block.successorTargets()) {
                if (!loopBody.contains(t)) {
                    loopExits.add(new LoopExit(block, t));
                }
            }
        }
        return loopExits;
    }

    static List<Loop> findLoops(Body body) {
        return body.blocks().stream().flatMap(b -> isLoop(b).stream()).toList();
    }
}
