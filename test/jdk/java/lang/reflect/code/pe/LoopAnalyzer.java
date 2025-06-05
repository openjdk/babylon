import jdk.incubator.code.Block;
import jdk.incubator.code.Body;

import java.util.*;

public class LoopAnalyzer {
    public record Loop(Block header, Block back, Set<Block> body, List<LoopExit> exits) {
    }

    public record LoopExit(Block exit, Block target) {
    }

    public static Optional<Loop> isLoop(Block header) {
        List<Block> backBranchTargets = header.predecessors()
                .stream().filter(p -> p.isDominatedBy(header))
                .toList();
        if (backBranchTargets.size() == 1) {
            Block back = backBranchTargets.getFirst();
            Set<Block> body = loopBody(header, back);
            List<LoopExit> exits = loopExits(header, body);
            return Optional.of(new Loop(header, back, body, exits));
        } else {
            return Optional.empty();
        }
    }

    static Set<Block> naturalLoopBody(Block header, Block back) {
        Deque<Block> stack = new ArrayDeque<>();
        stack.push(back);
        Set<Block> loopBody = new HashSet<>();
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

    static Set<Block> loopBody(Block header, Block back) {
        Set<Block> naturalLoopBody = naturalLoopBody(header, back);

        Set<Block> loopBody = new HashSet<>(naturalLoopBody);
        for (Block lb : naturalLoopBody) {
            for (Block lbs : lb.successorTargets()) {
                if (!naturalLoopBody.contains(lbs)) {
                    // Find if there is path from lbs to back that does not pass through header
                    Deque<Block> stack = new ArrayDeque<>();
                    stack.push(lbs);
                    Set<Block> visited = new HashSet<>();
                    while (!stack.isEmpty()) {
                        Block x = stack.pop();
                        if (!visited.add(x)) {
                            continue;
                        }

                        if (find(x, header, back)) {
                            loopBody.add(x);
                        }

                        stack.addAll(x.successorTargets());
                    }
                }
            }
        }

        return loopBody;
    }

    // Determine if there is a forward path from x to y that does not pass through n
    static boolean find(Block x, Block n, Block y) {
        if (x == n) {
            return false;
        }
        if (x == y) {
            return true;
        }

        boolean r = false;
        for (Block b : x.successorTargets()) {
            // Back branch
            if (x.isDominatedBy(b))
                return false;
            if (find(b, n, y)) {
                return true;
            }
        }
        return false;
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
