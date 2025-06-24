/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package oracle.code.onnx.compiler;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;

import java.util.*;

public class LoopAnalyzer {
    public record Loop(Block header, List<Block> latches, Set<Block> body, List<LoopExit> exits) {
    }

    public record LoopExit(Block exit, Block target) {
    }

    public static Optional<Loop> isLoop(Block header) {
        List<Block> latches = header.predecessors()
                .stream().filter(p -> p.isDominatedBy(header))
                .toList();
        if (latches.isEmpty()) {
            return Optional.empty();
        }

        Set<Block> body = new HashSet<>();
        for (Block latch : latches) {
            loopBody(body, header, latch);
        }
        List<LoopExit> exits = loopExits(body);
        return Optional.of(new Loop(header, latches, body, exits));
    }

    static Set<Block> naturalLoopBody(Set<Block> loopBody, Block header, Block latch) {
        Deque<Block> stack = new ArrayDeque<>();
        stack.push(latch);
        Block node;
        // Backward depth first search from latch to header
        while (!stack.isEmpty() && (node = stack.pop()) != header) {
            if (!loopBody.add(node)) {
                continue;
            }

            stack.addAll(node.predecessors().reversed());
        }
        loopBody.add(header);

        return loopBody;
    }

    static Set<Block> loopBody(Set<Block> loopBody, Block header, Block latch) {
        naturalLoopBody(loopBody, header, latch);

        Set<Block> extendedLoopBody = new HashSet<>();
        for (Block lb : loopBody) {
            for (Block lbs : lb.successorTargets()) {
                if (!loopBody.contains(lbs) && !extendedLoopBody.contains(lbs)) {
                    // Find if there is path from lbs to latch that does not pass through header
                    Deque<Block> stack = new ArrayDeque<>();
                    stack.push(lbs);
                    Set<Block> visited = new HashSet<>();
                    while (!stack.isEmpty()) {
                        Block x = stack.pop();
                        if (!visited.add(x)) {
                            continue;
                        }

                        if (find(x, header, latch)) {
                            extendedLoopBody.add(x);
                        }

                        stack.addAll(x.successorTargets());
                    }
                }
            }
        }

        loopBody.addAll(extendedLoopBody);
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

    static List<LoopExit> loopExits(Set<Block> loopBody) {
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
