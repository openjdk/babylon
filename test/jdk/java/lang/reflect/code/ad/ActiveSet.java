/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOps;
import java.util.*;

public final class ActiveSet {

    private ActiveSet() {
    }

    // Create active value set, in order, starting from the given block parameter
    // following its users and following through block arguments
    // to block parameters, and so on until the graph is traversed.

    public static Set<Value> activeSet(CoreOps.FuncOp f, Block.Parameter fv) {
        if (!f.body().blocks().get(0).parameters().contains(fv)) {
            throw new IllegalArgumentException("Arg is not defined by function");
        }
        Deque<Value> q = new ArrayDeque<>();
        q.push(fv);

        Set<Value> active = new TreeSet<>();
        while (!q.isEmpty()) {
            Value v = q.pop();
            if (active.contains(v)) {
                continue;
            }
            active.add(v);

            // @@@ assume uses are declared in order?
            //     if so can push to queue in reverse order
            for (Op.Result or : v.uses()) {
                q.push(or);
                Op op = or.op();

                if (op instanceof Op.Terminating) {
                    for (Block.Reference s : op.successors()) {
                        for (int i = 0; i < s.arguments().size(); i++) {
                            if (v == s.arguments().get(i)) {
                                Block b = s.targetBlock();
                                Block.Parameter ba = b.parameters().get(i);

                                // Processing of block arguments may result in out of order
                                // production of uses if two or more block arguments are added
                                // for the same successor argument
                                q.push(ba);
                            }
                        }
                    }
                }
            }
        }

        // Ensure non-active block arguments of successors are added to the
        // active set for blocks with corresponding active parameters
        // Backtracking is not performed on the values as they are not strictly
        // active set but may be required for initialization purposes.
        Set<Value> bactive = new LinkedHashSet<>();
        for (Value v : active) {
            if (v instanceof Block.Parameter ba) {
                Block b = ba.declaringBlock();
                int i = b.parameters().indexOf(ba);

                for (Block p : b.predecessors()) {
                    Op to = p.terminatingOp();
                    for (Block.Reference s : to.successors()) {
                        if (s.targetBlock() == b) {
                            Value arg = s.arguments().get(i);
                            if (!active.contains(arg)) {
                                bactive.add(arg);
                                bactive.add(to.result());
                            }
                        }
                    }
                }
            }
        }
        active.addAll(bactive);

        return active;
    }
}
