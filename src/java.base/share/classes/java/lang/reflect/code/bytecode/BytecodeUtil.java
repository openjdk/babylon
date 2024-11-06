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
package java.lang.reflect.code.bytecode;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Value;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

final class BytecodeUtil {

    /**
     * Returns {@code true} if this value is dominated by the given set of values {@code doms}.
     * <p>
     * The set dominates if every path from the entry node go through any member of the set.
     * <p>
     * First part checks individual dominance of every member of the set.
     * <p>
     * If no member of the set is individually dominant, the second part tries to find path
     * to the entry block bypassing all blocks from the tested set.
     * <p>
     * Implementation searches for the paths by traversing the value declaring block predecessors,
     * stopping at blocks where values from the tested set are declared or at blocks already processed.
     * Negative test result is returned when the entry block is reached.
     * Positive test result is returned when no path to the entry block is found.
     *
     * @param value the value
     * @param doms the dominating set of values
     * @return {@code true} if this value is dominated by the given set of values {@code dom}.
     * @throws IllegalStateException if the declaring block is partially built
     */
    public static boolean isDominatedBy(Value value, Set<? extends Value> doms) {
        if (doms.isEmpty()) {
            return false;
        }

        for (Value dom : doms) {
            if (value.isDominatedBy(dom)) {
                return true;
            }
        }

        Set<Block> stopBlocks = new HashSet<>();
        for (Value dom : doms) {
            stopBlocks.add(dom.declaringBlock());
        }

        Deque<Block> toProcess = new ArrayDeque<>();
        toProcess.add(value.declaringBlock());
        stopBlocks.add(value.declaringBlock());
        while (!toProcess.isEmpty()) {
            for (Block b : toProcess.pop().predecessors()) {
                if (b.isEntryBlock()) {
                    return false;
                }
                if (stopBlocks.add(b)) {
                    toProcess.add(b);
                }
            }
        }
        return true;
    }

    private BytecodeUtil() {
    }
}
