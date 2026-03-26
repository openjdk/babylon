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
package optkl.util;
import optkl.jdot.ui.JDot;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

public class Dag<N> implements LookupCarrier {

    final public MethodHandles.Lookup lookup;

    @Override
    public MethodHandles.Lookup lookup() {
        return lookup;
    }

    protected final Set<N> nodeSet = new HashSet<>();
    public final List<N> rankOrdered = new LinkedList<>();

    protected final Map<N, Set<N>> fromToNodes = new HashMap<>();
    public void view(String name, Function<N,String> dotNodeLabelRenderer) {
        JDot.digraph(name, $ ->
                fromToNodes.forEach((l, r) ->
                        r.forEach(e ->
                                $.edge(dotNodeLabelRenderer.apply(l), dotNodeLabelRenderer.apply(e))
                        )
                ));
    }

    public boolean isDag() {
        return fromToNodes.size()>1;
    }


    protected Dag(MethodHandles.Lookup lookup) {
        this.lookup = lookup;
    }

    protected void  computeIfAbsent(N from, N to, Consumer<N> ifAbsent){
        if (!nodeSet.contains(to)) {
            nodeSet.add(to);
            fromToNodes.put(to, new HashSet<>());
            ifAbsent.accept(to);
        }
        fromToNodes.get(from).add(to);
    }


    public  void closeRanks() {
        Map<N, Integer> outDegree = new HashMap<>();
        Map<N, List<N>> reverseEdges = new HashMap<>();

        for (N parent : fromToNodes.keySet()) {
            outDegree.put(parent, fromToNodes.get(parent).size());
            for (N child : fromToNodes.get(parent)) {
                reverseEdges.computeIfAbsent(child, k -> new ArrayList<>()).add(parent);
                outDegree.putIfAbsent(child, 0);
            }
        }
        final Queue<N> queue = new LinkedList<>();
        for (Map.Entry<N, Integer> entry : outDegree.entrySet()) {
            if (entry.getValue() == 0) {
                queue.add(entry.getKey());
            }
        }

        while (!queue.isEmpty()) {
            N current = queue.poll();
            rankOrdered.add(current);
            List<N> parents = reverseEdges.getOrDefault(current, Collections.emptyList());
            for (N parent : parents) {
                int remainingChildren = outDegree.get(parent) - 1;
                outDegree.put(parent, remainingChildren);
                if (remainingChildren == 0) {
                    queue.add(parent);
                }
            }
        }
    }
}
