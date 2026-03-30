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
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.function.Function;

public class Dag<N> extends BiMapOfSets<N> implements LookupCarrier {
    final public MethodHandles.Lookup lookup;

    @Override
    public MethodHandles.Lookup lookup() {
        return lookup;
    }

    public final List<N> rankOrdered = new LinkedList<>();

    public void view(String name, Function<N, String> dotNodeLabelRenderer) {
        JDot.digraph(name, $ ->
                fromTo.forEach((l, r) ->
                        r.forEach(e ->
                                $.edge(dotNodeLabelRenderer.apply(l), dotNodeLabelRenderer.apply(e))
                        )
                ));
    }

    public boolean isDag() {
        return fromTo.size() > 1;
    }

    protected Dag(MethodHandles.Lookup lookup) {
        this.lookup = lookup;
    }

    public void closeRanks() {
        Map<N, Integer> outDegree = new HashMap<>();

        fromTo.keySet().forEach(parent -> {
            outDegree.put(parent, fromTo.get(parent).size());
            fromTo.get(parent).forEach(child ->
                    outDegree.putIfAbsent(child, 0)
            );
        });

        Queue<N> queue = new LinkedList<>(outDegree.entrySet().stream().filter(e -> e.getValue() == 0).map(Map.Entry::getKey).toList());

        while (queue.poll() instanceof N current) { // basically a null check
            rankOrdered.add(current);
            toFrom.getOrDefault(current, Collections.emptySet()).forEach(parent -> {
                int remainingChildren = outDegree.get(parent) - 1;
                outDegree.put(parent, remainingChildren);
                if (remainingChildren == 0) {
                    queue.add(parent);
                }
            });
        }
    }
}
