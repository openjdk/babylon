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
package hat.optools;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

public class RootSet {

    public final Set<Op> set;
    public RootSet(Stream<Op> ops){
        this.set = getRootSet(ops);
    }
    static public Stream<OpWrapper<?>> rootsWithoutVarFuncDeclarationsOrYields(MethodHandles.Lookup lookup,Block block) {
        RootSet rootSet = new RootSet(block.ops().stream());
        return block.ops().stream()
                .filter(rootSet.set::contains).map(o->OpWrapper.wrap(lookup,o))
                .filter(w -> !(w instanceof VarFuncDeclarationOpWrapper))
                .filter(w -> !(w instanceof YieldOpWrapper))
                .map(o->(OpWrapper<?>) o);
    }
    private static Set<Op> getRootSet(Stream<Op> ops) {
         record Node<T extends Value>(T node, List<Node<T>> children) {
        }
        Set<Op> roots = new LinkedHashSet<>();
        Map<Op, Node<Value>> trees = new LinkedHashMap<>();
        Map<Value, Node<Value>> params = new HashMap<>();
        ops.forEach(op -> {
            List<Node<Value>> children = new ArrayList<>();
            for (Value operand : op.operands()) {
                if (operand instanceof Op.Result opr) {
                    children.add(trees.get(opr.op()));
                } else {
                    children.add(params.computeIfAbsent(operand, _ -> new Node<>(operand, List.of())));
                }
            }
            trees.put(op, new Node<>(op.result(), children));
        });

        trees.forEach((op, _) -> {
            if (op instanceof CoreOp.VarAccessOp.VarStoreOp) {
                Value value = op.operands().get(1);
                if (value.uses().size() < 2) {
                    roots.add(op);
                }
            } else if (op instanceof CoreOp.VarOp || op.result().uses().isEmpty()) {
                roots.add(op);
            }
        });
        return roots;
    }
}
