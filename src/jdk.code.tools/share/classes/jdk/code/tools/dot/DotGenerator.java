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

package jdk.code.tools.dot;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Value;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class DotGenerator {

    final Writer w;

    interface NodeProperty {
        String key();

        String value();

        default String toText() {
            return key() + "=" + value();
        }
    }

    static NodeProperty property(String key, String value) {
        return new NodeProperty() {
            @Override
            public String key() {
                return key;
            }

            @Override
            public String value() {
                return value;
            }
        };
    }

    static String properties(List<? extends NodeProperty> properties) {
        return properties.stream().map(NodeProperty::toText).collect(Collectors.joining(" ", "[", "]"));
    }

    static NodeProperty label(String name) {
        return new NodeProperty() {
            @Override
            public String key() {
                return "label";
            }

            @Override
            public String value() {
                return "\"" + name + "\"";
            }
        };
    }

    enum Shape implements NodeProperty {
        BOX("box"),
        ELLIPSE("ellipse"),
        HEXAGONE("hexagon"),
        INVERTED_TRAPEZIUM("invtrapezium");

        final String value;

        Shape(String value) {
            this.value = value;
        }


        @Override
        public String key() {
            return "shape";
        }

        @Override
        public String value() {
            return value;
        }
    }

    private DotGenerator(Writer w) {
        this.w = w;
    }

    void digraph() {
        write("digraph G {\n");
    }

    void node(Object o, String properties) {
        write("%s %s;\n", System.identityHashCode(o), properties);
    }

    void node(Object o, NodeProperty... properties) {
        node(o, List.of(properties));
    }

    void node(Object o, List<? extends NodeProperty> properties) {
        node(o, properties(properties));
    }

    void edge(Object from, Object to) {
        write("%s -> %s;\n", System.identityHashCode(from), System.identityHashCode(to));
    }

    void write(String format, Object... args) {
        write(w, format, args);
    }

    void end() {
        write(w, "}\n");
        try {
            w.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static void write(Writer w, String format, Object... args) {
        try {
            w.write(String.format(format, args));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }


    /**
     * Generates the representation tree for a given operation.
     *
     * @param op the operation
     * @param w  the writer to write the sr.dot file
     */
    public static void representationTree(Op op, Writer w) {
        DotGenerator dg = new DotGenerator(w);

        dg.digraph();

        op.traverse(null, (t, codeElement) -> switch (codeElement) {
            case Body b -> {
                dg.node(b, label(""), Shape.HEXAGONE, property("style", "filled"));

                dg.edge(b.parentOp(), b);

                yield null;
            }
            case Block b -> {
                dg.node(b, label(""), Shape.BOX);

                dg.edge(b.parentBody(), b);

                yield null;
            }
            case Op o -> {
                List<NodeProperty> ps;
                if (o instanceof Op.Terminating) {
                    ps = List.of(label(o.opName()), Shape.ELLIPSE, property("style", "filled"));
                } else {
                    ps = List.of(label(o.opName()), Shape.ELLIPSE);
                }
                dg.node(o, ps);
                if (o.parentBlock() != null) {
                    dg.edge(o.parentBlock(), o);
                }

                yield null;
            }
        });

        dg.end();
    }

    /**
     * Generates a body graph (CFG) for a given body.
     *
     * @param body the body
     * @param w    the writer to write the sr.dot file
     */
    public static void bodyGraph(Body body, Writer w) {
        DotGenerator dg = new DotGenerator(w);

        dg.digraph();

        Block eb = body.entryBlock();
        Deque<Block> stack = new ArrayDeque<>();
        Set<Block> visited = new HashSet<>();
        stack.push(eb);
        while (!stack.isEmpty()) {
            Block b = stack.pop();
            if (!visited.add(b)) {
                continue;
            }

            dg.node(b, label(""), Shape.BOX);

            List<Block.Reference> successors = b.terminatingOp().successors();
            for (Block.Reference s : successors) {
                dg.edge(b, s.targetBlock());

                stack.push(s.targetBlock());
            }
        }

        dg.end();
    }

    /**
     * Generates a body dominator tree for a given body.
     *
     * @param body the body
     * @param w    the writer to write the sr.dot file
     */
    public static void bodyDominatorTree(Body body, Writer w) {
        DotGenerator dg = new DotGenerator(w);

        dg.digraph();

        Block eb = body.entryBlock();
        Map<Block, Block> idoms = body.immediateDominators();

        for (Map.Entry<Block, Block> e : idoms.entrySet()) {
            Block child = e.getKey();
            Block parent = e.getValue();

            dg.node(child, label(""), Shape.BOX);

            if (child != eb) {
                dg.edge(parent, child);
            }
        }

        dg.end();
    }

    /**
     * Generates a body dominator tree for a given body, with the dominance
     * frontier set presented for each block.
     * <p>
     * The dominance frontier of a block, b say, is the set of blocks where the b's
     * dominance stops.
     *
     * @param body the body
     * @param w    the writer to write the sr.dot file
     */
    public static void bodyDominanceFrontierTree(Body body, Writer w) {
        DotGenerator dg = new DotGenerator(w);

        dg.digraph();

        Block eb = body.entryBlock();
        Map<Block, Block> idoms = body.immediateDominators();
        Map<Block, Set<Block>> df = body.dominanceFrontier();

        for (Map.Entry<Block, Block> e : idoms.entrySet()) {
            Block child = e.getKey();
            Block parent = e.getValue();

            Set<Block> frontiers = df.get(child);

            String s = frontiers == null || frontiers.isEmpty()
                    ? "[-]"
                    : frontiers.stream().map(b -> String.valueOf(b.index())).collect(Collectors.joining(",", "[", "]"));
            dg.node(child, label("" + "\n" + s), Shape.BOX);

            if (child != eb) {
                dg.edge(parent, child);
            }
        }

        dg.end();
    }

    /**
     * Generates a body data dependence dag for a given body.
     *
     * @param body  the body
     * @param names a map of block arguments to names
     * @param w     the writer to write the sr.dot file
     */
    public static void dataDependenceGraph(Body body, Map<Block.Parameter, String> names, Writer w) {
        dataDependenceGraph(body, names, false, w);
    }

    /**
     * Generates a body data dependence graph for a given body.
     *
     * @param body              the body
     * @param names             a map of block arguments to names
     * @param traverseblockArgs true if a graph is produced, otherwise a DAG
     * @param w                 the writer to write the sr.dot file
     */
    public static void dataDependenceGraph(Body body, Map<Block.Parameter, String> names,
                                           boolean traverseblockArgs, Writer w) {
        DotGenerator dg = new DotGenerator(w);

        dg.digraph();

        record Edge(Value from, Value to) {
        }

        Set<Value> visted = new HashSet<>();
        Set<Edge> vistedEdges = new HashSet<>();
        Deque<Value> stack = new ArrayDeque<>(getValues(body));
        while (!stack.isEmpty()) {
            Value v = stack.pop();
            if (!visted.add(v)) {
                continue;
            }

            if (v instanceof Op.Result or) {
                if (!or.op().operands().isEmpty() || !(or.op() instanceof Op.Terminating)) {
                    dg.node(v, label(or.op().opName()), Shape.INVERTED_TRAPEZIUM);
                }
            } else if (v instanceof Block.Parameter ba) {
                String n = names.get(v);
                if (n != null) {
                    dg.node(v, label(n), Shape.INVERTED_TRAPEZIUM,
                            property("style", "filled"));
                } else {
                    Block b = ba.declaringBlock();
                    dg.node(v, label("(" + b.parameters().indexOf(ba) + ")"), Shape.BOX,
                            property("style", "filled"));
                }
            }

            Set<Op.Result> uses = v.uses();
            stack.addAll(uses);
            for (Op.Result use : uses) {
                if (traverseblockArgs && use.op() instanceof Op.Terminating) {
                    for (Block.Reference s : use.op().successors()) {
                        int i = s.arguments().indexOf(v);
                        if (i != -1) {
                            Block.Parameter ba = s.targetBlock().parameters().get(i);

                            if (vistedEdges.add(new Edge(v, ba))) {
                                dg.edge(v, ba);
                            }
                            stack.add(ba);
                        }
                    }
                }

                if (use.op().operands().contains(v)) {
                    if (vistedEdges.add(new Edge(v, use))) {
                        dg.edge(v, use);
                    }
                }
            }
        }

        dg.end();
    }

    static List<Value> getValues(Body r) {
        return r.traverse(new ArrayList<>(), (values, codeElement) -> switch (codeElement) {
            case Block b -> {
                values.addAll(b.parameters());
                yield values;
            }
            case Op o -> {
                values.add(o.result());
                yield values;
            }
            default -> values;
        });
    }

}
