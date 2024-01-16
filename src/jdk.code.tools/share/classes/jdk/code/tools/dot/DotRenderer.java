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

import jdk.code.tools.renderer.CommonRenderer;

import java.io.Writer;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.util.*;

/**
 * Created by gfrost
 * http://www.graphviz.org/Documentation/dotguide.pdf
 */
public class DotRenderer extends CommonRenderer<DotRenderer> {
    public DotRenderer() {
        super();
    }

    static String sysident(Object o) {
        return Integer.toString(System.identityHashCode(o));
    }

    DotRenderer end() {
        return out().cbrace().nl();
    }

    public DotRenderer start(String name) {
        return append("digraph").space().append(name).obrace().in().nl();
    }

    public DotRenderer rankdir(String s) {
        return append("rankdir").equal().append(s).semicolon().nl();
    }

    public DotRenderer concentrate() {
        return append("concentrate=true").nl();
    }

    public DotRenderer newrank() {
        return append("newrank=true").nl();
    }

    public DotRenderer edgesFirst() {
        return append("outputorder=edgesfirst").nl();
    }


    public <T extends CommonRenderer<T>> DotRenderer graph(NestedRendererSAM<GraphRenderer> nb) {

        nb.build(new GraphRenderer(this)).end();
        return self();
    }


    public static class GraphRenderer extends CommonRenderer<GraphRenderer> {
        public GraphRenderer(DotRenderer dotRenderer) {
            super(dotRenderer);
        }

        GraphRenderer end() {
            return self();
        }

        public GraphRenderer node(String name, String shape, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, shape)).end();
            return self();
        }

        public GraphRenderer node(String name, String shape) {
            return node(name, shape, (n) -> n);
        }

        public GraphRenderer record(String name, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, "record")).end();
            return self();
        }

        public GraphRenderer record(String name) {
            return record(name, (n) -> n);
        }

        public GraphRenderer ellipse(String name, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, "ellipse")).end();
            return self();
        }

        public GraphRenderer ellipse(String name) {
            return ellipse(name, (n) -> n);
        }

        public GraphRenderer ellipse(Object o, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, sysident(o), "ellipse")).end();
            return self();
        }

        public GraphRenderer ellipse(Object o) {
            return ellipse(o, (n) -> n);
        }

        public GraphRenderer circle(String name, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, "circle")).end();
            return self();
        }

        public GraphRenderer circle(String name) {
            return circle(name, (n) -> n);
        }

        public GraphRenderer invertedtrapezium(String name, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, "invtrapezium")).end();
            return self();
        }

        public GraphRenderer invertedtrapezium(String name) {
            return invertedtrapezium(name, (n) -> n);
        }

        public GraphRenderer invertedtrapezium(Object o, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, sysident(o), "invtrapezium")).end();
            return self();
        }

        public GraphRenderer invertedtrapezium(Object o) {
            return invertedtrapezium(o, (n) -> n);
        }

        public GraphRenderer box(String name, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, "box")).end();
            return self();
        }

        public GraphRenderer box(String name) {
            return box(name, (n) -> n);
        }

        public GraphRenderer box(Object o, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, sysident(o), "box")).end();
            return self();
        }

        public GraphRenderer box(Object o) {
            return box(o, (n) -> n);
        }

        public GraphRenderer hexagon(String name, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, name, "hexagon")).end();
            return self();
        }

        public GraphRenderer hexagon(String name) {
            return hexagon(name, (n) -> n);
        }

        public GraphRenderer hexagon(Object o, NestedRendererSAM<NodeRenderer> sam) {
            sam.build(new NodeRenderer(this, sysident(o), "hexagon")).end();
            return self();
        }

        public GraphRenderer hexagon(Object o) {
            return hexagon(o, (n) -> n);
        }

        public static class NodeRenderer extends CommonRenderer<NodeRenderer> {
            NodeRenderer(GraphRenderer graphRenderer, String name, String shape) {
                super(graphRenderer);
                append(name).osbrace().append("shape").equal().oquot().append(shape).cquot().space();
                // append(name).osbrace().append("shape").equal().append(shape).space();
                //   append(name).osbrace();//.append("shape").equal().append(shape).space();
            }

            public NodeRenderer end() {
                return csbrace().semicolon().nl();
            }

            NodeRenderer label(String label, NestedRendererSAM<LabelRenderer> sam) {
                LabelRenderer renderer = new LabelRenderer(this, label);
                sam.build(renderer).end();
                return self();
            }

            NodeRenderer label(NestedRendererSAM<LabelRenderer> sam) {
                LabelRenderer renderer = new LabelRenderer(this, "");
                sam.build(renderer).end();
                return self();
            }

            NodeRenderer label(String label) {
                return label(label, (l) -> l);
            }


            public NodeRenderer color(String color) {
                return append("color").equal().oquot().append(color).cquot().space();
            }

            public NodeRenderer style(String style) {
                return append("style").equal().oquot().append(style).cquot().space();
            }

            public static class LabelRenderer extends CommonRenderer<LabelRenderer> {
                int count = 0;

                LabelRenderer(NodeRenderer nodeRenderer, String label) {
                    super(nodeRenderer);
                    append("label").equal().oquot().append(label);
                }

                public LabelRenderer end() {
                    return cquot().space();
                }

                LabelRenderer port(String label, String text) {
                    if (count > 0) {
                        pipe();
                    }
                    count++;
                    return lt().append(label).gt().append(text);
                }

                LabelRenderer label(String label, String text) {
                    if (count > 0) {
                        pipe();
                    }
                    count++;
                    return append(text);
                }

                LabelRenderer box(NestedRendererSAM<BoxRenderer> sam) {
                    sam.build(new BoxRenderer(this)).end();
                    count = 0;
                    return self();
                }

                static class BoxRenderer extends CommonRenderer<BoxRenderer> {
                    int count = 0;

                    BoxRenderer(LabelRenderer labelRenderer) {
                        super(labelRenderer);
                        pipe().obrace();
                    }

                    BoxRenderer(BoxRenderer boxRenderer) {
                        super(boxRenderer);
                        pipe().obrace();
                    }

                    BoxRenderer end() {
                        return cbrace().pipe();
                    }

                    BoxRenderer port(String label, String text) {
                        if (count > 0) {
                            pipe();
                        }
                        count++;
                        return lt().append(label).gt().append(text);
                    }

                    BoxRenderer label(String text) {
                        if (count > 0) {
                            pipe();
                        }
                        count++;
                        return append(text);
                    }

                    BoxRenderer box(NestedRendererSAM<BoxRenderer> sam) {
                        sam.build(new BoxRenderer(this)).end();
                        count = 0;
                        return self();
                    }
                }
            }
        }

        public GraphRenderer edge(String from, String to, NestedRendererSAM<EdgeRenderer> sam) {
            sam.build(new EdgeRenderer(this, from, to)).end();
            return self();
        }

        public GraphRenderer edge(String from, String to) {
            return edge(from, to, (n) -> n);
        }

        public GraphRenderer edge(Object from, Object to, NestedRendererSAM<EdgeRenderer> sam) {
            sam.build(new EdgeRenderer(this, sysident(from), sysident(to))).end();
            return self();
        }

        public GraphRenderer edge(Object from, Object to) {
            return edge(from, to, (n) -> n);
        }

        public static class EdgeRenderer extends CommonRenderer<EdgeRenderer> {
            EdgeRenderer(GraphRenderer graphRenderer, String from, String to) {
                super(graphRenderer);
                append(from).rarrow().append(to).osbrace();
            }

            public EdgeRenderer end() {
                return csbrace().semicolon().nl().self();
            }

            EdgeRenderer label(String label, NestedRendererSAM<LabelRenderer> sam) {
                LabelRenderer renderer = new LabelRenderer(this, label);
                sam.build(renderer).end();
                return self();
            }


            EdgeRenderer label(String label) {
                return label(label, (l) -> l);
            }

            public static class LabelRenderer extends CommonRenderer<LabelRenderer> {

                LabelRenderer(EdgeRenderer edgeRenderer, String label) {
                    super(edgeRenderer);
                    append("label").equal().oquot().append(label);
                }

                public LabelRenderer end() {
                    return cquot().space();
                }

            }
        }
    }

    public static void representationTree(Op op, Writer w) {
        new DotRenderer().writer(w).start("g").graph((g) -> {
            op.traverse(null, (t, codeElement) -> switch (codeElement) {
                case Body b -> {
                    g.hexagon(b, (n) -> n.label("").style("filled"));
                    g.edge(b.parentOp(), b);
                    yield null;
                }
                case Block b -> {
                    g.box(b, (n) -> n.label(""));
                    g.edge(b.parentBody(), b);
                    yield null;
                }
                case Op o -> {
                    if (o instanceof Op.Terminating) {
                        g.ellipse(o, (n) -> n.label(o.opName()).style("filled"));
                    } else {
                        g.ellipse(o, (n) -> n.label(o.opName()));
                    }
                    if (o.parentBlock() != null) {
                        g.edge(o.parentBlock(), o);
                    }
                    yield null;
                }
            });
            return g;
        }).end();

    }

    public static void bodyGraph(Body body, Writer w) {
        Block eb = body.entryBlock();
        Deque<Block> stack = new ArrayDeque<>();
        Set<Block> visited = new HashSet<>();
        stack.push(eb);
        new DotRenderer().writer(w).start("g").graph((g) -> {
            while (!stack.isEmpty()) {
                Block b = stack.pop();
                if (!visited.add(b)) {
                    continue;
                }

                g.box(b, (box) -> box.label(""));

                List<Block.Reference> successors = b.terminatingOp().successors();
                for (Block.Reference s : successors) {
                    g.edge(b, s.targetBlock());

                    stack.push(s.targetBlock());
                }
            }
            return g;
        }).end();

    }

    public static void bodyDominatorTree(Body body, Writer w) {
        Block eb = body.entryBlock();
        Map<Block, Block> idoms = body.immediateDominators();

        new DotRenderer().writer(w).start("g").graph((g) -> {
            for (Map.Entry<Block, Block> e : idoms.entrySet()) {
                Block child = e.getKey();
                Block parent = e.getValue();

                g.box(child, (b) -> b.label(""));

                if (child != eb) {
                    g.edge(parent, child);
                }
            }
            return g;
        }).end();
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


        record Edge(Value from, Value to) {
        }
        new DotRenderer().writer(w).start("SR").graph((g) -> {
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
                        g.invertedtrapezium(v, (node) -> node.label(or.op().opName()));
                    }
                } else if (v instanceof Block.Parameter ba) {
                    String n = names.get(v);
                    if (n != null) {
                        g.invertedtrapezium(v, (node) -> node.label(n).style("filled"));
                    } else {
                        Block b = ba.declaringBlock();

                        g.box(v, (node) -> node.label("(" + b.parameters().indexOf(ba) + ")").style("filled"));
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
                                    g.edge(v, ba);
                                }
                                stack.add(ba);
                            }
                        }
                    }

                    if (use.op().operands().contains(v)) {
                        if (vistedEdges.add(new Edge(v, use))) {
                            g.edge(v, use);
                        }
                    }
                }
            }
            return g;
        }).end();
    }

    private static List<Value> getValues(Body r) {
        return r.traverse(new ArrayList<>(), (values, codeElement) -> switch (codeElement) {
            case Block b -> {
                values.addAll(b.parameters());
                yield values;
            }
            case Op op -> {
                values.add(op.result());
                yield values;
            }
            default -> values;
        });
    }
}