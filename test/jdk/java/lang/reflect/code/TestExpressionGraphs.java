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

/*
 * @test
 * @run testng TestExpressionGraphs
 */

import org.testng.annotations.Test;

import java.io.Writer;
import java.lang.reflect.Method;
import java.lang.reflect.code.*;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.writer.OpWriter;
import java.lang.runtime.CodeReflection;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class TestExpressionGraphs {

    @CodeReflection
    static double sub(double a, double b) {
        return a - b;
    }

    @CodeReflection
    static double distance1(double a, double b) {
        return Math.abs(a - b);
    }

    @CodeReflection
    static double distance1a(final double a, final double b) {
        final double diff = a - b;
        final double result = Math.abs(diff);
        return result;
    }

    @CodeReflection
    static double distance1b(final double a, final double b) {
        final double diff = a - b;
        // Note, incorrect for negative zero values
        final double result = diff < 0d ? -diff : diff;
        return result;
    }

    @CodeReflection
    static double distanceN(double[] a, double[] b) {
        double sum = 0d;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2d);
        }
        return Math.sqrt(sum);
    }

    @CodeReflection
    static double squareDiff(double a, double b) {
        // a^2 - b^2 = (a + b) * (a - b)
        final double plus = a + b;
        final double minus = a - b;
        return plus * minus;
    }

    @Test
    void traverseSub() throws ReflectiveOperationException {
        // Get the reflective object for method sub
        Method m = TestExpressionGraphs.class.getDeclaredMethod(
                "sub", double.class, double.class);
        // Get the code model for method sub
        Optional<CoreOp.FuncOp> oModel = m.getCodeModel();
        CoreOp.FuncOp model = oModel.orElseThrow();

        // Depth-first search, reporting elements in pre-order
        model.traverse(null, (acc, codeElement) -> {
            // Count the depth of the code element by
            // traversing up the tree from child to parent
            int depth = 0;
            CodeElement<?, ?> parent = codeElement;
            while ((parent = parent.parent()) != null) depth++;
            // Print out code element class
            System.out.println("  ".repeat(depth) + codeElement.getClass());
            return acc;
        });

        // Stream of elements topologically sorted in depth-first search pre-order
        model.elements().forEach(codeElement -> {
            // Count the depth of the code element
            int depth = 0;
            CodeElement<?, ?> parent = codeElement;
            while ((parent = parent.parent()) != null) depth++;
            // Print out code element class
            System.out.println("  ".repeat(depth) + codeElement.getClass());
        });
    }

    @Test
    void traverseDistance1() throws ReflectiveOperationException {
        // Get the reflective object for method distance1
        Method m = TestExpressionGraphs.class.getDeclaredMethod(
                "distance1", double.class, double.class);
        // Get the code model for method distance1
        Optional<CoreOp.FuncOp> oModel = m.getCodeModel();
        CoreOp.FuncOp model = oModel.orElseThrow();

        // Depth-first search, reporting elements in pre-order
        model.traverse(null, (acc, codeElement) -> {
            // Count the depth of the code element by
            // traversing up the tree from child to parent
            int depth = 0;
            CodeElement<?, ?> parent = codeElement;
            while ((parent = parent.parent()) != null) depth++;
            // Print out code element class
            System.out.println("  ".repeat(depth) + codeElement.getClass());
            return acc;
        });

        // Stream of elements topologically sorted in depth-first search pre-order
        model.elements().forEach(codeElement -> {
            // Count the depth of the code element
            int depth = 0;
            CodeElement<?, ?> parent = codeElement;
            while ((parent = parent.parent()) != null) depth++;
            // Print out code element class
            System.out.println("  ".repeat(depth) + codeElement.getClass());
        });
    }


    @Test
    void printSub() {
        CoreOp.FuncOp model = getFuncOp("sub");
        print(model);
    }

    @Test
    void printDistance1() {
        CoreOp.FuncOp model = getFuncOp("distance1");
        print(model);
    }

    @Test
    void printDistance1a() {
        CoreOp.FuncOp model = getFuncOp("distance1a");
        print(model);
    }

    @Test
    void printDistance1b() {
        CoreOp.FuncOp model = getFuncOp("distance1b");
        print(model);
    }

    @Test
    void printDistanceN() {
        CoreOp.FuncOp model = getFuncOp("distanceN");
        print(model);
    }

    void print(CoreOp.FuncOp f) {
        System.out.println(f.toText());

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(f.toText());

        f = SSA.transform(f);
        System.out.println(f.toText());
    }


    @Test
    void graphsDistance1() {
        CoreOp.FuncOp model = getFuncOp("distance1");
        Function<CodeItem, String> names = names(model);
        System.out.println(printOpWriteVoid(names, model));

        // Create the expression graph for the terminating operation result
        Op.Result returnResult = model.body().entryBlock().terminatingOp().result();
        Node<Value> returnGraph = expressionGraph(returnResult);
        System.out.println("Expression graph for terminating operation result");
        // Transform from Node<Value> to Node<String> and print the graph
        System.out.println(returnGraph.transformValues(v -> printValue(names, v)));

        System.out.println("Use graphs for block parameters");
        for (Block.Parameter parameter : model.parameters()) {
            Node<Value> useNode = useGraph(parameter);
            System.out.println(useNode.transformValues(v -> printValue(names, v)));
        }

        // Create the expression graphs for all values
        Map<Value, Node<Value>> graphs = expressionGraphs(model);
        System.out.println("Expression graphs for all declared values in the model");
        graphs.values().forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });

        // The graphs for the terminating operation result are the same
        assert returnGraph.equals(graphs.get(returnGraph.value()));

        // Filter for root graphs, operation results with no uses
        List<Node<Value>> rootGraphs = graphs.values().stream()
                .filter(n -> n.value() instanceof Op.Result opr &&
                        switch (opr.op()) {
                            // An operation result with no uses
                            default -> opr.uses().isEmpty();
                        })
                .toList();
        System.out.println("Root expression graphs");
        rootGraphs.forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });
    }

    @Test
    void graphsDistance1a() {
        CoreOp.FuncOp f = getFuncOp("distance1a");
        Function<CodeItem, String> names = names(f);
        System.out.println(printOpWriteVoid(names, f));

        {
            Map<Value, Node<Value>> graphs = expressionGraphs(f);
            List<Node<Value>> rootGraphs = graphs.values().stream()
                    .filter(n -> n.value() instanceof Op.Result opr &&
                            switch (opr.op()) {
                                // An operation result with no uses
                                default -> opr.uses().isEmpty();
                            })
                    .toList();
            System.out.println("Root expression graphs");
            rootGraphs.forEach(n -> {
                System.out.println(n.transformValues(v -> printValue(names, v)));
            });
        }

        {
            Map<Value, Node<Value>> graphs = expressionGraphs(f);
            List<Node<Value>> rootGraphs = graphs.values().stream()
                    .filter(n -> n.value() instanceof Op.Result opr &&
                            switch (opr.op()) {
                                // Variable declarations modeling local variables
                                case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                                // An operation result with no uses
                                default -> opr.uses().isEmpty();
                            })
                    .toList();
            System.out.println("Root (with variable) expression graphs");
            rootGraphs.forEach(n -> {
                System.out.println(n.transformValues(v -> printValue(names, v)));
            });
        }

        Map<Value, Node<Value>> prunedGraphs = prunedExpressionGraphs(f);
        List<Node<Value>> prunedRootGraphs = prunedGraphs.values().stream()
                .filter(n -> n.value() instanceof Op.Result opr &&
                        switch (opr.op()) {
                            // Variable declarations modeling local variables
                            case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                            // An operation result with no uses
                            default -> opr.uses().isEmpty();
                        })
                .toList();
        System.out.println("Pruned root expression graphs");
        prunedRootGraphs.forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });
    }

    @Test
    void graphsDistance1b() {
        CoreOp.FuncOp f = getFuncOp("distance1b");
        Function<CodeItem, String> names = names(f);
        System.out.println(printOpWriteVoid(names, f));

        Map<Value, Node<Value>> prunedGraphs = prunedExpressionGraphs(f);
        List<Node<Value>> prunedRootGraphs = prunedGraphs.values().stream()
                .filter(n -> n.value() instanceof Op.Result opr &&
                        switch (opr.op()) {
                            // Variable declarations modeling declaration of local variables
                            case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                            // An operation result with no uses
                            default -> opr.uses().isEmpty();
                        })
                .toList();
        System.out.println("Pruned root expression graphs");
        prunedRootGraphs.forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });
    }

    @Test
    void graphsDistanceN() {
        CoreOp.FuncOp f = getFuncOp("distanceN");
        Function<CodeItem, String> names = names(f);
        System.out.println(printOpWriteVoid(names, f));

        Map<Value, Node<Value>> prunedGraphs = prunedExpressionGraphs(f);
        List<Node<Value>> prunedRootGraphs = prunedGraphs.values().stream()
                .filter(n -> n.value() instanceof Op.Result opr &&
                        switch (opr.op()) {
                            // Variable declarations modeling declaration of local variables
                            case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                            // An operation result with no uses
                            default -> opr.uses().isEmpty();
                        })
                .toList();
        System.out.println("Pruned root expression graphs");
        prunedRootGraphs.forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });
    }

    @Test
    void graphsSquareDiff() {
        CoreOp.FuncOp f = getFuncOp("squareDiff");
        Function<CodeItem, String> names = names(f);
        System.out.println(printOpWriteVoid(names, f));

        {
            Map<Value, Node<Value>> graphs = expressionGraphs(f);
            List<Node<Value>> rootGraphs = graphs.values().stream()
                    .filter(n -> n.value() instanceof Op.Result opr &&
                            switch (opr.op()) {
                                // An operation result with no uses
                                default -> opr.uses().isEmpty();
                            })
                    .toList();
            System.out.println("Root expression graphs");
            rootGraphs.forEach(n -> {
                System.out.println(n.transformValues(v -> printValue(names, v)));
            });
        }

        {
            Map<Value, Node<Value>> graphs = expressionGraphs(f);
            List<Node<Value>> rootGraphs = graphs.values().stream()
                    .filter(n -> n.value() instanceof Op.Result opr &&
                            switch (opr.op()) {
                                // Variable declarations modeling local variables
                                case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                                // An operation result with no uses
                                default -> opr.uses().isEmpty();
                            })
                    .toList();
            System.out.println("Root (with variable) expression graphs");
            rootGraphs.forEach(n -> {
                System.out.println(n.transformValues(v -> printValue(names, v)));
            });
        }

        Map<Value, Node<Value>> prunedGraphs = prunedExpressionGraphs(f);
        List<Node<Value>> prunedRootGraphs = prunedGraphs.values().stream()
                .filter(n -> n.value() instanceof Op.Result opr &&
                        switch (opr.op()) {
                            // Variable declarations modeling local variables
                            case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                            // An operation result with no uses
                            default -> opr.uses().isEmpty();
                        })
                .toList();
        System.out.println("Pruned root expression graphs");
        prunedRootGraphs.forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });
    }



    @CodeReflection
    static int h(int x) {
        x += 2;                                                 // Statement 1
        g(x);                                                   // Statement 2
        int y = 1 + g(x) + (x += 2) + (x > 2 ? x : 10);         // Statement 3
        for (                                                   // Statement 4
                int i = 0, j = 1;                               // Statements 4.1
                i < 3 && j < 3;
                i++, j++) {                                     // Statements 4.2
            System.out.println(i);                              // Statement 4.3
        }
        return x + y;                                           // Statement 5
    }

    static int g(int i) {
        return i;
    }

    @Test
    void graphsH() {
        CoreOp.FuncOp f = getFuncOp("h");
        Function<CodeItem, String> names = names(f);
        System.out.println(printOpWriteVoid(names, f));

        Map<Value, Node<Value>> graphs = prunedExpressionGraphs(f);
        List<Node<Value>> rootGraphs = graphs.values().stream()
                .filter(n -> n.value() instanceof Op.Result opr &&
                        switch (opr.op()) {
                            // Variable declarations modeling declaration of local variables
                            case CoreOp.VarOp vop -> vop.operands().get(0) instanceof Op.Result;
                            // Variable stores modeling assignment expressions whose result is used
                            case CoreOp.VarAccessOp.VarStoreOp vsop -> vsop.operands().get(1).uses().size() == 1;
                            // An operation result with no uses
                            default -> opr.uses().isEmpty();
                        })
                .toList();
        rootGraphs.forEach(n -> {
            System.out.println(n.transformValues(v -> printValue(names, v)));
        });
    }


    static String printValue(Function<CodeItem, String> names, Value v) {
        if (v instanceof Op.Result opr) {
            return printOpHeader(names, opr.op());
        } else {
            return "%" + names.apply(v) + " <block parameter>";
        }
    }

    static String printOpHeader(Function<CodeItem, String> names, Op op) {
        return OpWriter.toText(op,
                OpWriter.OpDescendantsOption.DROP_DESCENDANTS,
                OpWriter.VoidOpResultOption.WRITE_VOID,
                OpWriter.CodeItemNamerOption.of(names));
    }

    static String printOpWriteVoid(Function<CodeItem, String> names, Op op) {
        return OpWriter.toText(op,
                OpWriter.VoidOpResultOption.WRITE_VOID,
                OpWriter.CodeItemNamerOption.of(names));
    }

    static Function<CodeItem, String> names(Op op) {
        OpWriter w = new OpWriter(Writer.nullWriter(),
                OpWriter.VoidOpResultOption.WRITE_VOID);
        w.writeOp(op);
        return w.namer();
    }


    static Node<Value> expressionGraph(Value value) {
        return expressionGraph(new HashMap<>(), value);
    }

    static Node<Value> expressionGraph(Map<Value, Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        // Find the expression graphs for each operand
        List<Node<Value>> edges = new ArrayList<>();
        for (Value operand : value.dependsOn()) {
            edges.add(expressionGraph(operand));
        }
        Node<Value> node = new Node<>(value, edges);
        visited.put(value, node);
        return node;
    }

    static Node<Value> expressionGraphDetailed(Map<Value, Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        List<Node<Value>> edges;
        if (value instanceof Op.Result result) {
            edges = new ArrayList<>();
            // Find the expression graphs for each operand
            Set<Value> valueVisited = new HashSet<>();
            for (Value operand : result.op().operands()) {
                // Ensure an operand is visited only once
                if (valueVisited.add(operand)) {
                    edges.add(expressionGraph(operand));
                }
            }
            // TODO if terminating operation find expression graphs
            //      for each successor argument
        } else {
            assert value instanceof Block.Parameter;
            // A block parameter has no outgoing edges
            edges = List.of();
        }
        Node<Value> node = new Node<>(value, edges);
        visited.put(value, node);
        return node;
    }


    static Node<Value> useGraph(Value value) {
        return useGraph(new HashMap<>(), value);
    }

    static Node<Value> useGraph(Map<Value, Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        // Find the use graphs for each use
        List<Node<Value>> edges = new ArrayList<>();
        for (Op.Result use : value.uses()) {
            edges.add(useGraph(visited, use));
        }
        Node<Value> node = new Node<>(value, edges);
        visited.put(value, node);
        return node;
    }

    static Map<Value, Node<Value>> expressionGraphs(CoreOp.FuncOp f) {
        return expressionGraphs(f.body());
    }

    static Map<Value, Node<Value>> expressionGraphs(Body b) {
        // Traverse the model building structurally shared expression graphs
        return b.traverse(new LinkedHashMap<>(), (graphs, codeElement) -> {
            switch (codeElement) {
                case Body _ -> {
                    // Do nothing
                }
                case Block block -> {
                    // Create the expression graphs for each block parameter
                    // A block parameter has no outgoing edges
                    for (Block.Parameter parameter : block.parameters()) {
                        graphs.put(parameter, new Node<>(parameter, List.of()));
                    }
                }
                case Op op -> {
                    // Find the expression graphs for each operand
                    List<Node<Value>> edges = new ArrayList<>();
                    for (Value operand : op.result().dependsOn()) {
                        // Get expression graph for the operand
                        // It must be previously computed since we encounter the
                        // declaration of values before their use
                        edges.add(graphs.get(operand));
                    }
                    // Create the expression graph for this operation result
                    graphs.put(op.result(), new Node<>(op.result(), edges));
                }
            }
            return graphs;
        });
    }

    static Map<Value, Node<Value>> prunedExpressionGraphs(CoreOp.FuncOp f) {
        return prunedExpressionGraphs(f.body());
    }

    static Map<Value, Node<Value>> prunedExpressionGraphs(Body b) {
        // Traverse the model building structurally shared expression graphs
        return b.traverse(new LinkedHashMap<>(), (graphs, codeElement) -> {
            switch (codeElement) {
                case Body _ -> {
                    // Do nothing
                }
                case Block block -> {
                    // Create the expression graphs for each block parameter
                    // A block parameter has no outgoing edges
                    for (Block.Parameter parameter : block.parameters()) {
                        graphs.put(parameter, new Node<>(parameter, List.of()));
                    }
                }
                // Prune graph for variable load operation
                case CoreOp.VarAccessOp.VarLoadOp op -> {
                    // Ignore edge for the variable value operand
                    graphs.put(op.result(), new Node<>(op.result(), List.of()));
                }
                // Prune graph for variable store operation
                case CoreOp.VarAccessOp.VarStoreOp op -> {
                    // Ignore edge for the variable value operand
                    // Add edge for value to store
                    List<Node<Value>> edges = List.of(graphs.get(op.operands().get(1)));
                    graphs.put(op.result(), new Node<>(op.result(), edges));
                }
                case Op op -> {
                    // Find the expression graphs for each operand
                    List<Node<Value>> edges = new ArrayList<>();
                    for (Value operand : op.result().dependsOn()) {
                        // Get expression graph for the operand
                        // It must be previously computed since we encounter the
                        // declaration of values before their use
                        edges.add(graphs.get(operand));
                    }
                    // Create the expression graph for this operation result
                    graphs.put(op.result(), new Node<>(op.result(), edges));
                }
            }
            return graphs;
        });
    }

    record Node<T>(T value, List<Node<T>> edges) {
        <U> Node<U> transformValues(Function<T, U> f) {
            List<Node<U>> transformedEdges = new ArrayList<>();
            for (Node<T> edge : edges()) {
                transformedEdges.add(edge.transformValues(f));
            }
            return new Node<>(f.apply(value()), transformedEdges);
        }

        Node<T> transformGraph(Function<Node<T>, Node<T>> f) {
            Node<T> apply = f.apply(this);
            if (apply != this) {
                // The function returned a new node
                return apply;
            } else {
                // The function returned the same node
                // Apply the transformation to the children
                List<Node<T>> transformedEdges = new ArrayList<>();
                for (Node<T> edge : edges()) {
                    transformedEdges.add(edge.transformGraph(f));
                }
                boolean same = IntStream.range(0, edges().size())
                        .allMatch(i -> edges().get(i) ==
                                transformedEdges.get(i));
                if (same) {
                    return this;
                } else {
                    return new Node<>(this.value(), transformedEdges);
                }
            }
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            print(sb, "", "");
            return sb.toString();
        }

        private void print(StringBuilder sb, String prefix, String edgePrefix) {
            sb.append(prefix);
            sb.append(value);
            sb.append('\n');
            for (Iterator<Node<T>> it = edges.iterator(); it.hasNext(); ) {
                Node<T> edge = it.next();
                if (it.hasNext()) {
                    edge.print(sb, edgePrefix + "├── ", edgePrefix + "│   ");
                } else {
                    edge.print(sb, edgePrefix + "└── ", edgePrefix + "    ");
                }
            }
        }
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestExpressionGraphs.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

}
