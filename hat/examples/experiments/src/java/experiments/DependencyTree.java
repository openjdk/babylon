package experiments;

import hat.optools.RootSet;

import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

public class DependencyTree {

/*

    record NodePrev<T extends Value>(T node, List<Node<T>> children) {
        <U extends Value> Node<U> transform(Function<T, U> f) {
            List<Node<U>> children = new ArrayList<>();
            for (Node<T> child : children()) {
                children.add(child.transform(f));
            }
            return new Node<>(f.apply(node()), children);
        }
    }

    static Map<Op, Node<Value>> dependencyTrees(CoreOps.FuncOp f) {
        Map<Op, Node<Value>> trees = new LinkedHashMap<>();
        Map<Value, Node<Value>> params = new HashMap<>();
        f.body().traverse(null, (_, ce) -> {
            if (ce instanceof Op op) {
                List<Node<Value>> children = new ArrayList<>();
                for (Value operand : op.operands()) {
                    if (operand instanceof Op.Result opr) {
                        children.add(trees.get(opr.op()));
                    } else {
                        // Block parameter
                        children.add(params.computeIfAbsent(operand, _ -> new Node<>(operand, List.of())));
                    }
                }
                trees.put(op, new Node<>(op.result(), children));
            }
            return null;
        });
        return trees;
    }
*/

    /*
    static void printDependencyTrees(CoreOps.FuncOp f) {
        Map<Op, Node<Value>> trees = dependencyTrees(f);
        Map<CodeItem, String> names = OpWriter.computeGlobalNames(f);
        trees.forEach((op, valueNode) -> {
            if (op instanceof CoreOps.VarAccessOp.VarStoreOp) {
                Value value = op.operands().get(1);
                if (value.uses().size() > 1) {
                    System.out.println("Expression store: " + op);
                } else {
                    System.out.println("Root: " + op);
                }
                System.out.println("      " + valueNode.transform(names::get));
            }
            else if (op instanceof CoreOps.VarOp || op.result().uses().isEmpty()) {
                System.out.println("Root: " + op);
                System.out.println("      " + valueNode.transform(names::get));
            }
        });
    } */
/*
    static Set<Op> rootSet(CoreOps.FuncOp f) {
        Set<Op> roots = new LinkedHashSet<>();
        Map<Op, Node<Value>> trees = dependencyTrees(f);
     //   Map<CodeItem, String> names = OpWriter.computeGlobalNames(f);
        trees.forEach((op, valueNode) -> {
            if (op instanceof CoreOps.VarAccessOp.VarStoreOp) {
                Value value = op.operands().get(1);
                if (value.uses().size() > 1) {
                    //System.out.println("Expression store: " + op);
                } else {
                    roots.add(op);
                }
               // System.out.println("      " + valueNode.transform(names::get));
            }
            else if (op instanceof CoreOps.VarOp || op.result().uses().isEmpty()) {
                roots.add(op);
               // System.out.println("Root: " + op);
               // System.out.println("      " + valueNode.transform(names::get));
            }
        });
        return roots;
    }
*/

    /*
        static void printDependencyTree(CoreOps.FuncOp f) {
            Map<CodeItem, String> names = OpWriter.computeGlobalNames(f);

            f.traverse(null, (o, ce) -> {
                if (ce instanceof CoreOps.VarOp vop) {
                    Node<String> tree = dependencyTree(vop.result()).transform(names::get);
                    System.out.printf("Var def %s depends on %s\n",
                            vop.varName(), tree);
                } else if (ce instanceof CoreOps.VarAccessOp.VarStoreOp vsop) {
                    Node<String> tree = dependencyTree(vsop.result()).transform(names::get);
                    System.out.printf("Var store to %s depends on %s\n",
                            vsop.varOp().varName(), tree);
                } else if (ce instanceof CoreOps.ReturnOp rop) {
                    Node<String> tree = dependencyTree(rop.result()).transform(names::get);
                    System.out.printf("Return depends on %s\n",
                            tree);
                }
                return null;
            });
        }
    */
    static int g(int i) {
        return i;
    }

    @CodeReflection
    static void f() {
        int x = 0;
        x = 1;
        g(x);
        int y = g(x) + (x = 2);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(DependencyTree.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }


    public static void main(String[] args) {
        CoreOp.FuncOp f = getFuncOp("f");
        System.out.println(f.toText());

        Set<Op> roots = RootSet.getRootSet(f.body().entryBlock().ops().stream());
        f.body().entryBlock().ops().stream().filter(roots::contains).forEach(op -> {
            System.out.print(op.toText());
        });


    }

}
