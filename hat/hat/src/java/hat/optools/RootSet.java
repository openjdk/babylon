package hat.optools;

import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

//Recursive
/*
static Node<Value> dependencyTree(Value value) {
    // @@@ There should exactly one Node in the tree for a given Value
    List<Node<Value>> children = new ArrayList<>();
    for (Value dependencyOnValue : value.dependsOn()) {
        Node<Value> child;
        if (dependencyOnValue instanceof Op.Result or && or.op() instanceof CoreOps.VarAccessOp.VarLoadOp) {
            // Break the tree at a var load
            child = new Node<>(dependencyOnValue, List.of());
        } else {
            // Traverse backwards
            child = dependencyTree(dependencyOnValue); // recurses
        }
        children.add(child);
    }
    return new Node<>(value, children);
} */
public class RootSet {
    record Node<T extends Value>(T node, List<Node<T>> children) {
    }

    public static Set<Op> getRootSet(Stream<Op> ops) {
        Set<Op> roots = new LinkedHashSet<>();
        Map<Op, Node<Value>> trees = new LinkedHashMap<>();
        Map<Value, Node<Value>> params = new HashMap<>();
        ops.forEach(op -> {
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
        });

        trees.forEach((op, valueNode) -> {
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
