package jdk.incubator.code.internal;

import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.TreeScanner;

public class TreeScannerPrev extends TreeScanner {

    private JCTree currentNode;
    private JCTree prevNode;

    @Override
    public void scan(JCTree tree) {
        JCTree prevPrevNode = prevNode;
        prevNode = currentNode;
        currentNode = tree;
        try {
            super.scan(tree);
        } finally {
            currentNode = prevNode;
            prevNode = prevPrevNode;
        }
    }

    public void scan(JCTree tree, JCTree prevNode) {
        this.prevNode = null;
        currentNode = prevNode;
        scan(tree);
    }

    JCTree prevNode() {
        return prevNode;
    }
}
