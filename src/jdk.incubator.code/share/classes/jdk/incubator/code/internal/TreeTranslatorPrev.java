package jdk.incubator.code.internal;

import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.TreeTranslator;

public class TreeTranslatorPrev extends TreeTranslator {

    private JCTree currentNode;
    private JCTree prevNode;

    @Override
    public <T extends JCTree> T translate(T tree) {
        JCTree prevPrevNode = prevNode;
        prevNode = currentNode;
        currentNode = tree;
        try {
            return super.translate(tree);
        } finally {
            currentNode = prevNode;
            prevNode = prevPrevNode;
        }
    }

    JCTree currentNode() {
        return currentNode;
    }

    JCTree prevNode() {
        return prevNode;
    }
}
