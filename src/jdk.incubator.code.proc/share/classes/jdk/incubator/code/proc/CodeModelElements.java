package jdk.incubator.code.proc;

import com.sun.tools.javac.api.JavacScope;
import com.sun.tools.javac.api.JavacTrees;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.comp.Attr;
import com.sun.tools.javac.model.JavacElements;
import com.sun.tools.javac.processing.JavacProcessingEnvironment;
import com.sun.tools.javac.tree.JCTree.JCMethodDecl;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.util.Context;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.internal.ReflectMethods;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import java.util.Optional;

/**
 * Utility methods for extracting code models from program elements.
 */
public class CodeModelElements {

    final ReflectMethods reflectMethods;
    final Attr attr;
    final JavacElements elements;
    final JavacTrees trees;
    final TreeMaker make;

    CodeModelElements(ProcessingEnvironment processingEnvironment) {
        Context context = ((JavacProcessingEnvironment)processingEnvironment).getContext();
        reflectMethods = ReflectMethods.instance(context);
        attr = Attr.instance(context);
        elements = JavacElements.instance(context);
        trees = JavacTrees.instance(context);
        make = TreeMaker.instance(context);
    }
    /**
     * Returns the code model of provided executable element (if any).
     * <p>
     * If the executable element has a code model then it will be an instance of
     * {@code java.lang.reflect.code.op.CoreOps.FuncOp}.
     * Note: due to circular dependencies we cannot refer to the type explicitly.
     *
     * @implSpec The default implementation unconditionally returns an empty optional.
     * @param e the executable element.
     * @return the code model of the provided executable element (if any).
     */
    public Optional<FuncOp> createCodeModel(ExecutableElement e) {
        if (e.getModifiers().contains(Modifier.ABSTRACT) ||
                e.getModifiers().contains(Modifier.NATIVE)) {
            return Optional.empty();
        }

        try {
            JCMethodDecl methodTree = (JCMethodDecl)elements.getTree(e);
            JavacScope scope = trees.getScope(trees.getPath(e));
            ClassSymbol enclosingClass = (ClassSymbol) scope.getEnclosingClass();
            FuncOp op = attr.runWithAttributedMethod(scope.getEnv(), methodTree,
                    attribBlock -> reflectMethods.getMethodBody(enclosingClass, methodTree, attribBlock, make));
            return Optional.of(op);
        } catch (RuntimeException ex) {  // ReflectMethods.UnsupportedASTException
            // some other error occurred when attempting to attribute the method
            // @@@ better report of error
            ex.printStackTrace();
            return Optional.empty();
        }
    }

    /**
     * {@return a new instance of {@code CodeModelElements} for the provided processing environment}
     * @param processingEnvironment the annotation processing environment
     */
    public static CodeModelElements of(ProcessingEnvironment processingEnvironment) {
        return new CodeModelElements(processingEnvironment);
    }
}
