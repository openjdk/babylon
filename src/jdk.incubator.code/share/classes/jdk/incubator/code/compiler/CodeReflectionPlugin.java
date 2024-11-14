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

package jdk.incubator.code.compiler;

import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.Plugin;
import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskEvent.Kind;
import com.sun.tools.javac.api.BasicJavacTask;
import com.sun.tools.javac.code.Source;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.tree.TreeScanner;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import jdk.incubator.code.internal.ReflectMethods;

import javax.lang.model.element.TypeElement;
import java.util.HashMap;
import java.util.Map;

/**
 * A compiler plugin that processes methods annotated with the {@link jdk.incubator.codeReflection}
 * annotation, and saves their code model in the resulting AST.
 */
public class CodeReflectionPlugin implements Plugin {

    Context context;
    TreeMaker treeMaker;
    Runnable dropListener;

    /**
     * Plugin constructor
     */
    public CodeReflectionPlugin() { }

    @Override
    public String getName() {
        return "CodeReflection Plugin";
    }

    @Override
    public void init(JavacTask task, String... args) {
        this.context = ((BasicJavacTask)task).getContext();
        TaskListener taskListener = new TaskListener();
        task.addTaskListener(taskListener);
        dropListener = () -> task.removeTaskListener(taskListener);
    }

    @Override
    public boolean autoStart() {
        return true;
    }

    class TaskListener implements com.sun.source.util.TaskListener {
        @Override
        public void started(TaskEvent e) {
            if (e.getKind() == Kind.ENTER) {
                if (dropListener != null && !isCodeReflectionAvailable(context)) {
                    // do not process further events if code reflection module is not enabled
                    dropListener.run();
                    dropListener = null;
                }
            }
        }

        @Override
        public void finished(TaskEvent e) {
            if (e.getKind() == Kind.ANALYZE) {
                JCCompilationUnit jcCompilationUnit = (JCCompilationUnit)e.getCompilationUnit();
                if (Log.instance(context).nerrors == 0) {
                    treeMaker = TreeMaker.instance(context);
                    TreeMaker localMake = treeMaker.forToplevel(jcCompilationUnit);
                    ClassDeclFinder classDeclFinder = new ClassDeclFinder(e.getTypeElement());
                    classDeclFinder.scan(jcCompilationUnit);
                    ReflectMethods.instance(context)
                                  .translateTopLevelClass(classDeclFinder.classDecl, localMake);
                }
            }
        }
    }

    public static boolean isCodeReflectionAvailable(Context context) {
        Source source = Source.instance(context);
        if (!Source.Feature.REFLECT_METHODS.allowedInSource(source)) {
            // if source level is not latest, return false
            return false;
        }

        // if jdk.incubator.code is not in the module graph, skip
        return JavaCompiler.instance(context).hasCodeReflectionModule();
    }

    // A simple tree scanner that finds a class declaration tree given its type element.
    static class ClassDeclFinder extends TreeScanner {

        JCClassDecl classDecl;
        final TypeElement element;

        public ClassDeclFinder(TypeElement element) {
            this.element = element;
        }

        @Override
        public void visitClassDef(JCClassDecl tree) {
            if (tree.sym == element) {
                classDecl = tree;
            } else {
                super.visitClassDef(tree);
            }
        }
    }
}
