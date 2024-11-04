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

package jdk.incubator.code;

import com.sun.source.util.JavacTask;
import com.sun.source.util.Plugin;
import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskEvent.Kind;
import com.sun.tools.javac.api.BasicJavacTask;
import com.sun.tools.javac.api.JavacTaskImpl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.util.Context;
import jdk.incubator.code.internal.ReflectMethods;

/**
 * A vector test plugin
 */
public class CodeReflectionPlugin implements Plugin {

    Context context;

    /**
     * Plugin constructor
     */
    public CodeReflectionPlugin() {

    }

    @Override
    public String getName() {
        return "CodeReflection Plugin";
    }

    @Override
    public void init(JavacTask task, String... args) {
        System.out.println("Hello from " + getName());
        context = ((BasicJavacTask)task).getContext();
        task.addTaskListener(new TaskListener());
    }

    @Override
    public boolean autoStart() {
        return true;
    }

    class TaskListener implements com.sun.source.util.TaskListener {
        @Override
        public void started(TaskEvent e) {
            // do nothing
        }

        @Override
        public void finished(TaskEvent e) {
            if (e.getKind() == Kind.ANALYZE) {
                // end of attribution/flow analysis
                JCCompilationUnit jcCompilationUnit = (JCCompilationUnit)e.getCompilationUnit();
                TreeMaker localMake = TreeMaker.instance(context).forToplevel(jcCompilationUnit);
                ReflectMethods.instance(context).translateTopLevelClass(jcCompilationUnit, localMake);
            }
        }
    }
}
