/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
 * @summary Check presence/absence of code model in annotated elements
 * @library /tools/javac/lib
 * @modules java.compiler
 *          jdk.compiler
 *          jdk.incubator.code
 * @enablePreview
 * @build   JavacTestingAbstractProcessor TestElementReflection
 * @compile -processor TestElementReflection -proc:only TestElementReflection.java
 */

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;

import java.util.Optional;
import java.util.Set;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.element.*;

public class TestElementReflection extends JavacTestingAbstractProcessor {

    @interface HasModel {
        boolean value();
    }

    static class Test {
        @HasModel(false)
        Test() { } // constructor

        @Reflect
        @HasModel(true)
        void member_reflectable() { } // reflectable class member

        @HasModel(false)
        void member() { } // class member

        class Inner {
            @Reflect
            @HasModel(false)
            void inner_member_reflectable() { } // reflectable inner class member

            @HasModel(false)
            void inner_member() { } // inner class member
        }

        static class Nested {
            @Reflect
            @HasModel(true)
            void static_member_reflectable() { } // reflectable nested class member

            @HasModel(false)
            void static_member() { } // nested class member
        }
    }

    public boolean process(Set<? extends TypeElement> annotations,
                           RoundEnvironment roundEnvironment) {
        class Scan extends ElementScanner<Void,Void> {
            @Override
            public Void visitExecutable(ExecutableElement e, Void p) {
                HasModel hasModel = e.getAnnotation(HasModel.class);
                if (hasModel == null) {
                    return null; // skip
                }
                Optional<FuncOp> body = Op.ofElement(processingEnv, e);
                if (body.isPresent() && !hasModel.value()) {
                    throw new AssertionError(String.format("Unexpected model found for unsupported element %s",
                            toExecutableElementString(e)));
                } else if (body.isEmpty() && hasModel.value()) {
                    throw new AssertionError(String.format("Model not found for supported element %s",
                            toExecutableElementString(e)));
                }
                return null;
            }
        }
        Scan scan = new Scan();
        for (Element e : roundEnvironment.getRootElements()) {
            scan.scan(e);
        }
        return true;
    }

    static String toExecutableElementString(ExecutableElement e) {
        return e.getEnclosingElement() + "." + e;
    }
}
