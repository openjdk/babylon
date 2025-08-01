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

package jdk.incubator.code.extern;

import jdk.incubator.code.Op;

/**
 * An operation factory for constructing an {@link Op operation} from its
 * {@link ExternalizedOp external content}.
 */
@FunctionalInterface
public interface OpFactory {

    /**
     * Constructs an {@link Op operation} from its external content.
     * <p>
     * If there is no mapping from the operation's name to a concrete
     * class of an {@code Op} then this method returns null.
     *
     * @param def the operation's external content
     * @return the operation, otherwise null
     */
    Op constructOp(ExternalizedOp def);

    /**
     * Constructs an {@link Op operation} from its external content.
     * <p>
     * If there is no mapping from the operation's name to a concrete
     * class of an {@code Op} then this method throws UnsupportedOperationException.
     *
     * @param def the operation's external content
     * @return the operation
     * @throws UnsupportedOperationException if there is no mapping from the operation's
     *                                       name to a concrete class of an {@code Op}
     */
    default Op constructOpOrFail(ExternalizedOp def) {
        Op op = constructOp(def);
        if (op == null) {
            throw new UnsupportedOperationException("Unsupported operation: " + def.name());
        }

        return op;
    }

    /**
     * Composes this operation factory with another operation factory.
     * <p>
     * If there is no mapping in this operation factory then the result
     * of the other operation factory is returned.
     *
     * @param after the other operation factory.
     * @return the composed operation factory.
     */
    default OpFactory andThen(OpFactory after) {
        return def -> {
            Op op = constructOp(def);
            return op != null ? op : after.constructOp(def);
        };
    }

//    // Uncomment the following and execute using an exploded build like as follows to generate a factory method
//    // for enclosed concrete operations
//    // java --add-modules jdk.incubator.code jdk.incubator.code.extern.OpFactory jdk.incubator.code.dialect.core.CoreOp
//    static void main(String[] args) {
//        Class<?> enclosingOpClass = null;
//        try {
//            enclosingOpClass = Class.forName(args[0]);
//        } catch (ClassNotFoundException e) {
//            throw new RuntimeException(e);
//        }
//        generateSwitchExpression(enclosingOpClass, System.out);
//    }
//
//    static void generateSwitchExpression(Class<?> enclosingOpClass, java.io.PrintStream out) {
//        java.util.Map<String, java.lang.reflect.Executable> opNameMap = new java.util.TreeMap<>();
//        for (Class<?> opClass : enclosingOpClass.getNestMembers()) {
//            if (!Op.class.isAssignableFrom(opClass)) {
//                continue;
//            }
//            if (!java.lang.reflect.Modifier.isFinal(opClass.getModifiers())) {
//                continue;
//            }
//
//            var opDecl = opClass.getAnnotation(jdk.incubator.code.internal.OpDeclaration.class);
//            String name = opDecl.value();
//
//            var e = getOpConstructorExecutable(opClass);
//            opNameMap.put(name, e);
//        }
//
//        out.println("static Op createOp(ExternalizedOp def) {");
//        out.println("    Op op = switch (def.name()) {");
//        opNameMap.forEach((name, e) -> {
//            out.print("        case \"" + name + "\" -> ");
//            switch (e) {
//                case java.lang.reflect.Constructor<?> constructor -> {
//                    out.println("new " + name(enclosingOpClass, constructor.getDeclaringClass()) + "(def);");
//                }
//                case java.lang.reflect.Method method -> {
//                    out.println(name(enclosingOpClass, method.getDeclaringClass()) + "." + method.getName() + "(def);");
//                }
//            }
//        });
//        out.println("        default -> null;");
//        out.println("    };");
//        out.print(
//                """
//                    if (op != null) {
//                        op.setLocation(def.location());
//                    }
//                    return op;
//                """);
//        out.println("}");
//    }
//
//    private static java.lang.reflect.Executable getOpConstructorExecutable(Class<?> opClass) {
//        java.lang.reflect.Executable e = null;
//        try {
//            e = opClass.getDeclaredMethod("create", ExternalizedOp.class);
//        } catch (NoSuchMethodException _) {
//        }
//
//        if (e != null) {
//            if (!java.lang.reflect.Modifier.isStatic(e.getModifiers())) {
//                throw new InternalError("Operation constructor is not a static method: " + e);
//            }
//            return e;
//        }
//
//        try {
//            e = opClass.getDeclaredConstructor(ExternalizedOp.class);
//        } catch (NoSuchMethodException _) {
//            return null;
//        }
//
//        return e;
//    }
//
//    static String name(Class<?> enclosingOpClass, Class<?> declaringClass) {
//        return declaringClass.getCanonicalName().substring(enclosingOpClass.getCanonicalName().length() + 1);
//    }
}
