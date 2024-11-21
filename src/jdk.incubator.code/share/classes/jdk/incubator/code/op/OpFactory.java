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

package jdk.incubator.code.op;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import jdk.incubator.code.Op;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * An operation factory for constructing an {@link Op operation} from its
 * {@link ExternalizableOp.ExternalizedOp external content}.
 */
@FunctionalInterface
public interface OpFactory {

    /**
     * An operation declaration annotation.
     * <p>
     * This annotation may be declared on a concrete class implementing an {@link Op operation} whose name is a constant
     * that can be declared as this attribute's value.
     * <p>
     * Tooling can process declarations of this annotation to build a factory for constructing operations from their name.
     */
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.TYPE)
    @interface OpDeclaration {
        /**
         * {@return the operation name}
         */
        String value();
    }

    /**
     * A class value for lazily computing an operation factory for {@link Op operation} classes
     * annotated with {@link OpDeclaration} and enclosed within a given class to compute over.
     * <p>
     * Each enclosed class annotated with {@code OpDeclaration} must declare a public static method named {@code create}
     * with one parameter type of {@link ExternalizableOp.ExternalizedOp} and return type that is the concrete class type.
     * Alternatively, the concrete class must declare public constructor with one parameter type of
     * {@link ExternalizableOp.ExternalizedOp}.
     */
    ClassValue<OpFactory> OP_FACTORY = new ClassValue<>() {
        @Override
        protected OpFactory computeValue(Class<?> c) {
            // @@@ See https://bugs.openjdk.org/browse/JDK-8321207
            final Map<String, Class<? extends Op>> opMapping = createOpMapping(c);

            return def -> {
                var opClass = opMapping.get(def.name());
                if (opClass == null) {
                    return null;
                }

                return constructOp(opClass, def);
            };
        }
    };

    /**
     * Constructs an {@link Op operation} from its external content.
     * <p>
     * If there is no mapping from the operation's name to a concrete
     * class of an {@code Op} then this method returns null.
     *
     * @param def the operation's external content
     * @return the operation, otherwise null
     */
    Op constructOp(ExternalizableOp.ExternalizedOp def);

    /**
     * Constructs an {@link Op operation} from its external content.
     * <p>
     * If there is no mapping from the operation's name to a concrete
     * class of an {@code Op} then this method throws UnsupportedOperationException.
     *
     * @param def the operation's external content
     * @return the operation, otherwise null
     * @throws UnsupportedOperationException if there is no mapping from the operation's
     *                                       name to a concrete class of an {@code Op}
     */
    default Op constructOpOrFail(ExternalizableOp.ExternalizedOp def) {
        Op op = constructOp(def);
        if (op == null) {
            throw new UnsupportedOperationException("Unsupported operation: " + def.name());
        }

        return op;
    }

    /**
     * Compose this operation factory with another operation factory.
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

    private static Map<String, Class<? extends Op>> createOpMapping(Class<?> opClasses) {
        Map<String, Class<? extends Op>> mapping = new HashMap<>();
        for (Class<?> opClass : opClasses.getNestMembers()) {
            if (opClass.isAnnotationPresent(OpDeclaration.class)) {
                if (!Modifier.isPublic(opClass.getModifiers())) {
                    throw new InternalError("Operation class not public: " + opClass.getName());
                }

                if (!Op.class.isAssignableFrom(opClass)) {
                    throw new InternalError("Operation class is not assignable to Op: " + opClass);
                }

                MethodHandle handle = getOpConstructorMethodHandle(opClass);
                if (handle == null) {
                    throw new InternalError("Operation constructor for operation class not found: " + opClass.getName());
                }

                if (!Op.class.isAssignableFrom(handle.type().returnType())) {
                    throw new InternalError("Operation constructor does not return an Op: " + handle);
                }

                String opName = opClass.getAnnotation(OpDeclaration.class).value();
                @SuppressWarnings("unchecked")
                var opClassCast = (Class<Op>) opClass;
                mapping.put(opName, opClassCast);
            }
        }
        return mapping;
    }

    private static MethodHandle getOpConstructorMethodHandle(Class<?> opClass) {
        Method method = null;
        try {
            method = opClass.getMethod("create", ExternalizableOp.ExternalizedOp.class);
        } catch (NoSuchMethodException e) {
        }

        if (method != null) {
            if (!Modifier.isStatic(method.getModifiers())) {
                throw new InternalError("Operation constructor is not a static method: " + method);
            }

            try {
/*__return MethodHandles.lookup().unreflect(method);__*/                return MethodHandles.publicLookup().unreflect(method);
            } catch (IllegalAccessException e) {
                throw new InternalError("Inaccessible operation constructor for operation: " +
                        method);
            }
        }

        Constructor<?> constructor;
        try {
            constructor = opClass.getConstructor(ExternalizableOp.ExternalizedOp.class);
        } catch (NoSuchMethodException e) {
            return null;
        }

        try {
/*__return MethodHandles.lookup().unreflectConstructor(constructor);__*/            return MethodHandles.publicLookup().unreflectConstructor(constructor);
        } catch (IllegalAccessException e) {
            throw new InternalError("Inaccessible operation constructor for operation: " +
                    constructor);
        }
    }

    private static Op constructOp(Class<? extends Op> opClass, ExternalizableOp.ExternalizedOp opDef) {
        class Enclosed {
            private static final ClassValue<Function<ExternalizableOp.ExternalizedOp, Op>> OP_CONSTRUCTOR = new ClassValue<>() {
                @Override
                protected Function<ExternalizableOp.ExternalizedOp, Op> computeValue(Class<?> opClass) {
                    final MethodHandle opConstructorMH = getOpConstructorMethodHandle(opClass);
                    assert opConstructorMH != null;

                    return operationDefinition -> {
                        try {
                            return (Op) opConstructorMH.invoke(operationDefinition);
                        } catch (RuntimeException | Error e) {
                            throw e;
                        } catch (Throwable t) {
                            throw new RuntimeException(t);
                        }
                    };
                }
            };
        }
        return Enclosed.OP_CONSTRUCTOR.get(opClass).apply(opDef);
    }
}
