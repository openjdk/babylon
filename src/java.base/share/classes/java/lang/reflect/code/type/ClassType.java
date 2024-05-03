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

package java.lang.reflect.code.type;

import java.lang.constant.ClassDesc;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.code.TypeElement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A class type.
 */
public final class ClassType implements TypeVarRef.Owner, JavaType {
    // Fully qualified name
    private final ClassDesc type;

    private final List<JavaType> typeArguments;

    ClassType(ClassDesc type) {
        this(type, List.of());
    }

    ClassType(ClassDesc type, List<JavaType> typeArguments) {
        if (!type.isClassOrInterface()) {
            throw new IllegalArgumentException("Invalid base type: " + type);
        }
        this.type = type;
        this.typeArguments = List.copyOf(typeArguments);
    }

    @Override
    public Type resolve(Lookup lookup) throws ReflectiveOperationException {
        Class<?> baseType = type.resolveConstantDesc(lookup);
        List<Type> resolvedTypeArgs = new ArrayList<>();
        for (JavaType typearg : typeArguments) {
            resolvedTypeArgs.add(typearg.resolve(lookup));
        }
        return resolvedTypeArgs.isEmpty() ?
                baseType :
                makeReflectiveParameterizedType(baseType,
                        resolvedTypeArgs.toArray(new Type[0]), baseType.getDeclaringClass()); // @@@: generic owner is erased here
    }

    // Copied code in jdk.compiler module throws UOE
    private static ParameterizedType makeReflectiveParameterizedType(Class<?> base, Type[] typeArgs, Class<?> owner) {
/*__throw new UnsupportedOperationException();__*/        return sun.reflect.generics.reflectiveObjects.ParameterizedTypeImpl.make(base, typeArgs, owner);
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        List<TypeDefinition> args = typeArguments.stream()
                .map(TypeElement::toTypeDefinition)
                .toList();

        TypeDefinition td = new TypeDefinition(toClassName(), args);
        return td;
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ClassType typeDesc = (ClassType) o;

        if (!type.equals(typeDesc.type)) return false;
        return typeArguments.equals(typeDesc.typeArguments);
    }

    @Override
    public int hashCode() {
        int result = type.hashCode();
        result = 31 * result + typeArguments.hashCode();
        return result;
    }

    /**
     * {@return the unboxed primitive type associated with this class type (if any)}
     */
    public Optional<PrimitiveType> unbox() {
        class LazyHolder {
            static final Map<ClassType, PrimitiveType> wrapperToPrimitive = Map.of(
                    J_L_BYTE, BYTE,
                    J_L_SHORT, SHORT,
                    J_L_INTEGER, INT,
                    J_L_LONG, LONG,
                    J_L_FLOAT, FLOAT,
                    J_L_DOUBLE, DOUBLE,
                    J_L_CHARACTER, CHAR,
                    J_L_BOOLEAN, BOOLEAN
            );
        }
        return Optional.ofNullable(LazyHolder.wrapperToPrimitive.get(this));
    }

    @Override
    public JavaType erasure() {
        return rawType();
    }

    // Conversions

    /**
     * {@return a class type whose base type is the same as this class type, but without any
     * type arguments}
     */
    public ClassType rawType() {
        return new ClassType(type);
    }

    /**
     * {@return {@code true} if this class type has a non-empty type argument list}
     * @see ClassType#typeArguments()
     */
    public boolean hasTypeArguments() {
        return !typeArguments.isEmpty();
    }

    /**
     * {@return the type argument list associated with this class type}
     */
    public List<JavaType> typeArguments() {
        return typeArguments;
    }

    @Override
    public JavaType toBasicType() {
        return JavaType.J_L_OBJECT;
    }

    /**
     * {@return a human-readable name for this class type}
     */
    public String toClassName() {
        String pkg = type.packageName();
        return pkg.isEmpty() ?
                type.displayName() :
                String.format("%s.%s", pkg, type.displayName());
    }

    @Override
    public ClassDesc toNominalDescriptor() {
        return type;
    }
}
