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
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.WildcardType.BoundKind;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The symbolic description of a Java type.
 */
public sealed interface JavaType extends TypeElement permits ClassType, ArrayType,
                                                             PrimitiveType, WildcardType, TypeVarRef {

    // @@@ Share with general void type?
    JavaType VOID = new PrimitiveType("void");

    JavaType BOOLEAN = new PrimitiveType("boolean");

    JavaType J_L_BOOLEAN = new ClassType("java.lang.Boolean");

    JavaType BOOLEAN_ARRAY = new ArrayType(BOOLEAN);

    JavaType BYTE = new PrimitiveType("byte");

    JavaType J_L_BYTE = new ClassType("java.lang.Byte");

    JavaType BYTE_ARRAY = new ArrayType(BYTE);

    JavaType CHAR = new PrimitiveType("char");

    JavaType J_L_CHARACTER = new ClassType("java.lang.Character");

    JavaType CHAR_ARRAY = new ArrayType(CHAR);

    JavaType SHORT = new PrimitiveType("short");

    JavaType J_L_SHORT = new ClassType("java.lang.Short");

    JavaType SHORT_ARRAY = new ArrayType(SHORT);

    JavaType INT = new PrimitiveType("int");

    JavaType J_L_INTEGER = new ClassType("java.lang.Integer");

    JavaType INT_ARRAY = new ArrayType(INT);

    JavaType LONG = new PrimitiveType("long");

    JavaType J_L_LONG = new ClassType("java.lang.Long");

    JavaType LONG_ARRAY = new ArrayType(LONG);

    JavaType FLOAT = new PrimitiveType("float");

    JavaType J_L_FLOAT = new ClassType("java.lang.Float");

    JavaType FLOAT_ARRAY = new ArrayType(FLOAT);

    JavaType DOUBLE = new PrimitiveType("double");

    JavaType J_L_DOUBLE = new ClassType("java.lang.Double");

    JavaType DOUBLE_ARRAY = new ArrayType(DOUBLE);

    JavaType J_L_OBJECT = new ClassType("java.lang.Object");

    JavaType J_L_OBJECT_ARRAY = new ArrayType(J_L_OBJECT);

    JavaType J_L_CLASS = new ClassType("java.lang.Class");

    JavaType J_L_STRING = new ClassType("java.lang.String");

    JavaType J_L_STRING_TEMPLATE = new ClassType("java.lang.StringTemplate");

    JavaType J_L_STRING_TEMPLATE_PROCESSOR = new ClassType("java.lang.StringTemplate$Processor");

    JavaType J_U_LIST = new ClassType("java.util.List");

    //

    Map<TypeElement, TypeElement> primitiveToWrapper = Map.of(
            BYTE, J_L_BYTE,
            SHORT, J_L_SHORT,
            INT, J_L_INTEGER,
            LONG, J_L_LONG,
            FLOAT, J_L_FLOAT,
            DOUBLE, J_L_DOUBLE,
            CHAR, J_L_CHARACTER,
            BOOLEAN, J_L_BOOLEAN
    );

    static boolean isPrimitive(TypeElement te) {
        return primitiveToWrapper.containsKey(te);
    }

    static TypeElement getWrapperType(TypeElement te) {
        return primitiveToWrapper.get(te);
    };

    // Conversions

    JavaType toBasicType();


    String toNominalDescriptorString();

    default ClassDesc toNominalDescriptor() {
        return ClassDesc.ofDescriptor(toNominalDescriptorString());
    }

    default Class<?> resolve(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return (Class<?>) toNominalDescriptor().resolveConstantDesc(l);
    }

    /**
     * {@return the erasure of this Java type, as per JLS 4.6}
     */
    JavaType erasure();

    // Factories

    static JavaType type(Class<?> c) {
        if (c.isPrimitive()) {
            return new PrimitiveType(c.getName());
        } else if (c.isArray()) {
            return array(type(c.getComponentType()));
        } else {
            return new ClassType(c.getName());
        }
    }

    static JavaType type(Class<?> c, Class<?>... typeArguments) {
        return type(c, List.of(typeArguments));
    }

    static JavaType type(Class<?> c, List<Class<?>> typeArguments) {
        if (c.isPrimitive()) {
            throw new IllegalArgumentException("Cannot parameterize a primitive type");
        } else if (c.isArray()) {
            return array(type(c.getComponentType(), typeArguments));
        } else {
            return new ClassType(c.getName(),
                    typeArguments.stream().map(JavaType::type).toList());
        }
    }

    static JavaType ofNominalDescriptor(ClassDesc d) {
        return ofNominalDescriptorStringInternal(d.descriptorString(), 0);
    }

    static JavaType ofNominalDescriptorString(String d) {
        return ofNominalDescriptor(ClassDesc.ofDescriptor(d));
    }

    private static JavaType ofNominalDescriptorStringInternal(String descriptor, int i) {
        if (descriptor.charAt(i) == '[') {
            return new ArrayType(ofNominalDescriptorStringInternal(descriptor, i + 1));
        } else {
            return switch (descriptor.charAt(i)) {
                case 'V' -> JavaType.VOID;
                case 'I' -> JavaType.INT;
                case 'J' -> JavaType.LONG;
                case 'C' -> JavaType.CHAR;
                case 'S' -> JavaType.SHORT;
                case 'B' -> JavaType.BYTE;
                case 'F' -> JavaType.FLOAT;
                case 'D' -> JavaType.DOUBLE;
                case 'Z' -> JavaType.BOOLEAN;
                case 'L' -> {
                    // La.b.c.Class;
                    String typeName = descriptor.substring(i + 1, descriptor.length() - 1).replace('/', '.');
                    yield new ClassType(typeName);
                }
                default -> throw new InternalError();
            };
        }
    }

    static JavaType type(JavaType t, JavaType... typeArguments) {
        return type(t, List.of(typeArguments));
    }

    static JavaType type(JavaType t, List<JavaType> typeArguments) {
        return switch (t) {
            case ArrayType at -> array(type(at.componentType(), typeArguments));
            case ClassType ct when !ct.hasTypeArguments() -> new ClassType(ct.toClassName(), typeArguments);
            default -> throw new IllegalArgumentException("Cannot parameterize type: " + t);
        };
    }

    /**
     * Constructs an array type.
     *
     * @param elementType the array type's element type.
     * @return an array type.
     */
    static ArrayType array(JavaType elementType) {
        Objects.requireNonNull(elementType);
        return new ArrayType(elementType);
    }

    /**
     * Constructs an array type.
     *
     * @param elementType the array type's element type.
     * @param dims the array type dimension
     * @return an array type.
     * @throws IllegalArgumentException if {@code dims < 1}.
     */
    static ArrayType array(JavaType elementType, int dims) {
        Objects.requireNonNull(elementType);
        if (dims < 1) {
            throw new IllegalArgumentException("Invalid dimension: " + dims);
        }
        for (int i = 1 ; i < dims ; i++) {
            elementType = array(elementType);
        }
        return array(elementType);
    }

    /**
     * Constructs an unbounded wildcard type.
     *
     * @return an unbounded wildcard type.
     */
    static WildcardType wildcard() {
        return new WildcardType(BoundKind.EXTENDS, JavaType.J_L_OBJECT);
    }

    /**
     * Constructs a bounded wildcard type of the given kind.
     *
     * @return a bounded wildcard type.
     */
    static WildcardType wildcard(BoundKind kind, JavaType bound) {
        return new WildcardType(kind, bound);
    }

    /**
     * Constructs a reference to a class type-variable.
     *
     * @param bound the type-variable bound.
     * @param owner the class where the type-variable is declared.
     * @return a type-variable reference.
     */
    static TypeVarRef typeVarRef(String name, ClassType owner, JavaType bound) {
        return new TypeVarRef(name, owner, bound);
    }

    /**
     * Constructs a reference to a method type-variable.
     *
     * @param bound the type-variable bound.
     * @param owner the method where the type-variable is declared.
     * @return a type-variable reference.
     */
    static TypeVarRef typeVarRef(String name, MethodRef owner, JavaType bound) {
        return new TypeVarRef(name, owner, bound);
    }

    // Copied code in jdk.compiler module throws UOE
    static JavaType ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return (JavaType) CoreTypeFactory.JAVA_TYPE_FACTORY.constructType(java.lang.reflect.code.parser.impl.DescParser.parseTypeDefinition(s));
    }
}
