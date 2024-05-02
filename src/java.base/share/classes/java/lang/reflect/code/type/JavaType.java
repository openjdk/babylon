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
import java.lang.constant.ConstantDescs;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.Executable;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.GenericDeclaration;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.WildcardType.BoundKind;
import java.util.List;
import java.util.Objects;

/**
 * The symbolic description of a Java type. Java types can be classified as follows:
 * <ul>
 *     <li>{@linkplain PrimitiveType primitive types}, e.g. {@code int}, {@code void}</li>
 *     <li>{@linkplain ClassType class types}, e.g. {@code String}, {@code List<? extends Number>}</li>
 *     <li>{@linkplain ArrayType array types}, e.g. {@code Object[][]}, {@code List<Runnable>[]}</li>
 *     <li>{@linkplain WildcardType wildcard types}, e.g. {@code ? extends Number}, {@code ? super ArrayList<String>}</li>
 *     <li>{@linkplain TypeVarRef type-variables}, e.g. {@code T extends Runnable}</li>
 * </ul>
 * Java types can be constructed from either {@linkplain ClassDesc nominal descriptors} or
 * {@linkplain Type reflective type mirrors}. Conversely, Java types can be
 * {@linkplain #toNominalDescriptor() turned} into nominal descriptors,
 * or be {@linkplain #resolve(Lookup) resolved} into reflective type mirrors.
 * @sealedGraph
 */
public sealed interface JavaType extends TypeElement permits ClassType, ArrayType,
                                                             PrimitiveType, WildcardType, TypeVarRef {

    /** {@link JavaType} representing {@code void} */
    PrimitiveType VOID = new PrimitiveType(ConstantDescs.CD_void);

    /** {@link JavaType} representing {@code boolean} */
    PrimitiveType BOOLEAN = new PrimitiveType(ConstantDescs.CD_boolean);

    /** {@link JavaType} representing {@link Boolean} */
    ClassType J_L_BOOLEAN = new ClassType(ConstantDescs.CD_Boolean);

    /** {@link JavaType} representing {@code boolean[]} */
    ArrayType BOOLEAN_ARRAY = new ArrayType(BOOLEAN);

    /** {@link JavaType} representing {@code byte} */
    PrimitiveType BYTE = new PrimitiveType(ConstantDescs.CD_byte);

    /** {@link JavaType} representing {@link Byte} */
    ClassType J_L_BYTE = new ClassType(ConstantDescs.CD_Byte);

    /** {@link JavaType} representing {@code byte[]} */
    ArrayType BYTE_ARRAY = new ArrayType(BYTE);

    /** {@link JavaType} representing {@code char} */
    PrimitiveType CHAR = new PrimitiveType(ConstantDescs.CD_char);

    /** {@link JavaType} representing {@link Character} */
    ClassType J_L_CHARACTER = new ClassType(ConstantDescs.CD_Character);

    /** {@link JavaType} representing {@code char[]} */
    ArrayType CHAR_ARRAY = new ArrayType(CHAR);

    /** {@link JavaType} representing {@code short} */
    PrimitiveType SHORT = new PrimitiveType(ConstantDescs.CD_short);

    /** {@link JavaType} representing {@link Short} */
    ClassType J_L_SHORT = new ClassType(ConstantDescs.CD_Short);

    /** {@link JavaType} representing {@code short[]} */
    ArrayType SHORT_ARRAY = new ArrayType(SHORT);

    /** {@link JavaType} representing {@code int} */
    PrimitiveType INT = new PrimitiveType(ConstantDescs.CD_int);

    /** {@link JavaType} representing {@link Integer} */
    ClassType J_L_INTEGER = new ClassType(ConstantDescs.CD_Integer);

    /** {@link JavaType} representing {@code int[]} */
    ArrayType INT_ARRAY = new ArrayType(INT);

    /** {@link JavaType} representing {@code long} */
    PrimitiveType LONG = new PrimitiveType(ConstantDescs.CD_long);

    /** {@link JavaType} representing {@link Long} */
    ClassType J_L_LONG = new ClassType(ConstantDescs.CD_Long);

    /** {@link JavaType} representing {@code long[]} */
    ArrayType LONG_ARRAY = new ArrayType(LONG);

    /** {@link JavaType} representing {@code float} */
    PrimitiveType FLOAT = new PrimitiveType(ConstantDescs.CD_float);

    /** {@link JavaType} representing {@link Float} */
    ClassType J_L_FLOAT = new ClassType(ConstantDescs.CD_Float);

    /** {@link JavaType} representing {@code float[]} */
    ArrayType FLOAT_ARRAY = new ArrayType(FLOAT);

    /** {@link JavaType} representing {@code double} */
    PrimitiveType DOUBLE = new PrimitiveType(ConstantDescs.CD_double);

    /** {@link JavaType} representing {@link Double} */
    ClassType J_L_DOUBLE = new ClassType(ConstantDescs.CD_Double);

    /** {@link JavaType} representing {@code double[]} */
    ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);

    /** {@link JavaType} representing {@link Object} */
    ClassType J_L_OBJECT = new ClassType(ConstantDescs.CD_Object);

    /** {@link JavaType} representing {@link Object[]} */
    ArrayType J_L_OBJECT_ARRAY = new ArrayType(J_L_OBJECT);

    /** {@link JavaType} representing {@link Class} */
    ClassType J_L_CLASS = new ClassType(ConstantDescs.CD_Class);

    /** {@link JavaType} representing {@link String} */
    ClassType J_L_STRING = new ClassType(ConstantDescs.CD_String);

    /** {@link JavaType} representing {@link List} */
    ClassType J_U_LIST = new ClassType(ConstantDescs.CD_List);

    // Conversions

    /**
     * {@return the basic type associated with this Java type}. A basic type is one of the following
     * types:
     * <ul>
     *     <li>{@link JavaType#VOID}</li>
     *     <li>{@link JavaType#INT}</li>
     *     <li>{@link JavaType#LONG}</li>
     *     <li>{@link JavaType#FLOAT}</li>
     *     <li>{@link JavaType#DOUBLE}</li>
     *     <li>{@link JavaType#J_L_OBJECT}</li>
     * </ul>
     *
     */
    JavaType toBasicType();

    /**
     * {@return the nominal descriptor associated with this Java type}
     */
    ClassDesc toNominalDescriptor();

    /**
     * Resolve this Java type to a reflective type mirror.
     * @param lookup the lookup used to create the reflective type mirror
     * @return a reflective type mirror for this type
     * @throws ReflectiveOperationException if this Java type cannot be resolved
     */
    // @@@: this should return a reflective type mirror
    default Class<?> resolve(MethodHandles.Lookup lookup) throws ReflectiveOperationException {
        return toNominalDescriptor().resolveConstantDesc(lookup);
    }

    /**
     * {@return the erasure of this Java type, as per JLS 4.6}
     */
    JavaType erasure();

    // Factories

    /**
     * Constructs a Java type from a reflective type mirror.
     *
     * @param reflectiveType the reflective type mirror
     */
    static JavaType type(Type reflectiveType) {
        return switch (reflectiveType) {
            case Class<?> c -> type(c.describeConstable().get());
            case java.lang.reflect.WildcardType wt -> wt.getLowerBounds().length == 0 ?
                    wildcard(BoundKind.EXTENDS, type(wt.getUpperBounds()[0])) :
                    wildcard(BoundKind.SUPER, type(wt.getLowerBounds()[0]));
            case TypeVariable<?> tv -> typeVarRef(tv.getName(), owner(tv.getGenericDeclaration()), type(tv.getBounds()[0]));
            case GenericArrayType at -> array(type(at.getGenericComponentType()));
            default -> throw new InternalError();
        };
    }

    private static TypeVarRef.Owner owner(GenericDeclaration genDecl) {
        return switch (genDecl) {
            case Executable e -> MethodRef.method(e);
            case Class<?> t -> (ClassType)type(t);
            default -> throw new InternalError();
        };
    }

    /**
     * Constructs a Java type from a nominal descriptor.
     *
     * @param desc the nominal descriptor
     */
    static JavaType type(ClassDesc desc) {
        if (desc.isPrimitive()) {
            return switch (desc.descriptorString().charAt(0)) {
                case 'V' -> JavaType.VOID;
                case 'I' -> JavaType.INT;
                case 'J' -> JavaType.LONG;
                case 'C' -> JavaType.CHAR;
                case 'S' -> JavaType.SHORT;
                case 'B' -> JavaType.BYTE;
                case 'F' -> JavaType.FLOAT;
                case 'D' -> JavaType.DOUBLE;
                case 'Z' -> JavaType.BOOLEAN;
                default -> throw new InternalError();
            };
        } else if (desc.isArray()) {
            return array(type(desc.componentType()));
        } else {
            // class
            return new ClassType(desc, List.of());
        }
    }

    /**
     * Constructs a parameterized class type.
     *
     * @param type the base type of the parameterized type
     * @param typeArguments the type arguments of the parameterized type
     * @return a parameterized class type
     * @throws IllegalArgumentException if {@code type} is not a {@linkplain ClassType class type}
     * @throws IllegalArgumentException if {@code type} is {@linkplain ClassType class type} with
     * a non-empty {@linkplain ClassType#typeArguments() type argument list}.
     */
    static ClassType parameterized(JavaType type, JavaType... typeArguments) {
        return parameterized(type, List.of(typeArguments));
    }

    /**
     * Constructs a parameterized class type.
     *
     * @param type the base type of the parameterized type
     * @param typeArguments the type arguments of the parameterized type
     * @return a parameterized class type
     * @throws IllegalArgumentException if {@code type} is not a {@linkplain ClassType class type}
     * @throws IllegalArgumentException if {@code type} is {@linkplain ClassType class type} with
     * a non-empty {@linkplain ClassType#typeArguments() type argument list}.
     */
    static ClassType parameterized(JavaType type, List<JavaType> typeArguments) {
        if (!(type instanceof ClassType ct)) {
            throw new IllegalArgumentException("Not a class type: " + type);
        }
        if (ct.hasTypeArguments()) {
            throw new IllegalArgumentException("Already parameterized: " + type);
        }
        return new ClassType(type.toNominalDescriptor(), typeArguments);
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
     * Constructs a reference to a type-variable with the given owner.
     *
     * @param bound the type-variable bound.
     * @param owner the type-variable owner.
     * @return a type-variable reference.
     */
    static TypeVarRef typeVarRef(String name, TypeVarRef.Owner owner, JavaType bound) {
        return new TypeVarRef(name, owner, bound);
    }

    /**
     * Constructs a Java type from a string representation.
     * @param s string representation
     * @return a Java type corresponding to the provided string representation
     */
    // Copied code in jdk.compiler module throws UOE
    static JavaType ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return (JavaType) CoreTypeFactory.JAVA_TYPE_FACTORY.constructType(java.lang.reflect.code.parser.impl.DescParser.parseTypeDefinition(s));
    }
}
