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
 * The symbolic description of a Java type.
 */
public sealed interface JavaType extends TypeElement permits ClassType, ArrayType,
                                                             PrimitiveType, WildcardType, TypeVarRef {

    // @@@ Share with general void type?
    PrimitiveType VOID = new PrimitiveType(ConstantDescs.CD_void);

    PrimitiveType BOOLEAN = new PrimitiveType(ConstantDescs.CD_boolean);

    ClassType J_L_BOOLEAN = new ClassType(ConstantDescs.CD_Boolean);

    ArrayType BOOLEAN_ARRAY = new ArrayType(BOOLEAN);

    PrimitiveType BYTE = new PrimitiveType(ConstantDescs.CD_byte);

    ClassType J_L_BYTE = new ClassType(ConstantDescs.CD_Byte);

    ArrayType BYTE_ARRAY = new ArrayType(BYTE);

    PrimitiveType CHAR = new PrimitiveType(ConstantDescs.CD_char);

    ClassType J_L_CHARACTER = new ClassType(ConstantDescs.CD_Character);

    ArrayType CHAR_ARRAY = new ArrayType(CHAR);

    PrimitiveType SHORT = new PrimitiveType(ConstantDescs.CD_short);

    ClassType J_L_SHORT = new ClassType(ConstantDescs.CD_Short);

    ArrayType SHORT_ARRAY = new ArrayType(SHORT);

    PrimitiveType INT = new PrimitiveType(ConstantDescs.CD_int);

    ClassType J_L_INTEGER = new ClassType(ConstantDescs.CD_Integer);

    ArrayType INT_ARRAY = new ArrayType(INT);

    PrimitiveType LONG = new PrimitiveType(ConstantDescs.CD_long);

    ClassType J_L_LONG = new ClassType(ConstantDescs.CD_Long);

    ArrayType LONG_ARRAY = new ArrayType(LONG);

    PrimitiveType FLOAT = new PrimitiveType(ConstantDescs.CD_float);

    ClassType J_L_FLOAT = new ClassType(ConstantDescs.CD_Float);

    ArrayType FLOAT_ARRAY = new ArrayType(FLOAT);

    PrimitiveType DOUBLE = new PrimitiveType(ConstantDescs.CD_double);

    ClassType J_L_DOUBLE = new ClassType(ConstantDescs.CD_Double);

    ArrayType DOUBLE_ARRAY = new ArrayType(DOUBLE);

    ClassType J_L_OBJECT = new ClassType(ConstantDescs.CD_Object);

    ArrayType J_L_OBJECT_ARRAY = new ArrayType(J_L_OBJECT);

    ClassType J_L_CLASS = new ClassType(ConstantDescs.CD_Class);

    ClassType J_L_STRING = new ClassType(ConstantDescs.CD_String);

    ClassType J_U_LIST = new ClassType(ConstantDescs.CD_List);

    // Conversions

    JavaType toBasicType();

    ClassDesc toNominalDescriptor();

    default Class<?> resolve(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return toNominalDescriptor().resolveConstantDesc(l);
    }

    /**
     * {@return the erasure of this Java type, as per JLS 4.6}
     */
    JavaType erasure();

    // Factories

    static JavaType type(Type t) {
        return switch (t) {
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

    static JavaType type(ClassDesc d) {
        if (d.isPrimitive()) {
            return switch (d.descriptorString().charAt(0)) {
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
        } else if (d.isArray()) {
            return array(type(d.componentType()));
        } else {
            // class
            return new ClassType(d, List.of());
        }
    }

    static ClassType parameterized(JavaType type, JavaType... typeArguments) {
        return parameterized(type, List.of(typeArguments));
    }

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

    // Copied code in jdk.compiler module throws UOE
    static JavaType ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return (JavaType) CoreTypeFactory.JAVA_TYPE_FACTORY.constructType(java.lang.reflect.code.parser.impl.DescParser.parseTypeDefinition(s));
    }
}
