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
import java.lang.reflect.code.type.impl.JavaTypeImpl;
import java.util.List;

/**
 * The symbolic description of a Java type.
 */
// @@@ Extend from this interface to model Java types with more fidelity
public sealed interface JavaType extends TypeElement permits JavaTypeImpl {

    // @@@ Share with general void type?
    JavaType VOID = new JavaTypeImpl("void");

    JavaType BOOLEAN = new JavaTypeImpl("boolean");

    JavaType J_L_BOOLEAN = new JavaTypeImpl("java.lang.Boolean");

    JavaType BOOLEAN_ARRAY = new JavaTypeImpl("boolean", 1);

    JavaType BYTE = new JavaTypeImpl("byte");

    JavaType J_L_BYTE = new JavaTypeImpl("java.lang.Byte");

    JavaType BYTE_ARRAY = new JavaTypeImpl("byte", 1);

    JavaType CHAR = new JavaTypeImpl("char");

    JavaType J_L_CHARACTER = new JavaTypeImpl("java.lang.Character");

    JavaType CHAR_ARRAY = new JavaTypeImpl("char", 1);

    JavaType SHORT = new JavaTypeImpl("short");

    JavaType J_L_SHORT = new JavaTypeImpl("java.lang.Short");

    JavaType SHORT_ARRAY = new JavaTypeImpl("short", 1);

    JavaType INT = new JavaTypeImpl("int");

    JavaType J_L_INTEGER = new JavaTypeImpl("java.lang.Integer");

    JavaType INT_ARRAY = new JavaTypeImpl("int", 1);

    JavaType LONG = new JavaTypeImpl("long");

    JavaType J_L_LONG = new JavaTypeImpl("java.lang.Long");

    JavaType LONG_ARRAY = new JavaTypeImpl("long", 1);

    JavaType FLOAT = new JavaTypeImpl("float");

    JavaType J_L_FLOAT = new JavaTypeImpl("java.lang.Float");

    JavaType FLOAT_ARRAY = new JavaTypeImpl("float", 1);

    JavaType DOUBLE = new JavaTypeImpl("double");

    JavaType J_L_DOUBLE = new JavaTypeImpl("java.lang.Double");

    JavaType DOUBLE_ARRAY = new JavaTypeImpl("double", 1);

    JavaType J_L_OBJECT = new JavaTypeImpl("java.lang.Object");

    JavaType J_L_OBJECT_ARRAY = new JavaTypeImpl("java.lang.Object", 1);

    JavaType J_L_CLASS = new JavaTypeImpl("java.lang.Class");

    JavaType J_L_STRING = new JavaTypeImpl("java.lang.String");

    //

    boolean isArray();

    int dimensions();

    JavaType componentType();

    JavaType rawType();

    boolean hasTypeArguments();

    List<JavaType> typeArguments();

    // Conversions

    JavaType toBasicType();

    String toClassName();

    String toInternalName();

    String toNominalDescriptorString();

    default ClassDesc toNominalDescriptor() {
        return ClassDesc.ofDescriptor(toNominalDescriptorString());
    }

    default Class<?> resolve(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return (Class<?>) toNominalDescriptor().resolveConstantDesc(l);
    }

    // Factories

    static JavaType type(Class<?> c) {
        int dims = 0;
        if (c.isArray()) {
            while (c.isArray()) {
                c = c.getComponentType();
                dims++;
            }
        }
        return new JavaTypeImpl(c.getName(), dims);
    }

    static JavaType type(Class<?> c, Class<?>... typeArguments) {
        return type(c, List.of(typeArguments));
    }

    static JavaType type(Class<?> c, List<Class<?>> typeArguments) {
        int dims = 0;
        if (c.isArray()) {
            while (c.isArray()) {
                c = c.getComponentType();
                dims++;
            }
        }
        return new JavaTypeImpl(c.getName(), dims, typeArguments.stream().map(JavaType::type).toList());
    }

    static JavaType ofNominalDescriptor(ClassDesc d) {
        String descriptor = d.descriptorString();
        int i = 0;
        while (descriptor.charAt(i) == '[') {
            i++;
        }
        int dims = i;

        JavaType td = switch (descriptor.charAt(i)) {
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
                yield new JavaTypeImpl(typeName, 0);
            }
            default -> throw new InternalError();
        };

        return JavaType.type(td, dims);
    }

    static JavaType ofNominalDescriptorString(String d) {
        return ofNominalDescriptor(ClassDesc.ofDescriptor(d));
    }

    static JavaType type(JavaType t, JavaType... typeArguments) {
        return type(t, List.of(typeArguments));
    }

    static JavaType type(JavaType t, List<JavaType> typeArguments) {
        if (t.hasTypeArguments()) {
            throw new IllegalArgumentException("Type descriptor must not have type arguments: " + t);
        }
        JavaTypeImpl timpl = (JavaTypeImpl) t;
        return new JavaTypeImpl(timpl.type, timpl.dims, typeArguments);
    }

    static JavaType type(JavaType t, int dims, JavaType... typeArguments) {
        return type(t, dims, List.of(typeArguments));
    }

    static JavaType type(JavaType t, int dims, List<JavaType> typeArguments) {
        if (t.isArray()) {
            throw new IllegalArgumentException("Type descriptor must not be an array: " + t);
        }
        if (t.hasTypeArguments()) {
            throw new IllegalArgumentException("Type descriptor must not have type arguments: " + t);
        }
        JavaTypeImpl timpl = (JavaTypeImpl) t;
        return new JavaTypeImpl(timpl.type, dims, typeArguments);
    }

    // Copied code in jdk.compiler module throws UOE
    static JavaType ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return (JavaType) CoreTypeFactory.JAVA_TYPE_FACTORY.constructType(java.lang.reflect.code.parser.impl.DescParser.parseTypeDefinition(s));
    }
}
