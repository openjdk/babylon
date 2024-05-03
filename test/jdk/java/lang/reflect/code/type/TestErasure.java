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

/*
 * @test
 * @run testng TestErasure
 */

import static org.testng.Assert.*;
import org.testng.annotations.*;

import java.lang.reflect.code.type.ArrayType;
import java.lang.reflect.code.type.ClassType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.reflect.code.type.TypeVarRef;
import java.lang.reflect.code.type.WildcardType.BoundKind;
import java.util.ArrayList;
import java.util.List;

@Test
public class TestErasure {

    @Test(dataProvider = "typesAndErasures")
    public void testErasure(String testName, TypeAndErasure<?> typeAndErasure) {
        assertEquals(typeAndErasure.type.erasure(), typeAndErasure.erasure);
        assertEquals(typeAndErasure.type.toBasicType(), typeAndErasure.erasure.toBasicType());
        assertEquals(typeAndErasure.type.toNominalDescriptor(), typeAndErasure.erasure.toNominalDescriptor());
    }

    @DataProvider
    public static Object[][] typesAndErasures() {
        List<TypeAndErasure<?>> typeAndErasures = new ArrayList<>();
        typeAndErasures.addAll(primitives());
        typeAndErasures.addAll(references());
        typeAndErasures.addAll(genericReferences());
        typeAndErasures.addAll(arrays());
        typeAndErasures.addAll(typeVars());
        return typeAndErasures.stream()
                .map(t -> new Object[] { t.type.toString(), t })
                .toArray(Object[][]::new);
    }

    static List<TypeAndErasure<PrimitiveType>> primitives() {
        return List.of(
                new TypeAndErasure<>(JavaType.BOOLEAN, JavaType.BOOLEAN),
                new TypeAndErasure<>(JavaType.CHAR, JavaType.CHAR),
                new TypeAndErasure<>(JavaType.BYTE, JavaType.BYTE),
                new TypeAndErasure<>(JavaType.SHORT, JavaType.SHORT),
                new TypeAndErasure<>(JavaType.INT, JavaType.INT),
                new TypeAndErasure<>(JavaType.FLOAT, JavaType.FLOAT),
                new TypeAndErasure<>(JavaType.LONG, JavaType.LONG),
                new TypeAndErasure<>(JavaType.DOUBLE, JavaType.DOUBLE),
                new TypeAndErasure<>(JavaType.VOID, JavaType.VOID));
    }

    static List<TypeAndErasure<ClassType>> references() {
        return List.of(
                new TypeAndErasure<>(JavaType.J_L_STRING, JavaType.J_L_STRING),
                new TypeAndErasure<>(JavaType.J_L_OBJECT, JavaType.J_L_OBJECT));
    }

    static List<TypeAndErasure<ClassType>> genericReferences() {
        JavaType LIST = JavaType.type(List.class);
        List<TypeAndErasure<ClassType>> genericTypes = new ArrayList<>();
        BoundKind[] kinds = new BoundKind[] { null, BoundKind.EXTENDS, BoundKind.SUPER };
        for (BoundKind kind : kinds) {
            for (TypeAndErasure<ClassType> t : references()) {
                JavaType arg = t.type;
                if (kind != null) {
                    arg = JavaType.wildcard(kind, arg);
                }
                genericTypes.add(new TypeAndErasure<>(JavaType.parameterized(LIST, arg), LIST));
            }
            for (TypeAndErasure<PrimitiveType> t : primitives()) {
                JavaType arg = JavaType.array(t.type);
                if (kind != null) {
                    arg = JavaType.wildcard(kind, arg);
                }
                genericTypes.add(new TypeAndErasure<>(JavaType.parameterized(LIST, arg), LIST));
            }
        }
        return genericTypes;
    }

    static List<TypeAndErasure<ArrayType>> arrays() {
        List<TypeAndErasure<ArrayType>> arrayTypes = new ArrayList<>();
        for (int dims = 1 ; dims <= 3 ; dims++) {
            for (TypeAndErasure<PrimitiveType> t : primitives()) {
                if (t.type.isVoid()) continue; // void is not a valid array component type
                arrayTypes.add(new TypeAndErasure<>(JavaType.array(t.type, dims), JavaType.array(t.erasure, dims)));
            }
            for (TypeAndErasure<ClassType> t : references()) {
                arrayTypes.add(new TypeAndErasure<>(JavaType.array(t.type, dims), JavaType.array(t.erasure, dims)));
            }
            for (TypeAndErasure<ClassType> t : genericReferences()) {
                arrayTypes.add(new TypeAndErasure<>(JavaType.array(t.type, dims), JavaType.array(t.erasure, dims)));
            }
        }
        return arrayTypes;
    }

    static List<TypeAndErasure<TypeVarRef>> typeVars() {
        List<TypeAndErasure<TypeVarRef>> typeVars = new ArrayList<>();
        for (int dims = 1 ; dims <= 3 ; dims++) {
            for (TypeAndErasure<ClassType> t : references()) {
                typeVars.add(new TypeAndErasure<>(JavaType.typeVarRef("X", JavaType.J_L_OBJECT, t.type), t.erasure));
            }
            for (TypeAndErasure<ClassType> t : genericReferences()) {
                typeVars.add(new TypeAndErasure<>(JavaType.typeVarRef("X", JavaType.J_L_OBJECT, t.type), t.erasure));
            }
            for (TypeAndErasure<ArrayType> t : arrays()) {
                typeVars.add(new TypeAndErasure<>(JavaType.typeVarRef("X", JavaType.J_L_OBJECT, t.type), t.erasure));
            }
        }
        return typeVars;
    }

    record TypeAndErasure<T extends JavaType>(T type, JavaType erasure) { }
}
