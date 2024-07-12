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

package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.ifacemapper.accessor.AccessorInfo;
import hat.ifacemapper.accessor.Accessors;
import hat.util.Result;

import java.lang.constant.ClassDesc;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_CHAR;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class MapperUtil {

    private MapperUtil() {
    }

    private static final boolean DEBUG =
            Boolean.getBoolean("hat.ifacemapper.debug");

    public static final String SECRET_SEGMENT_METHOD_NAME = "$_$_$sEgMeNt$_$_$";
    public static final String SECRET_LAYOUT_METHOD_NAME = "$_$_$lAyOuT$_$_$";
    public static final String SECRET_BOUND_SCHEMA_METHOD_NAME = "$_$_$bOuNdScHeMa$_$_$";
    public static final String SECRET_OFFSET_METHOD_NAME = "$_$_$oFfSeT$_$_$";

    public static boolean isDebug() {
        return DEBUG;
    }

    public static <T> Class<T> requireImplementableInterfaceType(Class<T> type) {
        Objects.requireNonNull(type);
        if (!type.isInterface()) {
            throw newIae(type, "not an interface");
        }
        if (type.isHidden()) {
            throw newIae(type, "a hidden interface");
        }
        if (type.isSealed()) {
            throw newIae(type, "a sealed interface");
        }
        assertNotDeclaringTypeParameters(type);
        return type;
    }

    private static void assertNotDeclaringTypeParameters(Class<?> type) {
        if (type.getTypeParameters().length != 0) {
            throw newIae(type, "directly declaring type parameters: " + type.toGenericString());
        }
    }

    static IllegalArgumentException newIae(Class<?> type, String trailingInfo) {
        return new IllegalArgumentException(type.getName() + " is " + trailingInfo);
    }

    public static ClassDesc desc(Class<?> clazz) {
        return clazz.describeConstable()
                .orElseThrow();
    }

    public static boolean isSegmentMapperDiscoverable(Class<?> type, Method method) {
        return SegmentMapper.Discoverable.class.isAssignableFrom(type) &&
                method.getParameterCount() == 0 &&
                (method.getReturnType() == MemorySegment.class && method.getName().equals("segment") ||
                        method.getReturnType() == MemoryLayout.class && method.getName().equals("layout") ||
                        method.getReturnType() == BoundSchema.class && method.getName().equals("boundSchema") ||
                        method.getReturnType() == long.class && method.getName().equals("offset"));
    }

    static void assertMappingsCorrectAndTotal(Class<?> type,
                                              GroupLayout layout,
                                              Accessors accessors) {

        var nameMappingCounts = layout.memberLayouts().stream()
                .map(MemoryLayout::name)
                .flatMap(Optional::stream)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));

        List<AccessorInfo> allMethods = accessors.stream().toList();

        // Make sure we have all components distinctly mapped
        for (AccessorInfo component : allMethods) {
            String name = component.method().getName();
            switch (nameMappingCounts.getOrDefault(name, 0L).intValue()) {
                case 0 -> throw new IllegalArgumentException("No mapping for " +
                        type.getName() + "." + name +
                        " in layout " + layout);
                case 1 -> { /* Happy path */ }
                default -> throw new IllegalArgumentException("Duplicate mappings for " +
                        type.getName() + "." + name +
                        " in layout " + layout);
            }
        }

        // Make sure all methods of the type are mapped (totality)
        Set<Method> accessorMethods = allMethods.stream()
                .map(AccessorInfo::method)
                .collect(Collectors.toSet());

        var typeMethods = Arrays.stream(type.getMethods())
                .filter(m -> !MapperUtil.isSegmentMapperDiscoverable(type, m))
                .filter(m -> Modifier.isAbstract(m.getModifiers()))
                .toList();

        var missing = typeMethods.stream()
                .filter(Predicate.not(accessorMethods::contains))
                .toList();

        if (!missing.isEmpty()) {
            throw new IllegalArgumentException("Unable to map methods: " + missing);
        }

    }

    public static MemoryLayout primitiveToLayout(Class<?> type) {
        if (type == Integer.TYPE) {
            return JAVA_INT;
        } else if (type == Float.TYPE) {
            return JAVA_FLOAT;
        } else if (type == Long.TYPE) {
            return JAVA_LONG;
        } else if (type == Double.TYPE) {
            return JAVA_DOUBLE;
        } else if (type == Short.TYPE) {
            return JAVA_SHORT;
        } else if (type == Character.TYPE) {
            return JAVA_CHAR;
        } else if (type == Byte.TYPE) {
            return JAVA_BYTE;
        } else if (type == Boolean.TYPE) {
            return JAVA_BOOLEAN;
        } else {
            throw new IllegalStateException("Expecting primitive   " + type);
        }
    }


    static boolean isMappableIface(Class<?> clazz) {
        return  MappableIface.class.isAssignableFrom(clazz);
    }
    static boolean isBuffer(Class<?> clazz) {
        return  Buffer.class.isAssignableFrom(clazz);
    }

    static boolean isStruct(Class<?> clazz) {
        return  Buffer.Struct.class.isAssignableFrom(clazz);
    }
    public static boolean isMemorySegment(Class<?> clazz) {
        return  MemorySegment.class.isAssignableFrom(clazz);
    }

    static boolean isStructOrBuffer(Class<?> clazz) {
        return (isBuffer(clazz) || isStruct(clazz));
    }

    static boolean isUnion(Class<?> clazz) {
        return  Buffer.Union.class.isAssignableFrom(clazz);
    }

    static boolean isPrimitive(Class<?> clazz) {
        return  clazz.isPrimitive();
    }

    static boolean isVoid(Class<?> clazz) {
        return  clazz == Void.TYPE;
    }

    static boolean isLong(Class<?> clazz) {
        return  clazz == Long.TYPE;
    }

    /*
     * From the iface mapper
     * T foo()             getter iface|primitive  0 args                  , return T     returnType T
     * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
     * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
     * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
     */
    static Class<?> typeOf(Class<?> iface, String name) {
        var methods = iface.getDeclaredMethods();
        Result<Class<?>> typeResult = new Result<>();
        Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
            Class<?> returnType = matchingMethod.getReturnType();
            Class<?>[] paramTypes = matchingMethod.getParameterTypes();
            Class<?> thisType = null;
            if (paramTypes.length == 0 && (returnType.isInterface() || returnType.isPrimitive())) {
                thisType = returnType;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                thisType = paramTypes[0];
            } else if (paramTypes.length == 1 && isMemorySegment(paramTypes[0]) && returnType == Void.TYPE) {
                thisType = paramTypes[0];
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && (returnType.isInterface() || returnType.isPrimitive())) {
                thisType = returnType;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                thisType = paramTypes[1];
            } else {
                throw new IllegalStateException("Can't determine iface mapping type for " + matchingMethod);
            }
            if (!typeResult.isPresent() || typeResult.get().equals(thisType)) {
                typeResult.of(thisType);
            } else {
                throw new IllegalStateException("type mismatch for " + name);
            }
        });
        if (!typeResult.isPresent()) {
            throw new IllegalStateException("No type mapping for " + iface + " " + name);

        }
        return typeResult.get();
    }
}
