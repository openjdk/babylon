/*
 * Copyright (c) 2024, 2024 Oracle and/or its affiliates. All rights reserved.
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

package hat.ifacemapper.accessor;

import hat.ifacemapper.MapperUtil;
import jdk.internal.ValueBased;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * This class is used to create MethodInfo objects (which hold extensive additional information
 * for a method) and to organize them in a way they can easily be retrieved.
 */
@ValueBased
public final class Accessors {

    private final Map<AccessorInfo.Key, List<AccessorInfo>> keyToAccessorMap;
    private final Map<Method, AccessorInfo> methodToAccessorMap;

    private Accessors(Map<AccessorInfo.Key, List<AccessorInfo>> keyToAccessorMap) {
        this.keyToAccessorMap = keyToAccessorMap;
        this.methodToAccessorMap = keyToAccessorMap.values().stream()
                .flatMap(Collection::stream)
                .collect(Collectors.toMap(AccessorInfo::method, Function.identity()));
    }

    public boolean isEmpty() {
        return keyToAccessorMap.isEmpty();
    }

    public Optional<AccessorInfo> get(Method method) {
        return Optional.ofNullable(methodToAccessorMap.get(method));
    }

    public AccessorInfo getOrThrow(Method method) {
        AccessorInfo accessorInfo = methodToAccessorMap.get(method);
        if (accessorInfo == null) {
            throw new IllegalArgumentException("There is no method: " + method);
        }
        return accessorInfo;
    }

    public List<AccessorInfo> get(AccessorInfo.Key key) {
        return keyToAccessorMap.getOrDefault(key, Collections.emptyList());
    }

    public Stream<AccessorInfo> stream(Set<AccessorInfo.Key> set) {
        return set.stream()
                .map(keyToAccessorMap::get)
                .filter(Objects::nonNull)
                .flatMap(Collection::stream);
    }

    public Stream<AccessorInfo> stream(AccessorInfo.AccessorType accessorType) {
        return stream(key -> key.accessorType() == accessorType);
    }

    public Stream<AccessorInfo> stream(Predicate<AccessorInfo.Key> condition) {
        return Arrays.stream(AccessorInfo.Key.values())
                .filter(condition)
                .map(keyToAccessorMap::get)
                .filter(Objects::nonNull)
                .flatMap(Collection::stream);
    }

    public Stream<AccessorInfo> stream() {
        return keyToAccessorMap.values().stream()
                .flatMap(Collection::stream);
    }

    // The order of methods should conform to the order in which names
    // appears in the layout
    public static Accessors ofInterface(Class<?> type, GroupLayout layout) {
        Map<String, List<Method>> methods = Arrays.stream(type.getMethods())
                .filter(m -> !MapperUtil.isSegmentMapperDiscoverable(type, m))
                .collect(Collectors.groupingBy(Method::getName));

        return new Accessors(layout.memberLayouts().stream()
                .map(MemoryLayout::name)
                .filter(Optional::isPresent)
                // Get the name of the element layout
                .map(Optional::get)
                // Lookup class methods using the element name
                .map(methods::get)
                // Ignore unmapped elements
                .filter(Objects::nonNull)
                // Flatten the list of Methods (e.g. getters and setters)
                .flatMap(Collection::stream)
                // Only consider abstract methods (e.g. ignore default methods)
                .filter(m -> Modifier.isAbstract(m.getModifiers()))
                .map(m -> accessorInfo(type, layout, m))
                // Retain insertion order
                .collect(Collectors.groupingBy(AccessorInfo::key, LinkedHashMap::new, Collectors.toList())));
    }

    private static AccessorInfo accessorInfo(Class<?> type, GroupLayout layout, Method method) {
        AccessorInfo.AccessorType accessorType = isGetter(method)
                ? AccessorInfo.AccessorType.GETTER
                : AccessorInfo.AccessorType.SETTER;
        return accessorInfo(type, layout, method,  accessorType);
    }


    private static AccessorInfo accessorInfo(Class<?> type,
                                             GroupLayout layout,
                                             Method method,
                                             AccessorInfo.AccessorType accessorType) {


        Class<?> targetType = (accessorType == AccessorInfo.AccessorType.GETTER)
                ? method.getReturnType()
                : getterType(method);

        ValueType valueType = valueTypeFor(method, targetType);

        var elementPath = MemoryLayout.PathElement.groupElement(method.getName());
        MemoryLayout element;
        try {
            element = layout.select(elementPath);
        } catch (IllegalArgumentException iae) {
            throw new IllegalArgumentException("Unable to resolve '" + method + "' in " + layout, iae);
        }
        var offset = layout.byteOffset(elementPath);

        return switch (element) {
            case ValueLayout vl -> {
                if (!targetType.equals(vl.carrier())) {
                    throw new IllegalArgumentException("The type " + targetType + " for method " + method +
                            "does not match " + element);
                }
                yield new AccessorInfo(AccessorInfo.Key.of(Cardinality.SCALAR, valueType, accessorType),
                        method,  targetType, LayoutInfo.of(vl), offset);
            }
            case GroupLayout gl ->
                    new AccessorInfo(AccessorInfo.Key.of(Cardinality.SCALAR, valueType, accessorType),
                            method,  targetType, LayoutInfo.of(gl), offset);
            case SequenceLayout sl -> {
                AccessorInfo info = new AccessorInfo(AccessorInfo.Key.of(Cardinality.ARRAY, valueType, accessorType)
                        , method,  targetType, LayoutInfo.of(sl), offset);

                    // This is an interface mapper so, check the array access parameter count matches
                    int noDimensions = info.layoutInfo().arrayInfo().orElseThrow().dimensions().size();
                    // The last parameter for a setter is the new value
                    int expectedParameterIndexCount = method.getParameterCount() - (accessorType == AccessorInfo.AccessorType.SETTER ? 1 : 0);
                    if (expectedParameterIndexCount != noDimensions) {
                        throw new IllegalArgumentException(
                                "Sequence layout has a dimension of " + noDimensions +
                                        " and so, the method parameter count does not" +
                                        " match for: " + method);
                    }

                yield info;
            }
            default -> throw new IllegalArgumentException("Cannot map " + element + " for " + type);
        };
    }

    private static ValueType valueTypeFor(Method method, Class<?> targetType) {
        if (targetType.isArray()) {
            return valueTypeFor(method, targetType.getComponentType());
        }
        ValueType valueType;
        if (targetType.isPrimitive() || targetType.equals(MemorySegment.class)) {
            valueType = ValueType.VALUE;
        } else if (targetType.isInterface()) {
            valueType = ValueType.INTERFACE;
        } else {
            throw new IllegalArgumentException("Type " + targetType + " is neither a primitive value or an interface: " + method);
        }
        return valueType;
    }

    private static Class<?> getterType(Method method) {
        if (method.getParameterCount() == 0) {
            throw new IllegalArgumentException("A setter must take at least one argument: " + method);
        }
        return method.getParameterTypes()[method.getParameterCount() - 1];
    }

    private static boolean isGetter(Method method) {
        return method.getReturnType() != void.class;
    }

}
