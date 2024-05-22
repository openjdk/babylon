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


import hat.ifacemapper.accessor.Accessors;
import hat.ifacemapper.accessor.ValueType;
import hat.ifacemapper.accessor.AccessorInfo;
import jdk.internal.vm.annotation.Stable;

import java.lang.foreign.GroupLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

abstract class AbstractSegmentMapper<T> implements SegmentMapper<T> {

    @Stable
    private final MethodHandles.Lookup lookup;
    @Stable
    private final Class<T> type;
    @Stable
    private final GroupLayout layout;
    private final boolean leaf;
    private final MapperCache mapperCache;
    protected Accessors accessors;

    protected AbstractSegmentMapper(MethodHandles.Lookup lookup,
                                    Class<T> type,
                                    GroupLayout layout,
                                    boolean leaf,
                                    ValueType valueType, // This is always ValueType.Interface... !
                                    UnaryOperator<Class<T>> typeInvariantChecker,
                                    BiFunction<Class<?>, GroupLayout, Accessors> accessorFactory) {
        this.lookup = lookup;
        this.type = typeInvariantChecker.apply(type);
        this.layout = layout;
        this.leaf = leaf;
        this.mapperCache = MapperCache.of(lookup);
        this.accessors = accessorFactory.apply(type, layout);

        List<Method> unsupportedAccessors = accessors.stream(k -> !k.isSupportedFor(valueType))
                .map(AccessorInfo::method)
                .toList();
        if (!unsupportedAccessors.isEmpty()) {
            throw new IllegalArgumentException(
                    "The following accessors are not supported for " + valueType + ": " + unsupportedAccessors);
        }
        MapperUtil.assertMappingsCorrectAndTotal(type, layout, accessors);
    }

    @Override
    public final Class<T> type() {
        return type;
    }

    @Override
    public final GroupLayout layout() {
        return layout;
    }


    @Override
    public final String toString() {
        return getClass().getSimpleName() + "[" +
                "lookup=" + lookup + ", " +
                "type=" + type + ", " +
                "layout=" + layout + "]";
    }

    // Protected methods

    protected final MethodHandles.Lookup lookup() {
        return lookup;
    }

    protected final Accessors accessors() {
        return accessors;
    }

    protected final MapperCache mapperCache() {
        return mapperCache;
    }

    protected final boolean isLeaf() {
        return leaf;
    }

    // Abstract methods

    // -> (MemorySegment, long)T if isLeaf()
    // -> (MemorySegment, long)Object if !isLeaf()
    protected abstract MethodHandle computeGetHandle();

    // (MemorySegment, long, T)void if isLeaf()
    // (MemorySegment, long, Object)void if !isLeaf()
    protected abstract MethodHandle computeSetHandle();

}