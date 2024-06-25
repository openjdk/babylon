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


import hat.Schema;
import hat.ifacemapper.accessor.AccessorInfo;
import hat.ifacemapper.accessor.Accessors;
import hat.ifacemapper.accessor.ValueType;
import hat.ifacemapper.component.Util;
import jdk.internal.vm.annotation.Stable;

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassHierarchyResolver;
import java.lang.constant.ClassDesc;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;
import java.util.function.Function;

import static java.lang.classfile.ClassFile.ClassHierarchyResolverOption;

/**
 * A mapper that is matching components of an interface with elements in a GroupLayout.
 */
public final class SegmentInterfaceMapper<T>
        extends AbstractSegmentMapper<T>
        implements SegmentMapper<T> {

    private static final MethodHandles.Lookup LOCAL_LOOKUP = MethodHandles.lookup();

    @Stable
    private final Class<T> implClass;
    @Stable
    private final MethodHandle getHandle;
    @Stable
    private final MethodHandle setHandle;
    @Stable
    // Capability to extract the segment from an instance of the generated implClass
    private final MethodHandle segmentGetHandle;
    @Stable
    // Capability to extract the offset from an instance of the generated implClass
    private final MethodHandle offsetGetHandle;
    private final List<AffectedMemory> affectedMemories;

    private SegmentInterfaceMapper(MethodHandles.Lookup lookup,
                                   Class<T> type,
                                   GroupLayout layout,
                                   HatData hatData,
                                   boolean leaf,
                                   List<AffectedMemory> affectedMemories) {
        super(lookup, type, layout, hatData,leaf,
                ValueType.INTERFACE, MapperUtil::requireImplementableInterfaceType, Accessors::ofInterface);
        this.affectedMemories = affectedMemories;

        // Add affected memory for all the setters seen on this level (mutation)
        accessors().stream(AccessorInfo.AccessorType.SETTER)
                .map(AffectedMemory::from)
                .forEach(affectedMemories::add);

        this.implClass = generateClass();
        this.getHandle = computeGetHandle();
        this.setHandle = computeSetHandle();

        try {
            this.segmentGetHandle = lookup.unreflect(implClass.getMethod(MapperUtil.SECRET_SEGMENT_METHOD_NAME))
                    .asType(MethodType.methodType(MemorySegment.class, Object.class));
            this.offsetGetHandle = lookup.unreflect(implClass.getMethod(MapperUtil.SECRET_OFFSET_METHOD_NAME))
                    .asType(MethodType.methodType(long.class, Object.class));
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        // No need for this now
        this.accessors = null;
    }

    @Override
    public MethodHandle getHandle() {
        return getHandle;
    }

    @Override
    public MethodHandle setHandle() {
        return setHandle;
    }

    @Override
    public Optional<MemorySegment> segment(T source) {
        if (implClass == source.getClass()) {
            try {
                return Optional.of((MemorySegment) segmentGetHandle.invokeExact(source));
            } catch (Throwable _) {
            }
        }
        return Optional.empty();
    }

    @Override
    public OptionalLong offset(T source) {
        // Implicit null check of source
        if (implClass == source.getClass()) {
            try {
                return OptionalLong.of((long) offsetGetHandle.invokeExact(source));
            } catch (Throwable _) {
            }
        }
        return OptionalLong.empty();
    }

    @Override
    public <R> SegmentMapper<R> map(Class<R> newType, Function<? super T, ? extends R> toMapper) {
        return new Mapped<>(lookup(), newType ,layout(),hatData(), getHandle(), toMapper);
    }

    // @Override
    //  public <R> SegmentMapper<R> map(Class<R> newType,
    //Function<? super T, ? extends R> toMapper,
    // Function<? super R, ? extends T> fromMapper) {
    //  throw twoWayMappersUnsupported();
    //  }

    @Override
    protected MethodHandle computeGetHandle() {
        try {
            // (MemorySegment, long)void
            var ctor = lookup().findConstructor(implClass, MethodType.methodType(void.class, MemorySegment.class, GroupLayout.class, HatData.class,
            long.class));

            // try? var ctor = lookup().findConstructor(implClass, MethodType.methodType(void.class, MemorySegment.class, long.class));
            // -> (MemorySegment, long)Object
            ctor = ctor.asType(ctor.type().changeReturnType(Object.class));
            return ctor;
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("Unable to find constructor for " + implClass, e);
        }
    }

    // This method will return a MethodHandle that will update memory that
    // is mapped to a setter. Memory that is not mapped to a setter will be
    // unaffected.
    @Override
    protected MethodHandle computeSetHandle() {
        List<AffectedMemory> fragments = affectedMemories.stream()
                .sorted(Comparator.comparingLong(AffectedMemory::offset))
                .toList();

        fragments = AffectedMemory.coalesce(fragments);

        try {
            return switch (fragments.size()) {
                case 0 -> MethodHandles.empty(Util.SET_TYPE);
                case 1 -> {
                    MethodType mt = MethodType.methodType(void.class, MemorySegment.class, long.class, Object.class);
                    yield LOCAL_LOOKUP.findVirtual(SegmentInterfaceMapper.class, "setAll", mt)
                            .bindTo(this);
                }
                default -> {
                    MethodType mt = MethodType.methodType(void.class, MemorySegment.class, long.class, Object.class, List.class);
                    MethodHandle mh = LOCAL_LOOKUP.findVirtual(SegmentInterfaceMapper.class, "setFragments", mt)
                            .bindTo(this);
                    yield MethodHandles.insertArguments(mh, 3, fragments);
                }
            };
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("Unable to find setter", e);
        }
    }

    List<AffectedMemory> affectedMemories() {
        return affectedMemories;
    }

    // Private methods and classes

    private Class<T> generateClass() {
        String packageName = lookup().lookupClass().getPackageName();
        String className = packageName.isEmpty()
                ? ""
                : packageName + ".";
        className = className + type().getSimpleName() + "InterfaceMapper";
        ClassDesc classDesc = ClassDesc.of(className);
        ClassLoader loader = type().getClassLoader();

        // We need to materialize these methods so that the order is preserved
        // during generation of the class.
        List<AccessorInfo> virtualMethods = accessors().stream()
                .filter(mi -> mi.key().valueType().isVirtual())
                .toList();

        byte[] bytes = ClassFile.of(ClassHierarchyResolverOption.of(ClassHierarchyResolver.ofClassLoading(loader)))
                .build(classDesc, cb -> {
                    ByteCodeGenerator generator = ByteCodeGenerator.of(type(), classDesc, cb);

                    // public final XxInterfaceMapper implements Xx {
                    //     private final MemorySegment segment;
                    //     private final long offset;
                    generator.classDefinition();

                    // void XxInterfaceMapper(MemorySegment segment, long offset) {
                    //    this.segment = segment;
                    //    this.offset = offset;
                    // }
                    generator.constructor(layout().byteSize());

                    // MemorySegment $_$_$sEgMeNt$_$_$() {
                    //     return segment;
                    // }
                    generator.obscuredSegment();

                    // MemorySegment $_$_$lAyOuT$_$_$() {
                    //     return layout;
                    // }
                    generator.obscuredLayout();
                    // MemorySegment $_$_$bOuNdScHeMa$_$_$() {
                    //     return layout;
                    // }
                    generator.obscuredBoundSchema();

                    // long $_$_$oFfSeT$_$_$() {
                    //     return offset;
                    // }
                    generator.obscuredOffset();

                    // @Override
                    // <t> gX(c1, c2, ..., cN) {
                    //     long indexOffset = f(dimensions, c1, c2, ..., long cN);
                    //     return segment.get(JAVA_t, offset + elementOffset + indexOffset);
                    // }
                    accessors().stream(Set.of(AccessorInfo.Key.SCALAR_VALUE_GETTER, AccessorInfo.Key.ARRAY_VALUE_GETTER))
                            .forEach(generator::valueGetter);

                    // @Override
                    // void gX(c1, c2, ..., cN, <t> t) {
                    //     long indexOffset = f(dimensions, c1, c2, ..., long cN);
                    //     segment.set(JAVA_t, offset + elementOffset + indexOffset, t);
                    // }
                    accessors().stream(Set.of(AccessorInfo.Key.SCALAR_VALUE_SETTER, AccessorInfo.Key.ARRAY_VALUE_SETTER))
                            .forEach(generator::valueSetter);

                    for (int i = 0; i < virtualMethods.size(); i++) {
                        AccessorInfo a = virtualMethods.get(i);
                        switch (a.key().accessorType()) {
                            // @Override
                            // <T> T gX(long c1, long c2, ..., long cN) {
                            //     long indexOffset = f(dimensions, c1, c2, ..., long cN);
                            //     return (T) mh[x].invokeExact(segment, offset + elementOffset + indexOffset);
                            // }
                            case GETTER -> generator.invokeVirtualGetter(a, i);
                            // @Override
                            // <T> void gX(T t) {
                            //     long indexOffset = f(dimensions, c1, c2, ..., long cN);
                            //     mh[x].invokeExact(segment, offset + elementOffset + indexOffset, t);
                            // }
                            case SETTER -> generator.invokeVirtualSetter(a, i);
                        }
                    }

                    // @Override
                    // int hashCode() {
                    //     return System.identityHashCode(this);
                    // }
                    generator.hashCode_();

                    // @Override
                    // boolean equals(Object o) {
                    //     return this == o;
                    // }
                    generator.equals_();

                    //  @Override
                    //  public String toString() {
                    //      return "Foo[g0()=" + g0() + ", g1()=" + g1() + ... "]";
                    //  }
                    List<AccessorInfo> getters = accessors().stream(AccessorInfo.AccessorType.GETTER)
                            .toList();
                    generator.toString_(getters);
                });
        try {
            List<MethodHandle> classData = virtualMethods.stream()
                    .map(a -> switch (a.key()) {
                                case SCALAR_INTERFACE_GETTER, ARRAY_INTERFACE_GETTER ->
                                        mapperCache().interfaceGetMethodHandleFor(a, affectedMemories::add);
                                default -> throw new InternalError("Should not reach here " + a);
                            }
                    )
                    .toList();

            if (MapperUtil.isDebug()) {
                Path path = Path.of(classDesc.displayName() + ".class");
                try {
                    Files.write(path, bytes, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
                    System.out.println("Wrote class file " + path.toAbsolutePath());
                } catch (IOException e) {
                    System.out.println("Unable to write class file: " + path.toAbsolutePath() + " " + e.getMessage());
                }
            }

            @SuppressWarnings("unchecked")
            Class<T> c = (Class<T>) lookup()
                    .defineHiddenClassWithClassData(bytes, classData, true)
                    .lookupClass();
            return c;
        } catch (IllegalAccessException | VerifyError e) {
            throw new IllegalArgumentException("Unable to define proxy class for " + type() + " using " + layout(), e);
        }
    }

    private MethodHandle changeReturnTypeToObject(MethodHandle mh) {
        return mh.asType(mh.type().changeReturnType(Object.class));
    }

    private MethodHandle changeParam2ToObject(MethodHandle mh) {
        return mh.asType(mh.type().changeParameterType(2, Object.class));
    }

    // Invoked reflectively
    private void setAll(MemorySegment segment, long offset, T t) {
        MemorySegment srcSegment = segment(t)
                .orElseThrow(SegmentInterfaceMapper::notImplType);
        long srcOffset = offset(t)
                .orElseThrow(SegmentInterfaceMapper::notImplType);
        MemorySegment.copy(srcSegment, srcOffset, segment, offset, layout().byteSize());
    }

    // Invoked reflectively
    private void setFragments(MemorySegment segment, long offset, T t, List<AffectedMemory> fragments) {
        MemorySegment srcSegment = segment(t)
                .orElseThrow(SegmentInterfaceMapper::notImplType);
        long srcOffset = offset(t)
                .orElseThrow(SegmentInterfaceMapper::notImplType);
        for (AffectedMemory m : fragments) {
            MemorySegment.copy(srcSegment, srcOffset + m.offset(), segment, offset + m.offset(), m.size());
        }
    }

    private static IllegalArgumentException notImplType() {
        return new IllegalArgumentException("The provided object of type T is not created by this mapper.");
    }

    // Used to keep track of which memory shards gets accessed
    // by setters. We need this when computing the setHandle
    record AffectedMemory(long offset,
                          long size) {

        //  AffectedMemory {
        //   long offset;
        //  long size; // requireNonNegative(offset);
        // requireNonNegative(size);
        // }

        static AffectedMemory from(AccessorInfo mi) {
            return new AffectedMemory(mi.offset(), mi.layoutInfo().layout().byteSize());
        }

        AffectedMemory translate(long delta) {
            return new AffectedMemory(offset() + delta, size());
        }

        static List<AffectedMemory> coalesce(List<AffectedMemory> items) {
            List<AffectedMemory> result = new ArrayList<>();

            for (int i = 0; i < items.size(); i++) {
                AffectedMemory current = items.get(i);
                for (int j = i + 1; j < result.size(); j++) {
                    AffectedMemory next = items.get(j);
                    if (current.isBefore(next)) {
                        current = current.merge(next);
                    } else {
                        break;
                    }
                }
                result.add(current);
            }
            return result;
        }

        private boolean isBefore(AffectedMemory other) {
            return offset + size == other.offset();
        }

        private AffectedMemory merge(AffectedMemory other) {
            return new AffectedMemory(offset, size + other.size());
        }

    }

    public static <T> SegmentInterfaceMapper<T> create(MethodHandles.Lookup lookup,
                                                       Class<T> type,
                                                       GroupLayout layout,
                                                       HatData hatData) {
        return new SegmentInterfaceMapper<>(lookup, type,  layout, hatData, false, new ArrayList<>());
    }

    // Mapping

    /**
     * This class models composed record mappers.
     *
     * @param lookup    to use for reflective operations
     * @param type      new type to map to/from
     * @param layout    original layout
     * @param getHandle for get operations
     * @param toMapper  a function that goes from T to R
     * @param <T>       original mapper type
     * @param <R>       composed mapper type
     */
    record Mapped<T, R>(
            MethodHandles.Lookup lookup,
            @Override Class<R> type,
            @Override GroupLayout layout,
            @Override HatData hatData,
            @Override MethodHandle getHandle,
            Function<? super T, ? extends R> toMapper
    ) implements SegmentMapper<R> {

        static final MethodHandle SET_OPERATIONS_UNSUPPORTED;

        static {
            try {
                MethodType methodType = MethodType.methodType(void.class, MemorySegment.class, long.class, Object.class);
                SET_OPERATIONS_UNSUPPORTED = LOCAL_LOOKUP.findStatic(Mapped.class, "setOperationsUnsupported", methodType);
            } catch (ReflectiveOperationException e) {
                throw new ExceptionInInitializerError(e);
            }
        }

        Mapped(MethodHandles.Lookup lookup,
               Class<R> type,
               GroupLayout layout,
               HatData hatData,
               MethodHandle getHandle,
               Function<? super T, ? extends R> toMapper
        ) {
            this.lookup = lookup;
            this.type = type;
            this.hatData =hatData;
            this.layout = layout;
            this.toMapper = toMapper;
            MethodHandle toMh = findVirtual("mapTo").bindTo(this);
            this.getHandle = MethodHandles.filterReturnValue(getHandle, toMh);
        }

        @Override
        public MethodHandle setHandle() {
            return SET_OPERATIONS_UNSUPPORTED;
        }

        @Override
        public <R1> SegmentMapper<R1> map(Class<R1> newType,
                                          Function<? super R, ? extends R1> toMapper) {
            return new Mapped<>(lookup, newType,  layout(), hatData(), getHandle(), toMapper);
        }

        // Used reflective when obtaining a MethodHandle
        R mapTo(T t) {
            return toMapper.apply(t);
        }

        // Used reflective when obtaining a MethodHandle
        /*T mapFrom(R r) {
            return fromMapper.apply(r);
        }*/

        private static MethodHandle findVirtual(String name) {
            try {
                var mt = MethodType.methodType(Object.class, Object.class);
                return LOCAL_LOOKUP.findVirtual(Mapped.class, name, mt);
            } catch (ReflectiveOperationException e) {
                // Should not happen
                throw new InternalError(e);
            }
        }

        private static void setOperationsUnsupported(MemorySegment s, long o, Object t) {
            throw new UnsupportedOperationException("SegmentMapper::set operations are not supported for mapped interface mappers");
        }

    }


}