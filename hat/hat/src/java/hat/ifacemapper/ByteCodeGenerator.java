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

import hat.ifacemapper.accessor.AccessorInfo;
import hat.ifacemapper.accessor.ArrayInfo;
import hat.ifacemapper.accessor.ScalarInfo;
import jdk.internal.ValueBased;


import java.lang.classfile.Annotation;
import java.lang.classfile.ClassBuilder;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.attribute.RuntimeVisibleAnnotationsAttribute;
import java.lang.constant.ClassDesc;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.DynamicConstantDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.StringConcatFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static hat.ifacemapper.MapperUtil.SECRET_LAYOUT_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_OFFSET_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_SEGMENT_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.desc;
import static java.lang.classfile.ClassFile.ACC_FINAL;
import static java.lang.classfile.ClassFile.ACC_PRIVATE;
import static java.lang.classfile.ClassFile.ACC_PUBLIC;
import static java.lang.classfile.ClassFile.ACC_SUPER;
import static java.lang.constant.ConstantDescs.BSM_CLASS_DATA_AT;
import static java.lang.constant.ConstantDescs.CD_CallSite;
import static java.lang.constant.ConstantDescs.CD_MethodHandle;
import static java.lang.constant.ConstantDescs.CD_Object;
import static java.lang.constant.ConstantDescs.CD_String;
import static java.lang.constant.ConstantDescs.CD_boolean;
import static java.lang.constant.ConstantDescs.CD_int;
import static java.lang.constant.ConstantDescs.CD_long;
import static java.lang.constant.ConstantDescs.CD_void;
import static java.lang.constant.ConstantDescs.INIT_NAME;
import static java.lang.constant.ConstantDescs.MTD_void;
import static java.lang.constant.ConstantDescs.ofCallsiteBootstrap;

@ValueBased
final class ByteCodeGenerator {

    static final String SEGMENT_FIELD_NAME = "segment";
    static final String LAYOUT_FIELD_NAME = "layout";
    static final String OFFSET_FIELD_NAME = "offset";

    private static final ClassDesc VALUE_LAYOUTS_CLASS_DESC = desc(ValueLayout.class);
    private static final ClassDesc MEMORY_SEGMENT_CLASS_DESC = desc(MemorySegment.class);
    private static final ClassDesc LAYOUT_CLASS_DESC = desc(GroupLayout.class);

    private final Class<?> type;
    private final ClassBuilder cb;
    private final ClassDesc classDesc;
    private final ClassDesc interfaceClassDesc;

    private ByteCodeGenerator(Class<?> type, ClassDesc classDesc, ClassBuilder cb) {
        this.type = type;
        this.cb = cb;
        this.classDesc = classDesc;
        this.interfaceClassDesc = desc(type);
    }

    void classDefinition() {
        // @ValueBased
        Annotation valueBased = Annotation.of(desc(ValueBased.class));
        cb.with(RuntimeVisibleAnnotationsAttribute.of(valueBased));
        // public final
        cb.withFlags(ACC_PUBLIC | ACC_FINAL | ACC_SUPER);
        // extends Object
        cb.withSuperclass(CD_Object);
        List<ClassDesc> interfaces = new ArrayList<>();
        // implements "type"
        interfaces.add(interfaceClassDesc);
        if (SegmentMapper.Discoverable.class.isAssignableFrom(type)) {
            // implements SegmentMapper.Discoverable
            interfaces.add(desc(SegmentMapper.Discoverable.class));
        }
        cb.withInterfaceSymbols(interfaces);
        // private final MemorySegment segment;
        cb.withField(SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC, ACC_PRIVATE | ACC_FINAL);
        // private final GroupLayout layout;
        cb.withField(LAYOUT_FIELD_NAME, LAYOUT_CLASS_DESC, ACC_PRIVATE | ACC_FINAL);
        // private final long offset;
        cb.withField(OFFSET_FIELD_NAME, CD_long, ACC_PRIVATE | ACC_FINAL);
    }

    void constructor(long layoutByteSize) {
        final int THIS_VAR_SLOT = 0;
        final int SEGMENT_VAR_SLOT = 1;
        final int LAYOUT_VAR_SLOT = 2;
        final int OFFSET_VAR_SLOT = 3;
        cb.withMethodBody(INIT_NAME, MethodTypeDesc.of(CD_void, MEMORY_SEGMENT_CLASS_DESC, LAYOUT_CLASS_DESC, CD_long), ACC_PUBLIC,
                cob -> cob
                    .aload(THIS_VAR_SLOT)
                    .invokespecial(CD_Object, INIT_NAME, MTD_void, false) // Call Object's constructor

                    .aload(THIS_VAR_SLOT)
                    .aload(SEGMENT_VAR_SLOT)
                    .checkcast(MEMORY_SEGMENT_CLASS_DESC)
                    .putfield(classDesc, SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC) // this.segment = segment

                    .aload(THIS_VAR_SLOT)
                    .aload(LAYOUT_VAR_SLOT)
                    .checkcast(LAYOUT_CLASS_DESC)
                    .putfield(classDesc, LAYOUT_FIELD_NAME, LAYOUT_CLASS_DESC) // this.layout = layout

                    .aload(THIS_VAR_SLOT)
                    .lload(OFFSET_VAR_SLOT) // offset
                    .ldc(layoutByteSize)    // size
                    .aload(SEGMENT_VAR_SLOT) // segment
                    .invokeinterface(desc(MemorySegment.class), "byteSize", MethodTypeDesc.of(CD_long))

                    .invokestatic(desc(Objects.class), "checkFromIndexSize", MethodTypeDesc.of(CD_long, CD_long, CD_long, CD_long))

                    .putfield(classDesc, OFFSET_FIELD_NAME, CD_long) // this.offset = offset
                    .return_()
        );

    }

    void obscuredSegment() {
        cb.withMethodBody(SECRET_SEGMENT_METHOD_NAME, MethodTypeDesc.of(MEMORY_SEGMENT_CLASS_DESC), ACC_PUBLIC, cob ->
                cob.aload(0)
                        .getfield(classDesc, SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC)
                        .areturn()
        );
    }

    void obscuredLayout() {
        cb.withMethodBody(SECRET_LAYOUT_METHOD_NAME, MethodTypeDesc.of(LAYOUT_CLASS_DESC), ACC_PUBLIC, cob ->
                cob.aload(0)
                        .getfield(classDesc, LAYOUT_FIELD_NAME, LAYOUT_CLASS_DESC)
                        .areturn()
        );
    }
    void obscuredOffset() {
        cb.withMethodBody(SECRET_OFFSET_METHOD_NAME, MethodTypeDesc.of(CD_long), ACC_PUBLIC, cob ->
                cob.aload(0)
                        .getfield(classDesc, OFFSET_FIELD_NAME, CD_long)
                        .lreturn()
        );
    }
    void valueGetter(AccessorInfo info) {
        String name = info.method().getName();
        ClassDesc returnDesc = desc(info.type());
        ScalarInfo scalarInfo = info.layoutInfo().scalarInfo().orElseThrow();

        var getDesc = MethodTypeDesc.of(returnDesc, desc(scalarInfo.interfaceType()), CD_long);
        List<ClassDesc> parameterDesc = parameterDesc(info);

        cb.withMethodBody(name, MethodTypeDesc.of(returnDesc, parameterDesc), ACC_PUBLIC, cob -> {
                    cob.aload(0)
                            .getfield(classDesc, SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC)
                            .getstatic(VALUE_LAYOUTS_CLASS_DESC, scalarInfo.memberName(), desc(scalarInfo.interfaceType()));
                    offsetBlock(cob, info, classDesc)
                            .invokeinterface(MEMORY_SEGMENT_CLASS_DESC, "get", getDesc);
                    // ireturn(), dreturn() etc.
                    info.layoutInfo().returnOp().accept(cob);
                }
        );
    }

    void valueSetter(AccessorInfo info) {
        String name = info.method().getName();
        List<ClassDesc> parameterDesc = parameterDesc(info);

        ScalarInfo scalarInfo = info.layoutInfo().scalarInfo().orElseThrow();

        var setDesc = MethodTypeDesc.of(CD_void, desc(scalarInfo.interfaceType()), CD_long, desc(info.type()));

        cb.withMethodBody(name, MethodTypeDesc.of(CD_void, parameterDesc), ACC_PUBLIC, cob -> {
                    cob.aload(0)
                            .getfield(classDesc, SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC)
                            .getstatic(VALUE_LAYOUTS_CLASS_DESC, scalarInfo.memberName(), desc(scalarInfo.interfaceType()));
                    offsetBlock(cob, info, classDesc);
                    // iload, dload, etc.
                    info.layoutInfo().paramOp().accept(cob, valueSlotNumber(parameterDesc.size()));
                    cob.invokeinterface(MEMORY_SEGMENT_CLASS_DESC, "set", setDesc)
                            .return_();
                }
        );
    }

    void invokeVirtualGetter(AccessorInfo info,
                             int boostrapIndex) {

        var name = info.method().getName();
        var returnDesc = desc(info.type());

        DynamicConstantDesc<MethodHandle> desc = DynamicConstantDesc.of(
                BSM_CLASS_DATA_AT,
                boostrapIndex
        );

        List<ClassDesc> parameterDesc = parameterDesc(info);

        /*
        Was
        public ArgList.Arg args(long var1) {
            return (ArgList.Arg)((MethodHandle)"_").invokeExact(this.segment, this.offset + 8L + Objects.checkIndex(var1, 4L) * 4L);
        }
        Now
        public ArgList.Arg args(long var1) {
            MethodHandle var10000 = (MethodHandle)"_";
            return (ArgList.Arg)this.segment.invokeExact((MethodHandle)"_", this.layout, this.offset + 8L + Objects.checkIndex(var1, 4L) * 4L);
        }

         */

        cb.withMethodBody(name, MethodTypeDesc.of(returnDesc, parameterDesc), ACC_PUBLIC, cob -> {cob
                            .ldc(desc)
                            .checkcast(desc(MethodHandle.class)) // MethodHandle
                            .aload(0)
                            .getfield(classDesc, SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC) // MemorySegment
                            .aload(0)
                            .getfield(classDesc, LAYOUT_FIELD_NAME, LAYOUT_CLASS_DESC); // Layout

                    offsetBlock(cob, info, classDesc)
                            .invokevirtual(CD_MethodHandle, "invokeExact", MethodTypeDesc.of(CD_Object, MEMORY_SEGMENT_CLASS_DESC, LAYOUT_CLASS_DESC, CD_long))
                            .checkcast(returnDesc)
                            .areturn();
                }
        );
    }

    void invokeVirtualSetter(AccessorInfo info,
                             int boostrapIndex) {

        var name = info.method().getName();
        List<ClassDesc> parameterDesc = parameterDesc(info);

        DynamicConstantDesc<MethodHandle> desc = DynamicConstantDesc.of(
                BSM_CLASS_DATA_AT,
                boostrapIndex
        );

        cb.withMethodBody(name, MethodTypeDesc.of(CD_void, parameterDesc), ACC_PUBLIC, cob -> {
                    cob.ldc(desc)
                            .checkcast(desc(MethodHandle.class)) // MethodHandle
                            .aload(0)
                            .getfield(classDesc, SEGMENT_FIELD_NAME, MEMORY_SEGMENT_CLASS_DESC); // MemorySegment
                    offsetBlock(cob, info, classDesc)
                            .aload(valueSlotNumber(parameterDesc.size())) // Record
                            .checkcast(desc(Record.class))
                            .invokevirtual(CD_MethodHandle, "invokeExact", MethodTypeDesc.of(CD_void, MEMORY_SEGMENT_CLASS_DESC, CD_long, CD_Object))
                            .return_();
                }
        );
    }

    void hashCode_() {
        cb.withMethodBody("hashCode", MethodTypeDesc.of(CD_int), ACC_PUBLIC | ACC_FINAL, cob ->
                cob.aload(0)
                        .invokestatic(desc(System.class), "identityHashCode", MethodTypeDesc.of(CD_int, CD_Object))
                        .ireturn()
        );
    }

    void equals_() {
        cb.withMethodBody("equals", MethodTypeDesc.of(CD_boolean, CD_Object), ACC_PUBLIC | ACC_FINAL, cob -> {
                    Label l0 = cob.newLabel();
                    Label l1 = cob.newLabel();
                    cob.aload(0)
                            .aload(1)
                            .if_acmpne(l0)
                            .iconst_1()
                            .goto_(l1)
                            .labelBinding(l0)
                            .iconst_0()
                            .labelBinding(l1)
                            .ireturn()
                    ;
                }
        );
    }

    void toString_(List<AccessorInfo> getters) {

        // Foo[g0()=\u0001, g1()=\u0001, ...]
        var recipe = getters.stream()
                .map(m -> m.layoutInfo().arrayInfo()
                        .map(ai -> String.format("%s()=%s%s", m.method().getName(), m.type().getSimpleName(), ai.dimensions()))
                        .orElse(String.format("%s()=\u0001", m.method().getName()))
                )
                .collect(Collectors.joining(", ", type.getSimpleName() + "[", "]"));

        List<AccessorInfo> nonArrayGetters = getters.stream()
                .filter(i -> i.layoutInfo().arrayInfo().isEmpty())
                .toList();

        DirectMethodHandleDesc bootstrap = ofCallsiteBootstrap(
                desc(StringConcatFactory.class),
                "makeConcatWithConstants",
                CD_CallSite,
                CD_String, CD_Object.arrayType()
        );

        List<ClassDesc> getDescriptions = nonArrayGetters.stream()
                .map(AccessorInfo::type)
                .map(MapperUtil::desc)
                .toList();

        DynamicCallSiteDesc desc = DynamicCallSiteDesc.of(
                bootstrap,
                "toString",
                MethodTypeDesc.of(CD_String, getDescriptions), // String, g0, g1, ...
                recipe
        );

        cb.withMethodBody("toString",
                MethodTypeDesc.of(CD_String),
                ACC_PUBLIC | ACC_FINAL,
                cob -> {
                    for (int i = 0; i < nonArrayGetters.size(); i++) {
                        var name = nonArrayGetters.get(i).method().getName();
                        cob.aload(0);
                        // Method gi:()?
                        cob.invokevirtual(classDesc, name, MethodTypeDesc.of(getDescriptions.get(i)));
                    }
                    cob.invokedynamic(desc);
                    cob.areturn();
                });
    }

    // Generate code that calculates:
    // long indexOffset = f(dimensions, c1, c2, ..., long cN); // If an array, otherwise 0
    // "this.offset + layoutOffset + indexOffset"
    private static CodeBuilder offsetBlock(CodeBuilder cob,
                                           AccessorInfo info,
                                           ClassDesc classDesc) {
        cob.aload(0)
                .getfield(classDesc, OFFSET_FIELD_NAME, CD_long); // long

        if (info.offset() != 0) {
            cob.ldc(info.offset())
                    .ladd();
        }

        // If this is an array accessor, we need to
        // compute the adjusted indices offset
        info.layoutInfo().arrayInfo().ifPresent(
                ai -> reduceArrayIndexes(cob, ai)
        );

        return cob;
    }

    // Example:
    //   The dimensions are [3, 4] and the element byte size is 8 bytes
    //   reduce(2, 3) -> 2 * 4 * 8 + 3 * 8 = 88
    // public static long reduce(long i1, long i2) {
    //     long offset = Objects.checkIndex(i1, 3) * (8 * 4) +
    //     Objects.checkIndex(i2, 4) * 8;
    //     return offset;
    // }
    private static void reduceArrayIndexes(CodeBuilder cob,
                                           ArrayInfo arrayInfo) {
        long elementByteSize = arrayInfo.elementLayout().byteSize();
        // Check parameters and push scaled offsets on the stack
        for (int i = 0; i < arrayInfo.dimensions().size(); i++) {
            long dimension = arrayInfo.dimensions().get(i);
            long factor = arrayInfo.dimensions().stream()
                    .skip(i + 1)
                    .reduce(elementByteSize, Math::multiplyExact);

            cob.lload(1 + i * 2)
                    .ldc(dimension)
                    .invokestatic(desc(Objects.class), "checkIndex", MethodTypeDesc.of(CD_long, CD_long, CD_long))
                    .ldc(factor)
                    .lmul();
        }
        // Sum their values (including the value that existed *before* this method was invoked)
        for (int i = 0; i < arrayInfo.dimensions().size(); i++) {
            cob.ladd();
        }
    }

    private static int valueSlotNumber(int parameters) {
        return (parameters - 1) * 2 + 1;
    }

    private static List<ClassDesc> parameterDesc(AccessorInfo info) {
        // If it is an array, there is a number of long parameters
        List<ClassDesc> desc = info.layoutInfo().arrayInfo()
                .map(ai -> ai.dimensions().stream().map(_ -> CD_long).toList())
                .orElse(Collections.emptyList());

        if (info.key().accessorType() == AccessorInfo.AccessorType.SETTER) {
            // Add the trailing setter type
            desc = new ArrayList<>(desc);
            desc.add(desc(info.type()));
        }
        return List.copyOf(desc);
    }

    // Factory
    static ByteCodeGenerator of(Class<?> type, ClassDesc classDesc, ClassBuilder cb) {
        return new ByteCodeGenerator(type, classDesc, cb);
    }

}
