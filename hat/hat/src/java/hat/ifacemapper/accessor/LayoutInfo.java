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

package hat.ifacemapper.accessor;

import java.lang.classfile.CodeBuilder;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.ObjIntConsumer;

public record LayoutInfo(MemoryLayout layout,
                         Optional<ScalarInfo> scalarInfo,
                         Optional<ArrayInfo> arrayInfo,
                         Consumer<CodeBuilder> returnOp,
                         ObjIntConsumer<CodeBuilder> paramOp) {

    public static LayoutInfo of(ValueLayout layout) {
        return switch (layout) {
            // Todo: Remove boolean?
            case ValueLayout.OfBoolean bo ->
                    LayoutInfo.ofScalar(bo, "JAVA_BOOLEAN", ValueLayout.OfBoolean.class, CodeBuilder::ireturn, CodeBuilder::iload);
            case ValueLayout.OfByte by ->
                    LayoutInfo.ofScalar(by, "JAVA_BYTE", ValueLayout.OfByte.class, CodeBuilder::ireturn, CodeBuilder::iload);
            case ValueLayout.OfShort sh ->
                    LayoutInfo.ofScalar(sh, "JAVA_SHORT", ValueLayout.OfShort.class, CodeBuilder::ireturn, CodeBuilder::iload);
            case ValueLayout.OfChar ch ->
                    LayoutInfo.ofScalar(ch, "JAVA_CHAR", ValueLayout.OfChar.class, CodeBuilder::ireturn, CodeBuilder::iload);
            case ValueLayout.OfInt in ->
                    LayoutInfo.ofScalar(in, "JAVA_INT", ValueLayout.OfInt.class, CodeBuilder::ireturn, CodeBuilder::iload);
            case ValueLayout.OfFloat fl ->
                    LayoutInfo.ofScalar(fl, "JAVA_FLOAT", ValueLayout.OfFloat.class, CodeBuilder::freturn, CodeBuilder::fload);
            case ValueLayout.OfLong lo ->
                    LayoutInfo.ofScalar(lo, "JAVA_LONG", ValueLayout.OfLong.class, CodeBuilder::lreturn, CodeBuilder::lload);
            case ValueLayout.OfDouble db ->
                    LayoutInfo.ofScalar(db, "JAVA_DOUBLE", ValueLayout.OfDouble.class, CodeBuilder::dreturn, CodeBuilder::dload);
            case AddressLayout ad ->
                    LayoutInfo.ofScalar(ad, "ADDRESS", AddressLayout.class, CodeBuilder::areturn, CodeBuilder::aload);
        };
    }

    public static LayoutInfo of(GroupLayout layout) {
        return new LayoutInfo(layout, Optional.empty(), Optional.empty(), CodeBuilder::areturn, CodeBuilder::aload);
    }

    public static LayoutInfo of(SequenceLayout layout) {
        ArrayInfo arrayInfo = ArrayInfo.of(layout);
        LayoutInfo elementLayoutInfo = (arrayInfo.elementLayout() instanceof ValueLayout vl)
                ? of(vl)
                : null;
        Optional<ScalarInfo> scalarInfo = Optional.ofNullable(elementLayoutInfo)
                .flatMap(li -> li.scalarInfo);
        return scalarInfo
                .map(_ -> new LayoutInfo(layout, scalarInfo, Optional.of(arrayInfo), elementLayoutInfo.returnOp(), elementLayoutInfo.paramOp())
                ).orElse(new LayoutInfo(layout, scalarInfo, Optional.of(arrayInfo), CodeBuilder::areturn, CodeBuilder::aload));

    }

    private static <T extends ValueLayout> LayoutInfo ofScalar(T layout,
                                                               String memberName,
                                                               Class<T> interfaceType,
                                                               Consumer<CodeBuilder> returnOp,
                                                               ObjIntConsumer<CodeBuilder> paramOp) {
        return new LayoutInfo(layout, Optional.of(new ScalarInfo(memberName, interfaceType)), Optional.empty(), returnOp, paramOp);
    }

}
