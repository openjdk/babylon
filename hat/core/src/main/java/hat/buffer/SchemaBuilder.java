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
package hat.buffer;

import hat.codebuilders.CodeBuilder;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.UnionLayout;
import java.lang.foreign.ValueLayout;

public class SchemaBuilder extends CodeBuilder<SchemaBuilder> {
    SchemaBuilder layout(MemoryLayout layout) {
        either(layout.name().isPresent(), (_) -> identifier(layout.name().get()), (_) -> questionMark()).colon();
        switch (layout) {
            case StructLayout structLayout ->
                brace((_) -> separated(structLayout.memberLayouts(),(_)->comma(), this::layout));
            case UnionLayout unionLayout ->
                chevron((_) -> separated(unionLayout.memberLayouts(),(_)->bar(), this::layout));
            case ValueLayout valueLayout ->
                literal(ArgArray.valueLayoutToSchemaString(valueLayout));
            case PaddingLayout paddingLayout ->
                literal("x").literal(paddingLayout.byteSize());
            case SequenceLayout sequenceLayout ->
                sbrace((_) -> literal(sequenceLayout.elementCount()).colon().layout(sequenceLayout.elementLayout()));
        }
        return this;
    }

    public static String schema(Buffer buffer) {
            return new SchemaBuilder().literal(Buffer.getMemorySegment(buffer).byteSize())
                    .hash().layout(Buffer.getLayout(buffer)).toString();
    }
}
