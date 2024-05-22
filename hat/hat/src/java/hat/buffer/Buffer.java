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

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;

public interface Buffer {
    default MemorySegment memorySegment() {
        try {
            return (MemorySegment) getClass().getDeclaredMethod("$_$_$sEgMeNt$_$_$").invoke(this);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    default GroupLayout layout() {
        try {
            return (GroupLayout) getClass().getDeclaredMethod("$_$_$lAyOuT$_$_$").invoke(this);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    default String buildSchema(StringBuilder sb, MemoryLayout layout, SequenceLayout tailSequenceLayout) {
        sb.append((layout.name().isPresent()) ? layout.name().get() : "?").append(":");
        switch (layout) {
            case GroupLayout groupLayout -> {
                String prefix = groupLayout instanceof StructLayout ? "{" : "<";
                String suffix = groupLayout instanceof StructLayout ? "}" : ">";
                String separator = groupLayout instanceof StructLayout ? "," : "|";
                sb.append(prefix);
                boolean[] first = {true};
                groupLayout.memberLayouts().forEach(l -> {
                    if (!first[0]) {
                        sb.append(separator);
                    } else {
                        first[0] = false;
                    }
                    buildSchema(sb, l, tailSequenceLayout);
                });
                sb.append(suffix);
            }
            case ValueLayout valueLayout -> {
                sb.append(ArgArray.valueLayoutToSchemaString(valueLayout));
            }
            case PaddingLayout paddingLayout -> sb.append("x").append(paddingLayout.byteSize());
            case SequenceLayout sequenceLayout -> {

                sb.append('[');
                if (sequenceLayout.equals(tailSequenceLayout) && this instanceof IncompleteBuffer) {
                    sb.append("*");
                } else {
                    sb.append(sequenceLayout.elementCount());
                }
                sb.append(":");
                buildSchema(sb, sequenceLayout.elementLayout(), tailSequenceLayout);
                sb.append(']');
            }
        }
        return sb.toString();
    }


}
