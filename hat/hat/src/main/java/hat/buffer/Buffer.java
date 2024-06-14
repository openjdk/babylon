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

import hat.util.StreamCounter;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.UnionLayout;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;

import static hat.ifacemapper.MapperUtil.SECRET_LAYOUT_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_OFFSET_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_SEGMENT_METHOD_NAME;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface Buffer {
    static <T extends Buffer>MemorySegment getMemorySegment(T buffer){
        try {
            return (MemorySegment) buffer.getClass().getDeclaredMethod(SECRET_SEGMENT_METHOD_NAME).invoke(buffer);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

   static <T extends Buffer>MemoryLayout getLayout(T buffer){
       try {
           return (MemoryLayout) buffer.getClass().getDeclaredMethod(SECRET_LAYOUT_METHOD_NAME).invoke(buffer);
       } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
           throw new RuntimeException(e);
       }
   }

    static <T extends Buffer>long getOffset(T buffer){
        try {
            return (long) buffer.getClass().getDeclaredMethod(SECRET_OFFSET_METHOD_NAME).invoke(buffer);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    static <T extends Buffer> T setLength(T buffer, int length){
        Buffer.getMemorySegment(buffer).set(JAVA_INT, Buffer.getLayout(buffer).byteOffset(MemoryLayout.PathElement.groupElement("length")), length);
        return buffer;
    }

}
