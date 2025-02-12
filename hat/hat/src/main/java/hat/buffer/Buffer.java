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


import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.lang.reflect.InvocationTargetException;

import static hat.ifacemapper.MapperUtil.SECRET_BOUND_SCHEMA_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_LAYOUT_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_OFFSET_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_SEGMENT_METHOD_NAME;
import static hat.ifacemapper.SegmentMapper.MAGIC;

public interface Buffer extends MappableIface {

    interface Union extends MappableIface {
    }

    interface Struct extends MappableIface {
    }

    static <T extends Buffer> MemorySegment getMemorySegment(T buffer) {
       try {
            return (MemorySegment) buffer.getClass().getDeclaredMethod(SECRET_SEGMENT_METHOD_NAME).invoke(buffer);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    static <T extends Buffer> BoundSchema getBoundSchema(T buffer) {
        try {
            return (BoundSchema<?>) buffer.getClass().getDeclaredMethod(SECRET_BOUND_SCHEMA_METHOD_NAME).invoke(buffer);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    static <T extends Buffer> MemoryLayout getLayout(T buffer) {
        try {
            return (MemoryLayout) buffer.getClass().getDeclaredMethod(SECRET_LAYOUT_METHOD_NAME).invoke(buffer);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    static <T extends Buffer> long getOffset(T buffer) {
        try {
            return (long) buffer.getClass().getDeclaredMethod(SECRET_OFFSET_METHOD_NAME).invoke(buffer);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    /*
     * A hack for accessing the tailbits from a memory segment.  Hopefully I can get this from
     * ifacemapper
     * See SegmentMapper allocate and backend_ffi_shared/include/shared.h

        struct ifacebufferpayload_t{
          int javaDirty;
          int gpuDirty;
          int unused[2];
        };

        struct ifacebufferbitz_t{
           long magic1; // MAGIC
           ifacebufferpayload_t payload;
           long magic2; // MAGIC
        }
     */

    record Tail(Buffer buffer, MemorySegment segment, long offset ) implements MappableIface {

        public static final long tailLength = 4 * ValueLayout.JAVA_LONG.byteSize();
        public static final long magic1Offset =0;
        public static final long magic2Offset =3 * ValueLayout.JAVA_LONG.byteSize();
        public static final long javaDirtyOffset =ValueLayout.JAVA_LONG.byteSize();
        public static final long gpuDirtyOffset =ValueLayout.JAVA_LONG.byteSize()+ValueLayout.JAVA_INT.byteSize();

        public static Tail of(Buffer buffer) {
            MemorySegment segment = getMemorySegment(buffer);
            return new Tail(buffer, segment, segment.byteSize() - tailLength);
        }
        public long magic1(){
            return segment.get(ValueLayout.JAVA_LONG, offset +magic1Offset);
        }
        public int javaDirty(){
            return segment.get(ValueLayout.JAVA_INT, offset +javaDirtyOffset);
        }
        public boolean isJavaDirty(){
            return javaDirty()!=0;
        }
        public int gpuDirty(){
            return segment.get(ValueLayout.JAVA_INT, offset +gpuDirtyOffset);
        }
        public boolean isGpuDirty(){
            return gpuDirty()!=0;
        }

        public long magic2(){
            return segment.get(ValueLayout.JAVA_LONG, offset+ magic2Offset);
        }
        public boolean isOK() {
            if (magic1() != MAGIC) {
                System.out.println("magic1=" + magic1()+ " != " + MAGIC);
                return false;
            }
            if (magic2() != MAGIC) {
                System.out.println("magic2=" + magic2() + " != " + MAGIC);
                return false;
            }
            return true;
        }
    }


    default boolean isJavaDirty(){
        var seg = getMemorySegment(this);
        if (seg != null){
            long sizeInBytes = seg.byteSize();
            long magic1 = seg.get(ValueLayout.JAVA_LONG, sizeInBytes-4*ValueLayout.JAVA_LONG.byteSize());

            if (magic1 != MAGIC){
                System.out.println("magic1=" + magic1+" != "+MAGIC);
            }else{
                System.out.println("magic1=" + magic1+" != "+MAGIC);
            }
            return true;
        } else {
            return false;
        }
    }
}
