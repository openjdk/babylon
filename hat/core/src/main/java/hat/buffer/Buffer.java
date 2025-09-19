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
import hat.ifacemapper.BufferState;
import hat.ifacemapper.MappableIface;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.InvocationTargetException;

import static hat.ifacemapper.MapperUtil.SECRET_BOUND_SCHEMA_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_LAYOUT_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_OFFSET_METHOD_NAME;
import static hat.ifacemapper.MapperUtil.SECRET_SEGMENT_METHOD_NAME;

public interface Buffer extends MappableIface {

    default int getState(){
        return BufferState.of(this).getState();
    }
    default void setState(int newState ){
         BufferState.of(this).setState(newState);
    }

    default String getStateString(){
        return BufferState.of(this).getStateString();
    }
  //  default boolean isDeviceDirty(){
    //    return BufferState.of(this).isDeviceDirty();
   // }
   // default boolean isHostChecked(){
     //   return BufferState.of(this).isHostChecked();
   // }

   // default void clearDeviceDirty(){
   //      BufferState.of(this).clearDeviceDirty();
   // }
    //default void setHostDirty(){
      //  BufferState.of(this).setHostDirty(true);
   // }

   // default void setHostChecked(){
     //   BufferState.of(this).setHostChecked(true);
   // }

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

    static <T extends Buffer> BoundSchema<?> getBoundSchema(T buffer) {
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
}
