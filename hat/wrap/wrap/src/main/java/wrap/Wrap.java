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
package wrap;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public class Wrap {
    public interface Ptr{
        MemorySegment ptr();
        long sizeof();
    }
    public interface Arr extends Ptr{
        default long length(){
            return sizeof()/elementSize();
        }
        long elementSize();
    }
    public  record IntPtr(MemorySegment ptr) implements Ptr {
        public static IntPtr of(Arena arena, int value) {
            return new IntPtr(arena.allocateFrom(JAVA_INT, value));
        }

        public int set(int value) {
            ptr.set(JAVA_INT, 0, value);
            return value;
        }

        public int get() {
            return ptr.get(JAVA_INT, 0);
        }

        @Override public long sizeof(){
            return JAVA_INT.byteSize();
        }
    }

    public  record LongPtr(MemorySegment ptr)  implements Ptr{
        public static LongPtr of(Arena arena, long value) {
            return new LongPtr(arena.allocateFrom(JAVA_LONG, value));
        }

        public long set(long value) {
            ptr.set(JAVA_LONG, 0, value);
            return value;
        }

        public long get() {
            return ptr.get(JAVA_LONG, 0);
        }

        @Override
        public long sizeof(){
            return JAVA_LONG.byteSize();
        }
    }
    public  record DoublePtr(MemorySegment ptr)  implements Ptr{
        public static DoublePtr of(Arena arena, double value) {
            return new DoublePtr(arena.allocateFrom(JAVA_DOUBLE, value));
        }

        public double set(double value) {
            ptr.set(JAVA_DOUBLE, 0, value);
            return value;
        }

        public double get() {
            return ptr.get(JAVA_DOUBLE, 0);
        }

        @Override
        public long sizeof(){
            return JAVA_DOUBLE.byteSize();
        }
    }

    public  record PtrArr(MemorySegment ptr)  implements Arr{
        public static PtrArr of(Arena arena, int size) {
            return new PtrArr(arena.allocate(ADDRESS, size));
        }
        public static PtrArr of(Arena arena, MemorySegment ...memorySegments) {
            var ptrArray=  new PtrArr(arena.allocate(ADDRESS, memorySegments.length));
            for (int i = 0; i < memorySegments.length; i++) {
                ptrArray.set(i, memorySegments[i]);
            }
            return ptrArray;
        }

        public static PtrArr of(Arena arena, Ptr ...ptrs) {
            var ptrArray=  new PtrArr(arena.allocate(ADDRESS, ptrs.length));
            for (int i = 0; i < ptrs.length; i++) {
                ptrArray.set(i, ptrs[i].ptr());
            }
            return ptrArray;
        }
        public static PtrArr of(Arena arena, String ...strings) {
            var ptrArray=  new PtrArr(arena.allocate(ADDRESS, strings.length));
            for (int i = 0; i < strings.length; i++) {
                ptrArray.set(i, CStrPtr.of(arena,strings[i]).ptr());
            }
            return ptrArray;
        }


        public MemorySegment set(int idx, MemorySegment value) {
            ptr.set(AddressLayout.ADDRESS, idx* ADDRESS.byteSize(), value);
            return value;
        }

        public MemorySegment get(int idx) {
            return ptr.get(AddressLayout.ADDRESS, idx* ADDRESS.byteSize());
        }

        @Override
        public long sizeof(){
            return ptr.byteSize();
        }
        @Override public  long elementSize(){
            return AddressLayout.ADDRESS.byteSize();
        }
    }

    public  record CStrPtr(MemorySegment ptr, int len)  implements Ptr{
        public static CStrPtr of(Arena arena, int len) {
            return new CStrPtr(arena.allocate(JAVA_BYTE, len), len);
        }
        public static CStrPtr of(Arena arena, String str) {
            return new CStrPtr(arena.allocateFrom( str), str.length());
        }
        public static CStrPtr of( MemorySegment str) {
            return new CStrPtr(str, (int)str.byteSize());
        }

        public String get() {
            return ptr.getString(0);
        }
        @Override
        public long sizeof(){
            return JAVA_BYTE.byteSize();
        }

        @Override public String toString(){
            return get();
        }
    }

    public record FloatPtr(MemorySegment ptr)  implements Ptr{
        public static FloatPtr of(Arena arena, float value) {
            return new FloatPtr(arena.allocateFrom(JAVA_FLOAT, value));
        }

        public float set(float value) {
            ptr.set(JAVA_FLOAT, 0, value);
            return value;
        }

        public float get() {
            return ptr.get(JAVA_FLOAT, 0);
        }

        @Override
        public long sizeof(){
            return JAVA_FLOAT.byteSize();
        }
    }
    public record ShortPtr(MemorySegment ptr)  implements Ptr{
        public static ShortPtr of(Arena arena, short value) {
            return new ShortPtr(arena.allocateFrom(JAVA_SHORT, value));
        }

        public short set(short value) {
            ptr.set(JAVA_SHORT, 0, value);
            return value;
        }

        public short get() {
            return ptr.get(JAVA_SHORT, 0);
        }

        @Override
        public long sizeof(){
            return JAVA_SHORT.byteSize();
        }
    }

    public record FloatArr(MemorySegment ptr)  implements Arr{
        public static FloatArr of(Arena arena, int length) {
            return new FloatArr(arena.allocate(JAVA_FLOAT, length));
        }
        public static FloatArr of(Arena arena, float[] floats) {
            return new FloatArr(arena.allocateFrom(JAVA_FLOAT, floats));
        }

        public float set(int idx, float value) {
            ptr.set(JAVA_FLOAT, idx*JAVA_FLOAT.byteSize(), value);
            return value;
        }

        public float get(int idx) {
            return ptr.get(JAVA_FLOAT, JAVA_FLOAT.byteSize()*idx);
        }


        @Override public  long elementSize(){
            return JAVA_FLOAT.byteSize();
        }

        @Override
        public long sizeof(){
            return ptr.byteSize();
        }
    }

    public record IntArr(MemorySegment ptr)  implements Arr{
        public static IntArr of(Arena arena, int length) {
            return new IntArr(arena.allocate(JAVA_INT, length));
        }
        public static IntArr ofValues(Arena arena, int ...values ) {
            return of(arena, values);
        }
        public static IntArr of(Arena arena, int[] floats) {
            return new IntArr(arena.allocateFrom(JAVA_INT, floats));
        }

        public int set(int idx, int value) {
            ptr.set(JAVA_INT, idx*JAVA_INT.byteSize(), value);
            return value;
        }

        public int get(int idx) {
            return ptr.get(JAVA_INT, JAVA_INT.byteSize()*idx);
        }

        @Override public  long elementSize(){
            return JAVA_INT.byteSize();
        }

        @Override
        public long sizeof(){
            return ptr.byteSize();
        }
    }

    public record Float4Arr(MemorySegment ptr)  implements Arr{
        public record float4(float x, float y, float z, float w ) {
            public static float4 of(float x, float y, float z, float w) {
                return new float4(x, y, z, w);
            }
            static public  final float4 zero = new float4(0.f, 0.f, 0.f, 0.f);
            public static float4 of() {
                return zero;
            }
            public float4 sub(float4 rhs){
                return of(x-rhs.x,y-rhs.y,z-rhs.z,w-rhs.w);
            }
            public float4 add(float4 rhs){
                return of(x+rhs.x,y+rhs.y,z+rhs.z,w+rhs.w);
            }

            public float4 mul(float rhs) {
                return of(x*rhs,y*rhs,z*rhs,w*rhs);
            }
            public float4 mul(float4 rhs) {
                return of(x* rhs.x,y* rhs.y,z* rhs.z,w*rhs.w);
            }
        }


        public static Float4Arr of(Arena arena, int length) {
            return new Float4Arr(arena.allocate(JAVA_FLOAT, length*JAVA_FLOAT.byteSize()));
        }
        public static Float4Arr of(Arena arena, float[] floats) {
            return new Float4Arr(arena.allocateFrom(JAVA_FLOAT, floats));
        }

        public float4 get(int idx){
            return float4.of(
                    ptr.get(JAVA_FLOAT, elementSize()*idx),
                    ptr.get(JAVA_FLOAT, elementSize()*idx+JAVA_FLOAT.byteSize()),
                    ptr.get(JAVA_FLOAT, elementSize()*idx+JAVA_FLOAT.byteSize()*2),
                    ptr.get(JAVA_FLOAT, elementSize()*idx+JAVA_FLOAT.byteSize()*3));

        }
        public void set(int idx, float4 f4) {
            ptr.set(JAVA_FLOAT, idx*elementSize(), f4.x());
            ptr.set(JAVA_FLOAT, idx*elementSize()+JAVA_FLOAT.byteSize(), f4.y());
            ptr.set(JAVA_FLOAT, idx*elementSize()+JAVA_FLOAT.byteSize()*2, f4.z());
            ptr.set(JAVA_FLOAT, idx*elementSize()+JAVA_FLOAT.byteSize()*3, f4.w());
        }
        public float setx(int idx, float value) {
            ptr.set(JAVA_FLOAT, idx*elementSize(), value);
            return value;
        }
        public float sety(int idx, float value) {
            ptr.set(JAVA_FLOAT, idx*elementSize()+JAVA_FLOAT.byteSize(), value);
            return value;
        }
        public float setz(int idx, float value) {
            ptr.set(JAVA_FLOAT, idx*elementSize()+JAVA_FLOAT.byteSize()*2, value);
            return value;
        }
        public float setw(int idx, float value) {
            ptr.set(JAVA_FLOAT, idx*elementSize()+JAVA_FLOAT.byteSize()*3, value);
            return value;
        }

        public float getx(int idx) {
            return ptr.get(JAVA_FLOAT, elementSize()*idx);
        }
        public float gety(int idx) {
            return ptr.get(JAVA_FLOAT, elementSize()*idx+JAVA_FLOAT.byteSize());
        }
        public float getz(int idx) {
            return ptr.get(JAVA_FLOAT, elementSize()*idx+JAVA_FLOAT.byteSize()*2);
        }
        public float getw(int idx) {
            return ptr.get(JAVA_FLOAT, elementSize()*idx+JAVA_FLOAT.byteSize()*3);
        }

        @Override public  long elementSize(){
            return JAVA_FLOAT.byteSize()*4;
        }

        @Override
        public long sizeof(){
            return ptr.byteSize();
        }
    }

    public static void dump(MemorySegment s, int bytes){
        char[] chars = new char[16];
        boolean end=false;
        for (int i = 0; !end && i < bytes; i++) {
            int signed = s.get(ValueLayout.JAVA_BYTE, i);
            int unsigned =  ((signed<0)?signed+256:signed)&0xff;
            chars[i%16] = (char)unsigned;

            System.out.printf("%02x ", unsigned);
            if (unsigned == 0){
                end=true;
            }
            if (i>0 && i%16==0){
                System.out.print(" | ");
                for (int c=0; c<16; c++){
                    if (chars[c]<32){
                        System.out.print(switch (chars[c]){
                            case '\0'->"\\0";
                            case '\n'->"\\n";
                            case '\r'->"\\r";
                            default -> chars[c]+"";
                        });

                    }else {
                        System.out.print(chars[c]);
                    }
                }
                System.out.println();
            }
        }
    }
}
