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

import hat.Accelerator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_CHAR;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;


public interface ArgArray extends IncompleteBuffer {
    interface Arg {
        interface Value {
            interface Buf {
                MemoryLayout layout = MemoryLayout.structLayout(
                        ADDRESS.withName("address"), // segment ptr
                        JAVA_LONG.withName("bytes"),
                        ADDRESS.withName("vendorPtr"), // CLBuf *buf;   CUdeviceptr buf; void *buf ?
                        JAVA_BYTE.withName("access"), //0=??/1=RO/2=WO/3=RW
                        JAVA_BYTE.withName("state"), //0=UNKNOWN/1=GPUDIRTY/2=JAVADIRTY
                        MemoryLayout.paddingLayout(16 - JAVA_BYTE.byteSize() - JAVA_BYTE.byteSize())
                ).withName(Buf.class.getSimpleName());

                MemorySegment address();

                void address(MemorySegment address);

                long bytes();

                void bytes(long bytes);

                MemorySegment vendorPtr();

                void vendorPtr(MemorySegment vendorPtr);

                byte access();

                void access(byte access);

                byte state();

                void state(byte state);

            }

            MemoryLayout layout = MemoryLayout.unionLayout(
                    JAVA_BOOLEAN.withName("z1"),
                    JAVA_BYTE.withName("s8"),
                    JAVA_CHAR.withName("u16"),
                    JAVA_SHORT.withName("s16"),
                    JAVA_INT.withName("s32"),
                    JAVA_INT.withName("u32"),
                    JAVA_FLOAT.withName("f32"),
                    JAVA_LONG.withName("s64"),
                    JAVA_LONG.withName("u64"),
                    JAVA_DOUBLE.withName("f64"),
                    Buf.layout.withName("buf")
            ).withName(Value.class.getSimpleName());

            boolean z1();

            void z1(boolean z1);

            byte s8();

            void s8(byte s8);

            char u16();

            void u16(char u16);

            short s16();

            void s16(short s16);

            int u32();

            void u32(int u32);

            int s32();

            void s32(int s32);

            float f32();

            void f32(float f32);

            long u64();

            void u64(long u64);

            long s64();

            void s64(long s64);

            double f64();

            void f64(double f64);

            Buf buf();
        }

        MemoryLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("idx"),      //4
                JAVA_BYTE.withName("variant"), //5
                MemoryLayout.paddingLayout(16 - JAVA_INT.byteSize() - JAVA_BYTE.byteSize()),
                Value.layout.withName("value")
        ).withName(Arg.class.getSimpleName());

        int idx();

        void idx(int idx);

        byte variant();

        void variant(byte variant);

        Value value();

        default String asString() {
            switch (variant()) {
                case '&':
                    return Long.toHexString(u64());
                case 'F':
                    return Float.toString(f32());
                case 'I':
                    return Integer.toString(s32());
                case 'J':
                    return Long.toString(s64());
                case 'D':
                    return Double.toString(f64());
                case 'Z':
                    return Boolean.toString(z1());
                case 'B':
                    return Byte.toString(s8());
                case 'S':
                    return Short.toString(s16());
                case 'C':
                    return Character.toString(u16());
            }
            throw new IllegalStateException("what is this");
        }

        default boolean z1() {
            return value().z1();
        }

        default void z1(boolean z1) {
            variant((byte) 'Z');
            value().z1(z1);
        }

        default byte s8() {
            return value().s8();
        }

        default void s8(byte s8) {
            variant((byte) 'B');
            value().s8(s8);
        }

        default char u16() {
            return value().u16();
        }

        default void u16(char u16) {
            variant((byte) 'C');
            value().u16(u16);
        }

        default short s16() {
            return value().s16();
        }

        default void s16(short s16) {
            variant((byte) 'S');
            value().s16(s16);
        }

        default int s32() {
            return value().s32();
        }

        default void s32(int s32) {
            variant((byte) 'I');
            value().s32(s32);
        }

        default float f32() {
            return value().f32();
        }

        default void f32(float f32) {
            variant((byte) 'F');
            value().f32(f32);
        }

        default long s64() {
            return value().s64();
        }

        default void s64(long s64) {
            variant((byte) 'J');
            value().s64(s64);
        }

        default long u64() {
            return value().u64();
        }

        default void u64(long u64) {
            variant((byte) '&');
            value().u64(u64);
        }

        default double f64() {
            return value().f64();
        }

        default void f64(double f64) {
            variant((byte) 'D');
            value().f64(f64);
        }
    }

    static ArgArray create(Accelerator accelerator, MemoryLayout... layouts) {


        ArgArray argArray = SegmentMapper.of(accelerator.lookup, ArgArray.class,
                JAVA_INT.withName("argc"),
                MemoryLayout.paddingLayout(16 - JAVA_INT.byteSize()),
                MemoryLayout.sequenceLayout(layouts.length, Arg.layout.withName(Arg.class.getSimpleName())).withName("arg"),
                ADDRESS.withName("vendorPtr")
        ).allocate(accelerator.backend.arena());
        argArray.argc(layouts.length);
        for (int i = 0; i < layouts.length; i++) {
            MemoryLayout layout = layouts[i];
            Arg arg = argArray.arg(i);
            arg.idx(i);
            if (layout.equals(JAVA_BOOLEAN)) {
                arg.variant((byte) 'Z');
            } else if (layout.equals(JAVA_BYTE)) {
                arg.variant((byte) 'B');
            } else if (layout.equals(JAVA_CHAR)) {
                arg.variant((byte) 'C');
            } else if (layout.equals(JAVA_SHORT)) {
                arg.variant((byte) 'S');
            } else if (layout.equals(JAVA_INT)) {
                arg.variant((byte) 'I');
            } else if (layout.equals(JAVA_LONG)) {
                arg.variant((byte) 'J');
            } else if (layout.equals(JAVA_FLOAT)) {
                arg.variant((byte) 'F');
            } else if (layout.equals(JAVA_DOUBLE)) {
                arg.variant((byte) 'D');
            } else if (layout instanceof GroupLayout) {
                arg.variant((byte) '&');
            }
        }
        return argArray;
    }

    static String valueLayoutToSchemaString(ValueLayout valueLayout) {
        String descriptor = valueLayout.carrier().descriptorString();
        String schema = switch (descriptor) {
            case "Z" -> "Z";
            case "B" -> "S";
            case "C" -> "U";
            case "S" -> "S";
            case "I" -> "S";
            case "F" -> "F";
            case "D" -> "F";
            case "J" -> "S";
            default -> throw new IllegalStateException("Unexpected value: " + descriptor);
        } + valueLayout.byteSize() * 8;
        return (valueLayout.order().equals(ByteOrder.LITTLE_ENDIAN)) ? schema.toLowerCase() : schema;
    }

    static void update(ArgArray argArray, Object... args) {
        for (int i = 0; i < args.length; i++) {
            Object argObject = args[i];
            Arg arg = argArray.arg(i);
            arg.idx(i);
            switch (argObject) {
                case Boolean z1 -> arg.z1(z1);
                case Byte s8 -> arg.s8(s8);
                case Short s16 -> arg.s16(s16);
                case Character u16 -> arg.u16(u16);
                case Float f32 -> arg.f32(f32);
                case Integer s32 -> arg.s32(s32);
                case Long s64 -> arg.s64(s64);
                case Double f64 -> arg.f64(f64);
                case Buffer buffer -> {
                    MemorySegment segment = Buffer.getMemorySegment(buffer);
                    arg.variant((byte) '&');
                    Arg.Value value = arg.value();
                    Arg.Value.Buf buf = value.buf();
                    buf.address(segment);
                    buf.bytes(segment.byteSize());
                }
                default -> throw new IllegalStateException("Unexpected value: " + argObject);
            }

        }
    }

    static ArgArray create(Accelerator accelerator, Object... args) {
        String[] schemas = new String[args.length];
        StringBuilder argSchema = new StringBuilder();
        argSchema.append(args.length);
        for (int i = 0; i < args.length; i++) {
            Object argObject = args[i];
            schemas[i] = switch (argObject) {
                case Boolean z1 -> "(?:z1)";
                case Byte s8 -> "(?:s8)";
                case Short s16 -> "(?:s16)";
                case Character u16 -> "(?:u16)";
                case Float f32 -> "(?:f32)";
                case Integer s32 -> "(?:s32)";
                case Long s64 -> "(?:s64)";
                case Double f64 -> "(?:f64)";
                case CompleteBuffer buffer -> "(!:" + buffer.schema()+")";
                case IncompleteBuffer buffer -> "(?:" + buffer.schema()+")";
                default -> throw new IllegalStateException("Unexpected value: " + argObject);
            };
            if (i > 0) {
                argSchema.append(",");
            }
            argSchema.append(schemas[i]);
        }
        String schemaStr = argSchema.toString();

        ArgArray argArray = SegmentMapper.of(accelerator.lookup, ArgArray.class,
                JAVA_INT.withName("argc"),
                MemoryLayout.paddingLayout(16 - JAVA_INT.byteSize()),
                MemoryLayout.sequenceLayout(args.length, Arg.layout).withName("arg"),
                ADDRESS.withName("vendorPtr"),
                JAVA_INT.withName("schemaLen"),
                MemoryLayout.sequenceLayout(schemaStr.length() + 1, JAVA_BYTE).withName("schemaBytes")
        ).allocate(accelerator.backend.arena());
        argArray.argc(args.length);
        argArray.schemaBytes(schemaStr);
        for (int i = 0; i < args.length; i++) {
            Object argObject = args[i];
            Arg arg = argArray.arg(i);
            arg.idx(i);
            switch (argObject) {
                case Boolean z1 -> arg.z1(z1);
                case Byte s8 -> arg.s8(s8);
                case Short s16 -> arg.s16(s16);
                case Character u16 -> arg.u16(u16);
                case Float f32 -> arg.f32(f32);
                case Integer s32 -> arg.s32(s32);
                case Long s64 -> arg.s64(s64);
                case Double f64 -> arg.f64(f64);
                case Buffer buffer -> {
                    MemorySegment segment = Buffer.getMemorySegment(buffer);
                    arg.variant((byte) '&');
                    Arg.Value value = arg.value();
                    Arg.Value.Buf buf = value.buf();
                    buf.address(segment);
                    buf.bytes(segment.byteSize());
                }
                default -> throw new IllegalStateException("Unexpected value: " + argObject);
            }

        }
        return argArray;
    }

    int argc();

    void argc(int argc);

    int schemaLen();

    void schemaLen(int schemaLen);

    byte schemaBytes(long idx);

    void schemaBytes(long idx, byte b);

    default String schemaBytes() {
        byte[] bytes = new byte[schemaLen() + 1];
        for (int i = 0; i < schemaLen(); i++) {
            bytes[i] = schemaBytes(i);
        }
        bytes[bytes.length - 1] = '0';
        return new String(bytes);
    }

    default void schemaBytes(String schemaStr) {
        byte[] schemaStrBytes = schemaStr.getBytes();
        schemaLen(schemaStrBytes.length);
        // TODO:we should be able to copy into the segment here ;)
        for (int i = 0; i < schemaStrBytes.length; i++) {
            schemaBytes(i, schemaStrBytes[i]);
        }
        schemaBytes(schemaStrBytes.length, (byte) 0);
    }

    Arg arg(long idx);

    MemorySegment vendorPtr();

    void vendorPtr(MemorySegment vendorPtr);


    default String dump() {
        StringBuilder dump = new StringBuilder();
        dump.append("SchemaBytes:").append(schemaBytes()).append("\n");
        for (int argIndex = 0; argIndex < argc(); argIndex++) {
            Arg arg = arg(argIndex);
            dump.append(arg.asString()).append("\n");
        }
        return dump.toString();
    }

}
