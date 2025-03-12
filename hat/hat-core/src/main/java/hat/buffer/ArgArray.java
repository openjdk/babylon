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
import hat.ComputeContext;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.Schema;

import java.lang.annotation.Annotation;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.nio.ByteOrder;

import static hat.buffer.ArgArray.Arg.Value.Buf.UNKNOWN_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface ArgArray extends Buffer {
    interface Arg extends Buffer.Struct{
        interface Value extends Buffer.Union{
            interface Buf extends Buffer.Struct{
                 byte UNKNOWN_BYTE=(byte)0;
                 byte RO_BYTE =(byte)1<<1;
                 byte WO_BYTE =(byte)1<<2;
                 byte RW_BYTE =RO_BYTE|WO_BYTE;
                MemorySegment address();
                void address(MemorySegment address);
                long bytes();
                void bytes(long bytes);
                byte access();
                void access(byte access);
            }
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

    int argc();


    Arg arg(long idx);

    int schemaLen();

    byte schemaBytes(long idx);
    void schemaBytes(long idx, byte b);

    Schema<ArgArray> schema = Schema.of(ArgArray.class, s->s
            .arrayLen("argc").pad(12).array("arg", arg->arg
                            .fields("idx", "variant")
                            .pad(11/*(int)(16 - JAVA_INT.byteSize() - JAVA_BYTE.byteSize())*/)
                            .field("value", value->value
                                            .fields("z1","s8","u16","s16","s32","u32","f32","s64","u64","f64")
                                                    .field("buf", buf->buf
                                                            .fields("address","bytes",/*"vendorPtr",*/"access")
                                                            .pad((int)(16 - JAVA_BYTE.byteSize()/* - JAVA_BYTE.byteSize()*/))
                                                    )
                            )
                    )
         //   .field("vendorPtr")
            .arrayLen("schemaLen").array("schemaBytes")
    );

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

    static ArgArray create(Accelerator accelerator, KernelCallGraph kernelCallGraph, Object... args) {
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
                case Buffer buffer -> "(?:" +SchemaBuilder.schema(buffer)+")";
                default -> throw new IllegalStateException("Unexpected value: " + argObject + " Did you pass an interface which is neither a Complete or Incomplete buffer");
            };
            if (i > 0) {
                argSchema.append(",");
            }
            argSchema.append(schemas[i]);
        }
        String schemaStr = argSchema.toString();
        ArgArray argArray = schema.allocate(accelerator,args.length,schemaStr.length() + 1);
        byte[] schemaStrBytes = schemaStr.getBytes();
        for (int i = 0; i < schemaStrBytes.length; i++) {
            argArray.schemaBytes(i, schemaStrBytes[i]);
        }
        argArray.schemaBytes(schemaStrBytes.length, (byte) 0);
        update(argArray,kernelCallGraph,args);
        return argArray;
    }

    static void update(ArgArray argArray, KernelCallGraph kernelCallGraph, Object... args) {
        Annotation[][] parameterAnnotations = kernelCallGraph.entrypoint.getMethod().getParameterAnnotations();
        for (int i = 0; i < args.length; i++) {
            Object argObject = args[i];
            Arg arg = argArray.arg(i); // this should be invariant, but if we are called from create it will be 0 for all
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
                    Annotation[] annotations = parameterAnnotations[i];
                    byte accessByte = UNKNOWN_BYTE;
                    if (annotations.length > 0) {
                        for (Annotation annotation : annotations) {
                            accessByte = switch (annotation) {
                                case RO ro-> Arg.Value.Buf.RO_BYTE;
                                case RW rw -> Arg.Value.Buf.RW_BYTE;
                                case WO wo -> Arg.Value.Buf.WO_BYTE;
                                default -> throw new IllegalStateException("Unexpected value: " + annotation);
                            };
                        }
                    }else{
                        throw new IllegalArgumentException("Argument " + i + " has no access annotations");
                    }
                    MemorySegment segment = Buffer.getMemorySegment(buffer);
                    arg.variant((byte) '&');
                    Arg.Value value = arg.value();
                    Arg.Value.Buf buf = value.buf();
                    buf.address(segment);
                    buf.bytes(segment.byteSize());
                    buf.access(accessByte);
                }
                default -> throw new IllegalStateException("Unexpected value: " + argObject);
            }
        }
    }


    default String getSchemaBytes() {
        byte[] bytes = new byte[schemaLen() + 1];
        for (int i = 0; i < schemaLen(); i++) {
            bytes[i] = schemaBytes(i);
        }
        bytes[bytes.length - 1] = '0';
        return new String(bytes);
    }


    default String dump() {
        StringBuilder dump = new StringBuilder();
        dump.append("SchemaBytes:").append(getSchemaBytes()).append("\n");
        for (int argIndex = 0; argIndex < argc(); argIndex++) {
            Arg arg = arg(argIndex);
            dump.append(arg.asString()).append("\n");
        }
        return dump.toString();
    }

}
