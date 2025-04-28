/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx.proto;

import java.io.RandomAccessFile;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

public record OnnxProtoModel (
        @f(1) Long irVersion,
        @f(2) String producerName,
        @f(3) String producerVersion,
        @f(4) String domain,
        @f(5) Long modelVersion,
        @f(6) String docString,
        @f(7) GraphProto graph,
        @f(8) List<OperatorSetIdProto> opsetImports,
        @f(14) List<StringStringEntryProto> metadataProps,
        @f(20) List<TrainingInfoProto> trainingInfos,
        @f(25) List<FunctionProto> functions) {

    public record Attribute (
            @f(1) String name,
            @f(2) Float f,
            @f(3) Long i,
            @f(4) byte[] s,
            @f(5) TensorProto t,
            @f(6) GraphProto g,
            @f(7) List<float[]> floats,
            @f(8) List<long[]> ints,
            @f(9) List<byte[]> strings,
            @f(10) List<TensorProto> tensors,
            @f(11) List<GraphProto> graphs,
            @f(13) String docString,
            @f(14) TypeProto tp,
            @f(15) List<TypeProto> typeProtos,
            @f(20) Integer type,
            @f(21) String refAttrName,
            @f(22) SparseTensorProto sparseTensor,
            @f(23) List<SparseTensorProto> sparseTensors) {
    }

    public record ValueInfoProto (
            @f(1) String name,
            @f(2) TypeProto type,
            @f(3) String docString,
            @f(4) List<StringStringEntryProto> metadataProps) {
    }

    public record NodeProto (
            @f(1) List<String> inputs,
            @f(2) List<String> outputs,
            @f(3) String name,
            @f(4) String opType,
            @f(5) List<Attribute> attributes,
            @f(6) String docString,
            @f(7) String domain,
            @f(8) String overload,
            @f(9) List<StringStringEntryProto> metadataProps) {
    }

    public record TrainingInfoProto (
            @f(1) GraphProto initialization,
            @f(2) GraphProto algorithm,
            @f(3) List<StringStringEntryProto> initializationBindings,
            @f(4) List<StringStringEntryProto> updateBindings) {
    }

    public record StringStringEntryProto (
            @f(1) String key,
            @f(2) String value) {
    }

    public record TensorAnnotation (
            @f(1) String tensorName,
            @f(2) List<StringStringEntryProto> quantParameterTensorNames) {
    }

    public record GraphProto (
            @f(1) List<NodeProto> nodes,
            @f(2) String name,
            @f(5) List<TensorProto> initializers,
            @f(10) String docString,
            @f(11) List<ValueInfoProto> inputs,
            @f(12) List<ValueInfoProto> outputs,
            @f(13) List<ValueInfoProto> valueInfos,
            @f(14) List<TensorAnnotation> quantizationAnnotations,
            @f(15) List<SparseTensorProto> sparseInitializers,
            @f(16) List<StringStringEntryProto> metadataProps) {
    }

    public record TensorProto (
            @f(1) List<long[]> dims,
            @f(2) Integer dataType,
            @f(3) Segment segment,
            @f(4) List<float[]> floatData,
            @f(5) List<int[]> int32Data,
            @f(6) List<byte[]> stringData,
            @f(7) List<long[]> int64Data,
            @f(8) String name,
            @f(9) byte[] rawData,
            @f(10) List<double[]> doubleData,
            @f(11) List<long[]> uint64Data,
            @f(12) String docString,
            @f(13) List<StringStringEntryProto> externalData,
            @f(14) Long dataLocation,
            @f(16) List<StringStringEntryProto> metadataProps) {

        public record Segment (
                @f(1) Long begin,
                @f(2) Long end) {
        }
    }

    public record SparseTensorProto (
            @f(1) TensorProto values,
            @f(2) TensorProto indices,
            @f(3) List<long[]> dims) {
    }

    public record TensorShapeProto (
            @f(1) List<Dimension> dims) {

        public record Dimension (
                @f(1) Long dimValue,
                @f(2) String dimParam,
                @f(3) String denotation) {
        }
    }

    public record TypeProto (
            @f(1) Tensor tensorType,
            @f(4) Sequence sequenceType,
            @f(5) Map mapType,
            @f(6) String denotation,
            @f(8) SparseTensor sparseTensorType,
            @f(9) Optional optionalType) {

        public record Tensor (
                @f(1) Integer elemType,
                @f(2) TensorShapeProto shape) {
        }

        public record Sequence (
                @f(1) TypeProto elemType) {
        }

        public record Map (
                @f(1) Integer keyType,
                @f(2) TypeProto valueType) {
        }

        public record Optional (
                @f(1) TypeProto elemType) {
        }

        public record SparseTensor (
                @f(1) Integer elemType,
                @f(2) TensorShapeProto shape) {
        }
    }

    public record OperatorSetIdProto (
            @f(1) String domain,
            @f(2) Long version) {
    }

    public record FunctionProto (
            @f(1) String name,
            @f(4) List<String> inputs,
            @f(5) List<String> outputs,
            @f(6) List<String> attributes,
            @f(7) List<NodeProto> nodes,
            @f(8) String docString,
            @f(9) List<OperatorSetIdProto> opsetImports,
            @f(10) String domain,
            @f(11) List<Attribute> attributeProtos,
            @f(12) List<ValueInfoProto> valueInfos,
            @f(13) String overload,
            @f(14) List<StringStringEntryProto> metadataProps) {
    }

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.RECORD_COMPONENT)
    @interface f {
        int value();
    }

    private static long decodeVarint(ByteBuffer data) {
        long i, shift = 0, value = 0;
        do {
            value |= ((i = data.get()) & 0x7f) << shift;
            shift += 7;
        } while ((i & 0x80) != 0);
        return value;
    }

    private static int countVarInts(ByteBuffer data) {
        long end  = decodeVarint(data);
        int start = data.position();
        end += start;
        int count = 0;
        while (data.position() < end) {
            if ((data.get() & 0x80) == 0) count++;
        }
        data.position(start);
        return count;
    }

    private static int[] readPackedInts(ByteBuffer data) {
        var ret = new int[countVarInts(data)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (int)decodeVarint(data);
        }
        return ret;
    }

    private static long[] readPackedLongs(ByteBuffer data) {
        var ret = new long[countVarInts(data)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = decodeVarint(data);
        }
        return ret;
    }

    private static float[] readPackedFloats(ByteBuffer data) {
        var ret = new float[(int)(decodeVarint(data)/4)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data.getFloat();
        }
        return ret;
    }

    private static double[] readPackedDoubles(ByteBuffer data) {
        var ret = new double[(int)(decodeVarint(data)/8)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data.getDouble();
        }
        return ret;
    }

    private static byte[] readBytes(ByteBuffer data) {
        var bytes = new byte[(int)decodeVarint(data)];
        data.get(bytes);
        return bytes;
    }

    private static Object readData(Class<?> baseType, boolean packed, ByteBuffer bb) {
        if (baseType == Integer.class) {
            return (int)decodeVarint(bb);
        } else if (baseType == int[].class) {
            return packed ? readPackedInts(bb) : new int[]{(int)decodeVarint(bb)};
        } else if (baseType == Long.class) {
            return decodeVarint(bb);
        } else if (baseType == long[].class) {
            return packed ? readPackedLongs(bb) : new long[]{decodeVarint(bb)};
        } else if (baseType == Float.class) {
            return bb.getFloat();
        } else if (baseType == float[].class) {
            return packed ? readPackedFloats(bb) : new float[] {bb.getFloat()};
        } else if (baseType == Double.class) {
            return bb.getDouble();
        } else if (baseType == double[].class) {
            return packed ? readPackedDoubles(bb) : new double[] {bb.getDouble()};
        } else if (baseType == byte[].class) {
            return readBytes(bb);
        } else if (baseType == String.class) {
            return new String(readBytes(bb));
        } else {
            var size = decodeVarint(bb);
            int limit = bb.limit();
            var data = readFrom((Class<Record>)baseType, bb.limit(bb.position() + (int)size));
            bb.limit(limit);
            return data;
        }
    }

    private static int getRecordFieldIndex(RecordComponent[] rcs, int fieldIndex) {
        for (int i = 0; i < rcs.length; i++) {
            if (rcs[i].getAnnotation(f.class).value() == fieldIndex) {
                return i;
            }
        }
        throw new IllegalArgumentException("Field index " + fieldIndex + " not found in " + rcs[0].getDeclaringRecord());
    }

    private static <T> T readFrom(Class<T> type, ByteBuffer bb) {
        Object[] fieldsData = new Object[type.getRecordComponents().length];
        while (bb.remaining() > 0) {
            long tag = decodeVarint(bb);
            RecordComponent[] rcs = type.getRecordComponents();
            int rfi = getRecordFieldIndex(rcs, (int)tag >> 3);
            boolean packed = (tag & 7) == 2;
            RecordComponent rc = rcs[rfi];
            Class<?> rcType = rc.getType();
            if (rcType == List.class) {
                List list;
                if (fieldsData[rfi] instanceof List l) {
                    list = l;
                } else {
                    list = new ArrayList();
                    fieldsData[rfi] = list;
                }
                Class baseType = (Class)((ParameterizedType)rc.getGenericType()).getActualTypeArguments()[0];
                list.add(readData(baseType, packed, bb));
            } else {
                fieldsData[rfi] = readData(rcType, packed, bb);
            }
        }
        try {
            return (T)type.getDeclaredConstructors()[0].newInstance(fieldsData);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static void print(StringBuilder out, int indent, String name, Object value, boolean skipBigData) throws ReflectiveOperationException {
        if (value == null) return;
        out.append("  ".repeat(indent)).append(name);
        switch (value) {
            case List l -> {
                out.append(name.endsWith("s") ? ":" : "s:").append(System.lineSeparator());
                for (var el : l) print(out, indent + 1, "- " + (name.endsWith("s") ? name.substring(0, name.length() - 1) : name), el, skipBigData);
            }
            case Record r -> {
                out.append(':').append(System.lineSeparator());
                for (var rc : r.getClass().getRecordComponents()) {
                    print(out, indent + 2, rc.getName(), rc.getAccessor().invoke(r), skipBigData);
                }
            }
            case byte[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case long[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case float[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case double[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case String s ->
                out.append(": \"").append(s).append('"').append(System.lineSeparator());
            default ->
                out.append(": ").append(value).append(System.lineSeparator());
        }
    }

    private static final int SKIP_LIMIT = 1000;

    private static String checkSize(int size, Supplier<String> sup, boolean skipBigData) {
        return ": " + (skipBigData && size > SKIP_LIMIT ? "# skipped " + size + " values" : sup.get()) + System.lineSeparator();
    }

    public String toText() {
        return toText(true);
    }

    public String toText(boolean skipBigData) {
        try {
            var sb = new StringBuilder();
            print(sb, 0, "OnnxProtoModel", this, skipBigData);
            return sb.toString();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static OnnxProtoModel readFrom(byte[] onnxProtoModel) {
        return readFrom(ByteBuffer.wrap(onnxProtoModel));
    }

    public static OnnxProtoModel readFrom(ByteBuffer onnxProtoModel) {
        return readFrom(OnnxProtoModel.class, onnxProtoModel.order(ByteOrder.LITTLE_ENDIAN));
    }

    public static void main(String... args) throws Exception {
        for (var fName : args) {
            try (var in = new RandomAccessFile(fName, "r")) {
                OnnxProtoModel model = readFrom(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()));
                System.out.println(model.toText());
            }
        }
    }
}
