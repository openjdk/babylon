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

package oracle.code.onnx.opgen;

import oracle.code.json.*;
import oracle.code.onnx.OpSchema;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class OpSchemaParser {

    public static void main(String[] args) throws Exception {
        byte[] serSchemas = serialize(Path.of(
                "opgen/onnx-schema.json"));
        Files.write(Path.of("opgen/op-schemas.ser"), serSchemas, StandardOpenOption.CREATE_NEW);

        List<OpSchema> parse = parse(Path.of("opgen/onnx-schema.json"));
        for (OpSchema opSchema : parse) {
            for (OpSchema.Attribute attribute : opSchema.attributes()) {
                if (attribute.default_value() != null) {
                    System.out.println(attribute.name() + " : " + attribute.type() + " = " + attribute.default_value());
                }
            }

        }

    }

    static byte[] serialize(Path p) throws IOException {
        List<OpSchema> parse = parse(p);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream o = new ObjectOutputStream(baos);
        o.writeObject(parse);
        o.flush();
        return baos.toByteArray();
    }

    static List<OpSchema> parse(Path p) {
        String schemaString;
        try {
            schemaString = Files.readString(p);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        JsonValue schemaDoc = Json.parse(schemaString);
        return mapJsonArray((JsonArray) schemaDoc, OpSchema.class);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> T mapJsonValue(JsonValue v, Class<T> c, Type gt) {
        return switch (v) {
            case JsonBoolean b when c == boolean.class -> (T) (Boolean) b.value();

            case JsonNull _ when c == Object.class -> null;

            case JsonString s when c.isEnum() -> (T) Enum.valueOf((Class<Enum>) c, s.value());
            case JsonString s when c == String.class -> (T) s.value();
            case JsonString s when c == Object.class -> (T) s.value();

            case JsonNumber n when c == int.class -> (T) (Integer) n.value().intValue();
            case JsonNumber n when n.value() instanceof Integer i && c == Object.class -> (T) i;
            case JsonNumber n when n.value() instanceof Double d && c == Object.class -> (T) (Float) d.floatValue();

            case JsonArray a when c == List.class -> switch (gt) {
                case ParameterizedType pt when pt.getActualTypeArguments()[0] instanceof Class<?> tc ->
                        (T) mapJsonArray(a, tc);
                default -> throw new IllegalStateException();
            };

            case JsonObject o when Record.class.isAssignableFrom(c) -> (T) mapJsonObject(o, (Class<Record>) c);
            case JsonObject o when c == List.class -> switch (gt) {
                case ParameterizedType pt when pt.getActualTypeArguments()[0] instanceof Class<?> tc ->
                        (T) mapJsonObjectAsIfJsonArray(o, tc);
                default -> throw new IllegalStateException();
            };

            default -> throw new IllegalStateException(v + " " + c);
        };
    }

    static <T> List<T> mapJsonObjectAsIfJsonArray(JsonObject o, Class<T> ct) {
        return o.keys().values().stream().map(v -> mapJsonValue(v, ct, ct)).toList();
    }

    static <T> List<T> mapJsonArray(JsonArray a, Class<T> ct) {
        return a.values().stream().map(v -> mapJsonValue(v, ct, ct)).toList();
    }

    static <T extends Record> T mapJsonObject(JsonObject o, Class<T> r) {
        List<Object> rcInstances = new ArrayList<>();
        for (RecordComponent rc : r.getRecordComponents()) {
            JsonValue jsonValue = o.keys().get(rc.getName());
            if (jsonValue == null) {
                throw new IllegalStateException();
            }
            Object instance = mapJsonValue(jsonValue, rc.getType(), rc.getGenericType());
            rcInstances.add(instance);
        }

        Class<?>[] parameters = Stream.of(r.getRecordComponents())
                .map(RecordComponent::getType).toArray(Class[]::new);
        try {
            Constructor<T> declaredConstructor = r.getDeclaredConstructor(parameters);
            return declaredConstructor.newInstance(rcInstances.toArray());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
}
