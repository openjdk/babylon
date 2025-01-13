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
import java.util.Map;
import java.util.stream.Stream;

import static java.util.stream.Collectors.*;

public class OpSchemaParser {

    public static void main(String[] args) throws Exception {
        byte[] serSchemas = serialize(Path.of(
                "opgen/onnx-schema.json"));
        Files.write(Path.of("opgen/op-schemas.ser"), serSchemas, StandardOpenOption.CREATE_NEW);
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
        JSONValue schemaDoc = JSON.parse(schemaString);
        return mapJsonArray((JSONArray) schemaDoc, OpSchema.class);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> T mapJsonValue(JSONValue v, Class<T> c, Type gt) {
        return switch (v) {
            case JSONValue s when s.isString() && c.isEnum() -> (T) Enum.valueOf((Class<Enum>) c, s.asString());
            case JSONValue s when s.isString() && c == String.class -> (T) s.asString();
            case JSONValue s when s.isString() && c == Object.class -> (T) s.asString();
            case JSONValue n when n.isLong() && c == int.class -> (T) (Integer) (int) n.asLong();
            case JSONValue b when b.isBoolean() && c == boolean.class -> (T) (Boolean) b.asBoolean();
            case JSONArray a when c == List.class -> switch (gt) {
                case ParameterizedType pt when pt.getActualTypeArguments()[0] instanceof Class<?> tc ->
                        (T) mapJsonArray(a, tc);
                default -> throw new IllegalStateException();
            };
            case JSONObject o when Record.class.isAssignableFrom(c) -> (T) mapJsonObject(o, (Class<Record>) c);
            case JSONObject o when c == List.class -> switch (gt) {
                case ParameterizedType pt when pt.getActualTypeArguments()[0] instanceof Class<?> tc ->
                        (T) mapJsonObjectAsIfJsonArray(o, tc);
                default -> throw new IllegalStateException();
            };
            default -> throw new IllegalStateException();
        };
    }

    static <T> List<T> mapJsonObjectAsIfJsonArray(JSONObject o, Class<T> ct) {
        Map<String, JSONValue> map = o.fields().stream()
                .collect(toMap(JSONObject.Field::name, JSONObject.Field::value));
        return map.values().stream().map(v -> mapJsonValue(v, ct, ct)).toList();
    }

    static <T> List<T> mapJsonArray(JSONArray a, Class<T> ct) {
        return a.stream().map(v -> mapJsonValue(v, ct, ct)).toList();
    }

    static <T extends Record> T mapJsonObject(JSONObject o, Class<T> r) {
        Map<String, JSONValue> map = o.fields().stream()
                .collect(toMap(JSONObject.Field::name, JSONObject.Field::value));
        List<Object> rcInstances = new ArrayList<>();
        for (RecordComponent rc : r.getRecordComponents()) {
            JSONValue jsonValue = map.get(rc.getName());
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
