package oracle.code.onnx;

import java.io.*;
import java.util.*;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toCollection;

public final class OpSchemas {
    final SortedMap<String, SortedSet<OpSchema>> schemas;

    private OpSchemas() {
        List<OpSchema> schemas = load();
        this.schemas = schemas.stream().collect(groupingBy(
                OpSchema::name,
                TreeMap::new,
                toCollection(() -> new TreeSet<>(comparing(OpSchema::since_version).reversed())
                )));
    }

    @SuppressWarnings("unchecked")
    private static List<OpSchema> load() {
        try (InputStream is = OpSchemas.class.getResourceAsStream("op-schemas.ser")) {
            byte[] serSchemas = is.readAllBytes();
            ObjectInputStream oi = new ObjectInputStream(new ByteArrayInputStream(serSchemas));
            return (List<OpSchema>) oi.readObject();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private static final class Inner {
        static final OpSchemas INSTANCE = new OpSchemas();
    }

    static OpSchemas instance() {
        return Inner.INSTANCE;
    }

    static OpSchema get(String name) {
        SortedSet<OpSchema> opSchemas = instance().schemas.get(name);
        if (opSchemas == null) {
            throw new NoSuchElementException();
        }
        return opSchemas.getFirst();
    }

    public static void main(String[] args) {
        OpSchemas schemas = instance();

        schemas.schemas.values().stream().map(SortedSet::getFirst).forEach(opSchema -> {
            System.out.println(opSchema.name() + " " + opSchema.since_version() + " " + opSchema.type_constraints());
        });

        {
            OpSchema add = get("Add");
            System.out.println(add.name() + " " + add.since_version() + " " + add.type_constraints());
        }
    }
}
