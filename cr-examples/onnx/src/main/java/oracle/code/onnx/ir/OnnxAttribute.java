package oracle.code.onnx.ir;

import jdk.incubator.code.op.ExternalizableOp;

import java.util.*;
import java.util.function.Function;

interface OnnxAttribute {
    String name();

    Class<?> type();

    Object defaultValue();

    boolean optional();

    default void process(Map<String, Object> attrs, Object value) {
        if (value instanceof Optional<?> o) {
            value = o.orElse(null);
        }
        // @@@ Parse attribute from string value
        if (type().isInstance(value)) {
            attrs.put(name(), value);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    default <T> T access(Class<T> type, Map<String, Object> attrs) {
        Object value = attrs.get(name());
        if (value == null && !optional()) {
            throw new NoSuchElementException();
        }
        return type.cast(value);
    }

    static Map<String, Object> process(ExternalizableOp.ExternalizedOp eop,
                                       Function<String, OnnxAttribute> f) {
        Map<String, Object> attrs = new HashMap<>();
        for (Map.Entry<String, Object> e : eop.attributes().entrySet()) {
            f.apply(e.getKey()).process(attrs, e.getValue());
        }
        OnnxAttribute.validateRequired(EnumSet.allOf(OnnxOps.Col2Im.Attribute.class), attrs);
        return Map.copyOf(attrs);
    }

    static void validateRequired(Collection<? extends OnnxAttribute> attrSet, Map<String, Object> attrs) {
        for (OnnxAttribute a : attrSet) {
            if (!a.optional() && !attrs.containsKey(a.name())) {
                throw new NoSuchElementException(a.name());
            }
        }
    }
}
