package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The symbolic description of a tuple type
 */
public final class TupleType implements TypeElement {
    static final String NAME = "Tuple";

    final List<TypeElement> componentTypes;

    TupleType(List<? extends TypeElement> componentTypes) {
        this.componentTypes = List.copyOf(componentTypes);
    }

    public List<TypeElement> componentTypes() {
        return componentTypes;
    }

    @Override
    public String toString() {
        String cs = componentTypes.stream().map(TypeElement::toString)
                .collect(Collectors.joining(",", "<", ">"));
        return NAME + cs;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof TupleType that && componentTypes.equals(that.componentTypes);
    }

    @Override
    public int hashCode() {
        return componentTypes.hashCode();
    }

    public static TupleType tupleType(List<? extends TypeElement> componentTypes) {
        Objects.requireNonNull(componentTypes);
        return new TupleType(componentTypes);
    }

    public static TupleType tupleType(TypeElement... componentTypes) {
        return tupleType(List.of(componentTypes));
    }

    public static TupleType tupleTypeFromValues(Value... values) {
        return tupleType(Stream.of(values).map(Value::type).toList());
    }

    public static TupleType typeFromValues(List<? extends Value> values) {
        return tupleType(values.stream().map(Value::type).toList());
    }

}
