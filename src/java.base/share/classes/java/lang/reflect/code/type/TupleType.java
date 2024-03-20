package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * A tuple type.
 */
public final class TupleType implements TypeElement {
    static final String NAME = "Tuple";

    final List<TypeElement> componentTypes;

    TupleType(List<? extends TypeElement> componentTypes) {
        this.componentTypes = List.copyOf(componentTypes);
    }

    /**
     * {@return the tuple's component types, in order}
     */
    public List<TypeElement> componentTypes() {
        return componentTypes;
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        return new TypeDefinition(NAME, componentTypes.stream().map(TypeElement::toTypeDefinition).toList());
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
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

    /**
     * Constructs a tuple type.
     *
     * @param componentTypes the tuple type's component types.
     * @return a tuple type.
     */
    public static TupleType tupleType(List<? extends TypeElement> componentTypes) {
        Objects.requireNonNull(componentTypes);
        return new TupleType(componentTypes);
    }

    /**
     * Constructs a tuple type.
     *
     * @param componentTypes the tuple type's component types.
     * @return a tuple type.
     */
    public static TupleType tupleType(TypeElement... componentTypes) {
        return tupleType(List.of(componentTypes));
    }

    /**
     * Constructs a tuple type whose components are the types of
     * the given values.
     *
     * @param values the values.
     * @return a tuple type.
     */
    public static TupleType tupleTypeFromValues(List<? extends Value> values) {
        return tupleType(values.stream().map(Value::type).toList());
    }

    /**
     * Constructs a tuple type whose components are the types of
     * the given values.
     *
     * @param values the values.
     * @return a tuple type.
     */
    public static TupleType tupleTypeFromValues(Value... values) {
        return tupleType(Stream.of(values).map(Value::type).toList());
    }
}
