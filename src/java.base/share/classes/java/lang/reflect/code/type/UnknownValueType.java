package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Objects;

/**
 * An unknown value type. A value whose type is of the unknown value type has a value
 * of some other type but that value is unknown.
 */
public class UnknownValueType implements TypeElement {
    static final String NAME = "UnknownValue";

    final TypeElement valueType;

    UnknownValueType(TypeElement valueType) {
        this.valueType = valueType;
    }

    /**
     * {@return the unknown value type's value type}
     */
    public TypeElement valueType() {
        return valueType;
    }

    @Override
    public ExternalizedTypeElement externalize() {
        return new ExternalizedTypeElement(NAME, List.of(valueType.externalize()));
    }

    @Override
    public String toString() {
        return externalize().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof UnknownValueType that &&
                valueType.equals(that.valueType);
    }

    @Override
    public int hashCode() {
        return valueType.hashCode();
    }

    /**
     * Constructs an unknown value type.
     *
     * @param valueType the unknown value type's value type.
     * @return an unknown value type.
     */
    public static UnknownValueType unknownValueType(TypeElement valueType) {
        Objects.requireNonNull(valueType);
        return new UnknownValueType(valueType);
    }
}
