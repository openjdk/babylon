package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Objects;

/**
 * An undefined type.
 * <p>
 * A value whose type is the undefined type holds an unknown value whose type
 * is the undefined type's value type.
 */
public class UndefinedType implements TypeElement {
    static final String NAME = "Undefined";

    final TypeElement valueType;

    UndefinedType(TypeElement valueType) {
        this.valueType = valueType;
    }

    /**
     * {@return the undefined type's value type}
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
        return o instanceof UndefinedType that &&
                valueType.equals(that.valueType);
    }

    @Override
    public int hashCode() {
        return valueType.hashCode();
    }

    /**
     * Constructs an undefined type.
     *
     * @param valueType the undefined type's value type.
     * @return an undefined type.
     */
    public static UndefinedType undefinedType(TypeElement valueType) {
        Objects.requireNonNull(valueType);
        return new UndefinedType(valueType);
    }
}
