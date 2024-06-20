package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Objects;

/**
 * A variable type.
 */
public final class VarType implements TypeElement {
    static final String NAME = "Var";

    final TypeElement valueType;

    VarType(TypeElement valueType) {
        this.valueType = valueType;
    }

    /**
     * {@return the variable type's value type}
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
        return o instanceof VarType that &&
                valueType.equals(that.valueType);
    }

    @Override
    public int hashCode() {
        return valueType.hashCode();
    }

    /**
     * Constructs a variable type.
     *
     * @param valueType the variable's value type.
     * @return a variable type.
     */
    public static VarType varType(TypeElement valueType) {
        Objects.requireNonNull(valueType);
        return new VarType(valueType);
    }
}
