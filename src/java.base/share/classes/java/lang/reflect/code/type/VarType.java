package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Objects;

/**
 * A variable type.
 */
public final class VarType implements TypeElement {
    static final String NAME = "Var";

    final TypeElement variableType;

    VarType(TypeElement variableType) {
        this.variableType = variableType;
    }

    /**
     * {@return the variable type's value type}
     */
    public TypeElement valueType() {
        return variableType;
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        return new TypeDefinition(NAME, List.of(variableType.toTypeDefinition()));
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof VarType that &&
                variableType.equals(that.variableType);
    }

    @Override
    public int hashCode() {
        return variableType.hashCode();
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
