package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.Objects;

/**
 * The symbolic description of a variable type.
 */
public final class VarType implements TypeElement {
    static final String NAME = "Var";

    final TypeElement variableType;

    VarType(TypeElement variableType) {
        this.variableType = variableType;
    }

    public TypeElement variableType() {
        return variableType;
    }

    @Override
    public String toString() {
        return NAME + "<" + variableType + ">";
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

    public static VarType varType(TypeElement variableType) {
        Objects.requireNonNull(variableType);
        return new VarType(variableType);
    }
}
