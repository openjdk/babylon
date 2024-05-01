package java.lang.reflect.code.type;

import java.lang.reflect.code.CodeType;
import java.util.List;
import java.util.Objects;

/**
 * A variable type.
 */
public final class VarType implements CodeType {
    static final String NAME = "Var";

    final CodeType variableType;

    VarType(CodeType variableType) {
        this.variableType = variableType;
    }

    /**
     * {@return the variable type's value type}
     */
    public CodeType valueType() {
        return variableType;
    }

    @Override
    public ExternalizedCodeType externalize() {
        return new ExternalizedCodeType(NAME, List.of(variableType.externalize()));
    }

    @Override
    public String toString() {
        return externalize().toString();
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
    public static VarType varType(CodeType valueType) {
        Objects.requireNonNull(valueType);
        return new VarType(valueType);
    }
}
