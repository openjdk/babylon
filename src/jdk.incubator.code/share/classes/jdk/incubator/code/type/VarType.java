package jdk.incubator.code.type;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.parser.impl.DescParser;

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
        return DescParser.parseExTypeElem(toString());
    }

    @Override
    public String toString() {
        return NAME + "<" + valueType + ">";
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
