package jdk.incubator.code.dialect.core;

import jdk.incubator.code.ExternalizableTypeElement;
import jdk.incubator.code.TypeElement;

/**
 * A variable type.
 */
public final class VarType implements CoreType {
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
    public ExternalizableTypeElement.ExternalizedTypeElement externalize() {
        return ExternalizableTypeElement.ExternalizedTypeElement.of(NAME,
                valueType.externalize());
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
}
