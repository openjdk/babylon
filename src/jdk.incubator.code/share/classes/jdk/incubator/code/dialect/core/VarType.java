package jdk.incubator.code.dialect.core;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

/**
 * A variable type.
 */
public final class VarType implements CoreType {
    static final String NAME = "Var";

    final CodeType valueType;

    VarType(CodeType valueType) {
        this.valueType = valueType;
    }

    /**
     * {@return the variable type's value type}
     */
    public CodeType valueType() {
        return valueType;
    }

    @Override
    public ExternalizedCodeType externalize() {
        return ExternalizedCodeType.of(NAME,
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
