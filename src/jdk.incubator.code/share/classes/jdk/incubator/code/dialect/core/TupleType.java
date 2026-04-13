package jdk.incubator.code.dialect.core;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;

/**
 * A tuple type.
 */
public final class TupleType implements CoreType {
    static final String NAME = "Tuple";

    final List<CodeType> componentTypes;

    TupleType(List<? extends CodeType> componentTypes) {
        this.componentTypes = List.copyOf(componentTypes);
    }

    /**
     * {@return the tuple's component types, in order}
     */
    public List<CodeType> componentTypes() {
        return componentTypes;
    }

    @Override
    public ExternalizedCodeType externalize() {
        return ExternalizedCodeType.of(NAME,
                componentTypes.stream().map(CodeType::externalize).toList());
    }

    @Override
    public String toString() {
        return externalize().toString();
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
}
