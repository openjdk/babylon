package jdk.incubator.code.dialect.core;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.extern.ExternalizedTypeElement;

import java.util.List;

/**
 * A tuple type.
 */
public final class TupleType implements CoreType {
    static final String NAME = "Tuple";

    final List<TypeElement> componentTypes;

    TupleType(List<? extends TypeElement> componentTypes) {
        this.componentTypes = List.copyOf(componentTypes);
    }

    /**
     * {@return the tuple's component types, in order}
     */
    public List<TypeElement> componentTypes() {
        return componentTypes;
    }

    @Override
    public ExternalizedTypeElement externalize() {
        return ExternalizedTypeElement.of(NAME,
                componentTypes.stream().map(TypeElement::externalize).toList());
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
