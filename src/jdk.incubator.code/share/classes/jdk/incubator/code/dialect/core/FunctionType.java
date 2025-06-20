package jdk.incubator.code.dialect.core;

import jdk.incubator.code.ExternalizableTypeElement;
import jdk.incubator.code.TypeElement;

import java.util.List;
import java.util.stream.Stream;

/**
 * A function type.
 */
public final class FunctionType implements CoreType {
    // @@@ Change to "->" when the textual form supports it
    static final String NAME = "func";

    final TypeElement returnType;
    final List<TypeElement> parameterTypes;

    FunctionType(TypeElement returnType, List<? extends TypeElement> parameterTypes) {
        this.returnType = returnType;
        this.parameterTypes = List.copyOf(parameterTypes);
    }

    /**
     * {@return the function type's return type}
     */
    public TypeElement returnType() {
        return returnType;
    }

    /**
     * {@return the function type's parameter types}
     */
    public List<TypeElement> parameterTypes() {
        return parameterTypes;
    }

    @Override
    public ExternalizedTypeElement externalize() {
        return ExternalizedTypeElement.of(NAME,
                Stream.concat(Stream.of(returnType), parameterTypes.stream())
                        .map(TypeElement::externalize).toList());
    }

    @Override
    public String toString() {
        return externalize().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof FunctionType that &&
                returnType.equals(that.returnType) &&
                parameterTypes.equals(that.parameterTypes);
    }

    @Override
    public int hashCode() {
        int result = returnType.hashCode();
        result = 31 * result + parameterTypes.hashCode();
        return result;
    }
}
