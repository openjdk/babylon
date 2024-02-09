package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The symbolic description of a function type.
 */
public final class FunctionType implements TypeElement {
    static final String NAME = "->";

    final TypeElement returnType;
    final List<TypeElement> parameterTypes;

    FunctionType(TypeElement returnType, List<TypeElement> parameterTypes) {
        this.returnType = returnType;
        this.parameterTypes = List.copyOf(parameterTypes);
    }

    public TypeElement returnType() {
        return returnType;
    }

    public List<TypeElement> parameterTypes() {
        return parameterTypes;
    }

    @Override
    public String toString() {
        String cs = Stream.concat(Stream.of(returnType), parameterTypes.stream())
                .map(TypeElement::toString)
                .collect(Collectors.joining(",", "<", ">"));
        return NAME + cs;
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

    public static FunctionType functionType(TypeElement returnType, TypeElement... parameterTypes) {
        return functionType(returnType, List.of(parameterTypes));
    }

    public static FunctionType functionType(TypeElement returnType, List<TypeElement> parameterTypes) {
        Objects.requireNonNull(returnType);
        Objects.requireNonNull(parameterTypes);
        return new FunctionType(returnType, parameterTypes);
    }
}
