package jdk.incubator.code.type;

import jdk.incubator.code.TypeElement;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * A function type.
 */
public final class FunctionType implements TypeElement {
    // @@@ Change to "->" when the textual form supports it
    static final String NAME = "func";

    /**
     * The function type with no parameters, returning void.
     */
    // @@@ Uses JavaType
    public static final FunctionType VOID = functionType(JavaType.VOID);

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
        return new ExternalizedTypeElement(NAME,
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

    /**
     * Constructs a function type.
     *
     * @param returnType the function type's return type.
     * @param parameterTypes the function type's parameter types.
     * @return a function type.
     */
    public static FunctionType functionType(TypeElement returnType, List<? extends TypeElement> parameterTypes) {
        Objects.requireNonNull(returnType);
        Objects.requireNonNull(parameterTypes);
        return new FunctionType(returnType, parameterTypes);
    }
    /**
     * Constructs a function type.
     *
     * @param returnType the function type's return type.
     * @param parameterTypes the function type's parameter types.
     * @return a function type.
     */
    public static FunctionType functionType(TypeElement returnType, TypeElement... parameterTypes) {
        return functionType(returnType, List.of(parameterTypes));
    }

}
