package java.lang.reflect.code.type;

import java.lang.reflect.code.CodeType;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * A function type.
 */
public final class FunctionType implements CodeType {
    // @@@ Change to "->" when the textual form supports it
    static final String NAME = "func";

    /**
     * The function type with no parameters, returning void.
     */
    // @@@ Uses JavaType
    public static final FunctionType VOID = functionType(JavaType.VOID);

    final CodeType returnType;
    final List<CodeType> parameterTypes;

    FunctionType(CodeType returnType, List<? extends CodeType> parameterTypes) {
        this.returnType = returnType;
        this.parameterTypes = List.copyOf(parameterTypes);
    }

    /**
     * {@return the function type's return type}
     */
    public CodeType returnType() {
        return returnType;
    }

    /**
     * {@return the function type's parameter types}
     */
    public List<CodeType> parameterTypes() {
        return parameterTypes;
    }

    @Override
    public ExternalizedCodeType externalize() {
        return new ExternalizedCodeType(NAME,
                Stream.concat(Stream.of(returnType), parameterTypes.stream())
                        .map(CodeType::externalize).toList());
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
    public static FunctionType functionType(CodeType returnType, List<? extends CodeType> parameterTypes) {
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
    public static FunctionType functionType(CodeType returnType, CodeType... parameterTypes) {
        return functionType(returnType, List.of(parameterTypes));
    }

}
