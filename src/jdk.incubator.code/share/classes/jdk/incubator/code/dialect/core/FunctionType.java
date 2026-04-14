package jdk.incubator.code.dialect.core;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;
import java.util.stream.Stream;

/**
 * A function type.
 */
public final class FunctionType implements CoreType {
    // @@@ Change to "->" when the textual form supports it
    static final String NAME = "func";

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
        return ExternalizedCodeType.of(NAME,
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
}
