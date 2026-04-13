package jdk.incubator.code.dialect.core;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.Value;
import jdk.incubator.code.extern.ExternalizedCodeType;
import jdk.incubator.code.extern.CodeTypeFactory;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 *  The symbolic description of a core type.
 */
public sealed interface CoreType extends CodeType
        permits FunctionType, TupleType, VarType {

    /**
     * Creates a composed code type factory for core types and code types from the given
     * code type factory, where the core types can refer to code types from the
     * given code type factory.
     *
     * @param f the code type factory.
     * @return the composed code type factory.
     */
    static CodeTypeFactory coreTypeFactory(CodeTypeFactory f) {
        class CodeModelFactory implements CodeTypeFactory {
            final CodeTypeFactory thisThenF = this.andThen(f);

            @Override
            public CodeType constructType(ExternalizedCodeType tree) {
                return switch (tree.identifier()) {
                    case VarType.NAME -> {
                        if (tree.arguments().size() != 1) {
                            throw new IllegalArgumentException();
                        }

                        CodeType v = thisThenF.constructType(tree.arguments().getFirst());
                        if (v == null) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        yield varType(v);
                    }
                    case TupleType.NAME -> {
                        if (tree.arguments().isEmpty()) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }

                        List<CodeType> cs = new ArrayList<>(tree.arguments().size());
                        for (ExternalizedCodeType child : tree.arguments()) {
                            CodeType c = thisThenF.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException("Bad type: " + tree);
                            }
                            cs.add(c);
                        }
                        yield tupleType(cs);
                    }
                    case FunctionType.NAME -> {
                        if (tree.arguments().isEmpty()) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }

                        CodeType rt = thisThenF.constructType(tree.arguments().getFirst());
                        if (rt == null) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        List<CodeType> pts = new ArrayList<>(tree.arguments().size() - 1);
                        for (ExternalizedCodeType child : tree.arguments().subList(1, tree.arguments().size())) {
                            CodeType c = thisThenF.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException("Bad type: " + tree);
                            }
                            pts.add(c);
                        }
                        yield functionType(rt, pts);
                    }
                    default -> null;
                };
            }
        }
        if (f instanceof CodeModelFactory) {
            throw new IllegalArgumentException();
        }

        return new CodeModelFactory().thisThenF;
    }


    /**
     * The function type with no parameters, returning void.
     */
    // @@@ Uses JavaType
    FunctionType FUNCTION_TYPE_VOID = functionType(JavaType.VOID);

    /**
     * Constructs a function type.
     *
     * @param returnType the function type's return type.
     * @param parameterTypes the function type's parameter types.
     * @return a function type.
     */
    static FunctionType functionType(CodeType returnType, List<? extends CodeType> parameterTypes) {
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
    static FunctionType functionType(CodeType returnType, CodeType... parameterTypes) {
        return functionType(returnType, List.of(parameterTypes));
    }

    /**
     * Constructs a tuple type.
     *
     * @param componentTypes the tuple type's component types.
     * @return a tuple type.
     */
    static TupleType tupleType(CodeType... componentTypes) {
        return tupleType(List.of(componentTypes));
    }

    /**
     * Constructs a tuple type.
     *
     * @param componentTypes the tuple type's component types.
     * @return a tuple type.
     */
    static TupleType tupleType(List<? extends CodeType> componentTypes) {
        Objects.requireNonNull(componentTypes);
        return new TupleType(componentTypes);
    }

    /**
     * Constructs a tuple type whose components are the types of
     * the given values.
     *
     * @param values the values.
     * @return a tuple type.
     */
    static TupleType tupleTypeFromValues(Value... values) {
        return tupleType(Stream.of(values).map(Value::type).toList());
    }

    /**
     * Constructs a tuple type whose components are the types of
     * the given values.
     *
     * @param values the values.
     * @return a tuple type.
     */
    static TupleType tupleTypeFromValues(List<? extends Value> values) {
        return tupleType(values.stream().map(Value::type).toList());
    }

    /**
     * Constructs a variable type.
     *
     * @param valueType the variable's value type.
     * @return a variable type.
     */
    static VarType varType(CodeType valueType) {
        Objects.requireNonNull(valueType);
        return new VarType(valueType);
    }
}
