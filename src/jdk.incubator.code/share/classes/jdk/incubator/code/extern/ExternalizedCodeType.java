package jdk.incubator.code.extern;

import jdk.incubator.code.CodeType;

import java.util.List;
import java.util.stream.Collectors;

/**
 * A code type's externalized content in structured symbolic form.
 * <p>
 * A {@link CodeType code type} can be constructed from an externalized code type
 * using a {@link CodeTypeFactory}.
 *
 * @param identifier the externalized type's identifier
 * @param arguments  the externalized type's arguments
 */
public record ExternalizedCodeType(String identifier, List<ExternalizedCodeType> arguments) {

    /**
     * Constructs a new externalized type
     * @param identifier the externalized type's identifier
     * @param arguments  the externalized type's arguments
     */
    public ExternalizedCodeType {
        arguments = List.copyOf(arguments);
    }

    @Override
    public String toString() {
        return toString(this);
    }

    static String toString(ExternalizedCodeType t) {
        if (t.arguments.isEmpty()) {
            return t.identifier;
        }

        StringBuilder s = new StringBuilder();
        s.append(t.identifier);
        if (!t.arguments.isEmpty()) {
            String args = t.arguments.stream()
                    .map(Object::toString)
                    .collect(Collectors.joining(", ", "<", ">"));
            s.append(args);
        }

        return s.toString();
    }

    // Factories

    /**
     * Returns an externalized type with the given identifier.
     *
     * @param s the externalized type identifier
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s) {
        return new ExternalizedCodeType(s, List.of());
    }

    /**
     * Returns an externalized type with one type argument.
     *
     * @param s the externalized type identifier
     * @param a the type argument
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s,
                                          ExternalizedCodeType a) {
        return new ExternalizedCodeType(s, List.of(a));
    }

    /**
     * Returns an externalized type with two type arguments.
     *
     * @param s the externalized type identifier
     * @param a1 the first type argument
     * @param a2 the second type argument
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s,
                                          ExternalizedCodeType a1, ExternalizedCodeType a2) {
        return new ExternalizedCodeType(s, List.of(a1, a2));
    }

    /**
     * Returns an externalized type with three type arguments.
     *
     * @param s the externalized type identifier
     * @param a1 the first type argument
     * @param a2 the second type argument
     * @param a3 the third type argument
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s,
                                          ExternalizedCodeType a1, ExternalizedCodeType a2,
                                          ExternalizedCodeType a3) {
        return new ExternalizedCodeType(s, List.of(a1, a2, a3));
    }

    /**
     * Returns an externalized type with four type arguments.
     *
     * @param s the externalized type identifier
     * @param a1 the first type argument
     * @param a2 the second type argument
     * @param a3 the third type argument
     * @param a4 the fourth type argument
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s,
                                          ExternalizedCodeType a1, ExternalizedCodeType a2,
                                          ExternalizedCodeType a3, ExternalizedCodeType a4) {
        return new ExternalizedCodeType(s, List.of(a1, a2, a3, a4));
    }

    /**
     * Returns an externalized type with given type arguments.
     *
     * @param s the externalized type identifier
     * @param arguments the type arguments
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s,
                                          ExternalizedCodeType... arguments) {
        return new ExternalizedCodeType(s, List.of(arguments));
    }

    /**
     * Returns an externalized type with given type arguments.
     *
     * @param s the externalized type identifier
     * @param arguments the type arguments
     * @return the externalized type
     */
    public static ExternalizedCodeType of(String s,
                                          List<ExternalizedCodeType> arguments) {
        return new ExternalizedCodeType(s, arguments);
    }

    /**
     * Parses a string as an externalized code type.
     * <p>
     * For any given externalized code type, {@code ct}, the following
     * expression returns {@code true}.
     * {@snippet lang = java:
     * ct.equals(ExternalizedCodeType.ofString(ct.toString()));
     *}
     *
     * @param s the string
     * @return the externalized code type.
     */
    public static ExternalizedCodeType ofString(String s) {
        return jdk.incubator.code.extern.impl.DescParser.parseExCodeType(s);
    }
}
