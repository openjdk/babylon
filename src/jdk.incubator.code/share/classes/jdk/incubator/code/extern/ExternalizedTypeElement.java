package jdk.incubator.code.extern;

import jdk.incubator.code.TypeElement;

import java.util.List;
import java.util.stream.Collectors;

/**
 * A type element's externalized content in structured symbolic form.
 * <p>
 * A {@link TypeElement type element} can be constructed from an externalized type element
 * using a {@link TypeElementFactory}.
 *
 * @param identifier the externalized type's identifier
 * @param arguments  the externalized type's arguments
 */
public record ExternalizedTypeElement(String identifier, List<ExternalizedTypeElement> arguments) {

    /**
     * Constructs a new externalized type
     * @param identifier the externalized type's identifier
     * @param arguments  the externalized type's arguments
     */
    public ExternalizedTypeElement {
        arguments = List.copyOf(arguments);
    }

    @Override
    public String toString() {
        return toString(this);
    }

    static String toString(ExternalizedTypeElement t) {
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
    public static ExternalizedTypeElement of(String s) {
        return new ExternalizedTypeElement(s, List.of());
    }

    /**
     * Returns an externalized type with one type argument.
     *
     * @param s the externalized type identifier
     * @param a the type argument
     * @return the externalized type
     */
    public static ExternalizedTypeElement of(String s,
                                             ExternalizedTypeElement a) {
        return new ExternalizedTypeElement(s, List.of(a));
    }

    /**
     * Returns an externalized type with two type arguments.
     *
     * @param s the externalized type identifier
     * @param a1 the first type argument
     * @param a2 the second type argument
     * @return the externalized type
     */
    public static ExternalizedTypeElement of(String s,
                                             ExternalizedTypeElement a1, ExternalizedTypeElement a2) {
        return new ExternalizedTypeElement(s, List.of(a1, a2));
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
    public static ExternalizedTypeElement of(String s,
                                             ExternalizedTypeElement a1, ExternalizedTypeElement a2,
                                             ExternalizedTypeElement a3) {
        return new ExternalizedTypeElement(s, List.of(a1, a2, a3));
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
    public static ExternalizedTypeElement of(String s,
                                             ExternalizedTypeElement a1, ExternalizedTypeElement a2,
                                             ExternalizedTypeElement a3, ExternalizedTypeElement a4) {
        return new ExternalizedTypeElement(s, List.of(a1, a2, a3, a4));
    }

    /**
     * Returns an externalized type with given type arguments.
     *
     * @param s the externalized type identifier
     * @param arguments the type arguments
     * @return the externalized type
     */
    public static ExternalizedTypeElement of(String s,
                                             ExternalizedTypeElement... arguments) {
        return new ExternalizedTypeElement(s, List.of(arguments));
    }

    /**
     * Returns an externalized type with given type arguments.
     *
     * @param s the externalized type identifier
     * @param arguments the type arguments
     * @return the externalized type
     */
    public static ExternalizedTypeElement of(String s,
                                             List<ExternalizedTypeElement> arguments) {
        return new ExternalizedTypeElement(s, arguments);
    }

    /**
     * Parses a string as an externalized type element.
     * <p>
     * For any given externalized type element, {@code te}, the following
     * expression returns {@code true}.
     * {@snippet lang = java:
     * te.equals(ExternalizedTypeElement.ofString(te.toString()));
     * }
     *
     * @param s the string
     * @return the externalized code type.
     */
    public static ExternalizedTypeElement ofString(String s) {
        return jdk.incubator.code.extern.impl.DescParser.parseExTypeElem(s);
    }
}
