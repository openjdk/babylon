package jdk.incubator.code.extern;

import jdk.incubator.code.TypeElement;

import java.util.List;
import java.util.stream.Collectors;

/**
 * A type element that supports externalization of its content and reconstruction
 * via an instance of {@link ExternalizedTypeElement}.
 */
public interface ExternalizableTypeElement extends TypeElement {

    /**
     * A type element's externalized content in structured symbolic form.
     * <p>
     * A {@link TypeElement type element} can be constructed from an externalized type element
     * using a {@link TypeElementFactory}.
     *
     * @param identifier the externalized type's identifier
     * @param arguments  the externalized type's arguments
     */
    record ExternalizedTypeElement(String identifier, List<ExternalizedTypeElement> arguments) {

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

        public static ExternalizedTypeElement of(String s) {
            return new ExternalizedTypeElement(s, List.of());
        }

        public static ExternalizedTypeElement of(String s,
                                                 ExternalizedTypeElement a) {
            return new ExternalizedTypeElement(s, List.of(a));
        }

        public static ExternalizedTypeElement of(String s,
                                                 ExternalizedTypeElement a1, ExternalizedTypeElement a2) {
            return new ExternalizedTypeElement(s, List.of(a1, a2));
        }

        public static ExternalizedTypeElement of(String s,
                                                 ExternalizedTypeElement a1, ExternalizedTypeElement a2,
                                                 ExternalizedTypeElement a3) {
            return new ExternalizedTypeElement(s, List.of(a1, a2, a3));
        }

        public static ExternalizedTypeElement of(String s,
                                                 ExternalizedTypeElement a1, ExternalizedTypeElement a2,
                                                 ExternalizedTypeElement a3, ExternalizedTypeElement a4) {
            return new ExternalizedTypeElement(s, List.of(a1, a2, a3, a4));
        }

        public static ExternalizedTypeElement of(String s,
                                                 ExternalizedTypeElement... arguments) {
            return new ExternalizedTypeElement(s, List.of(arguments));
        }

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
         * te.equals(ExternalizableTypeElement.ExternalizedTypeElement.ofString(te.toString()));
         *}
         * @param s the string
         * @return the externalized code type.
         */
        public static ExternalizedTypeElement ofString(String s) {
            return jdk.incubator.code.extern.impl.DescParser.parseExTypeElem(s);
        }
    }

    /**
     * Externalizes this type element's content.
     *
     * @return the type element's content.
     */
    ExternalizedTypeElement externalize();

    /**
     * Externalizes the given type element.
     * <p>
     * If the type element is an instance of {@link ExternalizableTypeElement}
     * then the type element is externalized by invoking {@link #externalize()},
     * otherwise the type element is externalized with an identifier that is the
     * result of applying {@code toString} to the type element, and with no arguments.
     *
     * @param te the type element
     * @return the externalized type element
     */
    static ExternalizedTypeElement externalize(TypeElement te) {
        return te instanceof ExternalizableTypeElement ete
                ? ete.externalize()
                : ExternalizedTypeElement.of(te.toString());
    }
}
