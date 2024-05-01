package java.lang.reflect.code;

import java.lang.reflect.code.type.CodeTypeFactory;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A code (model) type, that defines a set of values.
 * <p>
 * A code type can be assigned to a {@link Value value} in a code model,
 * and implies the value is a member of the type's set.
 * <p>
 * The {@code equals} method should be used to check if two code types
 * are equal to each other.
 * @apiNote
 * Code model types enable reasoning statically about a code model,
 * approximating run time behaviour.
 */
public non-sealed interface CodeType extends CodeItem {
    // @@@ Common useful methods generally associated with properties of a type
    // e.g., arguments, is an array etc. (dimensions)

    /**
     * A code type's externalized content in structured symbolic form.
     * <p>
     * A {@link CodeType code type} can be constructed from an externalized code type
     * using a {@link CodeTypeFactory}.
     *
     * @param identifier the externalized type's identifier
     * @param arguments  the externalized type's arguments
     */
    record ExternalizedCodeType(String identifier, List<ExternalizedCodeType> arguments) {

        /**
         * {@inheritDoc}
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

            // Unpack array-like identifier [+
            int dimensions = 0;
            if (t.arguments.size() == 1) {
                dimensions = dimensions(t.identifier);
                if (dimensions > 0) {
                    t = t.arguments.getFirst();
                }
            }

            StringBuilder s = new StringBuilder();
            s.append(t.identifier);
            if (!t.arguments.isEmpty()) {
                String args = t.arguments.stream()
                        .map(Object::toString)
                        .collect(Collectors.joining(", ", "<", ">"));
                s.append(args);
            }

            // Write out array-like syntax at end []+
            if (dimensions > 0) {
                s.append("[]".repeat(dimensions));
            }

            return s.toString();
        }

        static int dimensions(String identifier) {
            if (!identifier.isEmpty() && identifier.charAt(0) == '[') {
                for (int i = 1; i < identifier.length(); i++) {
                    if (identifier.charAt(i) != '[') {
                        return 0;
                    }
                }
                return identifier.length();
            } else {
                return 0;
            }
        }

        // Factories

        /**
         * Parses a string as an externalized code type.
         * <p>
         * For any given code type, {@code ct}, the following
         * expression returns {@code true}.
         * {@snippet lang=java
         * ct.equals(CodeType.ofString(ct.toString()));
         * }
         * @param s the string
         * @return the externalized code type.
         */
        // Copied code in jdk.compiler module throws UOE
        public static ExternalizedCodeType ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseExternalizedCodeType(s);
        }
    }

    /**
     * Externalize this code type's content.
     *
     * @return the code type's content
     * @throws UnsupportedOperationException if the type is not externalizable
     */
    ExternalizedCodeType externalize();

    @Override
    String toString();

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
