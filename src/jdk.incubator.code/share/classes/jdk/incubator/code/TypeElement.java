package jdk.incubator.code;

/**
 * A type, that defines a set of values.
 * <p>
 * A type can be assigned to a {@link Value value} in a code model,
 * and implies the value is a member of the type's set.
 * <p>
 * The {@code equals} method should be used to check if two type elements
 * are equal to each other.
 * @apiNote
 * Code model types enable reasoning statically about a code model,
 * approximating run time behaviour.
 */
public non-sealed interface TypeElement extends CodeItem {
    // @@@ Common useful methods generally associated with properties of a type
    // e.g., arguments, is an array etc. (dimensions)

    /**
     * Externalizes this type element's content.
     *
     * @implSpec
     * The default implementation returns an externalized type element with
     * an identifier that is the result of applying {@code toString} to this object, and
     * with no arguments.
     *
     * @return the type element's content.
     */
    default ExternalizableTypeElement.ExternalizedTypeElement externalize() {
        // @@@ Certain externalizable type elements are composed of other type elements,
        // which may or may not be externalizable. OpWriter is designed to work with
        // non-externalizable type elements, but in such cases OpParser will fail
        // to parse what OpWriter produces.
        // @@@ Should this throw UnsupportedOperationException
        // @@@ Should this be a static helper method on ExternalizableTypeElement?
        return ExternalizableTypeElement.ExternalizedTypeElement.of(toString());
    }

    /**
     * Return a string representation of this Java type.
     */
    @Override
    String toString();

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
