package java.lang.reflect.code;

import java.lang.reflect.code.type.TypeDefinition;
import java.util.Optional;

/**
 * A type, that defines a set of values.
 * <p>
 * A type can be assigned to a {@link Value value} in a code model,
 * and implies the value is a member of the type's set.
 * @apiNote
 * Code model types enable reasoning statically about a code model,
 * approximating run time behaviour.
 */
public non-sealed interface TypeElement extends CodeItem {
    // @@@ Common useful methods generally associated with properties of a type
    // e.g., arguments, is an array etc. (dimensions)

    // @@@ Declare
    // toString, equals, hashCode

    /**
     * Converts this type element to a type definition.
     *
     * @return the type definition
     * @throws UnsupportedOperationException if the type element is not convertible
     */
    TypeDefinition toTypeDefinition();
}
