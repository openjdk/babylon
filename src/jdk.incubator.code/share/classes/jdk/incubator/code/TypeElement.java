package jdk.incubator.code;

import jdk.incubator.code.extern.ExternalizedTypeElement;

/**
 * A code item that classifies values.
 * <p>
 * A {@link Value value}, one of {@link Block.Parameter} or {@link Op.Result}, has a
 * {@code TypeElement} classifying that value.
 * A {@link Body} has a {@code TypeElement}, the {@link Body#yieldType yield type},
 * classifying values yielded from the body.
 * <p>
 * The {@code equals} method should be used to compare type elements.
 *
 * @apiNote
 * Type elements enable reasoning statically about a code model, approximating
 * run time behaviour.
 *
 * @see Value
 * @see Block.Parameter
 * @see Op.Result
 * @see Body#yieldType()
 */
public non-sealed interface TypeElement extends CodeItem {
    // @@@ Common useful methods generally associated with properties of a type
    // e.g., arguments, is an array etc. (dimensions)

    /**
     * Externalizes this type element's content.
     *
     * @return the type element's content.
     */
    ExternalizedTypeElement externalize();

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
