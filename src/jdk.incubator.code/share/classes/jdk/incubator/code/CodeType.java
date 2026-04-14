package jdk.incubator.code;

import jdk.incubator.code.extern.ExternalizedCodeType;

/**
 * A code type that classifies values.
 * <p>
 * A {@link Value value}, one of {@link Block.Parameter} or {@link Op.Result}, has a
 * {@code CodeType} classifying that value.
 * A {@link Body} has a {@code CodeType}, the {@link Body#yieldType yield type},
 * classifying values yielded from the body.
 * <p>
 * The {@code equals} method should be used to compare code types.
 *
 * @apiNote
 * Code types enable reasoning statically about a code model, approximating
 * run time behavior.
 *
 * @see Value
 * @see Block.Parameter
 * @see Op.Result
 * @see Body#yieldType()
 */
public non-sealed interface CodeType extends CodeItem {
    // @@@ Common useful methods generally associated with properties of a type
    // e.g., arguments, is an array etc. (dimensions)

    /**
     * Externalizes this code type's content.
     *
     * @return the code type's content.
     */
    ExternalizedCodeType externalize();

    /**
     * Return a string representation of this code type.
     */
    @Override
    String toString();

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
