package jdk.incubator.code.dialect;

import jdk.incubator.code.ExternalizableTypeElement;

/**
 * A dialect factory for constructing a dialect's operations and type elements from their
 * externalized form.
 *
 * @param opFactory the operation factory.
 * @param typeElementFactory the type element factory.
 */
public record DialectFactory(OpFactory opFactory, ExternalizableTypeElement.TypeElementFactory typeElementFactory) {

    // OpFactory
    // OpDeclaration
    // TypeElementFactory
}
