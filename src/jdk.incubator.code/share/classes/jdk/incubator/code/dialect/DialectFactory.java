package jdk.incubator.code.dialect;

/**
 * A dialect factory for constructing a dialect's operations and type elements from their
 * externalized form.
 *
 * @param opFactory the operation factory.
 * @param typeElementFactory the type element factory.
 */
public record DialectFactory(OpFactory opFactory, TypeElementFactory typeElementFactory) {

    // OpFactory
    // OpDeclaration
    // TypeElementFactory
}
