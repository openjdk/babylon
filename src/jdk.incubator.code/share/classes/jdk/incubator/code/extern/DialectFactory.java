package jdk.incubator.code.extern;

/**
 * A dialect factory for constructing a dialect's operations and code types from their
 * externalized form.
 *
 * @param opFactory the operation factory.
 * @param codeTypeFactory the code type factory.
 */
public record DialectFactory(OpFactory opFactory, CodeTypeFactory codeTypeFactory) {
}
