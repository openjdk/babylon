package jdk.incubator.code.internal;

import jdk.incubator.code.Op;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * An operation declaration annotation.
 * <p>
 * This annotation may be declared on a concrete class implementing an {@link Op operation} whose name is a constant
 * that can be declared as this attribute's value.
 * <p>
 * Tooling can process declarations of this annotation to build a factory for constructing operations from their name.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface OpDeclaration {
    /**
     * {@return the operation name}
     */
    String value();
}
