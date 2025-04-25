package jdk.incubator.code.type;

/**
 * A symbolic reference to a Java class member or type, commonly containing
 * symbolic names together with {@link JavaType symbolic descriptions} of Java types.
 * <p>
 * A symbolic Java reference can be resolved to a corresponding instance of its
 * reflected representation, much like the symbolic description of a Java type
 * can be resolved to an instance of {@link java.lang.reflect.Type Type}.
 */
public sealed interface JavaRef
    permits MethodRef, ConstructorRef, FieldRef, RecordTypeRef {
    // @@@ Extend from TypeElement
    //     - Uniform externalization of Java types and Java refs,
    //       therefore we don't require specific string representations
    //       and parser implementations. (Human readability of Java types
    //       and refs is a separate issue.)
    //       e.g., the description of a type-variable reference Java type
    //       (TypeVarRef) contains an owner, a description of Java class
    //       or a symbolic reference to a Java method or constructor.
    // @@@ Enhance TypeElement to traverse children
    //     - Uniform tree traversal and transformation independent of
    //       externalization.
    // @@@ Make RecordTypeRef.ComponentRef implement JavaRef
    //     - resolve to accessor method
    //     - (RecordTypeRef resolves to Class.)
    // @@@ AnnotatedElement is the common top type for resolved Java refs and types
}
