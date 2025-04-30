package jdk.incubator.code.type;

import jdk.incubator.code.TypeElement;

/**
 * A symbolic reference to a Java class member or a Java type including members,
 * commonly containing symbolic names together with {@link JavaType symbolic descriptions}
 * of Java types.
 * <p>
 * A symbolic Java reference can be resolved to a corresponding instance of its
 * reflected representation, much like the symbolic description of a Java type
 * can be resolved to an instance of {@link java.lang.reflect.Type Type}.
 */
public sealed interface JavaRef extends TypeElement
    permits MethodRef, ConstructorRef, FieldRef, RecordTypeRef {
    // @@@ Enhance TypeElement to traverse children
    //     - Uniform tree traversal and transformation independent of
    //       externalization.
    // @@@ Make RecordTypeRef.ComponentRef implement JavaRef?
    //     - resolve to RecordComponent
    //     - (RecordTypeRef resolves to Type.)
    // @@@ AnnotatedElement is the common top type for resolved Java refs and types
}
