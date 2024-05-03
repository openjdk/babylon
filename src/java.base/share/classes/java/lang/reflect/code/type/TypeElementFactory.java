package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;

/**
 * A type element factory for construction a {@link TypeElement} from its
 * {@link TypeElement.ExternalizedTypeElement external content}.
 */
@FunctionalInterface
public interface TypeElementFactory {

    /**
     * Constructs a {@link TypeElement} from its
     * {@link TypeElement.ExternalizedTypeElement external content}.
     * <p>
     * If there is no mapping from the external content to a type
     * element then this method returns {@code null}.
     *
     * @param tree the externalized type element.
     * @return the type element.
     */
    TypeElement constructType(TypeElement.ExternalizedTypeElement tree);

    /**
     * Compose this type element factory with another type element factory.
     * <p>
     * If there is no mapping in this type element factory then the result
     * of the other type element factory is returned.
     *
     * @param after the other type element factory.
     * @return the composed type element factory.
     */
    default TypeElementFactory andThen(TypeElementFactory after) {
        return t -> {
            TypeElement te = constructType(t);
            return te != null ? te : after.constructType(t);
        };
    }
}