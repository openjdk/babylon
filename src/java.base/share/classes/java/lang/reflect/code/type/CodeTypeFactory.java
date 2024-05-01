package java.lang.reflect.code.type;

import java.lang.reflect.code.CodeType;

/**
 * A code type factory for construction a {@link CodeType} from its
 * {@link CodeType.ExternalizedCodeType external content}.
 */
@FunctionalInterface
public interface CodeTypeFactory {

    /**
     * Constructs a {@link CodeType} from its
     * {@link CodeType.ExternalizedCodeType external content}.
     * <p>
     * If there is no mapping from the external content to a code
     * type then this method returns {@code null}.
     *
     * @param tree the externalized code type.
     * @return the code type.
     */
    CodeType constructType(CodeType.ExternalizedCodeType tree);

    /**
     * Compose this code type factory with another code type factory.
     * <p>
     * If there is no mapping in this code type factory then the result
     * of the other code type factory is returned.
     *
     * @param after the other code type factory.
     * @return the composed code type factory.
     */
    default CodeTypeFactory andThen(CodeTypeFactory after) {
        return t -> {
            CodeType te = constructType(t);
            return te != null ? te : after.constructType(t);
        };
    }
}