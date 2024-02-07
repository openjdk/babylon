package java.lang.reflect.code;

/**
 * A code item, one of {@link CodeElement}, or {@link Value}.
 */
public sealed interface CodeItem
        permits CodeElement, Value {
    // @@@ Common functionality between elements and values?
}
