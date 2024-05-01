package java.lang.reflect.code;

/**
 * A code item, one of {@link CodeElement}, {@link Value}, or {@link CodeType}.
 */
public sealed interface CodeItem
        permits CodeElement, Value, CodeType {
    // @@@ Common functionality between elements and values?
}
