package java.lang.reflect.code;

/**
 * A code item, one of {@link CodeElement}, {@link Value}, or {@link TypeElement}.
 * @sealedGraph
 */
public sealed interface CodeItem
        permits CodeElement, Value, TypeElement {
    // @@@ Common functionality between elements and values?
}
