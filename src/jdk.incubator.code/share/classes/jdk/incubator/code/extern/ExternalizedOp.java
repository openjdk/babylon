package jdk.incubator.code.extern;

import jdk.incubator.code.*;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * An operation's externalized state (a record) that can be utilized to construct an instance
 * of an {@link Op} associated with that state, such as the operation's name.
 *
 * @param name            the operation name
 * @param location        the source location associated with the operation, may be null
 * @param operands        the list of operands
 * @param successors      the list of successors
 * @param resultType      the operation result type
 * @param attributes      the operation's specific state as a map of attributes
 * @param bodyDefinitions the list of body builders for building the operation's bodies
 * @apiNote Deserializers of operations may utilize this record to construct operations,
 * thereby separating the specifics of deserializing from construction.
 */
public record ExternalizedOp(String name,
                             Location location,
                             List<Value> operands,
                             List<Block.Reference> successors,
                             TypeElement resultType,
                             Map<String, Object> attributes,
                             List<Body.Builder> bodyDefinitions) {

    /**
     * The attribute value that represents the external null value.
     */
    public static final Object NULL_ATTRIBUTE_VALUE = new Object();

    public ExternalizedOp {
        attributes = Map.copyOf(attributes);
    }

    /**
     * Gets an attribute value from the attributes map, converts the value by applying it
     * to mapping function, and returns the result.
     *
     * <p>If the attribute is a default attribute then this method first attempts to
     * get the attribute whose name is the empty string, otherwise if there is no such
     * attribute present or the attribute is not a default attribute then this method
     * attempts to get the attribute with the given name.
     *
     * <p>On successfully obtaining the attribute its value is converted by applying the value
     * to the mapping function. A {@code null} value is represented by the value
     * {@link ExternalizedOp#NULL_ATTRIBUTE_VALUE}.
     *
     * <p>If no attribute is present the {@code null} value is applied to the mapping function.
     *
     * @param name      the attribute name.
     * @param isDefault true if the attribute is a default attribute
     * @param <T>       the converted attribute value type
     * @return the converted attribute value
     */
    public <T> T extractAttributeValue(String name, boolean isDefault, Function<Object, T> mapper) {
        Object value = null;
        if (isDefault && attributes.containsKey("")) {
            value = attributes.get("");
            assert value != null;
        }

        if (value == null && attributes.containsKey(name)) {
            value = attributes.get(name);
            assert value != null;
        }

        return mapper.apply(value);
    }

    /**
     * Externalizes an operation's content.
     * <p>
     * If the operation is an instanceof {@code ExternalizableOp} then the operation's
     * specific content is externalized to an attribute map, otherwise the attribute map
     * is empty.
     *
     * @param cc the copy context
     * @param op the operation
     * @return the operation's content.
     */
    public static ExternalizedOp externalizeOp(CopyContext cc, Op op) {
        return new ExternalizedOp(
                op.externalizeOpName(),
                op.location(),
                cc.getValues(op.operands()),
                op.successors().stream().map(cc::getSuccessorOrCreate).toList(),
                op.resultType(),
                op.externalize(),
                op.bodies().stream().map(b -> b.copy(cc)).toList()
        );
    }
}
