/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package java.lang.reflect.code.op;

import java.lang.reflect.code.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * An operation that supports externalization of its content and reconstruction
 * via an instance of {@link ExternalizedOp}.
 * <p>
 * The specific content of an externalizable operation can be externalized to a
 * map of {@link #attributes attributes}, and is reconstructed from the
 * attributes component of an instance of {@link ExternalizedOp}.
 * <p>
 * An externalizable operation could be externalized via serialization to
 * a textual representation. That textual representation could then be deserialized,
 * via parsing, into an instance of {@link ExternalizedOp} from which a new
 * externalizable operation can be reconstructed that is identical to one that
 * was serialized.
 */
public abstract class ExternalizableOp extends Op {

    /**
     * An operation's externalized content (a record) that can be utilized to construct an instance
     * of an {@link ExternalizableOp} associated with the operation's name.
     *
     * @param name            the operation name
     * @param operands        the list of operands
     * @param successors      the list of successors
     * @param resultType      the operation result type
     * @param attributes      the operation's specific content as an attributes map, modifiable
     * @param bodyDefinitions the list of body builders for building the operation's bodies
     * @apiNote Deserializers of operations may utilize this record to construct operations,
     * thereby separating the specifics of deserializing from construction.
     */
    public record ExternalizedOp(String name,
                                 List<Value> operands,
                                 List<Block.Reference> successors,
                                 TypeElement resultType,
                                 Map<String, Object> attributes,
                                 List<Body.Builder> bodyDefinitions) {

        /**
         * Removes an attribute value from the attributes map, converts the value by applying it
         * to mapping function, and returns the result.
         *
         * <p>If the attribute is a default attribute then this method first attempts to
         * remove the attribute whose name is the empty string, otherwise if there is no such
         * attribute present or the attribute is not a default attribute then this method
         * attempts to remove the attribute with the given name.
         *
         * <p>On successful removal of the attribute its value is converted by applying the value
         * to the mapping function. A {@code null} value is represented the instance
         * {@link #NULL_ATTRIBUTE_VALUE}.
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
                value = attributes.remove("");
                assert value != null;
            }

            if (value == null && attributes.containsKey(name)) {
                value = attributes.remove(name);
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
                    op.opName(),
                    cc.getValues(op.operands()),
                    op.successors().stream().map(cc::getSuccessorOrCreate).toList(),
                    op.resultType(),
                    op instanceof ExternalizableOp exop ? new HashMap<>(exop.attributes()) : new HashMap<>(),
                    op.bodies().stream().map(b -> b.copy(cc)).toList()
            );
        }
    }

    /**
     * The attribute name associated with the location attribute.
     */
    public static final String ATTRIBUTE_LOCATION = "loc";

    /**
     * The attribute value that represents the external null value.
     */
    public static final Object NULL_ATTRIBUTE_VALUE = new Object();

    /**
     * Constructs an operation by copying given operation.
     *
     * @param that the operation to copy.
     * @param cc   the copy context.
     * @implSpec The default implementation calls the constructor with the operation's name, result type, and a list
     * values computed, in order, by mapping the operation's operands using the copy context.
     */
    protected ExternalizableOp(Op that, CopyContext cc) {
        super(that, cc);
    }

    /**
     * Constructs an operation with a name, operation result type, and list of operands.
     *
     * @param name     the operation name.
     * @param operands the list of operands, a copy of the list is performed if required.
     */
    protected ExternalizableOp(String name, List<? extends Value> operands) {
        super(name, operands);
    }

    /**
     * Constructs an operation from its external content.
     *
     * @param def the operation's external content.
     * @implSpec This implementation invokes the {@link Op#Op(String, List) constructor}
     * accepting the non-optional components of the operation's content, {@code name},
     * and {@code operands}:
     * <pre> {@code
     *  this(def.name(), def.operands());
     * }</pre>
     */
    @SuppressWarnings("this-escape")
    protected ExternalizableOp(ExternalizedOp def) {
        super(def.name(), def.operands());
        setLocation(extractLocation(def));
    }

    static Location extractLocation(ExternalizedOp def) {
        Object v = def.attributes().get(ATTRIBUTE_LOCATION);
        return switch (v) {
            case String s -> Location.fromString(s);
            case Location loc -> loc;
            case null -> null;
            default -> throw new UnsupportedOperationException("Unsupported location value:" + v);
        };
    }

    /**
     * Externalizes the operation's specific content as a map of attributes.
     *
     * <p>A null attribute value is represented by the constant
     * value {@link #NULL_ATTRIBUTE_VALUE}.
     *
     * @return the operation's attributes, as an unmodifiable map
     */
    public Map<String, Object> attributes() {
        Location l = location();
        return l == null ? Map.of() : Map.of(ATTRIBUTE_LOCATION, l);
    }
}
