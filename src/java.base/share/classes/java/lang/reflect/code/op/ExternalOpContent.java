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
import java.lang.reflect.code.TypeElement;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * An operation's external content (a record) that can be utilized to construct an instance
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
public record ExternalOpContent(String name,
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
     * to the mapping function.
     *
     * @param name      the attribute name.
     * @param isDefault true if the attribute is a default attribute
     * @param <T>       the converted attribute value type
     * @return the converted attribute value
     * @throws IllegalArgumentException if there is no attribute present
     */
    public <T> T extractAttributeValue(String name, boolean isDefault, Function<Object, T> mapper) {
        Object value = attributes.remove(isDefault ? "" : name);
        if (value == null) {
            if (!isDefault) {
                throw new IllegalArgumentException("Required attribute not present: "
                        + name);
            }

            value = attributes.remove(name);
        }

        return mapper.apply(value);
    }

    /**
     * Externalizes an operation to its external content.
     *
     * @param cc the copy context
     * @param op the operation
     * @return the operation's external content.
     */
    public static ExternalOpContent fromOp(CopyContext cc, Op op) {
        return new ExternalOpContent(
                op.opName(),
                cc.getValues(op.operands()),
                op.successors().stream().map(cc::getSuccessorOrCreate).toList(),
                op.resultType(),
                op instanceof ExternalizableOp exop ? new HashMap<>(exop.attributes()) : new HashMap<>(),
                op.bodies().stream().map(b -> b.copy(cc)).toList()
        );
    }
}
