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

package java.lang.reflect.code;

import java.util.List;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;

/**
 * A context utilized when transforming code models.
 * <p>
 * The context holds a mapping of input values to output values, input blocks to output block builders,
 * and input block references to output block references.
 * These mappings are built as an input model is transformed to produce an output model. Mappings are built implicitly
 * when an operation is transformed by copying, and can be explicitly added by the transformation when removing or
 * adding new operations.
 * <p>
 * Unless otherwise specified the passing of a {@code null} argument to the methods of this interface results in a
 * {@code NullPointerException}.
 */
public sealed interface CopyContext permits CopyContextImpl {

    // @@@ ?
    // CopyContext parent();


    // Value mappings

    /**
     * {@return the output value mapped to the input value}
     * <p>
     * If this context is not isolated and there is no value mapping in this context then this method will return
     * the result of calling {@code getValue} on the parent context, if present. Otherwise if this context is isolated
     * or there is no parent context, then there is no mapping.
     *
     * @param input the input value
     * @throws IllegalArgumentException if there is no mapping
     */
    Value getValue(Value input);

    /**
     * {@return the output value mapped to the input value or a default value if no mapping}
     *
     * @param input the input value
     * @param defaultValue the default value to return if no mapping
     */
    Value getValueOrDefault(Value input, Value defaultValue);

    /**
     * Maps an input value to an output value.
     * <p>
     * Uses of the input value will be mapped to the output value when transforming.
     *
     * @param input the input value
     * @param output the output value
     * @throws IllegalArgumentException if the output value is already bound
     */
    void mapValue(Value input, Value output);

    /**
     * Maps an input value to an output value, if no such mapping exists.
     * <p>
     * Uses of the input value will be mapped to the output value when transforming.
     *
     * @param input the input value
     * @param output the output value
     * @return the previous mapped value, or null of there was no mapping.
     * @throws IllegalArgumentException if the output value is already bound
     */
    // @@@ Is this needed?
    Value mapValueIfAbsent(Value input, Value output);

    /**
     * Returns a list of mapped output values by obtaining, in order, the output value for each element in the list
     * of input values.
     *
     * @param inputs the list of input values
     * @return a modifiable list of output values
     * @throws IllegalArgumentException if an input value has no mapping
     */
    // @@@ If getValue is modified to return null then this method should fail on null
    default List<Value> getValues(List<? extends Value> inputs) {
        return inputs.stream().map(this::getValue).collect(toList());
    }

    /**
     * Maps the list of input values, in order, to the corresponding list of output values, up to the number of
     * elements that is the minimum of the size of both lists.
     * <p>
     * Uses of an input value will be mapped to the corresponding output value when transforming.
     *
     * @param inputs the input values
     * @param outputs the output values.
     * @throws IllegalArgumentException if an output value is already bound
     */
    default void mapValues(List<? extends Value> inputs, List<? extends Value> outputs) {
        // @@@ sizes should be the same?
        for (int i = 0; i < Math.min(inputs.size(), outputs.size()); i++) {
            mapValue(inputs.get(i), outputs.get(i));
        }
    }


    // Block mappings

    /**
     * {@return the output block builder mapped to the input block, otherwise null if no mapping}
     *
     * @param input the input block
     */
    Block.Builder getBlock(Block input);

    /**
     * Maps an input block to an output block builder.
     * <p>
     * Uses of the input block will be mapped to the output block builder when transforming.
     *
     * @param input the input block
     * @param output the output block builder
     * @throws IllegalArgumentException if the output block is already bound
     */
    void mapBlock(Block input, Block.Builder output);


    // Successor mappings

    /**
     * {@return the output block reference mapped to the input block reference,
     * otherwise null if no mapping}
     *
     * @param input the input reference
     */
    Block.Reference getSuccessor(Block.Reference input);

    /**
     * Maps an input block reference to an output block reference.
     * <p>
     * Uses of the input block reference will be mapped to the output block reference when transforming.
     *
     * @param input the input block reference
     * @param output the output block reference
     * @throws IllegalArgumentException if the output block builder associated with the block reference or any of its
     * argument values are already bound
     */
    void mapSuccessor(Block.Reference input, Block.Reference output);

    /**
     * Returns a mapped output block reference, if present, otherwise creates a new, unmapped, reference from the input
     * block reference.
     * <p>
     * A new, unmapped reference, is created by obtaining the mapped output block builder from the input reference's
     * target block, and creating a successor from the output block builder with arguments that is the result of
     * obtaining the mapped values from the input reference's arguments.
     *
     * @param input the input block reference
     * @return the output block reference, if present, otherwise a created block reference
     * @throws IllegalArgumentException if a new reference is to be created and there is no mapped output block builder
     */
    default Block.Reference getSuccessorOrCreate(Block.Reference input) {
        Block.Reference successor = getSuccessor(input);
        if (successor != null) {
            return successor;
        }

        // Create successor
        Block.Builder outputBlock = getBlock(input.targetBlock());
        if (outputBlock == null) {
            throw new IllegalArgumentException("No mapping for input reference target block" + input.targetBlock());
        }
        return outputBlock.successor(getValues(input.arguments()));
    }


    // Properties mappings

    /**
     * {@return an object associated with a property key}
     *
     * @param key the property key
     */
    Object getProperty(Object key);

    /**
     * Associates an object with a property key.
     *
     * @param key the property key
     * @param value the associated object
     * @return the current associated object, or null if not associated
     */
    Object putProperty(Object key, Object value);

    /**
     * If the property key is not already associated with an object, attempts to compute the object using the
     * mapping function and associates it unless {@code null}.
     *
     * @param key the property key
     * @param mappingFunction the mapping function
     * @return the current (existing or computed) object associated with the property key,
     * or null if the computed object is null
     */
    Object computePropertyIfAbsent(Object key, Function<Object, Object> mappingFunction);


    // Factories

    /**
     * {@return a new isolated context initialized with no mappings and no parent }
     */
    static CopyContext create() {
        return new CopyContextImpl(null);
    }

    /**
     * {@return a new non-isolated context initialized with no mappings and a parent }
     * The returned context will query value and property mappings in the parent context
     * if a query of its value and property mappings yields no result.
     */
    static CopyContext create(CopyContext parent) {
        return new CopyContextImpl((CopyContextImpl) parent);
    }
}
