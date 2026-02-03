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

package jdk.incubator.code;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;

/**
 * A context utilized when transforming code models.
 * <p>
 * The context holds a mapping of input values to output values, input blocks to output block builders,
 * and input block references to output block references.
 * Mappings are defined as an input model is transformed to produce an output model. Mappings are defined
 * implicitly when an operation is transformed by copying, and can be explicitly defined when a transformation
 * removes operations or adds new operations.
 * <p>
 * Associating an input code item to its corresponding output requires that the output be part of an unbuilt code model,
 * specifically blocks connected to the output are unbuilt. An output value is a value whose declaring block is unbuilt.
 * An output block builder is a block builder that is unbuilt and therefore its block is unbuilt. An
 * output block reference is a block reference whose target block is unbuilt and whose arguments declaring blocks are
 * unbuilt.
 * <p>
 * Unless otherwise specified the passing of a {@code null} argument to the methods of this interface results in a
 * {@code NullPointerException}.
 */
public final class CodeContext {

    private static final Map<?, ?> EMPTY_MAP = Map.of();

    @SuppressWarnings("unchecked")
    private static <K, V> Map<K, V> emptyMap() {
        return (Map<K, V>) EMPTY_MAP;
    }

    private final CodeContext parent;
    private Map<Value, Value> valueMap;
    private Map<Block, Block.Builder> blockMap;
    private Map<Block.Reference, Block.Reference> successorMap;
    private Map<Object, Object> propertiesMap;

    private CodeContext(CodeContext that) {
        this.parent = that;
        this.blockMap = emptyMap();
        this.valueMap = emptyMap();
        this.successorMap = emptyMap();
        this.propertiesMap = emptyMap();
    }

    /**
     * {@return the parent context, otherwise {@code null} of there is no parent context.}
     */
    public CodeContext parent() {
        return parent;
    }

    // Value mappings

    /**
     * {@return the output value mapped to the input value}
     * <p>
     * If this context is not isolated and there is no value mapping in this context then this method will return
     * the result of calling {@code getValue} on the parent context, if present. Otherwise, if this context is isolated
     * or there is no parent context, then there is no mapping.
     *
     * @param input the input value
     * @throws IllegalArgumentException if there is no mapping
     */
    public Value getValue(Value input) {
        Value output = getValueOrNull(input);
        if (output != null) {
            return output;
        }
        throw new IllegalArgumentException("No mapping for input value: " + input);
    }

    /**
     * {@return the output value mapped to the input value or a default value if no mapping}
     *
     * @param input the input value
     * @param defaultValue the default value to return if no mapping
     */
    public Value getValueOrDefault(Value input, Value defaultValue) {
        Value output = getValueOrNull(input);
        if (output != null) {
            return output;
        }
        return defaultValue;
    }

    /**
     * Returns a list of mapped output values by obtaining, in order, the output value for each element in the list
     * of input values.
     *
     * @param inputs the list of input values
     * @return a modifiable list of output values
     * @throws IllegalArgumentException if an input value has no mapping
     */
    // @@@ If getValue is modified to return null then this method should fail on null
    public List<Value> getValues(List<? extends Value> inputs) {
        return inputs.stream().map(this::getValue).collect(toList());
    }

    private Value getValueOrNull(Value input) {
        Objects.requireNonNull(input);

        CodeContext p = this;
        do {
            Value output = p.valueMap.get(input);
            if (output != null) {
                return output;
            }
            p = p.parent;
        } while (p != null);

        return null;
    }

    /**
     * Maps an input value to an output value.
     * <p>
     * Uses of the input value will be mapped to the output value when transforming.
     *
     * @param input the input value
     * @param output the output value
     * @throws IllegalArgumentException if the output value's declaring block is built
     */
    public void mapValue(Value input, Value output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.isBuilt()) {
            throw new IllegalArgumentException("Output value bound: " + output);
        }

        if (valueMap == EMPTY_MAP) {
            valueMap = new HashMap<>();
        }
        valueMap.put(input, output);
    }

    /**
     * Maps an input value to an output value, if no such mapping exists.
     * <p>
     * Uses of the input value will be mapped to the output value when transforming.
     *
     * @param input the input value
     * @param output the output value
     * @return the previous mapped value, or null of there was no mapping.
     * @throws IllegalArgumentException if the output value's declaring block is built
     */
    // @@@ Is this needed?
    public Value mapValueIfAbsent(Value input, Value output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.isBuilt()) {
            throw new IllegalArgumentException("Output value is bound: " + output);
        }

        if (valueMap == EMPTY_MAP) {
            valueMap = new HashMap<>();
        }
        return valueMap.putIfAbsent(input, output);
    }

    /**
     * Maps the list of input values, in order, to the corresponding list of output values, up to the number of
     * elements that is the minimum of the size of both lists.
     * <p>
     * Uses of an input value will be mapped to the corresponding output value when transforming.
     *
     * @param inputs the input values
     * @param outputs the output values.
     * @throws IllegalArgumentException if an output value's declaring block is built
     */
    public void mapValues(List<? extends Value> inputs, List<? extends Value> outputs) {
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
    // @@@ throw IllegalArgumentException if there is no mapping?
    public Block.Builder getBlock(Block input) {
        Objects.requireNonNull(input);

        return blockMap.get(input);
    }

    /**
     * Maps an input block to an output block builder.
     * <p>
     * Uses of the input block will be mapped to the output block builder when transforming.
     *
     * @param input the input block
     * @param output the output block builder
     * @throws IllegalArgumentException if the output block builder's block is built
     */
    public void mapBlock(Block input, Block.Builder output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.target().isBuilt()) {
            throw new IllegalArgumentException("Output block builder is built: " + output);
        }

        if (blockMap == EMPTY_MAP) {
            blockMap = new HashMap<>();
        }
        blockMap.put(input, output);
    }


    // Successor mappings

    /**
     * {@return the output block reference mapped to the input block reference,
     * otherwise null if no mapping}
     *
     * @param input the input reference
     */
    // @@@ throw IllegalArgumentException if there is no mapping?
    public Block.Reference getSuccessor(Block.Reference input) {
        Objects.requireNonNull(input);

        return successorMap.get(input);
    }

    /**
     * Maps an input block reference to an output block reference.
     * <p>
     * Uses of the input block reference will be mapped to the output block reference when transforming.
     *
     * @param input the input block reference
     * @param output the output block reference
     * @throws IllegalArgumentException if the output block reference's target block is built or any of the
     * reference's arguments declaring blocks are built.
     */
    public void mapSuccessor(Block.Reference input, Block.Reference output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.target.isBuilt()) {
            throw new IllegalArgumentException("Output block reference target is built: " + output);
        }

        for (Value outputArgument : output.arguments()) {
            if (outputArgument.isBuilt()) {
                throw new IllegalArgumentException("Output block reference argument is bound: " + outputArgument);
            }
        }

        if (successorMap == EMPTY_MAP) {
            successorMap = new HashMap<>();
        }
        successorMap.put(input, output);
    }

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
     * @throws IllegalArgumentException if a new reference is to be created and there is no block mapping for the
     * input's target block or there is no value mapping for an input's argument
     */
    public Block.Reference getSuccessorOrCreate(Block.Reference input) {
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
     * {@return an object associated with a property key, or null if not associated.}
     *
     * @param key the property key
     */
    public Object getProperty(Object key) {
        CodeContext p = this;
        do {
            Object value = p.propertiesMap.get(key);
            if (value != null) {
                return value;
            }
            p = p.parent;
        } while (p != null);

        return null;
    }

    /**
     * Associates an object with a property key.
     *
     * @param key the property key
     * @param value the associated object
     * @return the current associated object, or null if not associated
     */
    public Object putProperty(Object key, Object value) {
        if (propertiesMap == EMPTY_MAP) {
            propertiesMap = new HashMap<>();
        }
        return propertiesMap.put(key, value);
    }

    /**
     * If the property key is not already associated with an object, attempts to compute the object using the
     * mapping function and associates it unless {@code null}.
     *
     * @param key the property key
     * @param mappingFunction the mapping function
     * @return the current (existing or computed) object associated with the property key,
     * or null if the computed object is null
     */
    public Object computePropertyIfAbsent(Object key, Function<Object, Object> mappingFunction) {
        if (propertiesMap == EMPTY_MAP) {
            propertiesMap = new HashMap<>();
        }
        Object value = getProperty(key);
        if (value != null) {
            return value;
        }
        propertiesMap.put(key, value = mappingFunction.apply(key));
        return value;
    }


    // Factories

    /**
     * {@return a new isolated context initialized with no mappings and no parent.}
     */
    public static CodeContext create() {
        return new CodeContext(null);
    }

    /**
     * {@return a new non-isolated context initialized with no mappings and a parent.}
     * The returned context will query value and property mappings in its parent context
     * if a query of its value and property mappings yields no result, and so on until
     * a context has no parent context.
     * <p>
     * The returned context only queries its own block and block reference mappings
     * and does not query the mappings in any ancestor context.
     */
    public static CodeContext create(CodeContext parent) {
        return new CodeContext(parent);
    }
}
