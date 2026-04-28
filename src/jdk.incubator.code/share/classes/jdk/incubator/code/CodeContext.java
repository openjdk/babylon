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

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;

/**
 * A context used when building and transforming code models.
 * <p>
 * A code context records correspondence between input code items and outputs during building and transformation of code
 * models. It holds mappings from:
 * <ul>
 * <li>
 * input values to output values;
 * <li>
 * input blocks to output block builders; and
 * <li>
 * input block references to output block references.
 * </ul>
 * <p>
 * Mappings are partial, since a mapping may not exist for a given input code item. Two forms of mapping lookup are
 * provided. One form returns the mapped output, if present, otherwise throws. The other form returns an optional
 * containing the mapped output, if present, otherwise an empty optional.
 * <p>
 * A code context may have a parent context, in which case it is referred to as a <i>child</i> context, otherwise it is
 * referred to as a <i>root</i> context.
 * <p>
 * Value mappings are looked up first in the current context and, if absent, in the parent context, if present, and so
 * on until a mapping is found or there is no parent context. Block and block reference mappings are local to the
 * current context and are not looked up in parent contexts.
 * <p>
 * Mappings are always added to the current context. They may be added implicitly when an operation is
 * {@link Block.Builder#op(Op) appended} to a block by
 * <a href="Block.Builder.html#transform-on-append">transform-on-append</a>. They may also be added explicitly when
 * building introduces outputs, or transformation removes, replaces, or introduces outputs.
 * <p>
 * The requirements for mapping an input code item depend on the kind of output:
 * <ul>
 * <li>
 * an output value requires that its declaring block is being built;
 * <li>
 * an output block builder requires that its block is being built;
 * <li>
 * an output block reference requires that its target block is being built, and each of the output block reference's
 * arguments requires that its declaring block is being built.
 * </ul>
 * <p>
 * A code context also supports properties that map arbitrary non-{@code null} keys mapped to arbitrary
 * non-{@code null} values. Properties are looked up first in the current context and, if absent, in the parent context,
 * and so on until a property is found or there is no parent context. Properties are always added to the current
 * context.
 * <p>
 * Unless otherwise specified the passing of a {@code null} argument to the methods of this class results in a
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
    private Map<Block.Reference, Block.Reference> referenceMap;
    private Map<Object, Object> propertiesMap;

    private CodeContext(CodeContext that) {
        this.parent = that;
        this.blockMap = emptyMap();
        this.valueMap = emptyMap();
        this.referenceMap = emptyMap();
        this.propertiesMap = emptyMap();
    }

    /**
     * {@return the parent context, otherwise {@code null} if this is a root context.}
     */
    public CodeContext parent() {
        return parent;
    }

    // Value mappings

    /**
     * {@return the output value mapped to the given input value}
     * <p>
     * The output value is looked up first in this context and, if absent, in the parent context, if present, and so on
     * until a mapping is found or there is no parent context.
     *
     * @param input the input value
     * @throws IllegalArgumentException if there is no mapping
     * @see #queryValue(Value)
     * @see #mapValue(Value, Value)
     */
    public Value getValue(Value input) {
        Value output = getValueOrNull(input);
        if (output != null) {
            return output;
        }
        throw new IllegalArgumentException("No mapping for input value: " + input);
    }

    /**
     * Queries the output value for the given input value.
     * <p>
     * If such an output value is present, this method returns an optional containing that value.
     * Otherwise, this method returns an empty optional.
     * <p>
     * The output value is looked up first in this context and, if absent, in the parent context, if present, and so on
     * until a mapping is found or there is no parent context.
     *
     * @param input the input value
     * @return an optional containing the output value mapped to the given input value, otherwise an empty optional
     * @see #getValue(Value)
     */
    public Optional<Value> queryValue(Value input) {
        Objects.requireNonNull(input);

        return Optional.ofNullable(getValueOrNull(input));
    }

    /**
     * Returns output values mapped to the given input values, in order.
     * <p>
     * The output value for each input value is obtained using {@link #getValue(Value)}.
     *
     * @param inputs the list of input values
     * @return a modifiable list of output values mapped to the given input values, in order
     * @throws IllegalArgumentException if an input value has no mapping
     * @see #getValue(Value)
     */
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
     * Maps the given input value to the given output value.
     * <p>
     * The mapping is added only to this context.
     *
     * @param input the input value
     * @param output the output value
     * @throws IllegalArgumentException if the output value's declaring block is built
     * @see #getValue(Value)
     */
    public void mapValue(Value input, Value output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.isBuilt()) {
            throw new IllegalArgumentException("Output value's declaring block is built: " + output);
        }

        if (valueMap == EMPTY_MAP) {
            valueMap = new HashMap<>();
        }
        valueMap.put(input, output);
    }

    /**
     * Maps the given input values, in order, to the given output values, up to the number of
     * elements that is the minimum of the size of both lists.
     * <p>
     * The mappings are added only to this context.
     *
     * @param inputs the input values
     * @param outputs the output values.
     * @throws IllegalArgumentException if an output value's declaring block is built
     * @see #mapValue(Value, Value)
     */
    public void mapValues(List<? extends Value> inputs, List<? extends Value> outputs) {
        // @@@ sizes should be the same?
        for (int i = 0; i < Math.min(inputs.size(), outputs.size()); i++) {
            mapValue(inputs.get(i), outputs.get(i));
        }
    }


    // Block mappings

    /**
     * {@return the output block builder mapped to the given input block}
     * <p>
     * The output block builder is looked up only in this context.
     *
     * @param input the input block
     * @throws IllegalArgumentException if there is no mapping
     * @see #queryBlock(Block)
     * @see #mapBlock(Block, Block.Builder)
     */
    public Block.Builder getBlock(Block input) {
        Objects.requireNonNull(input);

        Block.Builder output = blockMap.get(input);
        if (output == null) {
            throw new IllegalArgumentException("No mapping for input block: " + input);
        }

        return output;
    }

    /**
     * Queries the output block builder for the given input block.
     * <p>
     * The output block builder is looked up only in this context.
     *
     * @param input the input block
     * @return an optional containing the output block builder mapped to the given input block, otherwise an empty
     * optional
     */
    public Optional<Block.Builder> queryBlock(Block input) {
        Objects.requireNonNull(input);

        return Optional.ofNullable(blockMap.get(input));
    }

    /**
     * Maps the given input block to the given output block builder.
     * <p>
     * The mapping is added only to this context.
     *
     * @param input the input block
     * @param output the output block builder
     * @throws IllegalArgumentException if the output block builder's block is built
     * @see #getBlock(Block)
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


    // Derived body mappings

    /**
     * Queries the output body builder for the given input body.
     * <p>
     * This method queries the output block builder for the given input body's {@link Body#entryBlock() entry} block. If
     * such a block builder is present, this method returns an optional containing that block builder's
     * {@link Block.Builder#parentBody() parent} body builder. Otherwise, this method returns an empty optional.
     * <p>
     * The output block builder is looked up only in this context.
     *
     * @param input the input body
     * @return an optional containing the output body builder corresponding to the given input body, otherwise an empty
     * optional
     * @see #queryBlock(Block)
     */
    public Optional<Body.Builder> queryBody(Body input) {
        return queryBlock(input.entryBlock()).map(Block.Builder::parentBody);
    }


    // Reference mappings

    /**
     * {@return the output block reference mapped to the given input block reference}
     * <p>
     * The output block reference is looked up only in this context.
     *
     * @param input the input reference
     * @throws IllegalArgumentException if there is no mapping
     * @see #queryReference(Block.Reference)
     * @see #mapReference(Block.Reference, Block.Reference)
     */
    public Block.Reference getReference(Block.Reference input) {
        Objects.requireNonNull(input);

        Block.Reference output = referenceMap.get(input);
        if (output == null) {
            throw new IllegalArgumentException("No mapping for input block reference: " + input);
        }

        return output;
    }

    /**
     * Queries the output block reference for the given input block reference.
     * <p>
     * The output block reference is looked up only in this context.
     *
     * @param input the input block reference
     * @return an optional containing the output block reference mapped to the given input block reference, otherwise an
     * empty optional
     * @see #getReference(Block.Reference)
     */
    public Optional<Block.Reference> queryReference(Block.Reference input) {
        Objects.requireNonNull(input);

        return Optional.ofNullable(referenceMap.get(input));
    }

    /**
     * Maps the given input block reference to the given output block reference.
     * <p>
     * The mapping is added only to this context.
     *
     * @param input the input block reference
     * @param output the output block reference
     * @throws IllegalArgumentException if the output block reference's target block is built or any of the
     * reference's arguments declaring blocks are built.
     * @see #getReference(Block.Reference)
     */
    public void mapReference(Block.Reference input, Block.Reference output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.target.isBuilt()) {
            throw new IllegalArgumentException("Output block reference's target block is built: " + output);
        }

        for (Value outputArgument : output.arguments()) {
            if (outputArgument.isBuilt()) {
                throw new IllegalArgumentException("Output block reference argument's declaring block is built: " + outputArgument);
            }
        }

        if (referenceMap == EMPTY_MAP) {
            referenceMap = new HashMap<>();
        }
        referenceMap.put(input, output);
    }

    /**
     * Returns the output block reference mapped to the given input block reference, if present, otherwise creates
     * and returns a new output block reference from the input block reference.
     * <p>
     * A new output block reference is created by obtaining the output block builder mapped to the input reference's
     * target block, and creating a reference from the output block builder with arguments that are the output values
     * mapped to the input reference's arguments.
     * <p>
     * The output block reference and the output block builder are looked up only in this context. The output values
     * for the input reference's arguments are obtained using {@link #getValues(List)}.
     *
     * @param input the input block reference
     * @return the output block reference, if present, otherwise a newly created output block reference
     * @throws IllegalArgumentException if a new reference is to be created and there is no block mapping for the
     * input's target block or there is no value mapping for an input's argument
     * @see #queryReference(Block.Reference)
     */
    public Block.Reference getReferenceOrCreate(Block.Reference input) {
        Optional<Block.Reference> optionalOutput = queryReference(input);
        if (optionalOutput.isPresent()) {
            return optionalOutput.get();
        }

        // Create reference
        Block.Builder outputBlock = blockMap.get(input.targetBlock());
        if (outputBlock == null) {
            throw new IllegalArgumentException("No mapping for input reference target block" + input.targetBlock());
        }
        return outputBlock.reference(getValues(input.arguments()));
    }


    // Properties mappings

    /**
     * {@return an object associated with a property key, or {@code null} if not associated.}
     * <p>
     * The property is looked up first in this context and, if absent, in the parent context, if present, and so on
     * until a mapping is found or there is no parent context.
     *
     * @param key the property key
     * @see #putProperty(Object, Object)
     */
    public Object getProperty(Object key) {
        Objects.requireNonNull(key);

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
     * <p>
     * The property is added only to this context.
     *
     * @param key the property key
     * @param value the associated object
     * @return the current associated object, or {@code null} if not associated
     * @see #getProperty(Object)
     */
    public Object putProperty(Object key, Object value) {
        Objects.requireNonNull(key);
        Objects.requireNonNull(value);

        if (propertiesMap == EMPTY_MAP) {
            propertiesMap = new HashMap<>();
        }
        return propertiesMap.put(key, value);
    }

    /**
     * If the property key is not already associated with an object, attempts to compute the object using the
     * mapping function and associates it.
     * <p>
     * This method first obtains the property value using {@link #getProperty(Object)}. If no property is found, a new
     * non-{@code null} property value is computed and added only to this context.
     *
     * @param key the property key
     * @param mappingFunction the mapping function
     * @return the current (existing or computed) object associated with the property key
     * @see #getProperty(Object)
     * @see #putProperty(Object, Object)
     */
    public Object computePropertyIfAbsent(Object key, Function<Object, Object> mappingFunction) {
        Objects.requireNonNull(key);
        Objects.requireNonNull(mappingFunction);

        if (propertiesMap == EMPTY_MAP) {
            propertiesMap = new HashMap<>();
        }
        Object value = getProperty(key);
        if (value != null) {
            return value;
        }
        value = Objects.requireNonNull(mappingFunction.apply(key));
        propertiesMap.put(key, value);
        return value;
    }


    // Factories

    /**
     * {@return a new root context initialized with no local mappings, no properties, and no parent context.}
     * @see #create(CodeContext)
     */
    public static CodeContext create() {
        return new CodeContext(null);
    }

    /**
     * {@return a new child context initialized with the given parent context, no local mappings, and no properties.}
     * <p>
     * The returned context looks up value mappings and properties first in itself, and then in its parent context,
     * and so on, until a mapping is found or there is no parent context.
     * <p>
     * Block and block reference mappings are local to the returned context and are not looked up in parent contexts.
     * <p>
     * Mappings and properties added to the returned context are added only to that context.
     *
     * @param parent the parent code context
     */
    public static CodeContext create(CodeContext parent) {
        Objects.requireNonNull(parent);
        return new CodeContext(parent);
    }
}
