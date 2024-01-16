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

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

final class CopyContextImpl implements CopyContext {

    private static final Map<?, ?> EMPTY_MAP = Map.of();

    final CopyContextImpl parent;

    Map<Value, Value> valueMap;
    Map<Block, Block.Builder> blockMap;
    Map<Block.Reference, Block.Reference> successorMap;
    Map<Object, Object> propertiesMap;

    CopyContextImpl(CopyContextImpl that) {
        this.parent = that;
        this.blockMap = emptyMap();
        this.valueMap = emptyMap();
        this.successorMap = emptyMap();
        this.propertiesMap = emptyMap();
    }

    @SuppressWarnings("unchecked")
    private static <K, V> Map<K, V> emptyMap() {
        return (Map<K, V>) EMPTY_MAP;
    }


    // Values

    @Override
    public Value getValue(Value input) {
        Value output = getValueOrNull(input);
        if (output != null) {
            return output;
        }
        throw new IllegalArgumentException("No mapping for input value: " + input);
    }

    @Override
    public Value getValueOrDefault(Value input, Value defaultValue) {
        Value output = getValueOrNull(input);
        if (output != null) {
            return output;
        }
        return defaultValue;
    }

    private Value getValueOrNull(Value input) {
        Objects.requireNonNull(input);

        CopyContextImpl p = this;
        do {
            Value output = p.valueMap.get(input);
            if (output != null) {
                return output;
            }
            p = p.parent;
        } while (p != null);

        return null;
    }

    @Override
    public void mapValue(Value input, Value output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.isBound()) {
            throw new IllegalArgumentException("Output value bound: " + output);
        }

        if (valueMap == EMPTY_MAP) {
            valueMap = new HashMap<>();
        }
        valueMap.put(input, output);
    }

    @Override
    public Value mapValueIfAbsent(Value input, Value output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.isBound()) {
            throw new IllegalArgumentException("Output value is bound: " + output);
        }

        if (valueMap == EMPTY_MAP) {
            valueMap = new HashMap<>();
        }
        return valueMap.putIfAbsent(input, output);
    }


    // Blocks

    @Override
    public Block.Builder getBlock(Block input) {
        Objects.requireNonNull(input);

        return blockMap.get(input);
    }

    @Override
    public void mapBlock(Block input, Block.Builder output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.target().isBound()) {
            throw new IllegalArgumentException("Output block builder is built: " + output);
        }

        if (blockMap == EMPTY_MAP) {
            blockMap = new HashMap<>();
        }
        blockMap.put(input, output);
    }


    // Successors

    @Override
    public Block.Reference getSuccessor(Block.Reference input) {
        Objects.requireNonNull(input);

        return successorMap.get(input);
    }

    @Override
    public void mapSuccessor(Block.Reference input, Block.Reference output) {
        Objects.requireNonNull(input);
        Objects.requireNonNull(output);

        if (output.target.isBound()) {
            throw new IllegalArgumentException("Output block reference target is built: " + output);
        }

        for (Value outputArgument : output.arguments()) {
            if (outputArgument.isBound()) {
                throw new IllegalArgumentException("Output block reference argument is bound: " + outputArgument);
            }
        }

        if (successorMap == EMPTY_MAP) {
            successorMap = new HashMap<>();
        }
        successorMap.put(input, output);
    }


    // Properties

    @Override
    public Object getProperty(Object key) {
        CopyContextImpl p = this;
        do {
            Object value = p.propertiesMap.get(key);
            if (value != null) {
                return value;
            }
            p = p.parent;
        } while (p != null);

        return null;
    }

    @Override
    public Object putProperty(Object key, Object value) {
        if (propertiesMap == EMPTY_MAP) {
            propertiesMap = new HashMap<>();
        }
        return propertiesMap.put(key, value);
    }

    @Override
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
}
