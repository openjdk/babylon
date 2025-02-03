/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx;

import java.io.Serializable;
import java.util.List;

// https://onnx.ai/onnx/api/defs.html#class-opschema
public record OpSchema(
        String file,
        int line,
        SupportLevel support_level,
        String doc,
        int since_version,
        boolean deprecated,
        String domain,
        String name,
        int min_input,
        int max_input,
        int min_output,
        int max_output,
        List<Attribute> attributes,
        List<FormalParameter> inputs,
        List<FormalParameter> outputs,
        List<TypeConstraintParam> type_constraints,
        boolean has_function,
        boolean has_context_dependent_function,
        boolean has_data_propagation_function,
        boolean has_type_and_shape_inference_function
) implements Serializable {
    public enum SupportLevel implements Serializable {
        COMMON,
        EXPERIMENTAL
    }

    public enum AttributeType implements Serializable {
        FLOAT(float.class),
        INT(int.class),
        STRING(String.class),
        // @@@ proto
        TENSOR(byte[].class),
        // proto
        GRAPH(byte[].class),
        SPARSE_TENSOR(byte[].class),
        // @@@ Map<K, V>, Opaque, Optional<T>, Sequence<T>, SparseTensor<T>, Tensor<T>
        // OnnxTypeElement?
        TYPE_PROTO(Object.class),
        FLOATS(float[].class),
        INTS(int[].class),
        STRINGS(String[].class),
        // @@@ proto
        TENSORS(byte[][].class),
        // @@@ proto
        GRAPHS(byte[][].class),
        // @@@ proto
        SPARSE_TENSORS(byte[][].class),
        TYPE_PROTOS(Object[].class)
        ;

        final Class<?> type;

        AttributeType(Class<?> type) {
            this.type = type;
        }

        public Class<?> type() {
            return type;
        }
    }

    public record Attribute(
            String name,
            String description,
            AttributeType type,
            Object default_value,
            boolean required
    ) implements Serializable {
    }

    public enum FormalParameterOption implements Serializable {
        Single,
        Optional,
        Variadic
    }

    public enum DifferentiationCategory implements Serializable {
        Unknown,
        Differentiable,
        NonDifferentiable
    }

    public record FormalParameter(
            String name,
            // @@@ List<String>
            String types,
            String type_str,
            String description,
            FormalParameterOption option,
            boolean is_homogeneous,
            int min_arity,
            DifferentiationCategory differentiation_category
    ) implements Serializable {
    }

    public record TypeConstraintParam(
            String type_param_str,
            String description,
            List<String> allowed_type_strs
    ) implements Serializable {
    }
}
