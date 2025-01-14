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
