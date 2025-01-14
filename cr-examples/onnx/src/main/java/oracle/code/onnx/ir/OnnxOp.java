package oracle.code.onnx.ir;

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.ExternalizableOp;

import java.util.*;

public abstract class OnnxOp extends ExternalizableOp {

    public interface OnnxAttribute {
        String name();

        Class<?> type();

        Object defaultValue();

        boolean isOptional();

        default void process(Map<String, Object> attrs, Object value) {
            if (value instanceof Optional<?> o) {
                value = o.orElse(null);
            }
            // @@@ Parse attribute from string value
            if (type().isInstance(value)) {
                attrs.put(name(), value);
            } else {
                throw new UnsupportedOperationException();
            }
        }

        default <T> T access(Class<T> type, Map<String, Object> attrs) {
            Object value = attrs.get(name());
            if (value == null && !isOptional()) {
                throw new NoSuchElementException();
            }
            return type.cast(value);
        }

        static Map<String, Object> process(ExternalizedOp eop,
                                           OnnxAttribute[] attributes) {
            Map<String, Object> attrs = new HashMap<>();
            for (OnnxAttribute attribute : attributes) {
                Object v = eop.attributes().get(attribute.name());
                if (v == null && !attribute.isOptional()) {
                    throw new NoSuchElementException(attribute.name());
                }
                attribute.process(attrs, v);
            }

            return Map.copyOf(attrs);
        }

        static Map<String, Object> process(ExternalizedOp eop,
                                           List<OnnxAttribute> attributes) {
            Map<String, Object> attrs = new HashMap<>();
            for (OnnxAttribute attribute : attributes) {
                Object v = eop.attributes().get(attribute.name());
                if (v == null && !attribute.isOptional()) {
                    throw new NoSuchElementException(attribute.name());
                }
                attribute.process(attrs, v);
            }

            return Map.copyOf(attrs);
        }

        interface None extends OnnxAttribute {
            @Override
            default String name() {
                throw new UnsupportedOperationException();
            }

            @Override
            default Class<?> type() {
                throw new UnsupportedOperationException();
            }

            @Override
            default Object defaultValue() {
                throw new UnsupportedOperationException();
            }

            @Override
            default boolean isOptional() {
                throw new UnsupportedOperationException();
            }
        }

    }

    public interface OnnxTypeConstraint {
        String name();

        OnnxType.TypeVariable typeVariable();

        interface None extends OnnxTypeConstraint {
            @Override
            default String name() {
                throw new UnsupportedOperationException();
            }

            @Override
            default OnnxType.TypeVariable typeVariable() {
                throw new UnsupportedOperationException();
            }
        }
    }

    public interface OnnxParameter {
        enum Quantifier {
            REQUIRED, // Exactly once
            OPTIONAL, // Once or none
            VARIADIC, // One or more
            ;

            public boolean isOptional() {
                return this == OPTIONAL;
            }

            public boolean isRequired() {
                return this == REQUIRED;
            }

            public boolean isVariadoc() {
                return this == VARIADIC;
            }
        }

        String name();

        OnnxType type();

        Quantifier quantifier();

        interface None extends OnnxParameter {
            @Override
            default String name() {
                throw new UnsupportedOperationException();
            }

            @Override
            default OnnxType type() {
                throw new UnsupportedOperationException();
            }

            @Override
            default Quantifier quantifier() {
                throw new UnsupportedOperationException();
            }
        }
    }

    public interface OnnxSchema {
        String name();

        List<OnnxAttribute> attributes();

        List<OnnxTypeConstraint> typeConstraints();

        List<OnnxParameter> inputs();

        List<OnnxParameter> outputs();
    }

    record OnnxSchemaRecord(
            String name,
            List<OnnxAttribute> attributes,
            List<OnnxTypeConstraint> typeConstraints,
            List<OnnxParameter> inputs,
            List<OnnxParameter> outputs
    ) implements OnnxSchema {}

    static List<Value> concatValues(Value operand) {
        return List.of(operand);
    }

    static List<Value> concatValues(Value... operands) {
        return List.of(operands);
    }

    static List<Value> concatValues(List<Object> operands) {
        return concatValues(operands.toArray());
    }

    static List<Value> concatValues(Object... operands) {
        List<Value> l = new ArrayList<>();
        for (Object operand : operands) {
            switch (operand) {
                case Value v -> l.add(v);
                case Optional<?> ov -> {
                    if (ov.isPresent()) {
                        l.add((Value) ov.get());
                    }
                }
                case List<?> vs -> {
                    for (Object v : vs) {
                        l.add((Value) v);
                    }
                }
                default -> throw new UnsupportedOperationException();
            }
        }
        return l;
    }

    static final String ATTRIBUTE_OPTIONAL_INPUTS = "optional_inputs";
    static final String ATTRIBUTE_OPTIONAL_OUTPUTS = "optional_outputs";

    final Map<String, Object> onnxAttributes;
    final TypeElement resultType;
    final List<OnnxParameter> optionalInputArguments;
    final List<OnnxParameter> optionalOutputParameters;

    @SuppressWarnings("unchecked")
    OnnxOp(OnnxSchema schema, ExternalizedOp def) {
        super(def);

        this.onnxAttributes = schema.attributes().isEmpty()
                ? Map.of()
                : OnnxAttribute.process(def, schema.attributes());
        this.resultType = def.resultType();

        // @@@ Filter optional
        this.optionalInputArguments = def.extractAttributeValue(ATTRIBUTE_OPTIONAL_INPUTS,
                false, v -> switch (v) {
                    case List<?> s -> (List<OnnxParameter>) s;
                    case null -> List.of();
                    default -> throw new UnsupportedOperationException();
                });

        // @@@ Filter optional
        this.optionalOutputParameters = def.extractAttributeValue(ATTRIBUTE_OPTIONAL_OUTPUTS,
                false, v -> switch (v) {
                    case List<?> s -> (List<OnnxParameter>) s;
                    case null -> List.of();
                    default -> throw new UnsupportedOperationException();
                });
    }

    OnnxOp(OnnxOp that, CopyContext cc) {
        super(that, cc);

        this.onnxAttributes = Map.copyOf(that.onnxAttributes);
        this.resultType = that.resultType;
        this.optionalInputArguments = List.copyOf(that.optionalInputArguments);
        this.optionalOutputParameters = List.copyOf(that.optionalOutputParameters);
    }

    OnnxOp(OnnxSchema schema, TypeElement resultType,
           Set<? extends OnnxParameter> optionalOutputParameters,
           List<Object> inputArguments,
           List<Object> attributeValues) {
        super(schema.name(), concatValues(inputArguments));

        this.resultType = resultType;

        // Optional output parameters

        if (!optionalOutputParameters.isEmpty()) {
            List<OnnxParameter> l = new ArrayList<>();

            for (int i = 0; i < schema.outputs().size(); i++) {
                OnnxParameter p = schema.outputs().get(i);
                if (p.quantifier().isOptional()
                        && optionalOutputParameters.contains(p)) {
                    l.add(p);
                }
            }
            this.optionalOutputParameters = List.copyOf(l);
        } else {
            this.optionalOutputParameters = List.of();
        }

        // Optional input parameters

        if (!inputArguments.isEmpty()) {
            List<OnnxParameter> l = new ArrayList<>();

            for (int i = 0; i < schema.inputs().size(); i++) {
                OnnxParameter p = schema.inputs().get(i);
                if (p.quantifier().isOptional()) {
                    assert inputArguments.get(i) instanceof Optional;
                    if (inputArguments.get(i) instanceof Optional<?> optionalValue
                            && optionalValue.isPresent()) {
                        l.add(p);
                    }
                }
            }
            if (!l.isEmpty()) {
                this.optionalInputArguments = List.copyOf(l);
            } else {
                this.optionalInputArguments = List.of();
            }
        } else {
            this.optionalInputArguments = List.of();
        }

        // Attributes

        if (!attributeValues.isEmpty()) {
            Map<String, Object> attrs = new HashMap<>();
            assert schema.attributes().size() == attributeValues.size();
            for (int i = 0; i < schema.attributes().size(); i++) {
                schema.attributes().get(i).process(attrs, attributeValues.get(i));
            }
            this.onnxAttributes = Map.copyOf(attrs);
        } else {
            this.onnxAttributes = Map.of();
        }
    }

    @Override
    public TypeElement resultType() {
        return resultType;
    }

    @Override
    public Map<String, Object> attributes() {
        HashMap<String, Object> m = new HashMap<>(super.attributes());
        m.putAll(onnxAttributes);
        if (!optionalInputArguments.isEmpty()) {
            m.put(ATTRIBUTE_OPTIONAL_INPUTS, optionalInputArguments);
        }
        if (!optionalOutputParameters.isEmpty()) {
            m.put(ATTRIBUTE_OPTIONAL_OUTPUTS, optionalOutputParameters);
        }
        return m;
    }

    // @@@ Change to Map<OnnxAttribute, Object>
    public Map<String, Object> onnxAttributes() {
        return onnxAttributes;
    }

    public SequencedSet<OnnxParameter> onnxOutputs() {
        return Collections.emptyNavigableSet();
    }

    SequencedSet<OnnxParameter> onnxOutputs(OnnxSchema schema) {
        LinkedHashSet<OnnxParameter> s = new LinkedHashSet<>();
        for (OnnxParameter p : schema.outputs()) {
            if (!p.quantifier().isOptional() || optionalOutputParameters.contains(p)) {
                s.add(p);
            }
        }

        return s;
    }

    public SequencedMap<OnnxParameter, Object> onnxInputs() {
        return Collections.emptyNavigableMap();
    }

    SequencedMap<OnnxParameter, Object> onnxInputs(OnnxSchema schema, List<Object> inputArguments) {
        assert schema.inputs().size() == inputArguments.size();
        if (!inputArguments.isEmpty()) {
            SequencedMap<OnnxParameter, Object> inputs = new LinkedHashMap<>();
            for (int i = 0; i < schema.outputs().size(); i++) {
                inputs.put(schema.outputs().get(i), inputArguments.get(i));
            }
            return inputs;
        } else {
            return Collections.emptyNavigableMap();
        }
    }
}