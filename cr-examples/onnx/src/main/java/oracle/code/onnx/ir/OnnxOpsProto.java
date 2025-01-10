package oracle.code.onnx.ir;

import jdk.incubator.code.*;
import jdk.incubator.code.op.OpFactory;

import java.util.*;

public final class OnnxOpsProto {
    private OnnxOpsProto() {
    }

    public interface OnnxParameter {
        String name();
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

    @OpFactory.OpDeclaration(Add.NAME)
    public static final class Add extends OnnxOp {
        public static final String NAME = "Add";

        enum InputParameter implements OnnxParameter {
            A(null, false),
            B(null, false),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            InputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        enum OutputParameter implements OnnxParameter {
            C(null, false),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            OutputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        public Add(ExternalizedOp def) {
            super(def);
        }

        Add(Add that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Add transform(CopyContext cc, OpTransformer ot) {
            return new Add(this, cc);
        }

        Add(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }

        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> outputs = new LinkedHashSet<>();
            outputs.add(OutputParameter.C);
            return outputs;
        }

        // Operand accessors

        public SequencedMap<OnnxParameter, Value> onnxInputs() {
            SequencedMap<OnnxParameter, Value> inputs = new LinkedHashMap<>();
            inputs.put(InputParameter.A, A());
            inputs.put(InputParameter.B, B());
            return Collections.unmodifiableSequencedMap(inputs);
        }

        public Value A() {
            return operands().get(0);
        }

        public Value B() {
            return operands().get(1);
        }
    }

    public static Add Add(TypeElement resultType, Value A, Value B) {
        return new Add(resultType, A, B);
    }

    @OpFactory.OpDeclaration(AveragePool.NAME)
    public static final class AveragePool extends OnnxOp {
        public static final String NAME = "AveragePool";

        public enum Attribute implements OnnxAttribute {
            pads(int[].class, true, null),
            dilations(int[].class, true, null),
            auto_pad(String.class, true, "NOTSET"),
            count_include_pad(Integer.class, true, 0),
            ceil_mode(Integer.class, true, 0),
            strides(int[].class, true, null),
            kernel_shape(int[].class, false, null),
            ;

            final Class<?> t;
            final boolean optional;
            final Object defaultValue;

            Attribute(Class<?> type, boolean optional, Object defaultValue) {
                this.t = type;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }

            public Class<?> type() {
                return t;
            }

            public boolean optional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        final Map<String, Object> attributes;

        public AveragePool(ExternalizedOp def) {
            super(def);

            // @@@ Validate type constraints for X operand

            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }

        AveragePool(AveragePool that, CopyContext cc) {
            super(that, cc);

            this.attributes = Map.copyOf(that.attributes);
        }

        @Override
        public AveragePool transform(CopyContext cc, OpTransformer ot) {
            return new AveragePool(this, cc);
        }

        AveragePool(TypeElement resultType,
                    Value X,
                    Optional<int[]> pads,
                    Optional<int[]> dilations,
                    Optional<String> auto_pad,
                    Optional<Integer> count_include_pad,
                    Optional<Integer> ceil_mode,
                    Optional<int[]> strides,
                    int[] kernel_shape) {
            super(NAME, resultType, List.of(X));

            // @@@ Validate type constraints for X

            Map<String, Object> attrs = new HashMap<>();
            Attribute.pads.process(attrs, pads);
            Attribute.dilations.process(attrs, dilations);
            Attribute.auto_pad.process(attrs, auto_pad);
            Attribute.count_include_pad.process(attrs, count_include_pad);
            Attribute.ceil_mode.process(attrs, ceil_mode);
            Attribute.strides.process(attrs, strides);
            Attribute.kernel_shape.process(attrs, kernel_shape);
            this.attributes = Map.copyOf(attrs);
        }

        @Override
        public Map<String, Object> onnxAttributes() {
            return attributes;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.putAll(onnxAttributes());
            return Collections.unmodifiableMap(m);
        }

        // Operand accessors

        public Value X() {
            return operands().get(0);
        }

        // Attribute accessors

        public Optional<int[]> pads() {
            int[] pads = Attribute.pads.access(int[].class, attributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }

        public Optional<int[]> dilations() {
            int[] pads = Attribute.pads.access(int[].class, attributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }

        public Optional<String> auto_pad() {
            String auto_pad = Attribute.auto_pad.access(String.class, attributes);
            return Optional.ofNullable(auto_pad);
        }

        public Optional<Integer> count_include_pad() {
            Integer count_include_pad = Attribute.count_include_pad.access(Integer.class, attributes);
            return Optional.ofNullable(count_include_pad);
        }

        public Optional<Integer> ceil_mode() {
            Integer ceil_mode = Attribute.ceil_mode.access(Integer.class, attributes);
            return Optional.ofNullable(ceil_mode);
        }

        public Optional<int[]> strides() {
            int[] strides = Attribute.strides.access(int[].class, attributes);
            return Optional.ofNullable(strides).map(int[]::clone);
        }

        public int[] kernel_shape() {
            int[] kernel_shape = Attribute.kernel_shape.access(int[].class, attributes);
            return kernel_shape;
        }
    }

    public static AveragePool AveragePool(TypeElement resultType,
                                          Value X,
                                          Optional<int[]> pads,
                                          Optional<int[]> dilations,
                                          Optional<String> auto_pad,
                                          Optional<Integer> count_include_pad,
                                          Optional<Integer> ceil_mode,
                                          Optional<int[]> strides,
                                          int[] kernel_shape) {
        return new AveragePool(resultType, X, pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape);
    }


    @OpFactory.OpDeclaration(DFT.NAME)
    public static final class DFT extends OnnxOp {
        public static final String NAME = "DFT";

        public enum Attribute implements OnnxAttribute {
            inverse(Integer.class, true, 0),
            onesided(Integer.class, true, 0),
            ;

            final Class<?> t;
            final boolean optional;
            final Object defaultValue;

            Attribute(Class<?> type, boolean optional, Object defaultValue) {
                this.t = type;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }

            public Class<?> type() {
                return t;
            }

            public boolean optional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum InputParameter implements OnnxParameter {
            input(null, false),
            dft_length(null, true),
            axis(null, true),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            InputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        enum OutputParameter implements OnnxParameter {
            output(null, false),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            OutputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        static final String ATTRIBUTE_OPTIONAL_INPUTS = "optional_inputs";

        final List<InputParameter> optionalInputs;

        final Map<String, Object> attributes;

        @SuppressWarnings("unchecked")
        public DFT(ExternalizedOp def) {
            super(def);

            this.optionalInputs = def.extractAttributeValue(ATTRIBUTE_OPTIONAL_INPUTS,
                    false, v -> switch (v) {
                        case List<?> s -> (List<InputParameter>) s;
                        case null, default -> throw new UnsupportedOperationException();
                    });

            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }

        DFT(DFT that, CopyContext cc) {
            super(that, cc);

            this.optionalInputs = new ArrayList<>(that.optionalInputs);
            this.attributes = Map.copyOf(that.attributes);
        }

        @Override
        public DFT transform(CopyContext cc, OpTransformer ot) {
            return new DFT(this, cc);
        }

        DFT(TypeElement resultType,
            Value input,
            // Optional operands
            Optional<Value> dft_length,
            Optional<Value> axis,
            // Attributes
            Optional<Integer> inverse,
            Optional<Integer> onesided) {
            super(NAME, resultType, concatValues(input, dft_length, axis));

            this.optionalInputs = new ArrayList<>();
            if (dft_length.isPresent()) {
                optionalInputs.add(InputParameter.dft_length);
            }
            if (axis.isPresent()) {
                optionalInputs.add(InputParameter.axis);
            }

            Map<String, Object> attrs = new HashMap<>();
            Attribute.inverse.process(attrs, inverse);
            Attribute.onesided.process(attrs, onesided);
            this.attributes = Map.copyOf(attrs);
        }

        @Override
        public Map<String, Object> onnxAttributes() {
            return attributes;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.putAll(onnxAttributes());
            m.put(ATTRIBUTE_OPTIONAL_INPUTS, optionalInputs);
            return Collections.unmodifiableMap(m);
        }

        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> outputs = new LinkedHashSet<>();
            outputs.add(OutputParameter.output);
            return outputs;
        }

        // Operand accessors

        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            SequencedMap<OnnxParameter, Object> inputs = new LinkedHashMap<>();
            inputs.put(InputParameter.input, input());
            inputs.put(InputParameter.dft_length, dft_length());
            inputs.put(InputParameter.axis, axis());
            return Collections.unmodifiableSequencedMap(inputs);
        }

        public Value input() {
            return operands().get(0);
        }

        public Optional<Value> dft_length() {
            int i = optionalInputs.indexOf(InputParameter.dft_length);
            return i != -1
                    ? Optional.of(operands().get(1 + i))
                    : Optional.empty();
        }

        public Optional<Value> axis() {
            int i = optionalInputs.indexOf(InputParameter.axis);
            return i != -1
                    ? Optional.of(operands().get(1 + i))
                    : Optional.empty();
        }

        // Attribute accessors

        public Optional<Integer> inverse() {
            Integer inverse = Attribute.inverse.access(Integer.class, attributes);
            return Optional.ofNullable(inverse);
        }

        public Optional<Integer> onesided() {
            Integer onesided = Attribute.onesided.access(Integer.class, attributes);
            return Optional.ofNullable(onesided);
        }
    }

    public static DFT DFT(TypeElement resultType,
                          Value input,
                          Optional<Value> dft_length,
                          Optional<Value> axis,
                          Optional<Integer> inverse,
                          Optional<Integer> onesided) {
        return new DFT(resultType, input, dft_length, axis, inverse, onesided);
    }


    @OpFactory.OpDeclaration(SoftmaxCrossEntropyLoss.NAME)
    public static final class SoftmaxCrossEntropyLoss extends OnnxOp {
        public static final String NAME = "SoftmaxCrossEntropyLoss";

        public enum Attribute implements OnnxAttribute {
            ignore_index(Integer.class, true, 0),
            reduction(String.class, true, "mean"),
            ;

            final Class<?> t;
            final boolean optional;
            final Object defaultValue;

            Attribute(Class<?> type, boolean optional, Object defaultValue) {
                this.t = type;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }

            public Class<?> type() {
                return t;
            }

            public boolean optional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum InputParameter implements OnnxParameter {
            scores(null, false),
            labels(null, false),
            weights(null, true),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            InputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        enum OutputParameter implements OnnxParameter {
            output(null, false),
            log_prob(null, true),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            OutputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        static final String ATTRIBUTE_OPTIONAL_INPUTS = "optional_inputs";
        static final String ATTRIBUTE_OPTIONAL_OUTPUTS = "optional_outputs";

        final List<InputParameter> optionalInputs;
        final SequencedSet<OutputParameter> optionalOutputs;

        final Map<String, Object> attributes;

        @SuppressWarnings("unchecked")
        public SoftmaxCrossEntropyLoss(ExternalizedOp def) {
            super(def);

            this.optionalInputs = def.extractAttributeValue(ATTRIBUTE_OPTIONAL_INPUTS,
                    false, v -> switch (v) {
                        case List<?> s -> (List<InputParameter>) s;
                        case null, default -> throw new UnsupportedOperationException();
                    });

            this.optionalOutputs = def.extractAttributeValue(ATTRIBUTE_OPTIONAL_OUTPUTS,
                    false, v -> switch (v) {
                        case SequencedSet<?> s -> (SequencedSet<OutputParameter>) s;
                        case null, default -> throw new UnsupportedOperationException();
                    });

            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }

        SoftmaxCrossEntropyLoss(SoftmaxCrossEntropyLoss that, CopyContext cc) {
            super(that, cc);

            this.optionalInputs = new ArrayList<>(that.optionalInputs);
            this.optionalOutputs = new LinkedHashSet<>(that.optionalOutputs);
            this.attributes = Map.copyOf(that.attributes);
        }

        @Override
        public SoftmaxCrossEntropyLoss transform(CopyContext cc, OpTransformer ot) {
            return new SoftmaxCrossEntropyLoss(this, cc);
        }

        SoftmaxCrossEntropyLoss(TypeElement resultType,
                                SequencedSet<OutputParameter> optionalOutputs,
                                // Required Operands
                                Value scores,
                                Value labels,
                                // Optional operands
                                Optional<Value> weights,
                                // Attributes
                                Optional<Integer> ignore_index,
                                Optional<String> reduction) {
            super(NAME, resultType, concatValues(scores, labels, weights));

            this.optionalInputs = new ArrayList<>();
            if (weights.isPresent()) {
                optionalInputs.add(InputParameter.weights);
            }

            this.optionalOutputs = new LinkedHashSet<>();
            for (OutputParameter optionalOutput : optionalOutputs) {
                if (optionalOutput.optional) {
                    this.optionalOutputs.add(optionalOutput);
                }
            }

            Map<String, Object> attrs = new HashMap<>();
            Attribute.ignore_index.process(attrs, ignore_index);
            Attribute.reduction.process(attrs, reduction);
            this.attributes = Map.copyOf(attrs);
        }

//        public TypeElement resultType() {
//            List<TypeElement> types = new ArrayList<>();
//            types.add(operands().get(0).type());
//            if (optionalOutputs.contains(OutputParameter.log_prob)) {
//                types.add(operands().get(0).type());
//            }
//            if (types.size() == 1) {
//                return types.getFirst();
//            } else {
//                return TupleType.tupleType(types);
//            }
//        }

        @Override
        public Map<String, Object> onnxAttributes() {
            return attributes;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.putAll(onnxAttributes());
            m.put(ATTRIBUTE_OPTIONAL_INPUTS, optionalInputs);
            m.put(ATTRIBUTE_OPTIONAL_OUTPUTS, optionalOutputs);
            return Collections.unmodifiableMap(m);
        }

        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> inputs = new LinkedHashSet<>();
            inputs.add(OutputParameter.output);
            if (optionalOutputs.contains(OutputParameter.log_prob)) {
                inputs.add(OutputParameter.log_prob);
            }
            return inputs;
        }

        // Operand accessors

        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            SequencedMap<OnnxParameter, Object> inputs = new LinkedHashMap<>();
            inputs.put(InputParameter.scores, scores());
            inputs.put(InputParameter.labels, labels());
            inputs.put(InputParameter.weights, weights());
            return Collections.unmodifiableSequencedMap(inputs);
        }

        public Value scores() {
            return operands().get(0);
        }

        public Value labels() {
            return operands().get(1);
        }

        public Optional<Value> weights() {
            int i = optionalInputs.indexOf(InputParameter.weights);
            return i != -1
                    ? Optional.of(operands().get(2 + i))
                    : Optional.empty();
        }


        // Attribute accessors

        public Optional<Integer> ignore_index() {
            Integer ignore_index = Attribute.ignore_index.access(Integer.class, attributes);
            return Optional.ofNullable(ignore_index);
        }

        public Optional<String> reduction() {
            String reduction = Attribute.reduction.access(String.class, attributes);
            return Optional.ofNullable(reduction);
        }
    }

    public SoftmaxCrossEntropyLoss SoftmaxCrossEntropyLoss(TypeElement resultType,
                                                           SequencedSet<SoftmaxCrossEntropyLoss.OutputParameter> optionalOutputs,
                                                           // Required Operands
                                                           Value scores,
                                                           Value labels,
                                                           // Optional operands
                                                           Optional<Value> weights,
                                                           // Attributes
                                                           Optional<Integer> ignore_index,
                                                           Optional<String> reduction) {
        return new SoftmaxCrossEntropyLoss(resultType, optionalOutputs, scores, labels, weights, ignore_index, reduction);
    }


    @OpFactory.OpDeclaration(Adagrad.NAME)
    public static final class Adagrad extends OnnxOp {
        public static final String NAME = "Adagrad";

        public enum Attribute implements OnnxAttribute {
            norm_coefficient(Float.class, true, 0),
            decay_factor(Float.class, true, 0),
            epsilon(Float.class, true, 1e-06),
            ;

            final Class<?> t;
            final boolean optional;
            final Object defaultValue;

            Attribute(Class<?> type, boolean optional, Object defaultValue) {
                this.t = type;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }

            public Class<?> type() {
                return t;
            }

            public boolean optional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum InputParameter implements OnnxParameter {
            R(null, false),
            T(null, false),
            inputs(null, false),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            InputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        enum OutputParameter implements OnnxParameter {
            outputs(null, false),
            ;

            final List<TypeElement> typeConstraints;
            final boolean optional;

            OutputParameter(List<TypeElement> typeConstraints, boolean optional) {
                this.typeConstraints = typeConstraints;
                this.optional = optional;
            }
        }

        final Map<String, Object> attributes;

        @SuppressWarnings("unchecked")
        public Adagrad(ExternalizedOp def) {
            super(def);

            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }

        Adagrad(Adagrad that, CopyContext cc) {
            super(that, cc);

            this.attributes = Map.copyOf(that.attributes);
        }

        @Override
        public Adagrad transform(CopyContext cc, OpTransformer ot) {
            return new Adagrad(this, cc);
        }

        Adagrad(TypeElement resultType,
                // Required Operands
                Value R,
                Value T,
                List<Value> inputs,
                // Attributes
                Optional<Float> norm_coefficient,
                Optional<Float> decay_factor,
                Optional<Float> epsilon) {
            super(NAME, resultType, concatValues(R, T, inputs));

            Map<String, Object> attrs = new HashMap<>();
            Attribute.norm_coefficient.process(attrs, norm_coefficient);
            Attribute.decay_factor.process(attrs, decay_factor);
            Attribute.epsilon.process(attrs, epsilon);
            this.attributes = Map.copyOf(attrs);
        }

//        public TypeElement resultType() {
//            // @@@ Seq<T> size of N inputs or Tuple with N components
//            return operands().get(2).type();
//        }

        @Override
        public Map<String, Object> onnxAttributes() {
            return attributes;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.putAll(onnxAttributes());
            return Collections.unmodifiableMap(m);
        }

        public SequencedSet<OnnxParameter> outputs() {
            SequencedSet<OnnxParameter> outputs = new LinkedHashSet<>();
            outputs.add(OutputParameter.outputs);
            return outputs;
        }

        // Operand accessors

        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            SequencedMap<OnnxParameter, Object> inputs = new LinkedHashMap<>();
            inputs.put(InputParameter.R, R());
            inputs.put(InputParameter.T, T());
            inputs.put(InputParameter.inputs, inputs());
            return Collections.unmodifiableSequencedMap(inputs);
        }

        public Value R() {
            return operands().get(0);
        }

        public Value T() {
            return operands().get(1);
        }

        public List<Value> inputs() {
            return operands().subList(2, operands().size());
        }

        // Attribute accessors

        public Optional<Float> norm_coefficient() {
            Float norm_coefficient = Attribute.norm_coefficient.access(Float.class, attributes);
            return Optional.ofNullable(norm_coefficient);
        }

        public Optional<Float> decay_factor() {
            Float decay_factor = Attribute.decay_factor.access(Float.class, attributes);
            return Optional.ofNullable(decay_factor);
        }

        public Optional<Float> epsilon() {
            Float epsilon = Attribute.epsilon.access(Float.class, attributes);
            return Optional.ofNullable(epsilon);
        }
    }

    public Adagrad Adagrad(TypeElement resultType,
                           // Required Operands
                           Value R,
                           Value T,
                           List<Value> inputs,
                           // Attributes
                           Optional<Float> norm_coefficient,
                           Optional<Float> decay_factor,
                           Optional<Float> epsilon) {
        return new Adagrad(resultType, R, T, inputs, norm_coefficient, decay_factor, epsilon);
    }

}
