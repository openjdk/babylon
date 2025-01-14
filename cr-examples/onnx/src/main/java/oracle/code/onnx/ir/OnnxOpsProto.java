package oracle.code.onnx.ir;

import jdk.incubator.code.*;
import jdk.incubator.code.op.OpFactory;

import java.util.*;

public final class OnnxOpsProto {
    private OnnxOpsProto() {
    }

    public enum Attribute implements OnnxOp.OnnxAttribute.None {}

    @OpFactory.OpDeclaration(Add.NAME)
    public static final class Add extends OnnxOp {
        public static final String NAME = "Add";

        enum TypeConstraint implements OnnxTypeConstraint {
            T(new OnnxType.TypeVariable("T",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.UInt8Type()),
                            new OnnxType.TensorType(new OnnxType.UInt16Type())
                            // @@@ ...
                    )));

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        enum InputParameter implements OnnxParameter {
            A(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            B(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        enum OutputParameter implements OnnxParameter {
            C(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(Attribute.values()),
                List.of(TypeConstraint.values()),
                List.of(InputParameter.values()),
                List.of(OutputParameter.values())
        );

        public Add(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        Add(Add that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Add transform(CopyContext cc, OpTransformer ot) {
            return new Add(this, cc);
        }

        Add(TypeElement resultType, Value A, Value B) {
            super(SCHEMA, resultType,
                    Set.of(),
                    List.of(A, B),
                    List.of());
        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            return onnxOutputs(SCHEMA);
        }

        // Operand accessors

        @Override
        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            return onnxInputs(SCHEMA, List.of(A(), B()));
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

            public boolean isOptional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum TypeConstraint implements OnnxTypeConstraint {
            T(new OnnxType.TypeVariable("T",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.BFloat16Type()),
                            new OnnxType.TensorType(new OnnxType.Float16Type())
                            // ...
                    )));

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        enum InputParameter implements OnnxParameter {
            X(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        enum OutputParameter implements OnnxParameter {
            Y(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(OnnxOpsProto.Attribute.values()),
                List.of(Add.TypeConstraint.values()),
                List.of(Add.InputParameter.values()),
                List.of(Add.OutputParameter.values())
        );

        public AveragePool(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        AveragePool(AveragePool that, CopyContext cc) {
            super(that, cc);
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
            super(SCHEMA, resultType,
                    Set.of(),
                    List.of(X),
                    List.of(pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape));
        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> outputs = new LinkedHashSet<>();
            outputs.add(OutputParameter.Y);
            return outputs;
        }

        // Operand accessors

        @Override
        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            SequencedMap<OnnxParameter, Value> inputs = new LinkedHashMap<>();
            inputs.put(InputParameter.X, X());
            return Collections.unmodifiableSequencedMap(inputs);
        }

        public Value X() {
            return operands().get(0);
        }

        // Attribute accessors

        public Optional<int[]> pads() {
            int[] pads = Attribute.pads.access(int[].class, onnxAttributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }

        public Optional<int[]> dilations() {
            int[] pads = Attribute.pads.access(int[].class, onnxAttributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }

        public Optional<String> auto_pad() {
            String auto_pad = Attribute.auto_pad.access(String.class, onnxAttributes);
            return Optional.ofNullable(auto_pad);
        }

        public Optional<Integer> count_include_pad() {
            Integer count_include_pad = Attribute.count_include_pad.access(Integer.class, onnxAttributes);
            return Optional.ofNullable(count_include_pad);
        }

        public Optional<Integer> ceil_mode() {
            Integer ceil_mode = Attribute.ceil_mode.access(Integer.class, onnxAttributes);
            return Optional.ofNullable(ceil_mode);
        }

        public Optional<int[]> strides() {
            int[] strides = Attribute.strides.access(int[].class, onnxAttributes);
            return Optional.ofNullable(strides).map(int[]::clone);
        }

        public int[] kernel_shape() {
            int[] kernel_shape = Attribute.kernel_shape.access(int[].class, onnxAttributes);
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

            public boolean isOptional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum TypeConstraint implements OnnxTypeConstraint {
            T1(new OnnxType.TypeVariable("T1",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.BFloat16Type()),
                            new OnnxType.TensorType(new OnnxType.Float16Type())
                            // @@@ ...
                    ))),
            T2(new OnnxType.TypeVariable("T2",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.Int32Type()),
                            new OnnxType.TensorType(new OnnxType.Int64Type())
                            // @@@ ...
                    ))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        enum InputParameter implements OnnxParameter {
            input(TypeConstraint.T1.typeVariable(), Quantifier.REQUIRED),
            dft_length(TypeConstraint.T2.typeVariable(), Quantifier.OPTIONAL),
            axis(new OnnxType.TensorType(new OnnxType.Int64Type()), Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return null;
            }

            @Override
            public Quantifier quantifier() {
                return null;
            }
        }

        enum OutputParameter implements OnnxParameter {
            output(TypeConstraint.T1.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return null;
            }

            @Override
            public Quantifier quantifier() {
                return null;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(OnnxOpsProto.Attribute.values()),
                List.of(Add.TypeConstraint.values()),
                List.of(Add.InputParameter.values()),
                List.of(Add.OutputParameter.values())
        );

        @SuppressWarnings("unchecked")
        public DFT(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        DFT(DFT that, CopyContext cc) {
            super(that, cc);
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
            super(SCHEMA, resultType,
                    Set.of(),
                    List.of(input, dft_length, axis),
                    List.of(inverse, onesided));
        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> outputs = new LinkedHashSet<>();
            outputs.add(OutputParameter.output);
            return outputs;
        }

        // Operand accessors

        @Override
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
            int i = optionalInputArguments.indexOf(InputParameter.dft_length);
            return i != -1
                    ? Optional.of(operands().get(1 + i))
                    : Optional.empty();
        }

        public Optional<Value> axis() {
            int i = optionalInputArguments.indexOf(InputParameter.axis);
            return i != -1
                    ? Optional.of(operands().get(1 + i))
                    : Optional.empty();
        }

        // Attribute accessors

        public Optional<Integer> inverse() {
            Integer inverse = Attribute.inverse.access(Integer.class, onnxAttributes);
            return Optional.ofNullable(inverse);
        }

        public Optional<Integer> onesided() {
            Integer onesided = Attribute.onesided.access(Integer.class, onnxAttributes);
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

            public boolean isOptional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum TypeConstraint implements OnnxTypeConstraint {
            T(new OnnxType.TypeVariable("T",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.Float16Type()),
                            new OnnxType.TensorType(new OnnxType.Float32Type())
                            // @@@ ...
                    ))),
            Tind(new OnnxType.TypeVariable("Tind",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.Int32Type()),
                            new OnnxType.TensorType(new OnnxType.Int64Type())
                            // @@@ ...
                    ))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        enum InputParameter implements OnnxParameter {
            scores(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            labels(TypeConstraint.Tind.typeVariable(), Quantifier.REQUIRED),
            weights(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        enum OutputParameter implements OnnxParameter {
            output(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            log_prob(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(OnnxOpsProto.Attribute.values()),
                List.of(Add.TypeConstraint.values()),
                List.of(Add.InputParameter.values()),
                List.of(Add.OutputParameter.values())
        );

        @SuppressWarnings("unchecked")
        public SoftmaxCrossEntropyLoss(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        SoftmaxCrossEntropyLoss(SoftmaxCrossEntropyLoss that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SoftmaxCrossEntropyLoss transform(CopyContext cc, OpTransformer ot) {
            return new SoftmaxCrossEntropyLoss(this, cc);
        }

        SoftmaxCrossEntropyLoss(TypeElement resultType,
                                Set<OutputParameter> optionalOutputs,
                                // Required Operands
                                Value scores,
                                Value labels,
                                // Optional operands
                                Optional<Value> weights,
                                // Attributes
                                Optional<Integer> ignore_index,
                                Optional<String> reduction) {
            super(SCHEMA, resultType,
                    optionalOutputs,
                    List.of(scores, labels, weights),
                    List.of(ignore_index, reduction));
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
        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> inputs = new LinkedHashSet<>();
            inputs.add(OutputParameter.output);
            if (optionalOutputParameters.contains(OutputParameter.log_prob)) {
                inputs.add(OutputParameter.log_prob);
            }
            return inputs;
        }

        // Operand accessors

        @Override
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
            int i = optionalInputArguments.indexOf(InputParameter.weights);
            return i != -1
                    ? Optional.of(operands().get(2 + i))
                    : Optional.empty();
        }


        // Attribute accessors

        public Optional<Integer> ignore_index() {
            Integer ignore_index = Attribute.ignore_index.access(Integer.class, onnxAttributes);
            return Optional.ofNullable(ignore_index);
        }

        public Optional<String> reduction() {
            String reduction = Attribute.reduction.access(String.class, onnxAttributes);
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

            public boolean isOptional() {
                return optional;
            }

            public Object defaultValue() {
                return defaultValue;
            }
        }

        enum TypeConstraint implements OnnxTypeConstraint {
            T1(new OnnxType.TypeVariable("T1",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.Float32Type()),
                            new OnnxType.TensorType(new OnnxType.Float64Type())
                    ))),
            T2(new OnnxType.TypeVariable("T2",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.Int64Type())
                    ))),
            T3(new OnnxType.TypeVariable("T3",
                    List.of(
                            new OnnxType.TensorType(new OnnxType.Float32Type()),
                            new OnnxType.TensorType(new OnnxType.Float64Type())
                    ))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        enum InputParameter implements OnnxParameter {
            R(TypeConstraint.T1.typeVariable(), Quantifier.REQUIRED),
            T(TypeConstraint.T2.typeVariable(), Quantifier.REQUIRED),
            inputs(TypeConstraint.T3.typeVariable(), Quantifier.VARIADIC),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        enum OutputParameter implements OnnxParameter {
            outputs(TypeConstraint.T3.typeVariable(), Quantifier.VARIADIC),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(OnnxOpsProto.Attribute.values()),
                List.of(Add.TypeConstraint.values()),
                List.of(Add.InputParameter.values()),
                List.of(Add.OutputParameter.values())
        );

        @SuppressWarnings("unchecked")
        public Adagrad(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        Adagrad(Adagrad that, CopyContext cc) {
            super(that, cc);
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
            super(SCHEMA, resultType,
                    Set.of(),
                    List.of(R, T, inputs),
                    List.of(norm_coefficient, decay_factor, epsilon));
        }

//        public TypeElement resultType() {
//            // @@@ Seq<T> size of N inputs or Tuple with N components
//            return operands().get(2).type();
//        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            SequencedSet<OnnxParameter> outputs = new LinkedHashSet<>();
            outputs.add(OutputParameter.outputs);
            return outputs;
        }

        // Operand accessors

        @Override
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
            Float norm_coefficient = Attribute.norm_coefficient.access(Float.class, onnxAttributes);
            return Optional.ofNullable(norm_coefficient);
        }

        public Optional<Float> decay_factor() {
            Float decay_factor = Attribute.decay_factor.access(Float.class, onnxAttributes);
            return Optional.ofNullable(decay_factor);
        }

        public Optional<Float> epsilon() {
            Float epsilon = Attribute.epsilon.access(Float.class, onnxAttributes);
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
