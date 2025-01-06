package oracle.code.onnx.ir;

import jdk.incubator.code.*;
import jdk.incubator.code.op.OpFactory;

import java.util.*;

public final class OnnxOpsProto {
    private OnnxOpsProto() {
    }

    @OpFactory.OpDeclaration(Add.NAME)
    public static final class Add extends OnnxOp {
        public static final String NAME = "Add";

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

        Add(Value A, Value B) {
            super(NAME, List.of(A, B));
        }

        public TypeElement resultType() {
            return operands().get(0).type();
        }

        // Operand accessors

        public Value A() {
            return operands().get(0);
        }

        public Value B() {
            return operands().get(1);
        }
    }

    public static Add Add(Value A, Value B) {
        return new Add(A, B);
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

        AveragePool(Value X,
                    Optional<int[]> pads,
                    Optional<int[]> dilations,
                    Optional<String> auto_pad,
                    Optional<Integer> count_include_pad,
                    Optional<Integer> ceil_mode,
                    Optional<int[]> strides,
                    int[] kernel_shape) {
            super(NAME, List.of(X));

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

        public TypeElement resultType() {
            return operands().get(0).type();
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

    public static AveragePool AveragePool(Value X,
                                          Optional<int[]> pads,
                                          Optional<int[]> dilations,
                                          Optional<String> auto_pad,
                                          Optional<Integer> count_include_pad,
                                          Optional<Integer> ceil_mode,
                                          Optional<int[]> strides,
                                          int[] kernel_shape) {
        return new AveragePool(X, pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape);
    }
}

/*
Optional input parameters will need including modeling of what is
present/absent as an attribute e.g.,a bit set or names ordered by schema declaration,
with optional operands occurring after required operands.
This can be managed in the factory constructor and the operation access methods.
Variadic input parameters are the last parameters.

How do the return of optional output parameters get requested?
 */
