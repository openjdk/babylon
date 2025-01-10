// Auto-generated from ONNX op schema

package oracle.code.onnx.ir;

import jdk.incubator.code.*;
import jdk.incubator.code.op.OpFactory;
import oracle.code.onnx.Tensor;

import java.util.*;

public final class OnnxOps {
    
    private OnnxOps() {}
    
    @OpFactory.OpDeclaration(Abs.NAME)
    public static final class Abs extends OnnxOp {
        public static final String NAME = "Abs";
        
        public Abs(ExternalizedOp def) {
            super(def);
        }
        
        Abs(Abs that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Abs transform(CopyContext cc, OpTransformer ot) {
            return new Abs(this, cc);
        }
        
        Abs(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Abs Abs(TypeElement resultType, Value X) {
        return new Abs(resultType, X);
    }

    @OpFactory.OpDeclaration(Acos.NAME)
    public static final class Acos extends OnnxOp {
        public static final String NAME = "Acos";
        
        public Acos(ExternalizedOp def) {
            super(def);
        }
        
        Acos(Acos that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Acos transform(CopyContext cc, OpTransformer ot) {
            return new Acos(this, cc);
        }
        
        Acos(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Acos Acos(TypeElement resultType, Value input) {
        return new Acos(resultType, input);
    }

    @OpFactory.OpDeclaration(Acosh.NAME)
    public static final class Acosh extends OnnxOp {
        public static final String NAME = "Acosh";
        
        public Acosh(ExternalizedOp def) {
            super(def);
        }
        
        Acosh(Acosh that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Acosh transform(CopyContext cc, OpTransformer ot) {
            return new Acosh(this, cc);
        }
        
        Acosh(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Acosh Acosh(TypeElement resultType, Value input) {
        return new Acosh(resultType, input);
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
        
        Add(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
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

    @OpFactory.OpDeclaration(AffineGrid.NAME)
    public static final class AffineGrid extends OnnxOp {
        public static final String NAME = "AffineGrid";
        
        public enum Attribute implements OnnxAttribute {
            align_corners(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public AffineGrid(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        AffineGrid(AffineGrid that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public AffineGrid transform(CopyContext cc, OpTransformer ot) {
            return new AffineGrid(this, cc);
        }
        
        AffineGrid(TypeElement resultType, Value theta, Value size, Optional<Integer> align_corners) {
            super(NAME, resultType, List.of(theta, size));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.align_corners.process(attrs, align_corners);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value theta() {
            return operands().get(0);
        }
        
        public Value size() {
            return operands().get(1);
        }
        
        public Optional<Integer> align_corners() {
            Integer align_corners = Attribute.align_corners.access(Integer.class, attributes);
            return Optional.ofNullable(align_corners);
        }
        
    }
    
    public static AffineGrid AffineGrid(TypeElement resultType, Value theta, Value size, Optional<Integer> align_corners) {
        return new AffineGrid(resultType, theta, size, align_corners);
    }

    @OpFactory.OpDeclaration(And.NAME)
    public static final class And extends OnnxOp {
        public static final String NAME = "And";
        
        public And(ExternalizedOp def) {
            super(def);
        }
        
        And(And that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public And transform(CopyContext cc, OpTransformer ot) {
            return new And(this, cc);
        }
        
        And(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static And And(TypeElement resultType, Value A, Value B) {
        return new And(resultType, A, B);
    }

    @OpFactory.OpDeclaration(ArgMax.NAME)
    public static final class ArgMax extends OnnxOp {
        public static final String NAME = "ArgMax";
        
        public enum Attribute implements OnnxAttribute {
            keepdims(Integer.class, true, null),
            select_last_index(Integer.class, true, null),
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ArgMax(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ArgMax(ArgMax that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ArgMax transform(CopyContext cc, OpTransformer ot) {
            return new ArgMax(this, cc);
        }
        
        ArgMax(TypeElement resultType, Value data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
            super(NAME, resultType, List.of(data));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.keepdims.process(attrs, keepdims);
            Attribute.select_last_index.process(attrs, select_last_index);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Optional<Integer> keepdims() {
            Integer keepdims = Attribute.keepdims.access(Integer.class, attributes);
            return Optional.ofNullable(keepdims);
        }
        
        public Optional<Integer> select_last_index() {
            Integer select_last_index = Attribute.select_last_index.access(Integer.class, attributes);
            return Optional.ofNullable(select_last_index);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static ArgMax ArgMax(TypeElement resultType, Value data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
        return new ArgMax(resultType, data, keepdims, select_last_index, axis);
    }

    @OpFactory.OpDeclaration(ArgMin.NAME)
    public static final class ArgMin extends OnnxOp {
        public static final String NAME = "ArgMin";
        
        public enum Attribute implements OnnxAttribute {
            keepdims(Integer.class, true, null),
            select_last_index(Integer.class, true, null),
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ArgMin(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ArgMin(ArgMin that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ArgMin transform(CopyContext cc, OpTransformer ot) {
            return new ArgMin(this, cc);
        }
        
        ArgMin(TypeElement resultType, Value data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
            super(NAME, resultType, List.of(data));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.keepdims.process(attrs, keepdims);
            Attribute.select_last_index.process(attrs, select_last_index);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Optional<Integer> keepdims() {
            Integer keepdims = Attribute.keepdims.access(Integer.class, attributes);
            return Optional.ofNullable(keepdims);
        }
        
        public Optional<Integer> select_last_index() {
            Integer select_last_index = Attribute.select_last_index.access(Integer.class, attributes);
            return Optional.ofNullable(select_last_index);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static ArgMin ArgMin(TypeElement resultType, Value data, Optional<Integer> keepdims, Optional<Integer> select_last_index, Optional<Integer> axis) {
        return new ArgMin(resultType, data, keepdims, select_last_index, axis);
    }

    @OpFactory.OpDeclaration(ArrayFeatureExtractor.NAME)
    public static final class ArrayFeatureExtractor extends OnnxOp {
        public static final String NAME = "ArrayFeatureExtractor";
        
        public ArrayFeatureExtractor(ExternalizedOp def) {
            super(def);
        }
        
        ArrayFeatureExtractor(ArrayFeatureExtractor that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public ArrayFeatureExtractor transform(CopyContext cc, OpTransformer ot) {
            return new ArrayFeatureExtractor(this, cc);
        }
        
        ArrayFeatureExtractor(TypeElement resultType, Value X, Value Y) {
            super(NAME, resultType, List.of(X, Y));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
    }
    
    public static ArrayFeatureExtractor ArrayFeatureExtractor(TypeElement resultType, Value X, Value Y) {
        return new ArrayFeatureExtractor(resultType, X, Y);
    }

    @OpFactory.OpDeclaration(Asin.NAME)
    public static final class Asin extends OnnxOp {
        public static final String NAME = "Asin";
        
        public Asin(ExternalizedOp def) {
            super(def);
        }
        
        Asin(Asin that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Asin transform(CopyContext cc, OpTransformer ot) {
            return new Asin(this, cc);
        }
        
        Asin(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Asin Asin(TypeElement resultType, Value input) {
        return new Asin(resultType, input);
    }

    @OpFactory.OpDeclaration(Asinh.NAME)
    public static final class Asinh extends OnnxOp {
        public static final String NAME = "Asinh";
        
        public Asinh(ExternalizedOp def) {
            super(def);
        }
        
        Asinh(Asinh that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Asinh transform(CopyContext cc, OpTransformer ot) {
            return new Asinh(this, cc);
        }
        
        Asinh(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Asinh Asinh(TypeElement resultType, Value input) {
        return new Asinh(resultType, input);
    }

    @OpFactory.OpDeclaration(Atan.NAME)
    public static final class Atan extends OnnxOp {
        public static final String NAME = "Atan";
        
        public Atan(ExternalizedOp def) {
            super(def);
        }
        
        Atan(Atan that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Atan transform(CopyContext cc, OpTransformer ot) {
            return new Atan(this, cc);
        }
        
        Atan(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Atan Atan(TypeElement resultType, Value input) {
        return new Atan(resultType, input);
    }

    @OpFactory.OpDeclaration(Atanh.NAME)
    public static final class Atanh extends OnnxOp {
        public static final String NAME = "Atanh";
        
        public Atanh(ExternalizedOp def) {
            super(def);
        }
        
        Atanh(Atanh that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Atanh transform(CopyContext cc, OpTransformer ot) {
            return new Atanh(this, cc);
        }
        
        Atanh(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Atanh Atanh(TypeElement resultType, Value input) {
        return new Atanh(resultType, input);
    }

    @OpFactory.OpDeclaration(AveragePool.NAME)
    public static final class AveragePool extends OnnxOp {
        public static final String NAME = "AveragePool";
        
        public enum Attribute implements OnnxAttribute {
            pads(int[].class, true, null),
            dilations(int[].class, true, null),
            auto_pad(String.class, true, null),
            count_include_pad(Integer.class, true, null),
            ceil_mode(Integer.class, true, null),
            strides(int[].class, true, null),
            kernel_shape(int[].class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
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
        
        AveragePool(TypeElement resultType, Value X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> count_include_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.pads.process(attrs, pads.map(int[]::clone));
            Attribute.dilations.process(attrs, dilations.map(int[]::clone));
            Attribute.auto_pad.process(attrs, auto_pad);
            Attribute.count_include_pad.process(attrs, count_include_pad);
            Attribute.ceil_mode.process(attrs, ceil_mode);
            Attribute.strides.process(attrs, strides.map(int[]::clone));
            Attribute.kernel_shape.process(attrs, kernel_shape.clone());
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> pads() {
            int[] pads = Attribute.pads.access(int[].class, attributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }
        
        public Optional<int[]> dilations() {
            int[] dilations = Attribute.dilations.access(int[].class, attributes);
            return Optional.ofNullable(dilations).map(int[]::clone);
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
            return kernel_shape.clone();
        }
        
    }
    
    public static AveragePool AveragePool(TypeElement resultType, Value X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> count_include_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
        return new AveragePool(resultType, X, pads, dilations, auto_pad, count_include_pad, ceil_mode, strides, kernel_shape);
    }

    @OpFactory.OpDeclaration(BatchNormalization.NAME)
    public static final class BatchNormalization extends OnnxOp {
        public static final String NAME = "BatchNormalization";
        
        public enum Attribute implements OnnxAttribute {
            epsilon(Float.class, true, null),
            training_mode(Integer.class, true, null),
            momentum(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public BatchNormalization(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        BatchNormalization(BatchNormalization that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public BatchNormalization transform(CopyContext cc, OpTransformer ot) {
            return new BatchNormalization(this, cc);
        }
        
        BatchNormalization(TypeElement resultType, Value X, Value scale, Value B, Value input_mean, Value input_var, Optional<Float> epsilon, Optional<Integer> training_mode, Optional<Float> momentum) {
            super(NAME, resultType, List.of(X, scale, B, input_mean, input_var));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.epsilon.process(attrs, epsilon);
            Attribute.training_mode.process(attrs, training_mode);
            Attribute.momentum.process(attrs, momentum);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value scale() {
            return operands().get(1);
        }
        
        public Value B() {
            return operands().get(2);
        }
        
        public Value input_mean() {
            return operands().get(3);
        }
        
        public Value input_var() {
            return operands().get(4);
        }
        
        public Optional<Float> epsilon() {
            Float epsilon = Attribute.epsilon.access(Float.class, attributes);
            return Optional.ofNullable(epsilon);
        }
        
        public Optional<Integer> training_mode() {
            Integer training_mode = Attribute.training_mode.access(Integer.class, attributes);
            return Optional.ofNullable(training_mode);
        }
        
        public Optional<Float> momentum() {
            Float momentum = Attribute.momentum.access(Float.class, attributes);
            return Optional.ofNullable(momentum);
        }
        
    }
    
    public static BatchNormalization BatchNormalization(TypeElement resultType, Value X, Value scale, Value B, Value input_mean, Value input_var, Optional<Float> epsilon, Optional<Integer> training_mode, Optional<Float> momentum) {
        return new BatchNormalization(resultType, X, scale, B, input_mean, input_var, epsilon, training_mode, momentum);
    }

    @OpFactory.OpDeclaration(Bernoulli.NAME)
    public static final class Bernoulli extends OnnxOp {
        public static final String NAME = "Bernoulli";
        
        public enum Attribute implements OnnxAttribute {
            seed(Float.class, true, null),
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Bernoulli(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Bernoulli(Bernoulli that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Bernoulli transform(CopyContext cc, OpTransformer ot) {
            return new Bernoulli(this, cc);
        }
        
        Bernoulli(TypeElement resultType, Value input, Optional<Float> seed, Optional<Integer> dtype) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.seed.process(attrs, seed);
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Float> seed() {
            Float seed = Attribute.seed.access(Float.class, attributes);
            return Optional.ofNullable(seed);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static Bernoulli Bernoulli(TypeElement resultType, Value input, Optional<Float> seed, Optional<Integer> dtype) {
        return new Bernoulli(resultType, input, seed, dtype);
    }

    @OpFactory.OpDeclaration(Binarizer.NAME)
    public static final class Binarizer extends OnnxOp {
        public static final String NAME = "Binarizer";
        
        public enum Attribute implements OnnxAttribute {
            threshold(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Binarizer(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Binarizer(Binarizer that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Binarizer transform(CopyContext cc, OpTransformer ot) {
            return new Binarizer(this, cc);
        }
        
        Binarizer(TypeElement resultType, Value X, Optional<Float> threshold) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.threshold.process(attrs, threshold);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> threshold() {
            Float threshold = Attribute.threshold.access(Float.class, attributes);
            return Optional.ofNullable(threshold);
        }
        
    }
    
    public static Binarizer Binarizer(TypeElement resultType, Value X, Optional<Float> threshold) {
        return new Binarizer(resultType, X, threshold);
    }

    @OpFactory.OpDeclaration(BitShift.NAME)
    public static final class BitShift extends OnnxOp {
        public static final String NAME = "BitShift";
        
        public enum Attribute implements OnnxAttribute {
            direction(String.class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public BitShift(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        BitShift(BitShift that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public BitShift transform(CopyContext cc, OpTransformer ot) {
            return new BitShift(this, cc);
        }
        
        BitShift(TypeElement resultType, Value X, Value Y, String direction) {
            super(NAME, resultType, List.of(X, Y));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.direction.process(attrs, direction);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
        public String direction() {
            String direction = Attribute.direction.access(String.class, attributes);
            return direction;
        }
        
    }
    
    public static BitShift BitShift(TypeElement resultType, Value X, Value Y, String direction) {
        return new BitShift(resultType, X, Y, direction);
    }

    @OpFactory.OpDeclaration(BitwiseAnd.NAME)
    public static final class BitwiseAnd extends OnnxOp {
        public static final String NAME = "BitwiseAnd";
        
        public BitwiseAnd(ExternalizedOp def) {
            super(def);
        }
        
        BitwiseAnd(BitwiseAnd that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public BitwiseAnd transform(CopyContext cc, OpTransformer ot) {
            return new BitwiseAnd(this, cc);
        }
        
        BitwiseAnd(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static BitwiseAnd BitwiseAnd(TypeElement resultType, Value A, Value B) {
        return new BitwiseAnd(resultType, A, B);
    }

    @OpFactory.OpDeclaration(BitwiseNot.NAME)
    public static final class BitwiseNot extends OnnxOp {
        public static final String NAME = "BitwiseNot";
        
        public BitwiseNot(ExternalizedOp def) {
            super(def);
        }
        
        BitwiseNot(BitwiseNot that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public BitwiseNot transform(CopyContext cc, OpTransformer ot) {
            return new BitwiseNot(this, cc);
        }
        
        BitwiseNot(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static BitwiseNot BitwiseNot(TypeElement resultType, Value X) {
        return new BitwiseNot(resultType, X);
    }

    @OpFactory.OpDeclaration(BitwiseOr.NAME)
    public static final class BitwiseOr extends OnnxOp {
        public static final String NAME = "BitwiseOr";
        
        public BitwiseOr(ExternalizedOp def) {
            super(def);
        }
        
        BitwiseOr(BitwiseOr that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public BitwiseOr transform(CopyContext cc, OpTransformer ot) {
            return new BitwiseOr(this, cc);
        }
        
        BitwiseOr(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static BitwiseOr BitwiseOr(TypeElement resultType, Value A, Value B) {
        return new BitwiseOr(resultType, A, B);
    }

    @OpFactory.OpDeclaration(BitwiseXor.NAME)
    public static final class BitwiseXor extends OnnxOp {
        public static final String NAME = "BitwiseXor";
        
        public BitwiseXor(ExternalizedOp def) {
            super(def);
        }
        
        BitwiseXor(BitwiseXor that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public BitwiseXor transform(CopyContext cc, OpTransformer ot) {
            return new BitwiseXor(this, cc);
        }
        
        BitwiseXor(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static BitwiseXor BitwiseXor(TypeElement resultType, Value A, Value B) {
        return new BitwiseXor(resultType, A, B);
    }

    @OpFactory.OpDeclaration(BlackmanWindow.NAME)
    public static final class BlackmanWindow extends OnnxOp {
        public static final String NAME = "BlackmanWindow";
        
        public enum Attribute implements OnnxAttribute {
            periodic(Integer.class, true, null),
            output_datatype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public BlackmanWindow(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        BlackmanWindow(BlackmanWindow that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public BlackmanWindow transform(CopyContext cc, OpTransformer ot) {
            return new BlackmanWindow(this, cc);
        }
        
        BlackmanWindow(TypeElement resultType, Value size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
            super(NAME, resultType, List.of(size));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.periodic.process(attrs, periodic);
            Attribute.output_datatype.process(attrs, output_datatype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value size() {
            return operands().get(0);
        }
        
        public Optional<Integer> periodic() {
            Integer periodic = Attribute.periodic.access(Integer.class, attributes);
            return Optional.ofNullable(periodic);
        }
        
        public Optional<Integer> output_datatype() {
            Integer output_datatype = Attribute.output_datatype.access(Integer.class, attributes);
            return Optional.ofNullable(output_datatype);
        }
        
    }
    
    public static BlackmanWindow BlackmanWindow(TypeElement resultType, Value size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
        return new BlackmanWindow(resultType, size, periodic, output_datatype);
    }

    @OpFactory.OpDeclaration(Cast.NAME)
    public static final class Cast extends OnnxOp {
        public static final String NAME = "Cast";
        
        public enum Attribute implements OnnxAttribute {
            saturate(Integer.class, true, null),
            to(Integer.class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Cast(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Cast(Cast that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Cast transform(CopyContext cc, OpTransformer ot) {
            return new Cast(this, cc);
        }
        
        Cast(TypeElement resultType, Value input, Optional<Integer> saturate, int to) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.saturate.process(attrs, saturate);
            Attribute.to.process(attrs, to);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> saturate() {
            Integer saturate = Attribute.saturate.access(Integer.class, attributes);
            return Optional.ofNullable(saturate);
        }
        
        public int to() {
            Integer to = Attribute.to.access(Integer.class, attributes);
            return to;
        }
        
    }
    
    public static Cast Cast(TypeElement resultType, Value input, Optional<Integer> saturate, int to) {
        return new Cast(resultType, input, saturate, to);
    }

    @OpFactory.OpDeclaration(CastLike.NAME)
    public static final class CastLike extends OnnxOp {
        public static final String NAME = "CastLike";
        
        public enum Attribute implements OnnxAttribute {
            saturate(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public CastLike(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        CastLike(CastLike that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public CastLike transform(CopyContext cc, OpTransformer ot) {
            return new CastLike(this, cc);
        }
        
        CastLike(TypeElement resultType, Value input, Value target_type, Optional<Integer> saturate) {
            super(NAME, resultType, List.of(input, target_type));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.saturate.process(attrs, saturate);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value target_type() {
            return operands().get(1);
        }
        
        public Optional<Integer> saturate() {
            Integer saturate = Attribute.saturate.access(Integer.class, attributes);
            return Optional.ofNullable(saturate);
        }
        
    }
    
    public static CastLike CastLike(TypeElement resultType, Value input, Value target_type, Optional<Integer> saturate) {
        return new CastLike(resultType, input, target_type, saturate);
    }

    @OpFactory.OpDeclaration(CastMap.NAME)
    public static final class CastMap extends OnnxOp {
        public static final String NAME = "CastMap";
        
        public enum Attribute implements OnnxAttribute {
            map_form(String.class, true, null),
            cast_to(String.class, true, null),
            max_map(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public CastMap(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        CastMap(CastMap that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public CastMap transform(CopyContext cc, OpTransformer ot) {
            return new CastMap(this, cc);
        }
        
        CastMap(TypeElement resultType, Value X, Optional<String> map_form, Optional<String> cast_to, Optional<Integer> max_map) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.map_form.process(attrs, map_form);
            Attribute.cast_to.process(attrs, cast_to);
            Attribute.max_map.process(attrs, max_map);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> map_form() {
            String map_form = Attribute.map_form.access(String.class, attributes);
            return Optional.ofNullable(map_form);
        }
        
        public Optional<String> cast_to() {
            String cast_to = Attribute.cast_to.access(String.class, attributes);
            return Optional.ofNullable(cast_to);
        }
        
        public Optional<Integer> max_map() {
            Integer max_map = Attribute.max_map.access(Integer.class, attributes);
            return Optional.ofNullable(max_map);
        }
        
    }
    
    public static CastMap CastMap(TypeElement resultType, Value X, Optional<String> map_form, Optional<String> cast_to, Optional<Integer> max_map) {
        return new CastMap(resultType, X, map_form, cast_to, max_map);
    }

    @OpFactory.OpDeclaration(CategoryMapper.NAME)
    public static final class CategoryMapper extends OnnxOp {
        public static final String NAME = "CategoryMapper";
        
        public enum Attribute implements OnnxAttribute {
            cats_int64s(int[].class, true, null),
            cats_strings(String[].class, true, null),
            default_int64(Integer.class, true, null),
            default_string(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public CategoryMapper(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        CategoryMapper(CategoryMapper that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public CategoryMapper transform(CopyContext cc, OpTransformer ot) {
            return new CategoryMapper(this, cc);
        }
        
        CategoryMapper(TypeElement resultType, Value X, Optional<int[]> cats_int64s, Optional<String[]> cats_strings, Optional<Integer> default_int64, Optional<String> default_string) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.cats_int64s.process(attrs, cats_int64s.map(int[]::clone));
            Attribute.cats_strings.process(attrs, cats_strings.map(String[]::clone));
            Attribute.default_int64.process(attrs, default_int64);
            Attribute.default_string.process(attrs, default_string);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> cats_int64s() {
            int[] cats_int64s = Attribute.cats_int64s.access(int[].class, attributes);
            return Optional.ofNullable(cats_int64s).map(int[]::clone);
        }
        
        public Optional<String[]> cats_strings() {
            String[] cats_strings = Attribute.cats_strings.access(String[].class, attributes);
            return Optional.ofNullable(cats_strings).map(String[]::clone);
        }
        
        public Optional<Integer> default_int64() {
            Integer default_int64 = Attribute.default_int64.access(Integer.class, attributes);
            return Optional.ofNullable(default_int64);
        }
        
        public Optional<String> default_string() {
            String default_string = Attribute.default_string.access(String.class, attributes);
            return Optional.ofNullable(default_string);
        }
        
    }
    
    public static CategoryMapper CategoryMapper(TypeElement resultType, Value X, Optional<int[]> cats_int64s, Optional<String[]> cats_strings, Optional<Integer> default_int64, Optional<String> default_string) {
        return new CategoryMapper(resultType, X, cats_int64s, cats_strings, default_int64, default_string);
    }

    @OpFactory.OpDeclaration(Ceil.NAME)
    public static final class Ceil extends OnnxOp {
        public static final String NAME = "Ceil";
        
        public Ceil(ExternalizedOp def) {
            super(def);
        }
        
        Ceil(Ceil that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Ceil transform(CopyContext cc, OpTransformer ot) {
            return new Ceil(this, cc);
        }
        
        Ceil(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Ceil Ceil(TypeElement resultType, Value X) {
        return new Ceil(resultType, X);
    }

    @OpFactory.OpDeclaration(Celu.NAME)
    public static final class Celu extends OnnxOp {
        public static final String NAME = "Celu";
        
        public enum Attribute implements OnnxAttribute {
            alpha(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Celu(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Celu(Celu that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Celu transform(CopyContext cc, OpTransformer ot) {
            return new Celu(this, cc);
        }
        
        Celu(TypeElement resultType, Value X, Optional<Float> alpha) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
    }
    
    public static Celu Celu(TypeElement resultType, Value X, Optional<Float> alpha) {
        return new Celu(resultType, X, alpha);
    }

    @OpFactory.OpDeclaration(CenterCropPad.NAME)
    public static final class CenterCropPad extends OnnxOp {
        public static final String NAME = "CenterCropPad";
        
        public enum Attribute implements OnnxAttribute {
            axes(int[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public CenterCropPad(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        CenterCropPad(CenterCropPad that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public CenterCropPad transform(CopyContext cc, OpTransformer ot) {
            return new CenterCropPad(this, cc);
        }
        
        CenterCropPad(TypeElement resultType, Value input_data, Value shape, Optional<int[]> axes) {
            super(NAME, resultType, List.of(input_data, shape));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axes.process(attrs, axes.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input_data() {
            return operands().get(0);
        }
        
        public Value shape() {
            return operands().get(1);
        }
        
        public Optional<int[]> axes() {
            int[] axes = Attribute.axes.access(int[].class, attributes);
            return Optional.ofNullable(axes).map(int[]::clone);
        }
        
    }
    
    public static CenterCropPad CenterCropPad(TypeElement resultType, Value input_data, Value shape, Optional<int[]> axes) {
        return new CenterCropPad(resultType, input_data, shape, axes);
    }

    @OpFactory.OpDeclaration(Col2Im.NAME)
    public static final class Col2Im extends OnnxOp {
        public static final String NAME = "Col2Im";
        
        public enum Attribute implements OnnxAttribute {
            pads(int[].class, true, null),
            dilations(int[].class, true, null),
            strides(int[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Col2Im(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Col2Im(Col2Im that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Col2Im transform(CopyContext cc, OpTransformer ot) {
            return new Col2Im(this, cc);
        }
        
        Col2Im(TypeElement resultType, Value input, Value image_shape, Value block_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides) {
            super(NAME, resultType, List.of(input, image_shape, block_shape));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.pads.process(attrs, pads.map(int[]::clone));
            Attribute.dilations.process(attrs, dilations.map(int[]::clone));
            Attribute.strides.process(attrs, strides.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value image_shape() {
            return operands().get(1);
        }
        
        public Value block_shape() {
            return operands().get(2);
        }
        
        public Optional<int[]> pads() {
            int[] pads = Attribute.pads.access(int[].class, attributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }
        
        public Optional<int[]> dilations() {
            int[] dilations = Attribute.dilations.access(int[].class, attributes);
            return Optional.ofNullable(dilations).map(int[]::clone);
        }
        
        public Optional<int[]> strides() {
            int[] strides = Attribute.strides.access(int[].class, attributes);
            return Optional.ofNullable(strides).map(int[]::clone);
        }
        
    }
    
    public static Col2Im Col2Im(TypeElement resultType, Value input, Value image_shape, Value block_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides) {
        return new Col2Im(resultType, input, image_shape, block_shape, pads, dilations, strides);
    }

    @OpFactory.OpDeclaration(Compress.NAME)
    public static final class Compress extends OnnxOp {
        public static final String NAME = "Compress";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Compress(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Compress(Compress that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Compress transform(CopyContext cc, OpTransformer ot) {
            return new Compress(this, cc);
        }
        
        Compress(TypeElement resultType, Value input, Value condition, Optional<Integer> axis) {
            super(NAME, resultType, List.of(input, condition));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value condition() {
            return operands().get(1);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Compress Compress(TypeElement resultType, Value input, Value condition, Optional<Integer> axis) {
        return new Compress(resultType, input, condition, axis);
    }

    @OpFactory.OpDeclaration(ConcatFromSequence.NAME)
    public static final class ConcatFromSequence extends OnnxOp {
        public static final String NAME = "ConcatFromSequence";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, false, null),
            new_axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ConcatFromSequence(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ConcatFromSequence(ConcatFromSequence that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ConcatFromSequence transform(CopyContext cc, OpTransformer ot) {
            return new ConcatFromSequence(this, cc);
        }
        
        ConcatFromSequence(TypeElement resultType, Value input_sequence, int axis, Optional<Integer> new_axis) {
            super(NAME, resultType, List.of(input_sequence));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            Attribute.new_axis.process(attrs, new_axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input_sequence() {
            return operands().get(0);
        }
        
        public int axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return axis;
        }
        
        public Optional<Integer> new_axis() {
            Integer new_axis = Attribute.new_axis.access(Integer.class, attributes);
            return Optional.ofNullable(new_axis);
        }
        
    }
    
    public static ConcatFromSequence ConcatFromSequence(TypeElement resultType, Value input_sequence, int axis, Optional<Integer> new_axis) {
        return new ConcatFromSequence(resultType, input_sequence, axis, new_axis);
    }

    @OpFactory.OpDeclaration(Constant.NAME)
    public static final class Constant extends OnnxOp {
        public static final String NAME = "Constant";
        
        public enum Attribute implements OnnxAttribute {
            value_int(Integer.class, true, null),
            value_floats(float[].class, true, null),
            value_strings(String[].class, true, null),
            value_float(Float.class, true, null),
            value_string(String.class, true, null),
            value_ints(int[].class, true, null),
            sparse_value(Object.class, true, null),
            value(Tensor.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Constant(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Constant(Constant that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Constant transform(CopyContext cc, OpTransformer ot) {
            return new Constant(this, cc);
        }
        
        Constant(TypeElement resultType, Optional<Integer> value_int, Optional<float[]> value_floats, Optional<String[]> value_strings, Optional<Float> value_float, Optional<String> value_string, Optional<int[]> value_ints, Optional<Object> sparse_value, Optional<Tensor> value) {
            super(NAME, resultType, List.of());
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.value_int.process(attrs, value_int);
            Attribute.value_floats.process(attrs, value_floats.map(float[]::clone));
            Attribute.value_strings.process(attrs, value_strings.map(String[]::clone));
            Attribute.value_float.process(attrs, value_float);
            Attribute.value_string.process(attrs, value_string);
            Attribute.value_ints.process(attrs, value_ints.map(int[]::clone));
            Attribute.sparse_value.process(attrs, sparse_value);
            Attribute.value.process(attrs, value);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Optional<Integer> value_int() {
            Integer value_int = Attribute.value_int.access(Integer.class, attributes);
            return Optional.ofNullable(value_int);
        }
        
        public Optional<float[]> value_floats() {
            float[] value_floats = Attribute.value_floats.access(float[].class, attributes);
            return Optional.ofNullable(value_floats).map(float[]::clone);
        }
        
        public Optional<String[]> value_strings() {
            String[] value_strings = Attribute.value_strings.access(String[].class, attributes);
            return Optional.ofNullable(value_strings).map(String[]::clone);
        }
        
        public Optional<Float> value_float() {
            Float value_float = Attribute.value_float.access(Float.class, attributes);
            return Optional.ofNullable(value_float);
        }
        
        public Optional<String> value_string() {
            String value_string = Attribute.value_string.access(String.class, attributes);
            return Optional.ofNullable(value_string);
        }
        
        public Optional<int[]> value_ints() {
            int[] value_ints = Attribute.value_ints.access(int[].class, attributes);
            return Optional.ofNullable(value_ints).map(int[]::clone);
        }
        
        public Optional<Object> sparse_value() {
            Object sparse_value = Attribute.sparse_value.access(Object.class, attributes);
            return Optional.ofNullable(sparse_value);
        }
        
        public Optional<Tensor> value() {
            Tensor value = Attribute.value.access(Tensor.class, attributes);
            return Optional.ofNullable(value);
        }
        
    }
    
    public static Constant Constant(TypeElement resultType, Optional<Integer> value_int, Optional<float[]> value_floats, Optional<String[]> value_strings, Optional<Float> value_float, Optional<String> value_string, Optional<int[]> value_ints, Optional<Object> sparse_value, Optional<Tensor> value) {
        return new Constant(resultType, value_int, value_floats, value_strings, value_float, value_string, value_ints, sparse_value, value);
    }

    @OpFactory.OpDeclaration(ConstantOfShape.NAME)
    public static final class ConstantOfShape extends OnnxOp {
        public static final String NAME = "ConstantOfShape";
        
        public enum Attribute implements OnnxAttribute {
            value(Tensor.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ConstantOfShape(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ConstantOfShape(ConstantOfShape that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ConstantOfShape transform(CopyContext cc, OpTransformer ot) {
            return new ConstantOfShape(this, cc);
        }
        
        ConstantOfShape(TypeElement resultType, Value input, Optional<Tensor> value) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.value.process(attrs, value);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Tensor> value() {
            Tensor value = Attribute.value.access(Tensor.class, attributes);
            return Optional.ofNullable(value);
        }
        
    }
    
    public static ConstantOfShape ConstantOfShape(TypeElement resultType, Value input, Optional<Tensor> value) {
        return new ConstantOfShape(resultType, input, value);
    }

    @OpFactory.OpDeclaration(Cos.NAME)
    public static final class Cos extends OnnxOp {
        public static final String NAME = "Cos";
        
        public Cos(ExternalizedOp def) {
            super(def);
        }
        
        Cos(Cos that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Cos transform(CopyContext cc, OpTransformer ot) {
            return new Cos(this, cc);
        }
        
        Cos(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Cos Cos(TypeElement resultType, Value input) {
        return new Cos(resultType, input);
    }

    @OpFactory.OpDeclaration(Cosh.NAME)
    public static final class Cosh extends OnnxOp {
        public static final String NAME = "Cosh";
        
        public Cosh(ExternalizedOp def) {
            super(def);
        }
        
        Cosh(Cosh that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Cosh transform(CopyContext cc, OpTransformer ot) {
            return new Cosh(this, cc);
        }
        
        Cosh(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Cosh Cosh(TypeElement resultType, Value input) {
        return new Cosh(resultType, input);
    }

    @OpFactory.OpDeclaration(CumSum.NAME)
    public static final class CumSum extends OnnxOp {
        public static final String NAME = "CumSum";
        
        public enum Attribute implements OnnxAttribute {
            exclusive(Integer.class, true, null),
            reverse(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public CumSum(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        CumSum(CumSum that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public CumSum transform(CopyContext cc, OpTransformer ot) {
            return new CumSum(this, cc);
        }
        
        CumSum(TypeElement resultType, Value x, Value axis, Optional<Integer> exclusive, Optional<Integer> reverse) {
            super(NAME, resultType, List.of(x, axis));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.exclusive.process(attrs, exclusive);
            Attribute.reverse.process(attrs, reverse);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value x() {
            return operands().get(0);
        }
        
        public Value axis() {
            return operands().get(1);
        }
        
        public Optional<Integer> exclusive() {
            Integer exclusive = Attribute.exclusive.access(Integer.class, attributes);
            return Optional.ofNullable(exclusive);
        }
        
        public Optional<Integer> reverse() {
            Integer reverse = Attribute.reverse.access(Integer.class, attributes);
            return Optional.ofNullable(reverse);
        }
        
    }
    
    public static CumSum CumSum(TypeElement resultType, Value x, Value axis, Optional<Integer> exclusive, Optional<Integer> reverse) {
        return new CumSum(resultType, x, axis, exclusive, reverse);
    }

    @OpFactory.OpDeclaration(DepthToSpace.NAME)
    public static final class DepthToSpace extends OnnxOp {
        public static final String NAME = "DepthToSpace";
        
        public enum Attribute implements OnnxAttribute {
            mode(String.class, true, null),
            blocksize(Integer.class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public DepthToSpace(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        DepthToSpace(DepthToSpace that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public DepthToSpace transform(CopyContext cc, OpTransformer ot) {
            return new DepthToSpace(this, cc);
        }
        
        DepthToSpace(TypeElement resultType, Value input, Optional<String> mode, int blocksize) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            Attribute.blocksize.process(attrs, blocksize);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<String> mode() {
            String mode = Attribute.mode.access(String.class, attributes);
            return Optional.ofNullable(mode);
        }
        
        public int blocksize() {
            Integer blocksize = Attribute.blocksize.access(Integer.class, attributes);
            return blocksize;
        }
        
    }
    
    public static DepthToSpace DepthToSpace(TypeElement resultType, Value input, Optional<String> mode, int blocksize) {
        return new DepthToSpace(resultType, input, mode, blocksize);
    }

    @OpFactory.OpDeclaration(Det.NAME)
    public static final class Det extends OnnxOp {
        public static final String NAME = "Det";
        
        public Det(ExternalizedOp def) {
            super(def);
        }
        
        Det(Det that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Det transform(CopyContext cc, OpTransformer ot) {
            return new Det(this, cc);
        }
        
        Det(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Det Det(TypeElement resultType, Value X) {
        return new Det(resultType, X);
    }

    @OpFactory.OpDeclaration(DictVectorizer.NAME)
    public static final class DictVectorizer extends OnnxOp {
        public static final String NAME = "DictVectorizer";
        
        public enum Attribute implements OnnxAttribute {
            string_vocabulary(String[].class, true, null),
            int64_vocabulary(int[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public DictVectorizer(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        DictVectorizer(DictVectorizer that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public DictVectorizer transform(CopyContext cc, OpTransformer ot) {
            return new DictVectorizer(this, cc);
        }
        
        DictVectorizer(TypeElement resultType, Value X, Optional<String[]> string_vocabulary, Optional<int[]> int64_vocabulary) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.string_vocabulary.process(attrs, string_vocabulary.map(String[]::clone));
            Attribute.int64_vocabulary.process(attrs, int64_vocabulary.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String[]> string_vocabulary() {
            String[] string_vocabulary = Attribute.string_vocabulary.access(String[].class, attributes);
            return Optional.ofNullable(string_vocabulary).map(String[]::clone);
        }
        
        public Optional<int[]> int64_vocabulary() {
            int[] int64_vocabulary = Attribute.int64_vocabulary.access(int[].class, attributes);
            return Optional.ofNullable(int64_vocabulary).map(int[]::clone);
        }
        
    }
    
    public static DictVectorizer DictVectorizer(TypeElement resultType, Value X, Optional<String[]> string_vocabulary, Optional<int[]> int64_vocabulary) {
        return new DictVectorizer(resultType, X, string_vocabulary, int64_vocabulary);
    }

    @OpFactory.OpDeclaration(Div.NAME)
    public static final class Div extends OnnxOp {
        public static final String NAME = "Div";
        
        public Div(ExternalizedOp def) {
            super(def);
        }
        
        Div(Div that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Div transform(CopyContext cc, OpTransformer ot) {
            return new Div(this, cc);
        }
        
        Div(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Div Div(TypeElement resultType, Value A, Value B) {
        return new Div(resultType, A, B);
    }

    @OpFactory.OpDeclaration(DynamicQuantizeLinear.NAME)
    public static final class DynamicQuantizeLinear extends OnnxOp {
        public static final String NAME = "DynamicQuantizeLinear";
        
        public DynamicQuantizeLinear(ExternalizedOp def) {
            super(def);
        }
        
        DynamicQuantizeLinear(DynamicQuantizeLinear that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public DynamicQuantizeLinear transform(CopyContext cc, OpTransformer ot) {
            return new DynamicQuantizeLinear(this, cc);
        }
        
        DynamicQuantizeLinear(TypeElement resultType, Value x) {
            super(NAME, resultType, List.of(x));
        }
        
        public Value x() {
            return operands().get(0);
        }
        
    }
    
    public static DynamicQuantizeLinear DynamicQuantizeLinear(TypeElement resultType, Value x) {
        return new DynamicQuantizeLinear(resultType, x);
    }

    @OpFactory.OpDeclaration(Elu.NAME)
    public static final class Elu extends OnnxOp {
        public static final String NAME = "Elu";
        
        public enum Attribute implements OnnxAttribute {
            alpha(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Elu(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Elu(Elu that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Elu transform(CopyContext cc, OpTransformer ot) {
            return new Elu(this, cc);
        }
        
        Elu(TypeElement resultType, Value X, Optional<Float> alpha) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
    }
    
    public static Elu Elu(TypeElement resultType, Value X, Optional<Float> alpha) {
        return new Elu(resultType, X, alpha);
    }

    @OpFactory.OpDeclaration(Equal.NAME)
    public static final class Equal extends OnnxOp {
        public static final String NAME = "Equal";
        
        public Equal(ExternalizedOp def) {
            super(def);
        }
        
        Equal(Equal that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Equal transform(CopyContext cc, OpTransformer ot) {
            return new Equal(this, cc);
        }
        
        Equal(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Equal Equal(TypeElement resultType, Value A, Value B) {
        return new Equal(resultType, A, B);
    }

    @OpFactory.OpDeclaration(Erf.NAME)
    public static final class Erf extends OnnxOp {
        public static final String NAME = "Erf";
        
        public Erf(ExternalizedOp def) {
            super(def);
        }
        
        Erf(Erf that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Erf transform(CopyContext cc, OpTransformer ot) {
            return new Erf(this, cc);
        }
        
        Erf(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Erf Erf(TypeElement resultType, Value input) {
        return new Erf(resultType, input);
    }

    @OpFactory.OpDeclaration(Exp.NAME)
    public static final class Exp extends OnnxOp {
        public static final String NAME = "Exp";
        
        public Exp(ExternalizedOp def) {
            super(def);
        }
        
        Exp(Exp that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Exp transform(CopyContext cc, OpTransformer ot) {
            return new Exp(this, cc);
        }
        
        Exp(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Exp Exp(TypeElement resultType, Value input) {
        return new Exp(resultType, input);
    }

    @OpFactory.OpDeclaration(Expand.NAME)
    public static final class Expand extends OnnxOp {
        public static final String NAME = "Expand";
        
        public Expand(ExternalizedOp def) {
            super(def);
        }
        
        Expand(Expand that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Expand transform(CopyContext cc, OpTransformer ot) {
            return new Expand(this, cc);
        }
        
        Expand(TypeElement resultType, Value input, Value shape) {
            super(NAME, resultType, List.of(input, shape));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value shape() {
            return operands().get(1);
        }
        
    }
    
    public static Expand Expand(TypeElement resultType, Value input, Value shape) {
        return new Expand(resultType, input, shape);
    }

    @OpFactory.OpDeclaration(EyeLike.NAME)
    public static final class EyeLike extends OnnxOp {
        public static final String NAME = "EyeLike";
        
        public enum Attribute implements OnnxAttribute {
            dtype(Integer.class, true, null),
            k(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public EyeLike(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        EyeLike(EyeLike that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public EyeLike transform(CopyContext cc, OpTransformer ot) {
            return new EyeLike(this, cc);
        }
        
        EyeLike(TypeElement resultType, Value input, Optional<Integer> dtype, Optional<Integer> k) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.dtype.process(attrs, dtype);
            Attribute.k.process(attrs, k);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
        public Optional<Integer> k() {
            Integer k = Attribute.k.access(Integer.class, attributes);
            return Optional.ofNullable(k);
        }
        
    }
    
    public static EyeLike EyeLike(TypeElement resultType, Value input, Optional<Integer> dtype, Optional<Integer> k) {
        return new EyeLike(resultType, input, dtype, k);
    }

    @OpFactory.OpDeclaration(Flatten.NAME)
    public static final class Flatten extends OnnxOp {
        public static final String NAME = "Flatten";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Flatten(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Flatten(Flatten that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Flatten transform(CopyContext cc, OpTransformer ot) {
            return new Flatten(this, cc);
        }
        
        Flatten(TypeElement resultType, Value input, Optional<Integer> axis) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Flatten Flatten(TypeElement resultType, Value input, Optional<Integer> axis) {
        return new Flatten(resultType, input, axis);
    }

    @OpFactory.OpDeclaration(Floor.NAME)
    public static final class Floor extends OnnxOp {
        public static final String NAME = "Floor";
        
        public Floor(ExternalizedOp def) {
            super(def);
        }
        
        Floor(Floor that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Floor transform(CopyContext cc, OpTransformer ot) {
            return new Floor(this, cc);
        }
        
        Floor(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Floor Floor(TypeElement resultType, Value X) {
        return new Floor(resultType, X);
    }

    @OpFactory.OpDeclaration(Gather.NAME)
    public static final class Gather extends OnnxOp {
        public static final String NAME = "Gather";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Gather(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Gather(Gather that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Gather transform(CopyContext cc, OpTransformer ot) {
            return new Gather(this, cc);
        }
        
        Gather(TypeElement resultType, Value data, Value indices, Optional<Integer> axis) {
            super(NAME, resultType, List.of(data, indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value indices() {
            return operands().get(1);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Gather Gather(TypeElement resultType, Value data, Value indices, Optional<Integer> axis) {
        return new Gather(resultType, data, indices, axis);
    }

    @OpFactory.OpDeclaration(GatherElements.NAME)
    public static final class GatherElements extends OnnxOp {
        public static final String NAME = "GatherElements";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public GatherElements(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        GatherElements(GatherElements that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public GatherElements transform(CopyContext cc, OpTransformer ot) {
            return new GatherElements(this, cc);
        }
        
        GatherElements(TypeElement resultType, Value data, Value indices, Optional<Integer> axis) {
            super(NAME, resultType, List.of(data, indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value indices() {
            return operands().get(1);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static GatherElements GatherElements(TypeElement resultType, Value data, Value indices, Optional<Integer> axis) {
        return new GatherElements(resultType, data, indices, axis);
    }

    @OpFactory.OpDeclaration(GatherND.NAME)
    public static final class GatherND extends OnnxOp {
        public static final String NAME = "GatherND";
        
        public enum Attribute implements OnnxAttribute {
            batch_dims(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public GatherND(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        GatherND(GatherND that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public GatherND transform(CopyContext cc, OpTransformer ot) {
            return new GatherND(this, cc);
        }
        
        GatherND(TypeElement resultType, Value data, Value indices, Optional<Integer> batch_dims) {
            super(NAME, resultType, List.of(data, indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.batch_dims.process(attrs, batch_dims);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value indices() {
            return operands().get(1);
        }
        
        public Optional<Integer> batch_dims() {
            Integer batch_dims = Attribute.batch_dims.access(Integer.class, attributes);
            return Optional.ofNullable(batch_dims);
        }
        
    }
    
    public static GatherND GatherND(TypeElement resultType, Value data, Value indices, Optional<Integer> batch_dims) {
        return new GatherND(resultType, data, indices, batch_dims);
    }

    @OpFactory.OpDeclaration(Gelu.NAME)
    public static final class Gelu extends OnnxOp {
        public static final String NAME = "Gelu";
        
        public enum Attribute implements OnnxAttribute {
            approximate(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Gelu(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Gelu(Gelu that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Gelu transform(CopyContext cc, OpTransformer ot) {
            return new Gelu(this, cc);
        }
        
        Gelu(TypeElement resultType, Value X, Optional<String> approximate) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.approximate.process(attrs, approximate);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> approximate() {
            String approximate = Attribute.approximate.access(String.class, attributes);
            return Optional.ofNullable(approximate);
        }
        
    }
    
    public static Gelu Gelu(TypeElement resultType, Value X, Optional<String> approximate) {
        return new Gelu(resultType, X, approximate);
    }

    @OpFactory.OpDeclaration(GlobalAveragePool.NAME)
    public static final class GlobalAveragePool extends OnnxOp {
        public static final String NAME = "GlobalAveragePool";
        
        public GlobalAveragePool(ExternalizedOp def) {
            super(def);
        }
        
        GlobalAveragePool(GlobalAveragePool that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public GlobalAveragePool transform(CopyContext cc, OpTransformer ot) {
            return new GlobalAveragePool(this, cc);
        }
        
        GlobalAveragePool(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static GlobalAveragePool GlobalAveragePool(TypeElement resultType, Value X) {
        return new GlobalAveragePool(resultType, X);
    }

    @OpFactory.OpDeclaration(GlobalLpPool.NAME)
    public static final class GlobalLpPool extends OnnxOp {
        public static final String NAME = "GlobalLpPool";
        
        public enum Attribute implements OnnxAttribute {
            p(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public GlobalLpPool(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        GlobalLpPool(GlobalLpPool that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public GlobalLpPool transform(CopyContext cc, OpTransformer ot) {
            return new GlobalLpPool(this, cc);
        }
        
        GlobalLpPool(TypeElement resultType, Value X, Optional<Integer> p) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.p.process(attrs, p);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> p() {
            Integer p = Attribute.p.access(Integer.class, attributes);
            return Optional.ofNullable(p);
        }
        
    }
    
    public static GlobalLpPool GlobalLpPool(TypeElement resultType, Value X, Optional<Integer> p) {
        return new GlobalLpPool(resultType, X, p);
    }

    @OpFactory.OpDeclaration(GlobalMaxPool.NAME)
    public static final class GlobalMaxPool extends OnnxOp {
        public static final String NAME = "GlobalMaxPool";
        
        public GlobalMaxPool(ExternalizedOp def) {
            super(def);
        }
        
        GlobalMaxPool(GlobalMaxPool that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public GlobalMaxPool transform(CopyContext cc, OpTransformer ot) {
            return new GlobalMaxPool(this, cc);
        }
        
        GlobalMaxPool(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static GlobalMaxPool GlobalMaxPool(TypeElement resultType, Value X) {
        return new GlobalMaxPool(resultType, X);
    }

    @OpFactory.OpDeclaration(Greater.NAME)
    public static final class Greater extends OnnxOp {
        public static final String NAME = "Greater";
        
        public Greater(ExternalizedOp def) {
            super(def);
        }
        
        Greater(Greater that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Greater transform(CopyContext cc, OpTransformer ot) {
            return new Greater(this, cc);
        }
        
        Greater(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Greater Greater(TypeElement resultType, Value A, Value B) {
        return new Greater(resultType, A, B);
    }

    @OpFactory.OpDeclaration(GreaterOrEqual.NAME)
    public static final class GreaterOrEqual extends OnnxOp {
        public static final String NAME = "GreaterOrEqual";
        
        public GreaterOrEqual(ExternalizedOp def) {
            super(def);
        }
        
        GreaterOrEqual(GreaterOrEqual that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public GreaterOrEqual transform(CopyContext cc, OpTransformer ot) {
            return new GreaterOrEqual(this, cc);
        }
        
        GreaterOrEqual(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static GreaterOrEqual GreaterOrEqual(TypeElement resultType, Value A, Value B) {
        return new GreaterOrEqual(resultType, A, B);
    }

    @OpFactory.OpDeclaration(GridSample.NAME)
    public static final class GridSample extends OnnxOp {
        public static final String NAME = "GridSample";
        
        public enum Attribute implements OnnxAttribute {
            mode(String.class, true, null),
            align_corners(Integer.class, true, null),
            padding_mode(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public GridSample(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        GridSample(GridSample that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public GridSample transform(CopyContext cc, OpTransformer ot) {
            return new GridSample(this, cc);
        }
        
        GridSample(TypeElement resultType, Value X, Value grid, Optional<String> mode, Optional<Integer> align_corners, Optional<String> padding_mode) {
            super(NAME, resultType, List.of(X, grid));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            Attribute.align_corners.process(attrs, align_corners);
            Attribute.padding_mode.process(attrs, padding_mode);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value grid() {
            return operands().get(1);
        }
        
        public Optional<String> mode() {
            String mode = Attribute.mode.access(String.class, attributes);
            return Optional.ofNullable(mode);
        }
        
        public Optional<Integer> align_corners() {
            Integer align_corners = Attribute.align_corners.access(Integer.class, attributes);
            return Optional.ofNullable(align_corners);
        }
        
        public Optional<String> padding_mode() {
            String padding_mode = Attribute.padding_mode.access(String.class, attributes);
            return Optional.ofNullable(padding_mode);
        }
        
    }
    
    public static GridSample GridSample(TypeElement resultType, Value X, Value grid, Optional<String> mode, Optional<Integer> align_corners, Optional<String> padding_mode) {
        return new GridSample(resultType, X, grid, mode, align_corners, padding_mode);
    }

    @OpFactory.OpDeclaration(GroupNormalization.NAME)
    public static final class GroupNormalization extends OnnxOp {
        public static final String NAME = "GroupNormalization";
        
        public enum Attribute implements OnnxAttribute {
            epsilon(Float.class, true, null),
            stash_type(Integer.class, true, null),
            num_groups(Integer.class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public GroupNormalization(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        GroupNormalization(GroupNormalization that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public GroupNormalization transform(CopyContext cc, OpTransformer ot) {
            return new GroupNormalization(this, cc);
        }
        
        GroupNormalization(TypeElement resultType, Value X, Value scale, Value bias, Optional<Float> epsilon, Optional<Integer> stash_type, int num_groups) {
            super(NAME, resultType, List.of(X, scale, bias));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.epsilon.process(attrs, epsilon);
            Attribute.stash_type.process(attrs, stash_type);
            Attribute.num_groups.process(attrs, num_groups);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value scale() {
            return operands().get(1);
        }
        
        public Value bias() {
            return operands().get(2);
        }
        
        public Optional<Float> epsilon() {
            Float epsilon = Attribute.epsilon.access(Float.class, attributes);
            return Optional.ofNullable(epsilon);
        }
        
        public Optional<Integer> stash_type() {
            Integer stash_type = Attribute.stash_type.access(Integer.class, attributes);
            return Optional.ofNullable(stash_type);
        }
        
        public int num_groups() {
            Integer num_groups = Attribute.num_groups.access(Integer.class, attributes);
            return num_groups;
        }
        
    }
    
    public static GroupNormalization GroupNormalization(TypeElement resultType, Value X, Value scale, Value bias, Optional<Float> epsilon, Optional<Integer> stash_type, int num_groups) {
        return new GroupNormalization(resultType, X, scale, bias, epsilon, stash_type, num_groups);
    }

    @OpFactory.OpDeclaration(HammingWindow.NAME)
    public static final class HammingWindow extends OnnxOp {
        public static final String NAME = "HammingWindow";
        
        public enum Attribute implements OnnxAttribute {
            periodic(Integer.class, true, null),
            output_datatype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public HammingWindow(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        HammingWindow(HammingWindow that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public HammingWindow transform(CopyContext cc, OpTransformer ot) {
            return new HammingWindow(this, cc);
        }
        
        HammingWindow(TypeElement resultType, Value size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
            super(NAME, resultType, List.of(size));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.periodic.process(attrs, periodic);
            Attribute.output_datatype.process(attrs, output_datatype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value size() {
            return operands().get(0);
        }
        
        public Optional<Integer> periodic() {
            Integer periodic = Attribute.periodic.access(Integer.class, attributes);
            return Optional.ofNullable(periodic);
        }
        
        public Optional<Integer> output_datatype() {
            Integer output_datatype = Attribute.output_datatype.access(Integer.class, attributes);
            return Optional.ofNullable(output_datatype);
        }
        
    }
    
    public static HammingWindow HammingWindow(TypeElement resultType, Value size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
        return new HammingWindow(resultType, size, periodic, output_datatype);
    }

    @OpFactory.OpDeclaration(HannWindow.NAME)
    public static final class HannWindow extends OnnxOp {
        public static final String NAME = "HannWindow";
        
        public enum Attribute implements OnnxAttribute {
            periodic(Integer.class, true, null),
            output_datatype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public HannWindow(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        HannWindow(HannWindow that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public HannWindow transform(CopyContext cc, OpTransformer ot) {
            return new HannWindow(this, cc);
        }
        
        HannWindow(TypeElement resultType, Value size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
            super(NAME, resultType, List.of(size));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.periodic.process(attrs, periodic);
            Attribute.output_datatype.process(attrs, output_datatype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value size() {
            return operands().get(0);
        }
        
        public Optional<Integer> periodic() {
            Integer periodic = Attribute.periodic.access(Integer.class, attributes);
            return Optional.ofNullable(periodic);
        }
        
        public Optional<Integer> output_datatype() {
            Integer output_datatype = Attribute.output_datatype.access(Integer.class, attributes);
            return Optional.ofNullable(output_datatype);
        }
        
    }
    
    public static HannWindow HannWindow(TypeElement resultType, Value size, Optional<Integer> periodic, Optional<Integer> output_datatype) {
        return new HannWindow(resultType, size, periodic, output_datatype);
    }

    @OpFactory.OpDeclaration(HardSigmoid.NAME)
    public static final class HardSigmoid extends OnnxOp {
        public static final String NAME = "HardSigmoid";
        
        public enum Attribute implements OnnxAttribute {
            alpha(Float.class, true, null),
            beta(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public HardSigmoid(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        HardSigmoid(HardSigmoid that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public HardSigmoid transform(CopyContext cc, OpTransformer ot) {
            return new HardSigmoid(this, cc);
        }
        
        HardSigmoid(TypeElement resultType, Value X, Optional<Float> alpha, Optional<Float> beta) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            Attribute.beta.process(attrs, beta);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
        public Optional<Float> beta() {
            Float beta = Attribute.beta.access(Float.class, attributes);
            return Optional.ofNullable(beta);
        }
        
    }
    
    public static HardSigmoid HardSigmoid(TypeElement resultType, Value X, Optional<Float> alpha, Optional<Float> beta) {
        return new HardSigmoid(resultType, X, alpha, beta);
    }

    @OpFactory.OpDeclaration(HardSwish.NAME)
    public static final class HardSwish extends OnnxOp {
        public static final String NAME = "HardSwish";
        
        public HardSwish(ExternalizedOp def) {
            super(def);
        }
        
        HardSwish(HardSwish that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public HardSwish transform(CopyContext cc, OpTransformer ot) {
            return new HardSwish(this, cc);
        }
        
        HardSwish(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static HardSwish HardSwish(TypeElement resultType, Value X) {
        return new HardSwish(resultType, X);
    }

    @OpFactory.OpDeclaration(Hardmax.NAME)
    public static final class Hardmax extends OnnxOp {
        public static final String NAME = "Hardmax";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Hardmax(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Hardmax(Hardmax that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Hardmax transform(CopyContext cc, OpTransformer ot) {
            return new Hardmax(this, cc);
        }
        
        Hardmax(TypeElement resultType, Value input, Optional<Integer> axis) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Hardmax Hardmax(TypeElement resultType, Value input, Optional<Integer> axis) {
        return new Hardmax(resultType, input, axis);
    }

    @OpFactory.OpDeclaration(Identity.NAME)
    public static final class Identity extends OnnxOp {
        public static final String NAME = "Identity";
        
        public Identity(ExternalizedOp def) {
            super(def);
        }
        
        Identity(Identity that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Identity transform(CopyContext cc, OpTransformer ot) {
            return new Identity(this, cc);
        }
        
        Identity(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Identity Identity(TypeElement resultType, Value input) {
        return new Identity(resultType, input);
    }

    @OpFactory.OpDeclaration(ImageDecoder.NAME)
    public static final class ImageDecoder extends OnnxOp {
        public static final String NAME = "ImageDecoder";
        
        public enum Attribute implements OnnxAttribute {
            pixel_format(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ImageDecoder(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ImageDecoder(ImageDecoder that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ImageDecoder transform(CopyContext cc, OpTransformer ot) {
            return new ImageDecoder(this, cc);
        }
        
        ImageDecoder(TypeElement resultType, Value encoded_stream, Optional<String> pixel_format) {
            super(NAME, resultType, List.of(encoded_stream));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.pixel_format.process(attrs, pixel_format);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value encoded_stream() {
            return operands().get(0);
        }
        
        public Optional<String> pixel_format() {
            String pixel_format = Attribute.pixel_format.access(String.class, attributes);
            return Optional.ofNullable(pixel_format);
        }
        
    }
    
    public static ImageDecoder ImageDecoder(TypeElement resultType, Value encoded_stream, Optional<String> pixel_format) {
        return new ImageDecoder(resultType, encoded_stream, pixel_format);
    }

    @OpFactory.OpDeclaration(Imputer.NAME)
    public static final class Imputer extends OnnxOp {
        public static final String NAME = "Imputer";
        
        public enum Attribute implements OnnxAttribute {
            replaced_value_int64(Integer.class, true, null),
            replaced_value_float(Float.class, true, null),
            imputed_value_int64s(int[].class, true, null),
            imputed_value_floats(float[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Imputer(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Imputer(Imputer that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Imputer transform(CopyContext cc, OpTransformer ot) {
            return new Imputer(this, cc);
        }
        
        Imputer(TypeElement resultType, Value X, Optional<Integer> replaced_value_int64, Optional<Float> replaced_value_float, Optional<int[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.replaced_value_int64.process(attrs, replaced_value_int64);
            Attribute.replaced_value_float.process(attrs, replaced_value_float);
            Attribute.imputed_value_int64s.process(attrs, imputed_value_int64s.map(int[]::clone));
            Attribute.imputed_value_floats.process(attrs, imputed_value_floats.map(float[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> replaced_value_int64() {
            Integer replaced_value_int64 = Attribute.replaced_value_int64.access(Integer.class, attributes);
            return Optional.ofNullable(replaced_value_int64);
        }
        
        public Optional<Float> replaced_value_float() {
            Float replaced_value_float = Attribute.replaced_value_float.access(Float.class, attributes);
            return Optional.ofNullable(replaced_value_float);
        }
        
        public Optional<int[]> imputed_value_int64s() {
            int[] imputed_value_int64s = Attribute.imputed_value_int64s.access(int[].class, attributes);
            return Optional.ofNullable(imputed_value_int64s).map(int[]::clone);
        }
        
        public Optional<float[]> imputed_value_floats() {
            float[] imputed_value_floats = Attribute.imputed_value_floats.access(float[].class, attributes);
            return Optional.ofNullable(imputed_value_floats).map(float[]::clone);
        }
        
    }
    
    public static Imputer Imputer(TypeElement resultType, Value X, Optional<Integer> replaced_value_int64, Optional<Float> replaced_value_float, Optional<int[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
        return new Imputer(resultType, X, replaced_value_int64, replaced_value_float, imputed_value_int64s, imputed_value_floats);
    }

    @OpFactory.OpDeclaration(InstanceNormalization.NAME)
    public static final class InstanceNormalization extends OnnxOp {
        public static final String NAME = "InstanceNormalization";
        
        public enum Attribute implements OnnxAttribute {
            epsilon(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public InstanceNormalization(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        InstanceNormalization(InstanceNormalization that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public InstanceNormalization transform(CopyContext cc, OpTransformer ot) {
            return new InstanceNormalization(this, cc);
        }
        
        InstanceNormalization(TypeElement resultType, Value input, Value scale, Value B, Optional<Float> epsilon) {
            super(NAME, resultType, List.of(input, scale, B));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.epsilon.process(attrs, epsilon);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value scale() {
            return operands().get(1);
        }
        
        public Value B() {
            return operands().get(2);
        }
        
        public Optional<Float> epsilon() {
            Float epsilon = Attribute.epsilon.access(Float.class, attributes);
            return Optional.ofNullable(epsilon);
        }
        
    }
    
    public static InstanceNormalization InstanceNormalization(TypeElement resultType, Value input, Value scale, Value B, Optional<Float> epsilon) {
        return new InstanceNormalization(resultType, input, scale, B, epsilon);
    }

    @OpFactory.OpDeclaration(IsInf.NAME)
    public static final class IsInf extends OnnxOp {
        public static final String NAME = "IsInf";
        
        public enum Attribute implements OnnxAttribute {
            detect_negative(Integer.class, true, null),
            detect_positive(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public IsInf(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        IsInf(IsInf that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public IsInf transform(CopyContext cc, OpTransformer ot) {
            return new IsInf(this, cc);
        }
        
        IsInf(TypeElement resultType, Value X, Optional<Integer> detect_negative, Optional<Integer> detect_positive) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.detect_negative.process(attrs, detect_negative);
            Attribute.detect_positive.process(attrs, detect_positive);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> detect_negative() {
            Integer detect_negative = Attribute.detect_negative.access(Integer.class, attributes);
            return Optional.ofNullable(detect_negative);
        }
        
        public Optional<Integer> detect_positive() {
            Integer detect_positive = Attribute.detect_positive.access(Integer.class, attributes);
            return Optional.ofNullable(detect_positive);
        }
        
    }
    
    public static IsInf IsInf(TypeElement resultType, Value X, Optional<Integer> detect_negative, Optional<Integer> detect_positive) {
        return new IsInf(resultType, X, detect_negative, detect_positive);
    }

    @OpFactory.OpDeclaration(IsNaN.NAME)
    public static final class IsNaN extends OnnxOp {
        public static final String NAME = "IsNaN";
        
        public IsNaN(ExternalizedOp def) {
            super(def);
        }
        
        IsNaN(IsNaN that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public IsNaN transform(CopyContext cc, OpTransformer ot) {
            return new IsNaN(this, cc);
        }
        
        IsNaN(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static IsNaN IsNaN(TypeElement resultType, Value X) {
        return new IsNaN(resultType, X);
    }

    @OpFactory.OpDeclaration(LRN.NAME)
    public static final class LRN extends OnnxOp {
        public static final String NAME = "LRN";
        
        public enum Attribute implements OnnxAttribute {
            size(Integer.class, false, null),
            alpha(Float.class, true, null),
            bias(Float.class, true, null),
            beta(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LRN(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LRN(LRN that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LRN transform(CopyContext cc, OpTransformer ot) {
            return new LRN(this, cc);
        }
        
        LRN(TypeElement resultType, Value X, int size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.size.process(attrs, size);
            Attribute.alpha.process(attrs, alpha);
            Attribute.bias.process(attrs, bias);
            Attribute.beta.process(attrs, beta);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public int size() {
            Integer size = Attribute.size.access(Integer.class, attributes);
            return size;
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
        public Optional<Float> bias() {
            Float bias = Attribute.bias.access(Float.class, attributes);
            return Optional.ofNullable(bias);
        }
        
        public Optional<Float> beta() {
            Float beta = Attribute.beta.access(Float.class, attributes);
            return Optional.ofNullable(beta);
        }
        
    }
    
    public static LRN LRN(TypeElement resultType, Value X, int size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
        return new LRN(resultType, X, size, alpha, bias, beta);
    }

    @OpFactory.OpDeclaration(LabelEncoder.NAME)
    public static final class LabelEncoder extends OnnxOp {
        public static final String NAME = "LabelEncoder";
        
        public enum Attribute implements OnnxAttribute {
            values_strings(String[].class, true, null),
            keys_int64s(int[].class, true, null),
            keys_tensor(Tensor.class, true, null),
            keys_strings(String[].class, true, null),
            default_float(Float.class, true, null),
            keys_floats(float[].class, true, null),
            default_tensor(Tensor.class, true, null),
            default_int64(Integer.class, true, null),
            values_tensor(Tensor.class, true, null),
            values_int64s(int[].class, true, null),
            default_string(String.class, true, null),
            values_floats(float[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LabelEncoder(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LabelEncoder(LabelEncoder that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LabelEncoder transform(CopyContext cc, OpTransformer ot) {
            return new LabelEncoder(this, cc);
        }
        
        LabelEncoder(TypeElement resultType, Value X, Optional<String[]> values_strings, Optional<int[]> keys_int64s, Optional<Tensor> keys_tensor, Optional<String[]> keys_strings, Optional<Float> default_float, Optional<float[]> keys_floats, Optional<Tensor> default_tensor, Optional<Integer> default_int64, Optional<Tensor> values_tensor, Optional<int[]> values_int64s, Optional<String> default_string, Optional<float[]> values_floats) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.values_strings.process(attrs, values_strings.map(String[]::clone));
            Attribute.keys_int64s.process(attrs, keys_int64s.map(int[]::clone));
            Attribute.keys_tensor.process(attrs, keys_tensor);
            Attribute.keys_strings.process(attrs, keys_strings.map(String[]::clone));
            Attribute.default_float.process(attrs, default_float);
            Attribute.keys_floats.process(attrs, keys_floats.map(float[]::clone));
            Attribute.default_tensor.process(attrs, default_tensor);
            Attribute.default_int64.process(attrs, default_int64);
            Attribute.values_tensor.process(attrs, values_tensor);
            Attribute.values_int64s.process(attrs, values_int64s.map(int[]::clone));
            Attribute.default_string.process(attrs, default_string);
            Attribute.values_floats.process(attrs, values_floats.map(float[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String[]> values_strings() {
            String[] values_strings = Attribute.values_strings.access(String[].class, attributes);
            return Optional.ofNullable(values_strings).map(String[]::clone);
        }
        
        public Optional<int[]> keys_int64s() {
            int[] keys_int64s = Attribute.keys_int64s.access(int[].class, attributes);
            return Optional.ofNullable(keys_int64s).map(int[]::clone);
        }
        
        public Optional<Tensor> keys_tensor() {
            Tensor keys_tensor = Attribute.keys_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(keys_tensor);
        }
        
        public Optional<String[]> keys_strings() {
            String[] keys_strings = Attribute.keys_strings.access(String[].class, attributes);
            return Optional.ofNullable(keys_strings).map(String[]::clone);
        }
        
        public Optional<Float> default_float() {
            Float default_float = Attribute.default_float.access(Float.class, attributes);
            return Optional.ofNullable(default_float);
        }
        
        public Optional<float[]> keys_floats() {
            float[] keys_floats = Attribute.keys_floats.access(float[].class, attributes);
            return Optional.ofNullable(keys_floats).map(float[]::clone);
        }
        
        public Optional<Tensor> default_tensor() {
            Tensor default_tensor = Attribute.default_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(default_tensor);
        }
        
        public Optional<Integer> default_int64() {
            Integer default_int64 = Attribute.default_int64.access(Integer.class, attributes);
            return Optional.ofNullable(default_int64);
        }
        
        public Optional<Tensor> values_tensor() {
            Tensor values_tensor = Attribute.values_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(values_tensor);
        }
        
        public Optional<int[]> values_int64s() {
            int[] values_int64s = Attribute.values_int64s.access(int[].class, attributes);
            return Optional.ofNullable(values_int64s).map(int[]::clone);
        }
        
        public Optional<String> default_string() {
            String default_string = Attribute.default_string.access(String.class, attributes);
            return Optional.ofNullable(default_string);
        }
        
        public Optional<float[]> values_floats() {
            float[] values_floats = Attribute.values_floats.access(float[].class, attributes);
            return Optional.ofNullable(values_floats).map(float[]::clone);
        }
        
    }
    
    public static LabelEncoder LabelEncoder(TypeElement resultType, Value X, Optional<String[]> values_strings, Optional<int[]> keys_int64s, Optional<Tensor> keys_tensor, Optional<String[]> keys_strings, Optional<Float> default_float, Optional<float[]> keys_floats, Optional<Tensor> default_tensor, Optional<Integer> default_int64, Optional<Tensor> values_tensor, Optional<int[]> values_int64s, Optional<String> default_string, Optional<float[]> values_floats) {
        return new LabelEncoder(resultType, X, values_strings, keys_int64s, keys_tensor, keys_strings, default_float, keys_floats, default_tensor, default_int64, values_tensor, values_int64s, default_string, values_floats);
    }

    @OpFactory.OpDeclaration(LeakyRelu.NAME)
    public static final class LeakyRelu extends OnnxOp {
        public static final String NAME = "LeakyRelu";
        
        public enum Attribute implements OnnxAttribute {
            alpha(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LeakyRelu(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LeakyRelu(LeakyRelu that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LeakyRelu transform(CopyContext cc, OpTransformer ot) {
            return new LeakyRelu(this, cc);
        }
        
        LeakyRelu(TypeElement resultType, Value X, Optional<Float> alpha) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
    }
    
    public static LeakyRelu LeakyRelu(TypeElement resultType, Value X, Optional<Float> alpha) {
        return new LeakyRelu(resultType, X, alpha);
    }

    @OpFactory.OpDeclaration(Less.NAME)
    public static final class Less extends OnnxOp {
        public static final String NAME = "Less";
        
        public Less(ExternalizedOp def) {
            super(def);
        }
        
        Less(Less that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Less transform(CopyContext cc, OpTransformer ot) {
            return new Less(this, cc);
        }
        
        Less(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Less Less(TypeElement resultType, Value A, Value B) {
        return new Less(resultType, A, B);
    }

    @OpFactory.OpDeclaration(LessOrEqual.NAME)
    public static final class LessOrEqual extends OnnxOp {
        public static final String NAME = "LessOrEqual";
        
        public LessOrEqual(ExternalizedOp def) {
            super(def);
        }
        
        LessOrEqual(LessOrEqual that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public LessOrEqual transform(CopyContext cc, OpTransformer ot) {
            return new LessOrEqual(this, cc);
        }
        
        LessOrEqual(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static LessOrEqual LessOrEqual(TypeElement resultType, Value A, Value B) {
        return new LessOrEqual(resultType, A, B);
    }

    @OpFactory.OpDeclaration(LinearClassifier.NAME)
    public static final class LinearClassifier extends OnnxOp {
        public static final String NAME = "LinearClassifier";
        
        public enum Attribute implements OnnxAttribute {
            classlabels_ints(int[].class, true, null),
            post_transform(String.class, true, null),
            coefficients(float[].class, false, null),
            multi_class(Integer.class, true, null),
            intercepts(float[].class, true, null),
            classlabels_strings(String[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LinearClassifier(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LinearClassifier(LinearClassifier that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LinearClassifier transform(CopyContext cc, OpTransformer ot) {
            return new LinearClassifier(this, cc);
        }
        
        LinearClassifier(TypeElement resultType, Value X, Optional<int[]> classlabels_ints, Optional<String> post_transform, float[] coefficients, Optional<Integer> multi_class, Optional<float[]> intercepts, Optional<String[]> classlabels_strings) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.classlabels_ints.process(attrs, classlabels_ints.map(int[]::clone));
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.coefficients.process(attrs, coefficients.clone());
            Attribute.multi_class.process(attrs, multi_class);
            Attribute.intercepts.process(attrs, intercepts.map(float[]::clone));
            Attribute.classlabels_strings.process(attrs, classlabels_strings.map(String[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> classlabels_ints() {
            int[] classlabels_ints = Attribute.classlabels_ints.access(int[].class, attributes);
            return Optional.ofNullable(classlabels_ints).map(int[]::clone);
        }
        
        public Optional<String> post_transform() {
            String post_transform = Attribute.post_transform.access(String.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public float[] coefficients() {
            float[] coefficients = Attribute.coefficients.access(float[].class, attributes);
            return coefficients.clone();
        }
        
        public Optional<Integer> multi_class() {
            Integer multi_class = Attribute.multi_class.access(Integer.class, attributes);
            return Optional.ofNullable(multi_class);
        }
        
        public Optional<float[]> intercepts() {
            float[] intercepts = Attribute.intercepts.access(float[].class, attributes);
            return Optional.ofNullable(intercepts).map(float[]::clone);
        }
        
        public Optional<String[]> classlabels_strings() {
            String[] classlabels_strings = Attribute.classlabels_strings.access(String[].class, attributes);
            return Optional.ofNullable(classlabels_strings).map(String[]::clone);
        }
        
    }
    
    public static LinearClassifier LinearClassifier(TypeElement resultType, Value X, Optional<int[]> classlabels_ints, Optional<String> post_transform, float[] coefficients, Optional<Integer> multi_class, Optional<float[]> intercepts, Optional<String[]> classlabels_strings) {
        return new LinearClassifier(resultType, X, classlabels_ints, post_transform, coefficients, multi_class, intercepts, classlabels_strings);
    }

    @OpFactory.OpDeclaration(LinearRegressor.NAME)
    public static final class LinearRegressor extends OnnxOp {
        public static final String NAME = "LinearRegressor";
        
        public enum Attribute implements OnnxAttribute {
            post_transform(String.class, true, null),
            coefficients(float[].class, true, null),
            targets(Integer.class, true, null),
            intercepts(float[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LinearRegressor(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LinearRegressor(LinearRegressor that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LinearRegressor transform(CopyContext cc, OpTransformer ot) {
            return new LinearRegressor(this, cc);
        }
        
        LinearRegressor(TypeElement resultType, Value X, Optional<String> post_transform, Optional<float[]> coefficients, Optional<Integer> targets, Optional<float[]> intercepts) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.coefficients.process(attrs, coefficients.map(float[]::clone));
            Attribute.targets.process(attrs, targets);
            Attribute.intercepts.process(attrs, intercepts.map(float[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> post_transform() {
            String post_transform = Attribute.post_transform.access(String.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public Optional<float[]> coefficients() {
            float[] coefficients = Attribute.coefficients.access(float[].class, attributes);
            return Optional.ofNullable(coefficients).map(float[]::clone);
        }
        
        public Optional<Integer> targets() {
            Integer targets = Attribute.targets.access(Integer.class, attributes);
            return Optional.ofNullable(targets);
        }
        
        public Optional<float[]> intercepts() {
            float[] intercepts = Attribute.intercepts.access(float[].class, attributes);
            return Optional.ofNullable(intercepts).map(float[]::clone);
        }
        
    }
    
    public static LinearRegressor LinearRegressor(TypeElement resultType, Value X, Optional<String> post_transform, Optional<float[]> coefficients, Optional<Integer> targets, Optional<float[]> intercepts) {
        return new LinearRegressor(resultType, X, post_transform, coefficients, targets, intercepts);
    }

    @OpFactory.OpDeclaration(Log.NAME)
    public static final class Log extends OnnxOp {
        public static final String NAME = "Log";
        
        public Log(ExternalizedOp def) {
            super(def);
        }
        
        Log(Log that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Log transform(CopyContext cc, OpTransformer ot) {
            return new Log(this, cc);
        }
        
        Log(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Log Log(TypeElement resultType, Value input) {
        return new Log(resultType, input);
    }

    @OpFactory.OpDeclaration(LogSoftmax.NAME)
    public static final class LogSoftmax extends OnnxOp {
        public static final String NAME = "LogSoftmax";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LogSoftmax(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LogSoftmax(LogSoftmax that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LogSoftmax transform(CopyContext cc, OpTransformer ot) {
            return new LogSoftmax(this, cc);
        }
        
        LogSoftmax(TypeElement resultType, Value input, Optional<Integer> axis) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static LogSoftmax LogSoftmax(TypeElement resultType, Value input, Optional<Integer> axis) {
        return new LogSoftmax(resultType, input, axis);
    }

    @OpFactory.OpDeclaration(LpNormalization.NAME)
    public static final class LpNormalization extends OnnxOp {
        public static final String NAME = "LpNormalization";
        
        public enum Attribute implements OnnxAttribute {
            p(Integer.class, true, null),
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LpNormalization(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LpNormalization(LpNormalization that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LpNormalization transform(CopyContext cc, OpTransformer ot) {
            return new LpNormalization(this, cc);
        }
        
        LpNormalization(TypeElement resultType, Value input, Optional<Integer> p, Optional<Integer> axis) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.p.process(attrs, p);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> p() {
            Integer p = Attribute.p.access(Integer.class, attributes);
            return Optional.ofNullable(p);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static LpNormalization LpNormalization(TypeElement resultType, Value input, Optional<Integer> p, Optional<Integer> axis) {
        return new LpNormalization(resultType, input, p, axis);
    }

    @OpFactory.OpDeclaration(LpPool.NAME)
    public static final class LpPool extends OnnxOp {
        public static final String NAME = "LpPool";
        
        public enum Attribute implements OnnxAttribute {
            p(Integer.class, true, null),
            pads(int[].class, true, null),
            dilations(int[].class, true, null),
            auto_pad(String.class, true, null),
            ceil_mode(Integer.class, true, null),
            strides(int[].class, true, null),
            kernel_shape(int[].class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public LpPool(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        LpPool(LpPool that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public LpPool transform(CopyContext cc, OpTransformer ot) {
            return new LpPool(this, cc);
        }
        
        LpPool(TypeElement resultType, Value X, Optional<Integer> p, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.p.process(attrs, p);
            Attribute.pads.process(attrs, pads.map(int[]::clone));
            Attribute.dilations.process(attrs, dilations.map(int[]::clone));
            Attribute.auto_pad.process(attrs, auto_pad);
            Attribute.ceil_mode.process(attrs, ceil_mode);
            Attribute.strides.process(attrs, strides.map(int[]::clone));
            Attribute.kernel_shape.process(attrs, kernel_shape.clone());
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> p() {
            Integer p = Attribute.p.access(Integer.class, attributes);
            return Optional.ofNullable(p);
        }
        
        public Optional<int[]> pads() {
            int[] pads = Attribute.pads.access(int[].class, attributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }
        
        public Optional<int[]> dilations() {
            int[] dilations = Attribute.dilations.access(int[].class, attributes);
            return Optional.ofNullable(dilations).map(int[]::clone);
        }
        
        public Optional<String> auto_pad() {
            String auto_pad = Attribute.auto_pad.access(String.class, attributes);
            return Optional.ofNullable(auto_pad);
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
            return kernel_shape.clone();
        }
        
    }
    
    public static LpPool LpPool(TypeElement resultType, Value X, Optional<Integer> p, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
        return new LpPool(resultType, X, p, pads, dilations, auto_pad, ceil_mode, strides, kernel_shape);
    }

    @OpFactory.OpDeclaration(MatMul.NAME)
    public static final class MatMul extends OnnxOp {
        public static final String NAME = "MatMul";
        
        public MatMul(ExternalizedOp def) {
            super(def);
        }
        
        MatMul(MatMul that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public MatMul transform(CopyContext cc, OpTransformer ot) {
            return new MatMul(this, cc);
        }
        
        MatMul(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static MatMul MatMul(TypeElement resultType, Value A, Value B) {
        return new MatMul(resultType, A, B);
    }

    @OpFactory.OpDeclaration(MaxPool.NAME)
    public static final class MaxPool extends OnnxOp {
        public static final String NAME = "MaxPool";
        
        public enum Attribute implements OnnxAttribute {
            pads(int[].class, true, null),
            dilations(int[].class, true, null),
            auto_pad(String.class, true, null),
            ceil_mode(Integer.class, true, null),
            storage_order(Integer.class, true, null),
            strides(int[].class, true, null),
            kernel_shape(int[].class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public MaxPool(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        MaxPool(MaxPool that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public MaxPool transform(CopyContext cc, OpTransformer ot) {
            return new MaxPool(this, cc);
        }
        
        MaxPool(TypeElement resultType, Value X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<Integer> storage_order, Optional<int[]> strides, int[] kernel_shape) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.pads.process(attrs, pads.map(int[]::clone));
            Attribute.dilations.process(attrs, dilations.map(int[]::clone));
            Attribute.auto_pad.process(attrs, auto_pad);
            Attribute.ceil_mode.process(attrs, ceil_mode);
            Attribute.storage_order.process(attrs, storage_order);
            Attribute.strides.process(attrs, strides.map(int[]::clone));
            Attribute.kernel_shape.process(attrs, kernel_shape.clone());
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> pads() {
            int[] pads = Attribute.pads.access(int[].class, attributes);
            return Optional.ofNullable(pads).map(int[]::clone);
        }
        
        public Optional<int[]> dilations() {
            int[] dilations = Attribute.dilations.access(int[].class, attributes);
            return Optional.ofNullable(dilations).map(int[]::clone);
        }
        
        public Optional<String> auto_pad() {
            String auto_pad = Attribute.auto_pad.access(String.class, attributes);
            return Optional.ofNullable(auto_pad);
        }
        
        public Optional<Integer> ceil_mode() {
            Integer ceil_mode = Attribute.ceil_mode.access(Integer.class, attributes);
            return Optional.ofNullable(ceil_mode);
        }
        
        public Optional<Integer> storage_order() {
            Integer storage_order = Attribute.storage_order.access(Integer.class, attributes);
            return Optional.ofNullable(storage_order);
        }
        
        public Optional<int[]> strides() {
            int[] strides = Attribute.strides.access(int[].class, attributes);
            return Optional.ofNullable(strides).map(int[]::clone);
        }
        
        public int[] kernel_shape() {
            int[] kernel_shape = Attribute.kernel_shape.access(int[].class, attributes);
            return kernel_shape.clone();
        }
        
    }
    
    public static MaxPool MaxPool(TypeElement resultType, Value X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<Integer> storage_order, Optional<int[]> strides, int[] kernel_shape) {
        return new MaxPool(resultType, X, pads, dilations, auto_pad, ceil_mode, storage_order, strides, kernel_shape);
    }

    @OpFactory.OpDeclaration(MaxRoiPool.NAME)
    public static final class MaxRoiPool extends OnnxOp {
        public static final String NAME = "MaxRoiPool";
        
        public enum Attribute implements OnnxAttribute {
            spatial_scale(Float.class, true, null),
            pooled_shape(int[].class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public MaxRoiPool(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        MaxRoiPool(MaxRoiPool that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public MaxRoiPool transform(CopyContext cc, OpTransformer ot) {
            return new MaxRoiPool(this, cc);
        }
        
        MaxRoiPool(TypeElement resultType, Value X, Value rois, Optional<Float> spatial_scale, int[] pooled_shape) {
            super(NAME, resultType, List.of(X, rois));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.spatial_scale.process(attrs, spatial_scale);
            Attribute.pooled_shape.process(attrs, pooled_shape.clone());
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value rois() {
            return operands().get(1);
        }
        
        public Optional<Float> spatial_scale() {
            Float spatial_scale = Attribute.spatial_scale.access(Float.class, attributes);
            return Optional.ofNullable(spatial_scale);
        }
        
        public int[] pooled_shape() {
            int[] pooled_shape = Attribute.pooled_shape.access(int[].class, attributes);
            return pooled_shape.clone();
        }
        
    }
    
    public static MaxRoiPool MaxRoiPool(TypeElement resultType, Value X, Value rois, Optional<Float> spatial_scale, int[] pooled_shape) {
        return new MaxRoiPool(resultType, X, rois, spatial_scale, pooled_shape);
    }

    @OpFactory.OpDeclaration(MeanVarianceNormalization.NAME)
    public static final class MeanVarianceNormalization extends OnnxOp {
        public static final String NAME = "MeanVarianceNormalization";
        
        public enum Attribute implements OnnxAttribute {
            axes(int[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public MeanVarianceNormalization(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        MeanVarianceNormalization(MeanVarianceNormalization that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public MeanVarianceNormalization transform(CopyContext cc, OpTransformer ot) {
            return new MeanVarianceNormalization(this, cc);
        }
        
        MeanVarianceNormalization(TypeElement resultType, Value X, Optional<int[]> axes) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axes.process(attrs, axes.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> axes() {
            int[] axes = Attribute.axes.access(int[].class, attributes);
            return Optional.ofNullable(axes).map(int[]::clone);
        }
        
    }
    
    public static MeanVarianceNormalization MeanVarianceNormalization(TypeElement resultType, Value X, Optional<int[]> axes) {
        return new MeanVarianceNormalization(resultType, X, axes);
    }

    @OpFactory.OpDeclaration(MelWeightMatrix.NAME)
    public static final class MelWeightMatrix extends OnnxOp {
        public static final String NAME = "MelWeightMatrix";
        
        public enum Attribute implements OnnxAttribute {
            output_datatype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public MelWeightMatrix(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        MelWeightMatrix(MelWeightMatrix that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public MelWeightMatrix transform(CopyContext cc, OpTransformer ot) {
            return new MelWeightMatrix(this, cc);
        }
        
        MelWeightMatrix(TypeElement resultType, Value num_mel_bins, Value dft_length, Value sample_rate, Value lower_edge_hertz, Value upper_edge_hertz, Optional<Integer> output_datatype) {
            super(NAME, resultType, List.of(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.output_datatype.process(attrs, output_datatype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value num_mel_bins() {
            return operands().get(0);
        }
        
        public Value dft_length() {
            return operands().get(1);
        }
        
        public Value sample_rate() {
            return operands().get(2);
        }
        
        public Value lower_edge_hertz() {
            return operands().get(3);
        }
        
        public Value upper_edge_hertz() {
            return operands().get(4);
        }
        
        public Optional<Integer> output_datatype() {
            Integer output_datatype = Attribute.output_datatype.access(Integer.class, attributes);
            return Optional.ofNullable(output_datatype);
        }
        
    }
    
    public static MelWeightMatrix MelWeightMatrix(TypeElement resultType, Value num_mel_bins, Value dft_length, Value sample_rate, Value lower_edge_hertz, Value upper_edge_hertz, Optional<Integer> output_datatype) {
        return new MelWeightMatrix(resultType, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, output_datatype);
    }

    @OpFactory.OpDeclaration(Mish.NAME)
    public static final class Mish extends OnnxOp {
        public static final String NAME = "Mish";
        
        public Mish(ExternalizedOp def) {
            super(def);
        }
        
        Mish(Mish that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Mish transform(CopyContext cc, OpTransformer ot) {
            return new Mish(this, cc);
        }
        
        Mish(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Mish Mish(TypeElement resultType, Value X) {
        return new Mish(resultType, X);
    }

    @OpFactory.OpDeclaration(Mod.NAME)
    public static final class Mod extends OnnxOp {
        public static final String NAME = "Mod";
        
        public enum Attribute implements OnnxAttribute {
            fmod(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Mod(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Mod(Mod that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Mod transform(CopyContext cc, OpTransformer ot) {
            return new Mod(this, cc);
        }
        
        Mod(TypeElement resultType, Value A, Value B, Optional<Integer> fmod) {
            super(NAME, resultType, List.of(A, B));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.fmod.process(attrs, fmod);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
        public Optional<Integer> fmod() {
            Integer fmod = Attribute.fmod.access(Integer.class, attributes);
            return Optional.ofNullable(fmod);
        }
        
    }
    
    public static Mod Mod(TypeElement resultType, Value A, Value B, Optional<Integer> fmod) {
        return new Mod(resultType, A, B, fmod);
    }

    @OpFactory.OpDeclaration(Mul.NAME)
    public static final class Mul extends OnnxOp {
        public static final String NAME = "Mul";
        
        public Mul(ExternalizedOp def) {
            super(def);
        }
        
        Mul(Mul that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Mul transform(CopyContext cc, OpTransformer ot) {
            return new Mul(this, cc);
        }
        
        Mul(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Mul Mul(TypeElement resultType, Value A, Value B) {
        return new Mul(resultType, A, B);
    }

    @OpFactory.OpDeclaration(Multinomial.NAME)
    public static final class Multinomial extends OnnxOp {
        public static final String NAME = "Multinomial";
        
        public enum Attribute implements OnnxAttribute {
            seed(Float.class, true, null),
            sample_size(Integer.class, true, null),
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Multinomial(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Multinomial(Multinomial that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Multinomial transform(CopyContext cc, OpTransformer ot) {
            return new Multinomial(this, cc);
        }
        
        Multinomial(TypeElement resultType, Value input, Optional<Float> seed, Optional<Integer> sample_size, Optional<Integer> dtype) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.seed.process(attrs, seed);
            Attribute.sample_size.process(attrs, sample_size);
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Float> seed() {
            Float seed = Attribute.seed.access(Float.class, attributes);
            return Optional.ofNullable(seed);
        }
        
        public Optional<Integer> sample_size() {
            Integer sample_size = Attribute.sample_size.access(Integer.class, attributes);
            return Optional.ofNullable(sample_size);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static Multinomial Multinomial(TypeElement resultType, Value input, Optional<Float> seed, Optional<Integer> sample_size, Optional<Integer> dtype) {
        return new Multinomial(resultType, input, seed, sample_size, dtype);
    }

    @OpFactory.OpDeclaration(Neg.NAME)
    public static final class Neg extends OnnxOp {
        public static final String NAME = "Neg";
        
        public Neg(ExternalizedOp def) {
            super(def);
        }
        
        Neg(Neg that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Neg transform(CopyContext cc, OpTransformer ot) {
            return new Neg(this, cc);
        }
        
        Neg(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Neg Neg(TypeElement resultType, Value X) {
        return new Neg(resultType, X);
    }

    @OpFactory.OpDeclaration(NonZero.NAME)
    public static final class NonZero extends OnnxOp {
        public static final String NAME = "NonZero";
        
        public NonZero(ExternalizedOp def) {
            super(def);
        }
        
        NonZero(NonZero that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public NonZero transform(CopyContext cc, OpTransformer ot) {
            return new NonZero(this, cc);
        }
        
        NonZero(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static NonZero NonZero(TypeElement resultType, Value X) {
        return new NonZero(resultType, X);
    }

    @OpFactory.OpDeclaration(Normalizer.NAME)
    public static final class Normalizer extends OnnxOp {
        public static final String NAME = "Normalizer";
        
        public enum Attribute implements OnnxAttribute {
            norm(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Normalizer(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Normalizer(Normalizer that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Normalizer transform(CopyContext cc, OpTransformer ot) {
            return new Normalizer(this, cc);
        }
        
        Normalizer(TypeElement resultType, Value X, Optional<String> norm) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.norm.process(attrs, norm);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> norm() {
            String norm = Attribute.norm.access(String.class, attributes);
            return Optional.ofNullable(norm);
        }
        
    }
    
    public static Normalizer Normalizer(TypeElement resultType, Value X, Optional<String> norm) {
        return new Normalizer(resultType, X, norm);
    }

    @OpFactory.OpDeclaration(Not.NAME)
    public static final class Not extends OnnxOp {
        public static final String NAME = "Not";
        
        public Not(ExternalizedOp def) {
            super(def);
        }
        
        Not(Not that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Not transform(CopyContext cc, OpTransformer ot) {
            return new Not(this, cc);
        }
        
        Not(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Not Not(TypeElement resultType, Value X) {
        return new Not(resultType, X);
    }

    @OpFactory.OpDeclaration(OneHot.NAME)
    public static final class OneHot extends OnnxOp {
        public static final String NAME = "OneHot";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public OneHot(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        OneHot(OneHot that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public OneHot transform(CopyContext cc, OpTransformer ot) {
            return new OneHot(this, cc);
        }
        
        OneHot(TypeElement resultType, Value indices, Value depth, Value values, Optional<Integer> axis) {
            super(NAME, resultType, List.of(indices, depth, values));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value indices() {
            return operands().get(0);
        }
        
        public Value depth() {
            return operands().get(1);
        }
        
        public Value values() {
            return operands().get(2);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static OneHot OneHot(TypeElement resultType, Value indices, Value depth, Value values, Optional<Integer> axis) {
        return new OneHot(resultType, indices, depth, values, axis);
    }

    @OpFactory.OpDeclaration(OneHotEncoder.NAME)
    public static final class OneHotEncoder extends OnnxOp {
        public static final String NAME = "OneHotEncoder";
        
        public enum Attribute implements OnnxAttribute {
            cats_strings(String[].class, true, null),
            cats_int64s(int[].class, true, null),
            zeros(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public OneHotEncoder(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        OneHotEncoder(OneHotEncoder that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public OneHotEncoder transform(CopyContext cc, OpTransformer ot) {
            return new OneHotEncoder(this, cc);
        }
        
        OneHotEncoder(TypeElement resultType, Value X, Optional<String[]> cats_strings, Optional<int[]> cats_int64s, Optional<Integer> zeros) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.cats_strings.process(attrs, cats_strings.map(String[]::clone));
            Attribute.cats_int64s.process(attrs, cats_int64s.map(int[]::clone));
            Attribute.zeros.process(attrs, zeros);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String[]> cats_strings() {
            String[] cats_strings = Attribute.cats_strings.access(String[].class, attributes);
            return Optional.ofNullable(cats_strings).map(String[]::clone);
        }
        
        public Optional<int[]> cats_int64s() {
            int[] cats_int64s = Attribute.cats_int64s.access(int[].class, attributes);
            return Optional.ofNullable(cats_int64s).map(int[]::clone);
        }
        
        public Optional<Integer> zeros() {
            Integer zeros = Attribute.zeros.access(Integer.class, attributes);
            return Optional.ofNullable(zeros);
        }
        
    }
    
    public static OneHotEncoder OneHotEncoder(TypeElement resultType, Value X, Optional<String[]> cats_strings, Optional<int[]> cats_int64s, Optional<Integer> zeros) {
        return new OneHotEncoder(resultType, X, cats_strings, cats_int64s, zeros);
    }

    @OpFactory.OpDeclaration(OptionalGetElement.NAME)
    public static final class OptionalGetElement extends OnnxOp {
        public static final String NAME = "OptionalGetElement";
        
        public OptionalGetElement(ExternalizedOp def) {
            super(def);
        }
        
        OptionalGetElement(OptionalGetElement that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public OptionalGetElement transform(CopyContext cc, OpTransformer ot) {
            return new OptionalGetElement(this, cc);
        }
        
        OptionalGetElement(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static OptionalGetElement OptionalGetElement(TypeElement resultType, Value input) {
        return new OptionalGetElement(resultType, input);
    }

    @OpFactory.OpDeclaration(Or.NAME)
    public static final class Or extends OnnxOp {
        public static final String NAME = "Or";
        
        public Or(ExternalizedOp def) {
            super(def);
        }
        
        Or(Or that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Or transform(CopyContext cc, OpTransformer ot) {
            return new Or(this, cc);
        }
        
        Or(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Or Or(TypeElement resultType, Value A, Value B) {
        return new Or(resultType, A, B);
    }

    @OpFactory.OpDeclaration(PRelu.NAME)
    public static final class PRelu extends OnnxOp {
        public static final String NAME = "PRelu";
        
        public PRelu(ExternalizedOp def) {
            super(def);
        }
        
        PRelu(PRelu that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public PRelu transform(CopyContext cc, OpTransformer ot) {
            return new PRelu(this, cc);
        }
        
        PRelu(TypeElement resultType, Value X, Value slope) {
            super(NAME, resultType, List.of(X, slope));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value slope() {
            return operands().get(1);
        }
        
    }
    
    public static PRelu PRelu(TypeElement resultType, Value X, Value slope) {
        return new PRelu(resultType, X, slope);
    }

    @OpFactory.OpDeclaration(Pow.NAME)
    public static final class Pow extends OnnxOp {
        public static final String NAME = "Pow";
        
        public Pow(ExternalizedOp def) {
            super(def);
        }
        
        Pow(Pow that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Pow transform(CopyContext cc, OpTransformer ot) {
            return new Pow(this, cc);
        }
        
        Pow(TypeElement resultType, Value X, Value Y) {
            super(NAME, resultType, List.of(X, Y));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
    }
    
    public static Pow Pow(TypeElement resultType, Value X, Value Y) {
        return new Pow(resultType, X, Y);
    }

    @OpFactory.OpDeclaration(QLinearMatMul.NAME)
    public static final class QLinearMatMul extends OnnxOp {
        public static final String NAME = "QLinearMatMul";
        
        public QLinearMatMul(ExternalizedOp def) {
            super(def);
        }
        
        QLinearMatMul(QLinearMatMul that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public QLinearMatMul transform(CopyContext cc, OpTransformer ot) {
            return new QLinearMatMul(this, cc);
        }
        
        QLinearMatMul(TypeElement resultType, Value a, Value a_scale, Value a_zero_point, Value b, Value b_scale, Value b_zero_point, Value y_scale, Value y_zero_point) {
            super(NAME, resultType, List.of(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point));
        }
        
        public Value a() {
            return operands().get(0);
        }
        
        public Value a_scale() {
            return operands().get(1);
        }
        
        public Value a_zero_point() {
            return operands().get(2);
        }
        
        public Value b() {
            return operands().get(3);
        }
        
        public Value b_scale() {
            return operands().get(4);
        }
        
        public Value b_zero_point() {
            return operands().get(5);
        }
        
        public Value y_scale() {
            return operands().get(6);
        }
        
        public Value y_zero_point() {
            return operands().get(7);
        }
        
    }
    
    public static QLinearMatMul QLinearMatMul(TypeElement resultType, Value a, Value a_scale, Value a_zero_point, Value b, Value b_scale, Value b_zero_point, Value y_scale, Value y_zero_point) {
        return new QLinearMatMul(resultType, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point);
    }

    @OpFactory.OpDeclaration(RandomNormal.NAME)
    public static final class RandomNormal extends OnnxOp {
        public static final String NAME = "RandomNormal";
        
        public enum Attribute implements OnnxAttribute {
            shape(int[].class, false, null),
            seed(Float.class, true, null),
            mean(Float.class, true, null),
            scale(Float.class, true, null),
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public RandomNormal(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        RandomNormal(RandomNormal that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public RandomNormal transform(CopyContext cc, OpTransformer ot) {
            return new RandomNormal(this, cc);
        }
        
        RandomNormal(TypeElement resultType, int[] shape, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
            super(NAME, resultType, List.of());
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.shape.process(attrs, shape.clone());
            Attribute.seed.process(attrs, seed);
            Attribute.mean.process(attrs, mean);
            Attribute.scale.process(attrs, scale);
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public int[] shape() {
            int[] shape = Attribute.shape.access(int[].class, attributes);
            return shape.clone();
        }
        
        public Optional<Float> seed() {
            Float seed = Attribute.seed.access(Float.class, attributes);
            return Optional.ofNullable(seed);
        }
        
        public Optional<Float> mean() {
            Float mean = Attribute.mean.access(Float.class, attributes);
            return Optional.ofNullable(mean);
        }
        
        public Optional<Float> scale() {
            Float scale = Attribute.scale.access(Float.class, attributes);
            return Optional.ofNullable(scale);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static RandomNormal RandomNormal(TypeElement resultType, int[] shape, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
        return new RandomNormal(resultType, shape, seed, mean, scale, dtype);
    }

    @OpFactory.OpDeclaration(RandomNormalLike.NAME)
    public static final class RandomNormalLike extends OnnxOp {
        public static final String NAME = "RandomNormalLike";
        
        public enum Attribute implements OnnxAttribute {
            seed(Float.class, true, null),
            mean(Float.class, true, null),
            scale(Float.class, true, null),
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public RandomNormalLike(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        RandomNormalLike(RandomNormalLike that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public RandomNormalLike transform(CopyContext cc, OpTransformer ot) {
            return new RandomNormalLike(this, cc);
        }
        
        RandomNormalLike(TypeElement resultType, Value input, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.seed.process(attrs, seed);
            Attribute.mean.process(attrs, mean);
            Attribute.scale.process(attrs, scale);
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Float> seed() {
            Float seed = Attribute.seed.access(Float.class, attributes);
            return Optional.ofNullable(seed);
        }
        
        public Optional<Float> mean() {
            Float mean = Attribute.mean.access(Float.class, attributes);
            return Optional.ofNullable(mean);
        }
        
        public Optional<Float> scale() {
            Float scale = Attribute.scale.access(Float.class, attributes);
            return Optional.ofNullable(scale);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static RandomNormalLike RandomNormalLike(TypeElement resultType, Value input, Optional<Float> seed, Optional<Float> mean, Optional<Float> scale, Optional<Integer> dtype) {
        return new RandomNormalLike(resultType, input, seed, mean, scale, dtype);
    }

    @OpFactory.OpDeclaration(RandomUniform.NAME)
    public static final class RandomUniform extends OnnxOp {
        public static final String NAME = "RandomUniform";
        
        public enum Attribute implements OnnxAttribute {
            high(Float.class, true, null),
            shape(int[].class, false, null),
            seed(Float.class, true, null),
            low(Float.class, true, null),
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public RandomUniform(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        RandomUniform(RandomUniform that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public RandomUniform transform(CopyContext cc, OpTransformer ot) {
            return new RandomUniform(this, cc);
        }
        
        RandomUniform(TypeElement resultType, Optional<Float> high, int[] shape, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
            super(NAME, resultType, List.of());
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.high.process(attrs, high);
            Attribute.shape.process(attrs, shape.clone());
            Attribute.seed.process(attrs, seed);
            Attribute.low.process(attrs, low);
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Optional<Float> high() {
            Float high = Attribute.high.access(Float.class, attributes);
            return Optional.ofNullable(high);
        }
        
        public int[] shape() {
            int[] shape = Attribute.shape.access(int[].class, attributes);
            return shape.clone();
        }
        
        public Optional<Float> seed() {
            Float seed = Attribute.seed.access(Float.class, attributes);
            return Optional.ofNullable(seed);
        }
        
        public Optional<Float> low() {
            Float low = Attribute.low.access(Float.class, attributes);
            return Optional.ofNullable(low);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static RandomUniform RandomUniform(TypeElement resultType, Optional<Float> high, int[] shape, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
        return new RandomUniform(resultType, high, shape, seed, low, dtype);
    }

    @OpFactory.OpDeclaration(RandomUniformLike.NAME)
    public static final class RandomUniformLike extends OnnxOp {
        public static final String NAME = "RandomUniformLike";
        
        public enum Attribute implements OnnxAttribute {
            high(Float.class, true, null),
            seed(Float.class, true, null),
            low(Float.class, true, null),
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public RandomUniformLike(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        RandomUniformLike(RandomUniformLike that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public RandomUniformLike transform(CopyContext cc, OpTransformer ot) {
            return new RandomUniformLike(this, cc);
        }
        
        RandomUniformLike(TypeElement resultType, Value input, Optional<Float> high, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.high.process(attrs, high);
            Attribute.seed.process(attrs, seed);
            Attribute.low.process(attrs, low);
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Float> high() {
            Float high = Attribute.high.access(Float.class, attributes);
            return Optional.ofNullable(high);
        }
        
        public Optional<Float> seed() {
            Float seed = Attribute.seed.access(Float.class, attributes);
            return Optional.ofNullable(seed);
        }
        
        public Optional<Float> low() {
            Float low = Attribute.low.access(Float.class, attributes);
            return Optional.ofNullable(low);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static RandomUniformLike RandomUniformLike(TypeElement resultType, Value input, Optional<Float> high, Optional<Float> seed, Optional<Float> low, Optional<Integer> dtype) {
        return new RandomUniformLike(resultType, input, high, seed, low, dtype);
    }

    @OpFactory.OpDeclaration(Range.NAME)
    public static final class Range extends OnnxOp {
        public static final String NAME = "Range";
        
        public Range(ExternalizedOp def) {
            super(def);
        }
        
        Range(Range that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Range transform(CopyContext cc, OpTransformer ot) {
            return new Range(this, cc);
        }
        
        Range(TypeElement resultType, Value start, Value limit, Value delta) {
            super(NAME, resultType, List.of(start, limit, delta));
        }
        
        public Value start() {
            return operands().get(0);
        }
        
        public Value limit() {
            return operands().get(1);
        }
        
        public Value delta() {
            return operands().get(2);
        }
        
    }
    
    public static Range Range(TypeElement resultType, Value start, Value limit, Value delta) {
        return new Range(resultType, start, limit, delta);
    }

    @OpFactory.OpDeclaration(Reciprocal.NAME)
    public static final class Reciprocal extends OnnxOp {
        public static final String NAME = "Reciprocal";
        
        public Reciprocal(ExternalizedOp def) {
            super(def);
        }
        
        Reciprocal(Reciprocal that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Reciprocal transform(CopyContext cc, OpTransformer ot) {
            return new Reciprocal(this, cc);
        }
        
        Reciprocal(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Reciprocal Reciprocal(TypeElement resultType, Value X) {
        return new Reciprocal(resultType, X);
    }

    @OpFactory.OpDeclaration(RegexFullMatch.NAME)
    public static final class RegexFullMatch extends OnnxOp {
        public static final String NAME = "RegexFullMatch";
        
        public enum Attribute implements OnnxAttribute {
            pattern(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public RegexFullMatch(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        RegexFullMatch(RegexFullMatch that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public RegexFullMatch transform(CopyContext cc, OpTransformer ot) {
            return new RegexFullMatch(this, cc);
        }
        
        RegexFullMatch(TypeElement resultType, Value X, Optional<String> pattern) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.pattern.process(attrs, pattern);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> pattern() {
            String pattern = Attribute.pattern.access(String.class, attributes);
            return Optional.ofNullable(pattern);
        }
        
    }
    
    public static RegexFullMatch RegexFullMatch(TypeElement resultType, Value X, Optional<String> pattern) {
        return new RegexFullMatch(resultType, X, pattern);
    }

    @OpFactory.OpDeclaration(Relu.NAME)
    public static final class Relu extends OnnxOp {
        public static final String NAME = "Relu";
        
        public Relu(ExternalizedOp def) {
            super(def);
        }
        
        Relu(Relu that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Relu transform(CopyContext cc, OpTransformer ot) {
            return new Relu(this, cc);
        }
        
        Relu(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Relu Relu(TypeElement resultType, Value X) {
        return new Relu(resultType, X);
    }

    @OpFactory.OpDeclaration(Reshape.NAME)
    public static final class Reshape extends OnnxOp {
        public static final String NAME = "Reshape";
        
        public enum Attribute implements OnnxAttribute {
            allowzero(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Reshape(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Reshape(Reshape that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Reshape transform(CopyContext cc, OpTransformer ot) {
            return new Reshape(this, cc);
        }
        
        Reshape(TypeElement resultType, Value data, Value shape, Optional<Integer> allowzero) {
            super(NAME, resultType, List.of(data, shape));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.allowzero.process(attrs, allowzero);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value shape() {
            return operands().get(1);
        }
        
        public Optional<Integer> allowzero() {
            Integer allowzero = Attribute.allowzero.access(Integer.class, attributes);
            return Optional.ofNullable(allowzero);
        }
        
    }
    
    public static Reshape Reshape(TypeElement resultType, Value data, Value shape, Optional<Integer> allowzero) {
        return new Reshape(resultType, data, shape, allowzero);
    }

    @OpFactory.OpDeclaration(ReverseSequence.NAME)
    public static final class ReverseSequence extends OnnxOp {
        public static final String NAME = "ReverseSequence";
        
        public enum Attribute implements OnnxAttribute {
            time_axis(Integer.class, true, null),
            batch_axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ReverseSequence(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ReverseSequence(ReverseSequence that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ReverseSequence transform(CopyContext cc, OpTransformer ot) {
            return new ReverseSequence(this, cc);
        }
        
        ReverseSequence(TypeElement resultType, Value input, Value sequence_lens, Optional<Integer> time_axis, Optional<Integer> batch_axis) {
            super(NAME, resultType, List.of(input, sequence_lens));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.time_axis.process(attrs, time_axis);
            Attribute.batch_axis.process(attrs, batch_axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value sequence_lens() {
            return operands().get(1);
        }
        
        public Optional<Integer> time_axis() {
            Integer time_axis = Attribute.time_axis.access(Integer.class, attributes);
            return Optional.ofNullable(time_axis);
        }
        
        public Optional<Integer> batch_axis() {
            Integer batch_axis = Attribute.batch_axis.access(Integer.class, attributes);
            return Optional.ofNullable(batch_axis);
        }
        
    }
    
    public static ReverseSequence ReverseSequence(TypeElement resultType, Value input, Value sequence_lens, Optional<Integer> time_axis, Optional<Integer> batch_axis) {
        return new ReverseSequence(resultType, input, sequence_lens, time_axis, batch_axis);
    }

    @OpFactory.OpDeclaration(RoiAlign.NAME)
    public static final class RoiAlign extends OnnxOp {
        public static final String NAME = "RoiAlign";
        
        public enum Attribute implements OnnxAttribute {
            mode(String.class, true, null),
            output_width(Integer.class, true, null),
            spatial_scale(Float.class, true, null),
            coordinate_transformation_mode(String.class, true, null),
            sampling_ratio(Integer.class, true, null),
            output_height(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public RoiAlign(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        RoiAlign(RoiAlign that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public RoiAlign transform(CopyContext cc, OpTransformer ot) {
            return new RoiAlign(this, cc);
        }
        
        RoiAlign(TypeElement resultType, Value X, Value rois, Value batch_indices, Optional<String> mode, Optional<Integer> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Integer> sampling_ratio, Optional<Integer> output_height) {
            super(NAME, resultType, List.of(X, rois, batch_indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            Attribute.output_width.process(attrs, output_width);
            Attribute.spatial_scale.process(attrs, spatial_scale);
            Attribute.coordinate_transformation_mode.process(attrs, coordinate_transformation_mode);
            Attribute.sampling_ratio.process(attrs, sampling_ratio);
            Attribute.output_height.process(attrs, output_height);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value rois() {
            return operands().get(1);
        }
        
        public Value batch_indices() {
            return operands().get(2);
        }
        
        public Optional<String> mode() {
            String mode = Attribute.mode.access(String.class, attributes);
            return Optional.ofNullable(mode);
        }
        
        public Optional<Integer> output_width() {
            Integer output_width = Attribute.output_width.access(Integer.class, attributes);
            return Optional.ofNullable(output_width);
        }
        
        public Optional<Float> spatial_scale() {
            Float spatial_scale = Attribute.spatial_scale.access(Float.class, attributes);
            return Optional.ofNullable(spatial_scale);
        }
        
        public Optional<String> coordinate_transformation_mode() {
            String coordinate_transformation_mode = Attribute.coordinate_transformation_mode.access(String.class, attributes);
            return Optional.ofNullable(coordinate_transformation_mode);
        }
        
        public Optional<Integer> sampling_ratio() {
            Integer sampling_ratio = Attribute.sampling_ratio.access(Integer.class, attributes);
            return Optional.ofNullable(sampling_ratio);
        }
        
        public Optional<Integer> output_height() {
            Integer output_height = Attribute.output_height.access(Integer.class, attributes);
            return Optional.ofNullable(output_height);
        }
        
    }
    
    public static RoiAlign RoiAlign(TypeElement resultType, Value X, Value rois, Value batch_indices, Optional<String> mode, Optional<Integer> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Integer> sampling_ratio, Optional<Integer> output_height) {
        return new RoiAlign(resultType, X, rois, batch_indices, mode, output_width, spatial_scale, coordinate_transformation_mode, sampling_ratio, output_height);
    }

    @OpFactory.OpDeclaration(Round.NAME)
    public static final class Round extends OnnxOp {
        public static final String NAME = "Round";
        
        public Round(ExternalizedOp def) {
            super(def);
        }
        
        Round(Round that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Round transform(CopyContext cc, OpTransformer ot) {
            return new Round(this, cc);
        }
        
        Round(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Round Round(TypeElement resultType, Value X) {
        return new Round(resultType, X);
    }

    @OpFactory.OpDeclaration(SVMClassifier.NAME)
    public static final class SVMClassifier extends OnnxOp {
        public static final String NAME = "SVMClassifier";
        
        public enum Attribute implements OnnxAttribute {
            prob_b(float[].class, true, null),
            kernel_params(float[].class, true, null),
            kernel_type(String.class, true, null),
            classlabels_ints(int[].class, true, null),
            post_transform(String.class, true, null),
            rho(float[].class, true, null),
            coefficients(float[].class, true, null),
            support_vectors(float[].class, true, null),
            vectors_per_class(int[].class, true, null),
            prob_a(float[].class, true, null),
            classlabels_strings(String[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public SVMClassifier(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        SVMClassifier(SVMClassifier that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public SVMClassifier transform(CopyContext cc, OpTransformer ot) {
            return new SVMClassifier(this, cc);
        }
        
        SVMClassifier(TypeElement resultType, Value X, Optional<float[]> prob_b, Optional<float[]> kernel_params, Optional<String> kernel_type, Optional<int[]> classlabels_ints, Optional<String> post_transform, Optional<float[]> rho, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<int[]> vectors_per_class, Optional<float[]> prob_a, Optional<String[]> classlabels_strings) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.prob_b.process(attrs, prob_b.map(float[]::clone));
            Attribute.kernel_params.process(attrs, kernel_params.map(float[]::clone));
            Attribute.kernel_type.process(attrs, kernel_type);
            Attribute.classlabels_ints.process(attrs, classlabels_ints.map(int[]::clone));
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.rho.process(attrs, rho.map(float[]::clone));
            Attribute.coefficients.process(attrs, coefficients.map(float[]::clone));
            Attribute.support_vectors.process(attrs, support_vectors.map(float[]::clone));
            Attribute.vectors_per_class.process(attrs, vectors_per_class.map(int[]::clone));
            Attribute.prob_a.process(attrs, prob_a.map(float[]::clone));
            Attribute.classlabels_strings.process(attrs, classlabels_strings.map(String[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<float[]> prob_b() {
            float[] prob_b = Attribute.prob_b.access(float[].class, attributes);
            return Optional.ofNullable(prob_b).map(float[]::clone);
        }
        
        public Optional<float[]> kernel_params() {
            float[] kernel_params = Attribute.kernel_params.access(float[].class, attributes);
            return Optional.ofNullable(kernel_params).map(float[]::clone);
        }
        
        public Optional<String> kernel_type() {
            String kernel_type = Attribute.kernel_type.access(String.class, attributes);
            return Optional.ofNullable(kernel_type);
        }
        
        public Optional<int[]> classlabels_ints() {
            int[] classlabels_ints = Attribute.classlabels_ints.access(int[].class, attributes);
            return Optional.ofNullable(classlabels_ints).map(int[]::clone);
        }
        
        public Optional<String> post_transform() {
            String post_transform = Attribute.post_transform.access(String.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public Optional<float[]> rho() {
            float[] rho = Attribute.rho.access(float[].class, attributes);
            return Optional.ofNullable(rho).map(float[]::clone);
        }
        
        public Optional<float[]> coefficients() {
            float[] coefficients = Attribute.coefficients.access(float[].class, attributes);
            return Optional.ofNullable(coefficients).map(float[]::clone);
        }
        
        public Optional<float[]> support_vectors() {
            float[] support_vectors = Attribute.support_vectors.access(float[].class, attributes);
            return Optional.ofNullable(support_vectors).map(float[]::clone);
        }
        
        public Optional<int[]> vectors_per_class() {
            int[] vectors_per_class = Attribute.vectors_per_class.access(int[].class, attributes);
            return Optional.ofNullable(vectors_per_class).map(int[]::clone);
        }
        
        public Optional<float[]> prob_a() {
            float[] prob_a = Attribute.prob_a.access(float[].class, attributes);
            return Optional.ofNullable(prob_a).map(float[]::clone);
        }
        
        public Optional<String[]> classlabels_strings() {
            String[] classlabels_strings = Attribute.classlabels_strings.access(String[].class, attributes);
            return Optional.ofNullable(classlabels_strings).map(String[]::clone);
        }
        
    }
    
    public static SVMClassifier SVMClassifier(TypeElement resultType, Value X, Optional<float[]> prob_b, Optional<float[]> kernel_params, Optional<String> kernel_type, Optional<int[]> classlabels_ints, Optional<String> post_transform, Optional<float[]> rho, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<int[]> vectors_per_class, Optional<float[]> prob_a, Optional<String[]> classlabels_strings) {
        return new SVMClassifier(resultType, X, prob_b, kernel_params, kernel_type, classlabels_ints, post_transform, rho, coefficients, support_vectors, vectors_per_class, prob_a, classlabels_strings);
    }

    @OpFactory.OpDeclaration(SVMRegressor.NAME)
    public static final class SVMRegressor extends OnnxOp {
        public static final String NAME = "SVMRegressor";
        
        public enum Attribute implements OnnxAttribute {
            kernel_type(String.class, true, null),
            kernel_params(float[].class, true, null),
            n_supports(Integer.class, true, null),
            rho(float[].class, true, null),
            post_transform(String.class, true, null),
            coefficients(float[].class, true, null),
            support_vectors(float[].class, true, null),
            one_class(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public SVMRegressor(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        SVMRegressor(SVMRegressor that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public SVMRegressor transform(CopyContext cc, OpTransformer ot) {
            return new SVMRegressor(this, cc);
        }
        
        SVMRegressor(TypeElement resultType, Value X, Optional<String> kernel_type, Optional<float[]> kernel_params, Optional<Integer> n_supports, Optional<float[]> rho, Optional<String> post_transform, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<Integer> one_class) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.kernel_type.process(attrs, kernel_type);
            Attribute.kernel_params.process(attrs, kernel_params.map(float[]::clone));
            Attribute.n_supports.process(attrs, n_supports);
            Attribute.rho.process(attrs, rho.map(float[]::clone));
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.coefficients.process(attrs, coefficients.map(float[]::clone));
            Attribute.support_vectors.process(attrs, support_vectors.map(float[]::clone));
            Attribute.one_class.process(attrs, one_class);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> kernel_type() {
            String kernel_type = Attribute.kernel_type.access(String.class, attributes);
            return Optional.ofNullable(kernel_type);
        }
        
        public Optional<float[]> kernel_params() {
            float[] kernel_params = Attribute.kernel_params.access(float[].class, attributes);
            return Optional.ofNullable(kernel_params).map(float[]::clone);
        }
        
        public Optional<Integer> n_supports() {
            Integer n_supports = Attribute.n_supports.access(Integer.class, attributes);
            return Optional.ofNullable(n_supports);
        }
        
        public Optional<float[]> rho() {
            float[] rho = Attribute.rho.access(float[].class, attributes);
            return Optional.ofNullable(rho).map(float[]::clone);
        }
        
        public Optional<String> post_transform() {
            String post_transform = Attribute.post_transform.access(String.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public Optional<float[]> coefficients() {
            float[] coefficients = Attribute.coefficients.access(float[].class, attributes);
            return Optional.ofNullable(coefficients).map(float[]::clone);
        }
        
        public Optional<float[]> support_vectors() {
            float[] support_vectors = Attribute.support_vectors.access(float[].class, attributes);
            return Optional.ofNullable(support_vectors).map(float[]::clone);
        }
        
        public Optional<Integer> one_class() {
            Integer one_class = Attribute.one_class.access(Integer.class, attributes);
            return Optional.ofNullable(one_class);
        }
        
    }
    
    public static SVMRegressor SVMRegressor(TypeElement resultType, Value X, Optional<String> kernel_type, Optional<float[]> kernel_params, Optional<Integer> n_supports, Optional<float[]> rho, Optional<String> post_transform, Optional<float[]> coefficients, Optional<float[]> support_vectors, Optional<Integer> one_class) {
        return new SVMRegressor(resultType, X, kernel_type, kernel_params, n_supports, rho, post_transform, coefficients, support_vectors, one_class);
    }

    @OpFactory.OpDeclaration(Scaler.NAME)
    public static final class Scaler extends OnnxOp {
        public static final String NAME = "Scaler";
        
        public enum Attribute implements OnnxAttribute {
            offset(float[].class, true, null),
            scale(float[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Scaler(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Scaler(Scaler that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Scaler transform(CopyContext cc, OpTransformer ot) {
            return new Scaler(this, cc);
        }
        
        Scaler(TypeElement resultType, Value X, Optional<float[]> offset, Optional<float[]> scale) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.offset.process(attrs, offset.map(float[]::clone));
            Attribute.scale.process(attrs, scale.map(float[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<float[]> offset() {
            float[] offset = Attribute.offset.access(float[].class, attributes);
            return Optional.ofNullable(offset).map(float[]::clone);
        }
        
        public Optional<float[]> scale() {
            float[] scale = Attribute.scale.access(float[].class, attributes);
            return Optional.ofNullable(scale).map(float[]::clone);
        }
        
    }
    
    public static Scaler Scaler(TypeElement resultType, Value X, Optional<float[]> offset, Optional<float[]> scale) {
        return new Scaler(resultType, X, offset, scale);
    }

    @OpFactory.OpDeclaration(Scatter.NAME)
    public static final class Scatter extends OnnxOp {
        public static final String NAME = "Scatter";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Scatter(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Scatter(Scatter that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Scatter transform(CopyContext cc, OpTransformer ot) {
            return new Scatter(this, cc);
        }
        
        Scatter(TypeElement resultType, Value data, Value indices, Value updates, Optional<Integer> axis) {
            super(NAME, resultType, List.of(data, indices, updates));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value indices() {
            return operands().get(1);
        }
        
        public Value updates() {
            return operands().get(2);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Scatter Scatter(TypeElement resultType, Value data, Value indices, Value updates, Optional<Integer> axis) {
        return new Scatter(resultType, data, indices, updates, axis);
    }

    @OpFactory.OpDeclaration(ScatterElements.NAME)
    public static final class ScatterElements extends OnnxOp {
        public static final String NAME = "ScatterElements";
        
        public enum Attribute implements OnnxAttribute {
            reduction(String.class, true, null),
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ScatterElements(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ScatterElements(ScatterElements that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ScatterElements transform(CopyContext cc, OpTransformer ot) {
            return new ScatterElements(this, cc);
        }
        
        ScatterElements(TypeElement resultType, Value data, Value indices, Value updates, Optional<String> reduction, Optional<Integer> axis) {
            super(NAME, resultType, List.of(data, indices, updates));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.reduction.process(attrs, reduction);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value indices() {
            return operands().get(1);
        }
        
        public Value updates() {
            return operands().get(2);
        }
        
        public Optional<String> reduction() {
            String reduction = Attribute.reduction.access(String.class, attributes);
            return Optional.ofNullable(reduction);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static ScatterElements ScatterElements(TypeElement resultType, Value data, Value indices, Value updates, Optional<String> reduction, Optional<Integer> axis) {
        return new ScatterElements(resultType, data, indices, updates, reduction, axis);
    }

    @OpFactory.OpDeclaration(ScatterND.NAME)
    public static final class ScatterND extends OnnxOp {
        public static final String NAME = "ScatterND";
        
        public enum Attribute implements OnnxAttribute {
            reduction(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ScatterND(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ScatterND(ScatterND that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ScatterND transform(CopyContext cc, OpTransformer ot) {
            return new ScatterND(this, cc);
        }
        
        ScatterND(TypeElement resultType, Value data, Value indices, Value updates, Optional<String> reduction) {
            super(NAME, resultType, List.of(data, indices, updates));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.reduction.process(attrs, reduction);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value indices() {
            return operands().get(1);
        }
        
        public Value updates() {
            return operands().get(2);
        }
        
        public Optional<String> reduction() {
            String reduction = Attribute.reduction.access(String.class, attributes);
            return Optional.ofNullable(reduction);
        }
        
    }
    
    public static ScatterND ScatterND(TypeElement resultType, Value data, Value indices, Value updates, Optional<String> reduction) {
        return new ScatterND(resultType, data, indices, updates, reduction);
    }

    @OpFactory.OpDeclaration(Selu.NAME)
    public static final class Selu extends OnnxOp {
        public static final String NAME = "Selu";
        
        public enum Attribute implements OnnxAttribute {
            alpha(Float.class, true, null),
            gamma(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Selu(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Selu(Selu that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Selu transform(CopyContext cc, OpTransformer ot) {
            return new Selu(this, cc);
        }
        
        Selu(TypeElement resultType, Value X, Optional<Float> alpha, Optional<Float> gamma) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            Attribute.gamma.process(attrs, gamma);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
        public Optional<Float> gamma() {
            Float gamma = Attribute.gamma.access(Float.class, attributes);
            return Optional.ofNullable(gamma);
        }
        
    }
    
    public static Selu Selu(TypeElement resultType, Value X, Optional<Float> alpha, Optional<Float> gamma) {
        return new Selu(resultType, X, alpha, gamma);
    }

    @OpFactory.OpDeclaration(SequenceAt.NAME)
    public static final class SequenceAt extends OnnxOp {
        public static final String NAME = "SequenceAt";
        
        public SequenceAt(ExternalizedOp def) {
            super(def);
        }
        
        SequenceAt(SequenceAt that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public SequenceAt transform(CopyContext cc, OpTransformer ot) {
            return new SequenceAt(this, cc);
        }
        
        SequenceAt(TypeElement resultType, Value input_sequence, Value position) {
            super(NAME, resultType, List.of(input_sequence, position));
        }
        
        public Value input_sequence() {
            return operands().get(0);
        }
        
        public Value position() {
            return operands().get(1);
        }
        
    }
    
    public static SequenceAt SequenceAt(TypeElement resultType, Value input_sequence, Value position) {
        return new SequenceAt(resultType, input_sequence, position);
    }

    @OpFactory.OpDeclaration(SequenceEmpty.NAME)
    public static final class SequenceEmpty extends OnnxOp {
        public static final String NAME = "SequenceEmpty";
        
        public enum Attribute implements OnnxAttribute {
            dtype(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public SequenceEmpty(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        SequenceEmpty(SequenceEmpty that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public SequenceEmpty transform(CopyContext cc, OpTransformer ot) {
            return new SequenceEmpty(this, cc);
        }
        
        SequenceEmpty(TypeElement resultType, Optional<Integer> dtype) {
            super(NAME, resultType, List.of());
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.dtype.process(attrs, dtype);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Optional<Integer> dtype() {
            Integer dtype = Attribute.dtype.access(Integer.class, attributes);
            return Optional.ofNullable(dtype);
        }
        
    }
    
    public static SequenceEmpty SequenceEmpty(TypeElement resultType, Optional<Integer> dtype) {
        return new SequenceEmpty(resultType, dtype);
    }

    @OpFactory.OpDeclaration(SequenceLength.NAME)
    public static final class SequenceLength extends OnnxOp {
        public static final String NAME = "SequenceLength";
        
        public SequenceLength(ExternalizedOp def) {
            super(def);
        }
        
        SequenceLength(SequenceLength that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public SequenceLength transform(CopyContext cc, OpTransformer ot) {
            return new SequenceLength(this, cc);
        }
        
        SequenceLength(TypeElement resultType, Value input_sequence) {
            super(NAME, resultType, List.of(input_sequence));
        }
        
        public Value input_sequence() {
            return operands().get(0);
        }
        
    }
    
    public static SequenceLength SequenceLength(TypeElement resultType, Value input_sequence) {
        return new SequenceLength(resultType, input_sequence);
    }

    @OpFactory.OpDeclaration(Shape.NAME)
    public static final class Shape extends OnnxOp {
        public static final String NAME = "Shape";
        
        public enum Attribute implements OnnxAttribute {
            start(Integer.class, true, null),
            end(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Shape(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Shape(Shape that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Shape transform(CopyContext cc, OpTransformer ot) {
            return new Shape(this, cc);
        }
        
        Shape(TypeElement resultType, Value data, Optional<Integer> start, Optional<Integer> end) {
            super(NAME, resultType, List.of(data));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.start.process(attrs, start);
            Attribute.end.process(attrs, end);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Optional<Integer> start() {
            Integer start = Attribute.start.access(Integer.class, attributes);
            return Optional.ofNullable(start);
        }
        
        public Optional<Integer> end() {
            Integer end = Attribute.end.access(Integer.class, attributes);
            return Optional.ofNullable(end);
        }
        
    }
    
    public static Shape Shape(TypeElement resultType, Value data, Optional<Integer> start, Optional<Integer> end) {
        return new Shape(resultType, data, start, end);
    }

    @OpFactory.OpDeclaration(Shrink.NAME)
    public static final class Shrink extends OnnxOp {
        public static final String NAME = "Shrink";
        
        public enum Attribute implements OnnxAttribute {
            lambd(Float.class, true, null),
            bias(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Shrink(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Shrink(Shrink that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Shrink transform(CopyContext cc, OpTransformer ot) {
            return new Shrink(this, cc);
        }
        
        Shrink(TypeElement resultType, Value input, Optional<Float> lambd, Optional<Float> bias) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.lambd.process(attrs, lambd);
            Attribute.bias.process(attrs, bias);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Float> lambd() {
            Float lambd = Attribute.lambd.access(Float.class, attributes);
            return Optional.ofNullable(lambd);
        }
        
        public Optional<Float> bias() {
            Float bias = Attribute.bias.access(Float.class, attributes);
            return Optional.ofNullable(bias);
        }
        
    }
    
    public static Shrink Shrink(TypeElement resultType, Value input, Optional<Float> lambd, Optional<Float> bias) {
        return new Shrink(resultType, input, lambd, bias);
    }

    @OpFactory.OpDeclaration(Sigmoid.NAME)
    public static final class Sigmoid extends OnnxOp {
        public static final String NAME = "Sigmoid";
        
        public Sigmoid(ExternalizedOp def) {
            super(def);
        }
        
        Sigmoid(Sigmoid that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Sigmoid transform(CopyContext cc, OpTransformer ot) {
            return new Sigmoid(this, cc);
        }
        
        Sigmoid(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Sigmoid Sigmoid(TypeElement resultType, Value X) {
        return new Sigmoid(resultType, X);
    }

    @OpFactory.OpDeclaration(Sign.NAME)
    public static final class Sign extends OnnxOp {
        public static final String NAME = "Sign";
        
        public Sign(ExternalizedOp def) {
            super(def);
        }
        
        Sign(Sign that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Sign transform(CopyContext cc, OpTransformer ot) {
            return new Sign(this, cc);
        }
        
        Sign(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Sign Sign(TypeElement resultType, Value input) {
        return new Sign(resultType, input);
    }

    @OpFactory.OpDeclaration(Sin.NAME)
    public static final class Sin extends OnnxOp {
        public static final String NAME = "Sin";
        
        public Sin(ExternalizedOp def) {
            super(def);
        }
        
        Sin(Sin that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Sin transform(CopyContext cc, OpTransformer ot) {
            return new Sin(this, cc);
        }
        
        Sin(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Sin Sin(TypeElement resultType, Value input) {
        return new Sin(resultType, input);
    }

    @OpFactory.OpDeclaration(Sinh.NAME)
    public static final class Sinh extends OnnxOp {
        public static final String NAME = "Sinh";
        
        public Sinh(ExternalizedOp def) {
            super(def);
        }
        
        Sinh(Sinh that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Sinh transform(CopyContext cc, OpTransformer ot) {
            return new Sinh(this, cc);
        }
        
        Sinh(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Sinh Sinh(TypeElement resultType, Value input) {
        return new Sinh(resultType, input);
    }

    @OpFactory.OpDeclaration(Size.NAME)
    public static final class Size extends OnnxOp {
        public static final String NAME = "Size";
        
        public Size(ExternalizedOp def) {
            super(def);
        }
        
        Size(Size that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Size transform(CopyContext cc, OpTransformer ot) {
            return new Size(this, cc);
        }
        
        Size(TypeElement resultType, Value data) {
            super(NAME, resultType, List.of(data));
        }
        
        public Value data() {
            return operands().get(0);
        }
        
    }
    
    public static Size Size(TypeElement resultType, Value data) {
        return new Size(resultType, data);
    }

    @OpFactory.OpDeclaration(Softmax.NAME)
    public static final class Softmax extends OnnxOp {
        public static final String NAME = "Softmax";
        
        public enum Attribute implements OnnxAttribute {
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Softmax(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Softmax(Softmax that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Softmax transform(CopyContext cc, OpTransformer ot) {
            return new Softmax(this, cc);
        }
        
        Softmax(TypeElement resultType, Value input, Optional<Integer> axis) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Softmax Softmax(TypeElement resultType, Value input, Optional<Integer> axis) {
        return new Softmax(resultType, input, axis);
    }

    @OpFactory.OpDeclaration(Softplus.NAME)
    public static final class Softplus extends OnnxOp {
        public static final String NAME = "Softplus";
        
        public Softplus(ExternalizedOp def) {
            super(def);
        }
        
        Softplus(Softplus that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Softplus transform(CopyContext cc, OpTransformer ot) {
            return new Softplus(this, cc);
        }
        
        Softplus(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Softplus Softplus(TypeElement resultType, Value X) {
        return new Softplus(resultType, X);
    }

    @OpFactory.OpDeclaration(Softsign.NAME)
    public static final class Softsign extends OnnxOp {
        public static final String NAME = "Softsign";
        
        public Softsign(ExternalizedOp def) {
            super(def);
        }
        
        Softsign(Softsign that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Softsign transform(CopyContext cc, OpTransformer ot) {
            return new Softsign(this, cc);
        }
        
        Softsign(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Softsign Softsign(TypeElement resultType, Value input) {
        return new Softsign(resultType, input);
    }

    @OpFactory.OpDeclaration(SpaceToDepth.NAME)
    public static final class SpaceToDepth extends OnnxOp {
        public static final String NAME = "SpaceToDepth";
        
        public enum Attribute implements OnnxAttribute {
            blocksize(Integer.class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public SpaceToDepth(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        SpaceToDepth(SpaceToDepth that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public SpaceToDepth transform(CopyContext cc, OpTransformer ot) {
            return new SpaceToDepth(this, cc);
        }
        
        SpaceToDepth(TypeElement resultType, Value input, int blocksize) {
            super(NAME, resultType, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.blocksize.process(attrs, blocksize);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public int blocksize() {
            Integer blocksize = Attribute.blocksize.access(Integer.class, attributes);
            return blocksize;
        }
        
    }
    
    public static SpaceToDepth SpaceToDepth(TypeElement resultType, Value input, int blocksize) {
        return new SpaceToDepth(resultType, input, blocksize);
    }

    @OpFactory.OpDeclaration(Sqrt.NAME)
    public static final class Sqrt extends OnnxOp {
        public static final String NAME = "Sqrt";
        
        public Sqrt(ExternalizedOp def) {
            super(def);
        }
        
        Sqrt(Sqrt that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Sqrt transform(CopyContext cc, OpTransformer ot) {
            return new Sqrt(this, cc);
        }
        
        Sqrt(TypeElement resultType, Value X) {
            super(NAME, resultType, List.of(X));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
    }
    
    public static Sqrt Sqrt(TypeElement resultType, Value X) {
        return new Sqrt(resultType, X);
    }

    @OpFactory.OpDeclaration(StringConcat.NAME)
    public static final class StringConcat extends OnnxOp {
        public static final String NAME = "StringConcat";
        
        public StringConcat(ExternalizedOp def) {
            super(def);
        }
        
        StringConcat(StringConcat that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public StringConcat transform(CopyContext cc, OpTransformer ot) {
            return new StringConcat(this, cc);
        }
        
        StringConcat(TypeElement resultType, Value X, Value Y) {
            super(NAME, resultType, List.of(X, Y));
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
    }
    
    public static StringConcat StringConcat(TypeElement resultType, Value X, Value Y) {
        return new StringConcat(resultType, X, Y);
    }

    @OpFactory.OpDeclaration(StringNormalizer.NAME)
    public static final class StringNormalizer extends OnnxOp {
        public static final String NAME = "StringNormalizer";
        
        public enum Attribute implements OnnxAttribute {
            is_case_sensitive(Integer.class, true, null),
            locale(String.class, true, null),
            stopwords(String[].class, true, null),
            case_change_action(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public StringNormalizer(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        StringNormalizer(StringNormalizer that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public StringNormalizer transform(CopyContext cc, OpTransformer ot) {
            return new StringNormalizer(this, cc);
        }
        
        StringNormalizer(TypeElement resultType, Value X, Optional<Integer> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.is_case_sensitive.process(attrs, is_case_sensitive);
            Attribute.locale.process(attrs, locale);
            Attribute.stopwords.process(attrs, stopwords.map(String[]::clone));
            Attribute.case_change_action.process(attrs, case_change_action);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> is_case_sensitive() {
            Integer is_case_sensitive = Attribute.is_case_sensitive.access(Integer.class, attributes);
            return Optional.ofNullable(is_case_sensitive);
        }
        
        public Optional<String> locale() {
            String locale = Attribute.locale.access(String.class, attributes);
            return Optional.ofNullable(locale);
        }
        
        public Optional<String[]> stopwords() {
            String[] stopwords = Attribute.stopwords.access(String[].class, attributes);
            return Optional.ofNullable(stopwords).map(String[]::clone);
        }
        
        public Optional<String> case_change_action() {
            String case_change_action = Attribute.case_change_action.access(String.class, attributes);
            return Optional.ofNullable(case_change_action);
        }
        
    }
    
    public static StringNormalizer StringNormalizer(TypeElement resultType, Value X, Optional<Integer> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
        return new StringNormalizer(resultType, X, is_case_sensitive, locale, stopwords, case_change_action);
    }

    @OpFactory.OpDeclaration(StringSplit.NAME)
    public static final class StringSplit extends OnnxOp {
        public static final String NAME = "StringSplit";
        
        public enum Attribute implements OnnxAttribute {
            delimiter(String.class, true, null),
            maxsplit(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public StringSplit(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        StringSplit(StringSplit that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public StringSplit transform(CopyContext cc, OpTransformer ot) {
            return new StringSplit(this, cc);
        }
        
        StringSplit(TypeElement resultType, Value X, Optional<String> delimiter, Optional<Integer> maxsplit) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.delimiter.process(attrs, delimiter);
            Attribute.maxsplit.process(attrs, maxsplit);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> delimiter() {
            String delimiter = Attribute.delimiter.access(String.class, attributes);
            return Optional.ofNullable(delimiter);
        }
        
        public Optional<Integer> maxsplit() {
            Integer maxsplit = Attribute.maxsplit.access(Integer.class, attributes);
            return Optional.ofNullable(maxsplit);
        }
        
    }
    
    public static StringSplit StringSplit(TypeElement resultType, Value X, Optional<String> delimiter, Optional<Integer> maxsplit) {
        return new StringSplit(resultType, X, delimiter, maxsplit);
    }

    @OpFactory.OpDeclaration(Sub.NAME)
    public static final class Sub extends OnnxOp {
        public static final String NAME = "Sub";
        
        public Sub(ExternalizedOp def) {
            super(def);
        }
        
        Sub(Sub that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Sub transform(CopyContext cc, OpTransformer ot) {
            return new Sub(this, cc);
        }
        
        Sub(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Sub Sub(TypeElement resultType, Value A, Value B) {
        return new Sub(resultType, A, B);
    }

    @OpFactory.OpDeclaration(Tan.NAME)
    public static final class Tan extends OnnxOp {
        public static final String NAME = "Tan";
        
        public Tan(ExternalizedOp def) {
            super(def);
        }
        
        Tan(Tan that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Tan transform(CopyContext cc, OpTransformer ot) {
            return new Tan(this, cc);
        }
        
        Tan(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Tan Tan(TypeElement resultType, Value input) {
        return new Tan(resultType, input);
    }

    @OpFactory.OpDeclaration(Tanh.NAME)
    public static final class Tanh extends OnnxOp {
        public static final String NAME = "Tanh";
        
        public Tanh(ExternalizedOp def) {
            super(def);
        }
        
        Tanh(Tanh that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Tanh transform(CopyContext cc, OpTransformer ot) {
            return new Tanh(this, cc);
        }
        
        Tanh(TypeElement resultType, Value input) {
            super(NAME, resultType, List.of(input));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
    }
    
    public static Tanh Tanh(TypeElement resultType, Value input) {
        return new Tanh(resultType, input);
    }

    @OpFactory.OpDeclaration(TfIdfVectorizer.NAME)
    public static final class TfIdfVectorizer extends OnnxOp {
        public static final String NAME = "TfIdfVectorizer";
        
        public enum Attribute implements OnnxAttribute {
            ngram_counts(int[].class, false, null),
            min_gram_length(Integer.class, false, null),
            pool_strings(String[].class, true, null),
            mode(String.class, false, null),
            max_gram_length(Integer.class, false, null),
            max_skip_count(Integer.class, false, null),
            pool_int64s(int[].class, true, null),
            weights(float[].class, true, null),
            ngram_indexes(int[].class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public TfIdfVectorizer(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        TfIdfVectorizer(TfIdfVectorizer that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public TfIdfVectorizer transform(CopyContext cc, OpTransformer ot) {
            return new TfIdfVectorizer(this, cc);
        }
        
        TfIdfVectorizer(TypeElement resultType, Value X, int[] ngram_counts, int min_gram_length, Optional<String[]> pool_strings, String mode, int max_gram_length, int max_skip_count, Optional<int[]> pool_int64s, Optional<float[]> weights, int[] ngram_indexes) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.ngram_counts.process(attrs, ngram_counts.clone());
            Attribute.min_gram_length.process(attrs, min_gram_length);
            Attribute.pool_strings.process(attrs, pool_strings.map(String[]::clone));
            Attribute.mode.process(attrs, mode);
            Attribute.max_gram_length.process(attrs, max_gram_length);
            Attribute.max_skip_count.process(attrs, max_skip_count);
            Attribute.pool_int64s.process(attrs, pool_int64s.map(int[]::clone));
            Attribute.weights.process(attrs, weights.map(float[]::clone));
            Attribute.ngram_indexes.process(attrs, ngram_indexes.clone());
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public int[] ngram_counts() {
            int[] ngram_counts = Attribute.ngram_counts.access(int[].class, attributes);
            return ngram_counts.clone();
        }
        
        public int min_gram_length() {
            Integer min_gram_length = Attribute.min_gram_length.access(Integer.class, attributes);
            return min_gram_length;
        }
        
        public Optional<String[]> pool_strings() {
            String[] pool_strings = Attribute.pool_strings.access(String[].class, attributes);
            return Optional.ofNullable(pool_strings).map(String[]::clone);
        }
        
        public String mode() {
            String mode = Attribute.mode.access(String.class, attributes);
            return mode;
        }
        
        public int max_gram_length() {
            Integer max_gram_length = Attribute.max_gram_length.access(Integer.class, attributes);
            return max_gram_length;
        }
        
        public int max_skip_count() {
            Integer max_skip_count = Attribute.max_skip_count.access(Integer.class, attributes);
            return max_skip_count;
        }
        
        public Optional<int[]> pool_int64s() {
            int[] pool_int64s = Attribute.pool_int64s.access(int[].class, attributes);
            return Optional.ofNullable(pool_int64s).map(int[]::clone);
        }
        
        public Optional<float[]> weights() {
            float[] weights = Attribute.weights.access(float[].class, attributes);
            return Optional.ofNullable(weights).map(float[]::clone);
        }
        
        public int[] ngram_indexes() {
            int[] ngram_indexes = Attribute.ngram_indexes.access(int[].class, attributes);
            return ngram_indexes.clone();
        }
        
    }
    
    public static TfIdfVectorizer TfIdfVectorizer(TypeElement resultType, Value X, int[] ngram_counts, int min_gram_length, Optional<String[]> pool_strings, String mode, int max_gram_length, int max_skip_count, Optional<int[]> pool_int64s, Optional<float[]> weights, int[] ngram_indexes) {
        return new TfIdfVectorizer(resultType, X, ngram_counts, min_gram_length, pool_strings, mode, max_gram_length, max_skip_count, pool_int64s, weights, ngram_indexes);
    }

    @OpFactory.OpDeclaration(ThresholdedRelu.NAME)
    public static final class ThresholdedRelu extends OnnxOp {
        public static final String NAME = "ThresholdedRelu";
        
        public enum Attribute implements OnnxAttribute {
            alpha(Float.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ThresholdedRelu(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ThresholdedRelu(ThresholdedRelu that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ThresholdedRelu transform(CopyContext cc, OpTransformer ot) {
            return new ThresholdedRelu(this, cc);
        }
        
        ThresholdedRelu(TypeElement resultType, Value X, Optional<Float> alpha) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
    }
    
    public static ThresholdedRelu ThresholdedRelu(TypeElement resultType, Value X, Optional<Float> alpha) {
        return new ThresholdedRelu(resultType, X, alpha);
    }

    @OpFactory.OpDeclaration(Tile.NAME)
    public static final class Tile extends OnnxOp {
        public static final String NAME = "Tile";
        
        public Tile(ExternalizedOp def) {
            super(def);
        }
        
        Tile(Tile that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Tile transform(CopyContext cc, OpTransformer ot) {
            return new Tile(this, cc);
        }
        
        Tile(TypeElement resultType, Value input, Value repeats) {
            super(NAME, resultType, List.of(input, repeats));
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value repeats() {
            return operands().get(1);
        }
        
    }
    
    public static Tile Tile(TypeElement resultType, Value input, Value repeats) {
        return new Tile(resultType, input, repeats);
    }

    @OpFactory.OpDeclaration(TopK.NAME)
    public static final class TopK extends OnnxOp {
        public static final String NAME = "TopK";
        
        public enum Attribute implements OnnxAttribute {
            largest(Integer.class, true, null),
            sorted(Integer.class, true, null),
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public TopK(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        TopK(TopK that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public TopK transform(CopyContext cc, OpTransformer ot) {
            return new TopK(this, cc);
        }
        
        TopK(TypeElement resultType, Value X, Value K, Optional<Integer> largest, Optional<Integer> sorted, Optional<Integer> axis) {
            super(NAME, resultType, List.of(X, K));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.largest.process(attrs, largest);
            Attribute.sorted.process(attrs, sorted);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value K() {
            return operands().get(1);
        }
        
        public Optional<Integer> largest() {
            Integer largest = Attribute.largest.access(Integer.class, attributes);
            return Optional.ofNullable(largest);
        }
        
        public Optional<Integer> sorted() {
            Integer sorted = Attribute.sorted.access(Integer.class, attributes);
            return Optional.ofNullable(sorted);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static TopK TopK(TypeElement resultType, Value X, Value K, Optional<Integer> largest, Optional<Integer> sorted, Optional<Integer> axis) {
        return new TopK(resultType, X, K, largest, sorted, axis);
    }

    @OpFactory.OpDeclaration(Transpose.NAME)
    public static final class Transpose extends OnnxOp {
        public static final String NAME = "Transpose";
        
        public enum Attribute implements OnnxAttribute {
            perm(int[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Transpose(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Transpose(Transpose that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Transpose transform(CopyContext cc, OpTransformer ot) {
            return new Transpose(this, cc);
        }
        
        Transpose(TypeElement resultType, Value data, Optional<int[]> perm) {
            super(NAME, resultType, List.of(data));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.perm.process(attrs, perm.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Optional<int[]> perm() {
            int[] perm = Attribute.perm.access(int[].class, attributes);
            return Optional.ofNullable(perm).map(int[]::clone);
        }
        
    }
    
    public static Transpose Transpose(TypeElement resultType, Value data, Optional<int[]> perm) {
        return new Transpose(resultType, data, perm);
    }

    @OpFactory.OpDeclaration(TreeEnsemble.NAME)
    public static final class TreeEnsemble extends OnnxOp {
        public static final String NAME = "TreeEnsemble";
        
        public enum Attribute implements OnnxAttribute {
            aggregate_function(Integer.class, true, null),
            nodes_hitrates(Tensor.class, true, null),
            nodes_featureids(int[].class, false, null),
            nodes_falseleafs(int[].class, false, null),
            post_transform(Integer.class, true, null),
            nodes_trueleafs(int[].class, false, null),
            nodes_modes(Tensor.class, false, null),
            nodes_falsenodeids(int[].class, false, null),
            nodes_truenodeids(int[].class, false, null),
            leaf_weights(Tensor.class, false, null),
            leaf_targetids(int[].class, false, null),
            tree_roots(int[].class, false, null),
            n_targets(Integer.class, true, null),
            nodes_missing_value_tracks_true(int[].class, true, null),
            membership_values(Tensor.class, true, null),
            nodes_splits(Tensor.class, false, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public TreeEnsemble(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        TreeEnsemble(TreeEnsemble that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public TreeEnsemble transform(CopyContext cc, OpTransformer ot) {
            return new TreeEnsemble(this, cc);
        }
        
        TreeEnsemble(TypeElement resultType, Value X, Optional<Integer> aggregate_function, Optional<Tensor> nodes_hitrates, int[] nodes_featureids, int[] nodes_falseleafs, Optional<Integer> post_transform, int[] nodes_trueleafs, Tensor nodes_modes, int[] nodes_falsenodeids, int[] nodes_truenodeids, Tensor leaf_weights, int[] leaf_targetids, int[] tree_roots, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<Tensor> membership_values, Tensor nodes_splits) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.aggregate_function.process(attrs, aggregate_function);
            Attribute.nodes_hitrates.process(attrs, nodes_hitrates);
            Attribute.nodes_featureids.process(attrs, nodes_featureids.clone());
            Attribute.nodes_falseleafs.process(attrs, nodes_falseleafs.clone());
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.nodes_trueleafs.process(attrs, nodes_trueleafs.clone());
            Attribute.nodes_modes.process(attrs, nodes_modes);
            Attribute.nodes_falsenodeids.process(attrs, nodes_falsenodeids.clone());
            Attribute.nodes_truenodeids.process(attrs, nodes_truenodeids.clone());
            Attribute.leaf_weights.process(attrs, leaf_weights);
            Attribute.leaf_targetids.process(attrs, leaf_targetids.clone());
            Attribute.tree_roots.process(attrs, tree_roots.clone());
            Attribute.n_targets.process(attrs, n_targets);
            Attribute.nodes_missing_value_tracks_true.process(attrs, nodes_missing_value_tracks_true.map(int[]::clone));
            Attribute.membership_values.process(attrs, membership_values);
            Attribute.nodes_splits.process(attrs, nodes_splits);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> aggregate_function() {
            Integer aggregate_function = Attribute.aggregate_function.access(Integer.class, attributes);
            return Optional.ofNullable(aggregate_function);
        }
        
        public Optional<Tensor> nodes_hitrates() {
            Tensor nodes_hitrates = Attribute.nodes_hitrates.access(Tensor.class, attributes);
            return Optional.ofNullable(nodes_hitrates);
        }
        
        public int[] nodes_featureids() {
            int[] nodes_featureids = Attribute.nodes_featureids.access(int[].class, attributes);
            return nodes_featureids.clone();
        }
        
        public int[] nodes_falseleafs() {
            int[] nodes_falseleafs = Attribute.nodes_falseleafs.access(int[].class, attributes);
            return nodes_falseleafs.clone();
        }
        
        public Optional<Integer> post_transform() {
            Integer post_transform = Attribute.post_transform.access(Integer.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public int[] nodes_trueleafs() {
            int[] nodes_trueleafs = Attribute.nodes_trueleafs.access(int[].class, attributes);
            return nodes_trueleafs.clone();
        }
        
        public Tensor nodes_modes() {
            Tensor nodes_modes = Attribute.nodes_modes.access(Tensor.class, attributes);
            return nodes_modes;
        }
        
        public int[] nodes_falsenodeids() {
            int[] nodes_falsenodeids = Attribute.nodes_falsenodeids.access(int[].class, attributes);
            return nodes_falsenodeids.clone();
        }
        
        public int[] nodes_truenodeids() {
            int[] nodes_truenodeids = Attribute.nodes_truenodeids.access(int[].class, attributes);
            return nodes_truenodeids.clone();
        }
        
        public Tensor leaf_weights() {
            Tensor leaf_weights = Attribute.leaf_weights.access(Tensor.class, attributes);
            return leaf_weights;
        }
        
        public int[] leaf_targetids() {
            int[] leaf_targetids = Attribute.leaf_targetids.access(int[].class, attributes);
            return leaf_targetids.clone();
        }
        
        public int[] tree_roots() {
            int[] tree_roots = Attribute.tree_roots.access(int[].class, attributes);
            return tree_roots.clone();
        }
        
        public Optional<Integer> n_targets() {
            Integer n_targets = Attribute.n_targets.access(Integer.class, attributes);
            return Optional.ofNullable(n_targets);
        }
        
        public Optional<int[]> nodes_missing_value_tracks_true() {
            int[] nodes_missing_value_tracks_true = Attribute.nodes_missing_value_tracks_true.access(int[].class, attributes);
            return Optional.ofNullable(nodes_missing_value_tracks_true).map(int[]::clone);
        }
        
        public Optional<Tensor> membership_values() {
            Tensor membership_values = Attribute.membership_values.access(Tensor.class, attributes);
            return Optional.ofNullable(membership_values);
        }
        
        public Tensor nodes_splits() {
            Tensor nodes_splits = Attribute.nodes_splits.access(Tensor.class, attributes);
            return nodes_splits;
        }
        
    }
    
    public static TreeEnsemble TreeEnsemble(TypeElement resultType, Value X, Optional<Integer> aggregate_function, Optional<Tensor> nodes_hitrates, int[] nodes_featureids, int[] nodes_falseleafs, Optional<Integer> post_transform, int[] nodes_trueleafs, Tensor nodes_modes, int[] nodes_falsenodeids, int[] nodes_truenodeids, Tensor leaf_weights, int[] leaf_targetids, int[] tree_roots, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<Tensor> membership_values, Tensor nodes_splits) {
        return new TreeEnsemble(resultType, X, aggregate_function, nodes_hitrates, nodes_featureids, nodes_falseleafs, post_transform, nodes_trueleafs, nodes_modes, nodes_falsenodeids, nodes_truenodeids, leaf_weights, leaf_targetids, tree_roots, n_targets, nodes_missing_value_tracks_true, membership_values, nodes_splits);
    }

    @OpFactory.OpDeclaration(TreeEnsembleClassifier.NAME)
    public static final class TreeEnsembleClassifier extends OnnxOp {
        public static final String NAME = "TreeEnsembleClassifier";
        
        public enum Attribute implements OnnxAttribute {
            classlabels_int64s(int[].class, true, null),
            class_ids(int[].class, true, null),
            nodes_hitrates(float[].class, true, null),
            nodes_featureids(int[].class, true, null),
            nodes_treeids(int[].class, true, null),
            class_weights_as_tensor(Tensor.class, true, null),
            post_transform(String.class, true, null),
            nodes_modes(String[].class, true, null),
            nodes_falsenodeids(int[].class, true, null),
            classlabels_strings(String[].class, true, null),
            nodes_truenodeids(int[].class, true, null),
            nodes_nodeids(int[].class, true, null),
            nodes_hitrates_as_tensor(Tensor.class, true, null),
            class_weights(float[].class, true, null),
            base_values_as_tensor(Tensor.class, true, null),
            nodes_missing_value_tracks_true(int[].class, true, null),
            class_nodeids(int[].class, true, null),
            class_treeids(int[].class, true, null),
            base_values(float[].class, true, null),
            nodes_values(float[].class, true, null),
            nodes_values_as_tensor(Tensor.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public TreeEnsembleClassifier(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        TreeEnsembleClassifier(TreeEnsembleClassifier that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public TreeEnsembleClassifier transform(CopyContext cc, OpTransformer ot) {
            return new TreeEnsembleClassifier(this, cc);
        }
        
        TreeEnsembleClassifier(TypeElement resultType, Value X, Optional<int[]> classlabels_int64s, Optional<int[]> class_ids, Optional<float[]> nodes_hitrates, Optional<int[]> nodes_featureids, Optional<int[]> nodes_treeids, Optional<Tensor> class_weights_as_tensor, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<int[]> nodes_falsenodeids, Optional<String[]> classlabels_strings, Optional<int[]> nodes_truenodeids, Optional<int[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<float[]> class_weights, Optional<Tensor> base_values_as_tensor, Optional<int[]> nodes_missing_value_tracks_true, Optional<int[]> class_nodeids, Optional<int[]> class_treeids, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.classlabels_int64s.process(attrs, classlabels_int64s.map(int[]::clone));
            Attribute.class_ids.process(attrs, class_ids.map(int[]::clone));
            Attribute.nodes_hitrates.process(attrs, nodes_hitrates.map(float[]::clone));
            Attribute.nodes_featureids.process(attrs, nodes_featureids.map(int[]::clone));
            Attribute.nodes_treeids.process(attrs, nodes_treeids.map(int[]::clone));
            Attribute.class_weights_as_tensor.process(attrs, class_weights_as_tensor);
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.nodes_modes.process(attrs, nodes_modes.map(String[]::clone));
            Attribute.nodes_falsenodeids.process(attrs, nodes_falsenodeids.map(int[]::clone));
            Attribute.classlabels_strings.process(attrs, classlabels_strings.map(String[]::clone));
            Attribute.nodes_truenodeids.process(attrs, nodes_truenodeids.map(int[]::clone));
            Attribute.nodes_nodeids.process(attrs, nodes_nodeids.map(int[]::clone));
            Attribute.nodes_hitrates_as_tensor.process(attrs, nodes_hitrates_as_tensor);
            Attribute.class_weights.process(attrs, class_weights.map(float[]::clone));
            Attribute.base_values_as_tensor.process(attrs, base_values_as_tensor);
            Attribute.nodes_missing_value_tracks_true.process(attrs, nodes_missing_value_tracks_true.map(int[]::clone));
            Attribute.class_nodeids.process(attrs, class_nodeids.map(int[]::clone));
            Attribute.class_treeids.process(attrs, class_treeids.map(int[]::clone));
            Attribute.base_values.process(attrs, base_values.map(float[]::clone));
            Attribute.nodes_values.process(attrs, nodes_values.map(float[]::clone));
            Attribute.nodes_values_as_tensor.process(attrs, nodes_values_as_tensor);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> classlabels_int64s() {
            int[] classlabels_int64s = Attribute.classlabels_int64s.access(int[].class, attributes);
            return Optional.ofNullable(classlabels_int64s).map(int[]::clone);
        }
        
        public Optional<int[]> class_ids() {
            int[] class_ids = Attribute.class_ids.access(int[].class, attributes);
            return Optional.ofNullable(class_ids).map(int[]::clone);
        }
        
        public Optional<float[]> nodes_hitrates() {
            float[] nodes_hitrates = Attribute.nodes_hitrates.access(float[].class, attributes);
            return Optional.ofNullable(nodes_hitrates).map(float[]::clone);
        }
        
        public Optional<int[]> nodes_featureids() {
            int[] nodes_featureids = Attribute.nodes_featureids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_featureids).map(int[]::clone);
        }
        
        public Optional<int[]> nodes_treeids() {
            int[] nodes_treeids = Attribute.nodes_treeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_treeids).map(int[]::clone);
        }
        
        public Optional<Tensor> class_weights_as_tensor() {
            Tensor class_weights_as_tensor = Attribute.class_weights_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(class_weights_as_tensor);
        }
        
        public Optional<String> post_transform() {
            String post_transform = Attribute.post_transform.access(String.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public Optional<String[]> nodes_modes() {
            String[] nodes_modes = Attribute.nodes_modes.access(String[].class, attributes);
            return Optional.ofNullable(nodes_modes).map(String[]::clone);
        }
        
        public Optional<int[]> nodes_falsenodeids() {
            int[] nodes_falsenodeids = Attribute.nodes_falsenodeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_falsenodeids).map(int[]::clone);
        }
        
        public Optional<String[]> classlabels_strings() {
            String[] classlabels_strings = Attribute.classlabels_strings.access(String[].class, attributes);
            return Optional.ofNullable(classlabels_strings).map(String[]::clone);
        }
        
        public Optional<int[]> nodes_truenodeids() {
            int[] nodes_truenodeids = Attribute.nodes_truenodeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_truenodeids).map(int[]::clone);
        }
        
        public Optional<int[]> nodes_nodeids() {
            int[] nodes_nodeids = Attribute.nodes_nodeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_nodeids).map(int[]::clone);
        }
        
        public Optional<Tensor> nodes_hitrates_as_tensor() {
            Tensor nodes_hitrates_as_tensor = Attribute.nodes_hitrates_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(nodes_hitrates_as_tensor);
        }
        
        public Optional<float[]> class_weights() {
            float[] class_weights = Attribute.class_weights.access(float[].class, attributes);
            return Optional.ofNullable(class_weights).map(float[]::clone);
        }
        
        public Optional<Tensor> base_values_as_tensor() {
            Tensor base_values_as_tensor = Attribute.base_values_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(base_values_as_tensor);
        }
        
        public Optional<int[]> nodes_missing_value_tracks_true() {
            int[] nodes_missing_value_tracks_true = Attribute.nodes_missing_value_tracks_true.access(int[].class, attributes);
            return Optional.ofNullable(nodes_missing_value_tracks_true).map(int[]::clone);
        }
        
        public Optional<int[]> class_nodeids() {
            int[] class_nodeids = Attribute.class_nodeids.access(int[].class, attributes);
            return Optional.ofNullable(class_nodeids).map(int[]::clone);
        }
        
        public Optional<int[]> class_treeids() {
            int[] class_treeids = Attribute.class_treeids.access(int[].class, attributes);
            return Optional.ofNullable(class_treeids).map(int[]::clone);
        }
        
        public Optional<float[]> base_values() {
            float[] base_values = Attribute.base_values.access(float[].class, attributes);
            return Optional.ofNullable(base_values).map(float[]::clone);
        }
        
        public Optional<float[]> nodes_values() {
            float[] nodes_values = Attribute.nodes_values.access(float[].class, attributes);
            return Optional.ofNullable(nodes_values).map(float[]::clone);
        }
        
        public Optional<Tensor> nodes_values_as_tensor() {
            Tensor nodes_values_as_tensor = Attribute.nodes_values_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(nodes_values_as_tensor);
        }
        
    }
    
    public static TreeEnsembleClassifier TreeEnsembleClassifier(TypeElement resultType, Value X, Optional<int[]> classlabels_int64s, Optional<int[]> class_ids, Optional<float[]> nodes_hitrates, Optional<int[]> nodes_featureids, Optional<int[]> nodes_treeids, Optional<Tensor> class_weights_as_tensor, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<int[]> nodes_falsenodeids, Optional<String[]> classlabels_strings, Optional<int[]> nodes_truenodeids, Optional<int[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<float[]> class_weights, Optional<Tensor> base_values_as_tensor, Optional<int[]> nodes_missing_value_tracks_true, Optional<int[]> class_nodeids, Optional<int[]> class_treeids, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
        return new TreeEnsembleClassifier(resultType, X, classlabels_int64s, class_ids, nodes_hitrates, nodes_featureids, nodes_treeids, class_weights_as_tensor, post_transform, nodes_modes, nodes_falsenodeids, classlabels_strings, nodes_truenodeids, nodes_nodeids, nodes_hitrates_as_tensor, class_weights, base_values_as_tensor, nodes_missing_value_tracks_true, class_nodeids, class_treeids, base_values, nodes_values, nodes_values_as_tensor);
    }

    @OpFactory.OpDeclaration(TreeEnsembleRegressor.NAME)
    public static final class TreeEnsembleRegressor extends OnnxOp {
        public static final String NAME = "TreeEnsembleRegressor";
        
        public enum Attribute implements OnnxAttribute {
            aggregate_function(String.class, true, null),
            nodes_hitrates(float[].class, true, null),
            target_weights_as_tensor(Tensor.class, true, null),
            nodes_featureids(int[].class, true, null),
            target_treeids(int[].class, true, null),
            nodes_treeids(int[].class, true, null),
            post_transform(String.class, true, null),
            nodes_modes(String[].class, true, null),
            target_weights(float[].class, true, null),
            nodes_falsenodeids(int[].class, true, null),
            target_ids(int[].class, true, null),
            nodes_truenodeids(int[].class, true, null),
            target_nodeids(int[].class, true, null),
            nodes_nodeids(int[].class, true, null),
            nodes_hitrates_as_tensor(Tensor.class, true, null),
            base_values_as_tensor(Tensor.class, true, null),
            n_targets(Integer.class, true, null),
            nodes_missing_value_tracks_true(int[].class, true, null),
            base_values(float[].class, true, null),
            nodes_values(float[].class, true, null),
            nodes_values_as_tensor(Tensor.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public TreeEnsembleRegressor(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        TreeEnsembleRegressor(TreeEnsembleRegressor that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public TreeEnsembleRegressor transform(CopyContext cc, OpTransformer ot) {
            return new TreeEnsembleRegressor(this, cc);
        }
        
        TreeEnsembleRegressor(TypeElement resultType, Value X, Optional<String> aggregate_function, Optional<float[]> nodes_hitrates, Optional<Tensor> target_weights_as_tensor, Optional<int[]> nodes_featureids, Optional<int[]> target_treeids, Optional<int[]> nodes_treeids, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<float[]> target_weights, Optional<int[]> nodes_falsenodeids, Optional<int[]> target_ids, Optional<int[]> nodes_truenodeids, Optional<int[]> target_nodeids, Optional<int[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<Tensor> base_values_as_tensor, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.aggregate_function.process(attrs, aggregate_function);
            Attribute.nodes_hitrates.process(attrs, nodes_hitrates.map(float[]::clone));
            Attribute.target_weights_as_tensor.process(attrs, target_weights_as_tensor);
            Attribute.nodes_featureids.process(attrs, nodes_featureids.map(int[]::clone));
            Attribute.target_treeids.process(attrs, target_treeids.map(int[]::clone));
            Attribute.nodes_treeids.process(attrs, nodes_treeids.map(int[]::clone));
            Attribute.post_transform.process(attrs, post_transform);
            Attribute.nodes_modes.process(attrs, nodes_modes.map(String[]::clone));
            Attribute.target_weights.process(attrs, target_weights.map(float[]::clone));
            Attribute.nodes_falsenodeids.process(attrs, nodes_falsenodeids.map(int[]::clone));
            Attribute.target_ids.process(attrs, target_ids.map(int[]::clone));
            Attribute.nodes_truenodeids.process(attrs, nodes_truenodeids.map(int[]::clone));
            Attribute.target_nodeids.process(attrs, target_nodeids.map(int[]::clone));
            Attribute.nodes_nodeids.process(attrs, nodes_nodeids.map(int[]::clone));
            Attribute.nodes_hitrates_as_tensor.process(attrs, nodes_hitrates_as_tensor);
            Attribute.base_values_as_tensor.process(attrs, base_values_as_tensor);
            Attribute.n_targets.process(attrs, n_targets);
            Attribute.nodes_missing_value_tracks_true.process(attrs, nodes_missing_value_tracks_true.map(int[]::clone));
            Attribute.base_values.process(attrs, base_values.map(float[]::clone));
            Attribute.nodes_values.process(attrs, nodes_values.map(float[]::clone));
            Attribute.nodes_values_as_tensor.process(attrs, nodes_values_as_tensor);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> aggregate_function() {
            String aggregate_function = Attribute.aggregate_function.access(String.class, attributes);
            return Optional.ofNullable(aggregate_function);
        }
        
        public Optional<float[]> nodes_hitrates() {
            float[] nodes_hitrates = Attribute.nodes_hitrates.access(float[].class, attributes);
            return Optional.ofNullable(nodes_hitrates).map(float[]::clone);
        }
        
        public Optional<Tensor> target_weights_as_tensor() {
            Tensor target_weights_as_tensor = Attribute.target_weights_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(target_weights_as_tensor);
        }
        
        public Optional<int[]> nodes_featureids() {
            int[] nodes_featureids = Attribute.nodes_featureids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_featureids).map(int[]::clone);
        }
        
        public Optional<int[]> target_treeids() {
            int[] target_treeids = Attribute.target_treeids.access(int[].class, attributes);
            return Optional.ofNullable(target_treeids).map(int[]::clone);
        }
        
        public Optional<int[]> nodes_treeids() {
            int[] nodes_treeids = Attribute.nodes_treeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_treeids).map(int[]::clone);
        }
        
        public Optional<String> post_transform() {
            String post_transform = Attribute.post_transform.access(String.class, attributes);
            return Optional.ofNullable(post_transform);
        }
        
        public Optional<String[]> nodes_modes() {
            String[] nodes_modes = Attribute.nodes_modes.access(String[].class, attributes);
            return Optional.ofNullable(nodes_modes).map(String[]::clone);
        }
        
        public Optional<float[]> target_weights() {
            float[] target_weights = Attribute.target_weights.access(float[].class, attributes);
            return Optional.ofNullable(target_weights).map(float[]::clone);
        }
        
        public Optional<int[]> nodes_falsenodeids() {
            int[] nodes_falsenodeids = Attribute.nodes_falsenodeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_falsenodeids).map(int[]::clone);
        }
        
        public Optional<int[]> target_ids() {
            int[] target_ids = Attribute.target_ids.access(int[].class, attributes);
            return Optional.ofNullable(target_ids).map(int[]::clone);
        }
        
        public Optional<int[]> nodes_truenodeids() {
            int[] nodes_truenodeids = Attribute.nodes_truenodeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_truenodeids).map(int[]::clone);
        }
        
        public Optional<int[]> target_nodeids() {
            int[] target_nodeids = Attribute.target_nodeids.access(int[].class, attributes);
            return Optional.ofNullable(target_nodeids).map(int[]::clone);
        }
        
        public Optional<int[]> nodes_nodeids() {
            int[] nodes_nodeids = Attribute.nodes_nodeids.access(int[].class, attributes);
            return Optional.ofNullable(nodes_nodeids).map(int[]::clone);
        }
        
        public Optional<Tensor> nodes_hitrates_as_tensor() {
            Tensor nodes_hitrates_as_tensor = Attribute.nodes_hitrates_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(nodes_hitrates_as_tensor);
        }
        
        public Optional<Tensor> base_values_as_tensor() {
            Tensor base_values_as_tensor = Attribute.base_values_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(base_values_as_tensor);
        }
        
        public Optional<Integer> n_targets() {
            Integer n_targets = Attribute.n_targets.access(Integer.class, attributes);
            return Optional.ofNullable(n_targets);
        }
        
        public Optional<int[]> nodes_missing_value_tracks_true() {
            int[] nodes_missing_value_tracks_true = Attribute.nodes_missing_value_tracks_true.access(int[].class, attributes);
            return Optional.ofNullable(nodes_missing_value_tracks_true).map(int[]::clone);
        }
        
        public Optional<float[]> base_values() {
            float[] base_values = Attribute.base_values.access(float[].class, attributes);
            return Optional.ofNullable(base_values).map(float[]::clone);
        }
        
        public Optional<float[]> nodes_values() {
            float[] nodes_values = Attribute.nodes_values.access(float[].class, attributes);
            return Optional.ofNullable(nodes_values).map(float[]::clone);
        }
        
        public Optional<Tensor> nodes_values_as_tensor() {
            Tensor nodes_values_as_tensor = Attribute.nodes_values_as_tensor.access(Tensor.class, attributes);
            return Optional.ofNullable(nodes_values_as_tensor);
        }
        
    }
    
    public static TreeEnsembleRegressor TreeEnsembleRegressor(TypeElement resultType, Value X, Optional<String> aggregate_function, Optional<float[]> nodes_hitrates, Optional<Tensor> target_weights_as_tensor, Optional<int[]> nodes_featureids, Optional<int[]> target_treeids, Optional<int[]> nodes_treeids, Optional<String> post_transform, Optional<String[]> nodes_modes, Optional<float[]> target_weights, Optional<int[]> nodes_falsenodeids, Optional<int[]> target_ids, Optional<int[]> nodes_truenodeids, Optional<int[]> target_nodeids, Optional<int[]> nodes_nodeids, Optional<Tensor> nodes_hitrates_as_tensor, Optional<Tensor> base_values_as_tensor, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<float[]> base_values, Optional<float[]> nodes_values, Optional<Tensor> nodes_values_as_tensor) {
        return new TreeEnsembleRegressor(resultType, X, aggregate_function, nodes_hitrates, target_weights_as_tensor, nodes_featureids, target_treeids, nodes_treeids, post_transform, nodes_modes, target_weights, nodes_falsenodeids, target_ids, nodes_truenodeids, target_nodeids, nodes_nodeids, nodes_hitrates_as_tensor, base_values_as_tensor, n_targets, nodes_missing_value_tracks_true, base_values, nodes_values, nodes_values_as_tensor);
    }

    @OpFactory.OpDeclaration(Unique.NAME)
    public static final class Unique extends OnnxOp {
        public static final String NAME = "Unique";
        
        public enum Attribute implements OnnxAttribute {
            sorted(Integer.class, true, null),
            axis(Integer.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Unique(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Unique(Unique that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Unique transform(CopyContext cc, OpTransformer ot) {
            return new Unique(this, cc);
        }
        
        Unique(TypeElement resultType, Value X, Optional<Integer> sorted, Optional<Integer> axis) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.sorted.process(attrs, sorted);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> sorted() {
            Integer sorted = Attribute.sorted.access(Integer.class, attributes);
            return Optional.ofNullable(sorted);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
    }
    
    public static Unique Unique(TypeElement resultType, Value X, Optional<Integer> sorted, Optional<Integer> axis) {
        return new Unique(resultType, X, sorted, axis);
    }

    @OpFactory.OpDeclaration(Unsqueeze.NAME)
    public static final class Unsqueeze extends OnnxOp {
        public static final String NAME = "Unsqueeze";
        
        public Unsqueeze(ExternalizedOp def) {
            super(def);
        }
        
        Unsqueeze(Unsqueeze that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Unsqueeze transform(CopyContext cc, OpTransformer ot) {
            return new Unsqueeze(this, cc);
        }
        
        Unsqueeze(TypeElement resultType, Value data, Value axes) {
            super(NAME, resultType, List.of(data, axes));
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value axes() {
            return operands().get(1);
        }
        
    }
    
    public static Unsqueeze Unsqueeze(TypeElement resultType, Value data, Value axes) {
        return new Unsqueeze(resultType, data, axes);
    }

    @OpFactory.OpDeclaration(Upsample.NAME)
    public static final class Upsample extends OnnxOp {
        public static final String NAME = "Upsample";
        
        public enum Attribute implements OnnxAttribute {
            mode(String.class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public Upsample(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        Upsample(Upsample that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public Upsample transform(CopyContext cc, OpTransformer ot) {
            return new Upsample(this, cc);
        }
        
        Upsample(TypeElement resultType, Value X, Value scales, Optional<String> mode) {
            super(NAME, resultType, List.of(X, scales));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value scales() {
            return operands().get(1);
        }
        
        public Optional<String> mode() {
            String mode = Attribute.mode.access(String.class, attributes);
            return Optional.ofNullable(mode);
        }
        
    }
    
    public static Upsample Upsample(TypeElement resultType, Value X, Value scales, Optional<String> mode) {
        return new Upsample(resultType, X, scales, mode);
    }

    @OpFactory.OpDeclaration(Where.NAME)
    public static final class Where extends OnnxOp {
        public static final String NAME = "Where";
        
        public Where(ExternalizedOp def) {
            super(def);
        }
        
        Where(Where that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Where transform(CopyContext cc, OpTransformer ot) {
            return new Where(this, cc);
        }
        
        Where(TypeElement resultType, Value condition, Value X, Value Y) {
            super(NAME, resultType, List.of(condition, X, Y));
        }
        
        public Value condition() {
            return operands().get(0);
        }
        
        public Value X() {
            return operands().get(1);
        }
        
        public Value Y() {
            return operands().get(2);
        }
        
    }
    
    public static Where Where(TypeElement resultType, Value condition, Value X, Value Y) {
        return new Where(resultType, condition, X, Y);
    }

    @OpFactory.OpDeclaration(Xor.NAME)
    public static final class Xor extends OnnxOp {
        public static final String NAME = "Xor";
        
        public Xor(ExternalizedOp def) {
            super(def);
        }
        
        Xor(Xor that, CopyContext cc) {
            super(that, cc);
        }
        
        @Override
        public Xor transform(CopyContext cc, OpTransformer ot) {
            return new Xor(this, cc);
        }
        
        Xor(TypeElement resultType, Value A, Value B) {
            super(NAME, resultType, List.of(A, B));
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
    }
    
    public static Xor Xor(TypeElement resultType, Value A, Value B) {
        return new Xor(resultType, A, B);
    }

    @OpFactory.OpDeclaration(ZipMap.NAME)
    public static final class ZipMap extends OnnxOp {
        public static final String NAME = "ZipMap";
        
        public enum Attribute implements OnnxAttribute {
            classlabels_int64s(int[].class, true, null),
            classlabels_strings(String[].class, true, null),
            ;
            
            final Class<?> type_;
            final boolean optional;
            final Object defaultValue;
            
            Attribute(Class<?> type_, boolean optional, Object defaultValue) {
                this.type_ = type_;
                this.optional = optional;
                this.defaultValue = defaultValue;
                assert optional || defaultValue == null;
            }
            
            public Class<?> type() {
                return type_;
            }
            
            public boolean optional() {
                return optional;
            }
            
            public Object defaultValue() {
                return defaultValue;
            }
        }
        
        final Map<String, Object> attributes;
        
        public ZipMap(ExternalizedOp def) {
            super(def);
            
            this.attributes = OnnxAttribute.process(def, Attribute::valueOf);
        }
        
        ZipMap(ZipMap that, CopyContext cc) {
            super(that, cc);
        
            this.attributes = Map.copyOf(that.attributes);
        }
        
        @Override
        public ZipMap transform(CopyContext cc, OpTransformer ot) {
            return new ZipMap(this, cc);
        }
        
        ZipMap(TypeElement resultType, Value X, Optional<int[]> classlabels_int64s, Optional<String[]> classlabels_strings) {
            super(NAME, resultType, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.classlabels_int64s.process(attrs, classlabels_int64s.map(int[]::clone));
            Attribute.classlabels_strings.process(attrs, classlabels_strings.map(String[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> classlabels_int64s() {
            int[] classlabels_int64s = Attribute.classlabels_int64s.access(int[].class, attributes);
            return Optional.ofNullable(classlabels_int64s).map(int[]::clone);
        }
        
        public Optional<String[]> classlabels_strings() {
            String[] classlabels_strings = Attribute.classlabels_strings.access(String[].class, attributes);
            return Optional.ofNullable(classlabels_strings).map(String[]::clone);
        }
        
    }
    
    public static ZipMap ZipMap(TypeElement resultType, Value X, Optional<int[]> classlabels_int64s, Optional<String[]> classlabels_strings) {
        return new ZipMap(resultType, X, classlabels_int64s, classlabels_strings);
    }

}
