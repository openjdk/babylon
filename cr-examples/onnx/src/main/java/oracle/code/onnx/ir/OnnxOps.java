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
        
        Abs(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Acos(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Acosh(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        AffineGrid(Value theta, Value size, Optional<Integer> align_corners) {
            super(NAME, List.of(theta, size));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.align_corners.process(attrs, align_corners);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        ArrayFeatureExtractor(Value X, Value Y) {
            super(NAME, List.of(X, Y));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
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
        
        Asin(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Asinh(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Atan(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Atanh(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        AveragePool(Value X, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> count_include_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
            super(NAME, List.of(X));
            
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
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Binarizer(Value X, Optional<Float> threshold) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.threshold.process(attrs, threshold);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> threshold() {
            Float threshold = Attribute.threshold.access(Float.class, attributes);
            return Optional.ofNullable(threshold);
        }
        
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
        
        BitShift(Value X, Value Y, String direction) {
            super(NAME, List.of(X, Y));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.direction.process(attrs, direction);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        BitwiseAnd(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        BitwiseNot(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        BitwiseOr(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        BitwiseXor(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        CastLike(Value input, Value target_type, Optional<Integer> saturate) {
            super(NAME, List.of(input, target_type));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.saturate.process(attrs, saturate);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(1).type();
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
        
        Ceil(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Celu(Value X, Optional<Float> alpha) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
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
        
        CenterCropPad(Value input_data, Value shape, Optional<int[]> axes) {
            super(NAME, List.of(input_data, shape));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axes.process(attrs, axes.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Col2Im(Value input, Value image_shape, Value block_shape, Optional<int[]> pads, Optional<int[]> dilations, Optional<int[]> strides) {
            super(NAME, List.of(input, image_shape, block_shape));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.pads.process(attrs, pads.map(int[]::clone));
            Attribute.dilations.process(attrs, dilations.map(int[]::clone));
            Attribute.strides.process(attrs, strides.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Compress(Value input, Value condition, Optional<Integer> axis) {
            super(NAME, List.of(input, condition));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Cos(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Cosh(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        CumSum(Value x, Value axis, Optional<Integer> exclusive, Optional<Integer> reverse) {
            super(NAME, List.of(x, axis));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.exclusive.process(attrs, exclusive);
            Attribute.reverse.process(attrs, reverse);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        DepthToSpace(Value input, Optional<String> mode, int blocksize) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            Attribute.blocksize.process(attrs, blocksize);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Det(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Div(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        Elu(Value X, Optional<Float> alpha) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
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
        
        Erf(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Exp(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Expand(Value input, Value shape) {
            super(NAME, List.of(input, shape));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value shape() {
            return operands().get(1);
        }
        
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
        
        Flatten(Value input, Optional<Integer> axis) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
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
        
        Floor(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Gather(Value data, Value indices, Optional<Integer> axis) {
            super(NAME, List.of(data, indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        GatherElements(Value data, Value indices, Optional<Integer> axis) {
            super(NAME, List.of(data, indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        GatherND(Value data, Value indices, Optional<Integer> batch_dims) {
            super(NAME, List.of(data, indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.batch_dims.process(attrs, batch_dims);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Gelu(Value X, Optional<String> approximate) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.approximate.process(attrs, approximate);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<String> approximate() {
            String approximate = Attribute.approximate.access(String.class, attributes);
            return Optional.ofNullable(approximate);
        }
        
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
        
        GlobalAveragePool(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        GlobalLpPool(Value X, Optional<Integer> p) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.p.process(attrs, p);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Integer> p() {
            Integer p = Attribute.p.access(Integer.class, attributes);
            return Optional.ofNullable(p);
        }
        
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
        
        GlobalMaxPool(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        GridSample(Value X, Value grid, Optional<String> mode, Optional<Integer> align_corners, Optional<String> padding_mode) {
            super(NAME, List.of(X, grid));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            Attribute.align_corners.process(attrs, align_corners);
            Attribute.padding_mode.process(attrs, padding_mode);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        GroupNormalization(Value X, Value scale, Value bias, Optional<Float> epsilon, Optional<Integer> stash_type, int num_groups) {
            super(NAME, List.of(X, scale, bias));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.epsilon.process(attrs, epsilon);
            Attribute.stash_type.process(attrs, stash_type);
            Attribute.num_groups.process(attrs, num_groups);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        HardSigmoid(Value X, Optional<Float> alpha, Optional<Float> beta) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            Attribute.beta.process(attrs, beta);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        HardSwish(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Hardmax(Value input, Optional<Integer> axis) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
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
        
        Identity(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Imputer(Value X, Optional<Integer> replaced_value_int64, Optional<Float> replaced_value_float, Optional<int[]> imputed_value_int64s, Optional<float[]> imputed_value_floats) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.replaced_value_int64.process(attrs, replaced_value_int64);
            Attribute.replaced_value_float.process(attrs, replaced_value_float);
            Attribute.imputed_value_int64s.process(attrs, imputed_value_int64s.map(int[]::clone));
            Attribute.imputed_value_floats.process(attrs, imputed_value_floats.map(float[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        InstanceNormalization(Value input, Value scale, Value B, Optional<Float> epsilon) {
            super(NAME, List.of(input, scale, B));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.epsilon.process(attrs, epsilon);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        LRN(Value X, int size, Optional<Float> alpha, Optional<Float> bias, Optional<Float> beta) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.size.process(attrs, size);
            Attribute.alpha.process(attrs, alpha);
            Attribute.bias.process(attrs, bias);
            Attribute.beta.process(attrs, beta);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        LeakyRelu(Value X, Optional<Float> alpha) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
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
        
        Log(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        LogSoftmax(Value input, Optional<Integer> axis) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
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
        
        LpNormalization(Value input, Optional<Integer> p, Optional<Integer> axis) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.p.process(attrs, p);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        LpPool(Value X, Optional<Integer> p, Optional<int[]> pads, Optional<int[]> dilations, Optional<String> auto_pad, Optional<Integer> ceil_mode, Optional<int[]> strides, int[] kernel_shape) {
            super(NAME, List.of(X));
            
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
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        MatMul(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        MaxRoiPool(Value X, Value rois, Optional<Float> spatial_scale, int[] pooled_shape) {
            super(NAME, List.of(X, rois));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.spatial_scale.process(attrs, spatial_scale);
            Attribute.pooled_shape.process(attrs, pooled_shape.clone());
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        MeanVarianceNormalization(Value X, Optional<int[]> axes) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axes.process(attrs, axes.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<int[]> axes() {
            int[] axes = Attribute.axes.access(int[].class, attributes);
            return Optional.ofNullable(axes).map(int[]::clone);
        }
        
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
        
        Mish(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Mod(Value A, Value B, Optional<Integer> fmod) {
            super(NAME, List.of(A, B));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.fmod.process(attrs, fmod);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Mul(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        Neg(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Not(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        OneHot(Value indices, Value depth, Value values, Optional<Integer> axis) {
            super(NAME, List.of(indices, depth, values));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(2).type();
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
        
        PRelu(Value X, Value slope) {
            super(NAME, List.of(X, slope));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value slope() {
            return operands().get(1);
        }
        
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
        
        Pow(Value X, Value Y) {
            super(NAME, List.of(X, Y));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
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
        
        QLinearMatMul(Value a, Value a_scale, Value a_zero_point, Value b, Value b_scale, Value b_zero_point, Value y_scale, Value y_zero_point) {
            super(NAME, List.of(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(7).type();
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
        
        Range(Value start, Value limit, Value delta) {
            super(NAME, List.of(start, limit, delta));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Reciprocal(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Relu(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Reshape(Value data, Value shape, Optional<Integer> allowzero) {
            super(NAME, List.of(data, shape));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.allowzero.process(attrs, allowzero);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        ReverseSequence(Value input, Value sequence_lens, Optional<Integer> time_axis, Optional<Integer> batch_axis) {
            super(NAME, List.of(input, sequence_lens));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.time_axis.process(attrs, time_axis);
            Attribute.batch_axis.process(attrs, batch_axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        RoiAlign(Value X, Value rois, Value batch_indices, Optional<String> mode, Optional<Integer> output_width, Optional<Float> spatial_scale, Optional<String> coordinate_transformation_mode, Optional<Integer> sampling_ratio, Optional<Integer> output_height) {
            super(NAME, List.of(X, rois, batch_indices));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            Attribute.output_width.process(attrs, output_width);
            Attribute.spatial_scale.process(attrs, spatial_scale);
            Attribute.coordinate_transformation_mode.process(attrs, coordinate_transformation_mode);
            Attribute.sampling_ratio.process(attrs, sampling_ratio);
            Attribute.output_height.process(attrs, output_height);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Round(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Scatter(Value data, Value indices, Value updates, Optional<Integer> axis) {
            super(NAME, List.of(data, indices, updates));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        ScatterElements(Value data, Value indices, Value updates, Optional<String> reduction, Optional<Integer> axis) {
            super(NAME, List.of(data, indices, updates));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.reduction.process(attrs, reduction);
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        ScatterND(Value data, Value indices, Value updates, Optional<String> reduction) {
            super(NAME, List.of(data, indices, updates));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.reduction.process(attrs, reduction);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Selu(Value X, Optional<Float> alpha, Optional<Float> gamma) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            Attribute.gamma.process(attrs, gamma);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Shrink(Value input, Optional<Float> lambd, Optional<Float> bias) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.lambd.process(attrs, lambd);
            Attribute.bias.process(attrs, bias);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Sigmoid(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Sign(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Sin(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Sinh(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Softmax(Value input, Optional<Integer> axis) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.axis.process(attrs, axis);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Optional<Integer> axis() {
            Integer axis = Attribute.axis.access(Integer.class, attributes);
            return Optional.ofNullable(axis);
        }
        
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
        
        Softplus(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        Softsign(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        SpaceToDepth(Value input, int blocksize) {
            super(NAME, List.of(input));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.blocksize.process(attrs, blocksize);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public int blocksize() {
            Integer blocksize = Attribute.blocksize.access(Integer.class, attributes);
            return blocksize;
        }
        
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
        
        Sqrt(Value X) {
            super(NAME, List.of(X));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
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
        
        StringConcat(Value X, Value Y) {
            super(NAME, List.of(X, Y));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Value Y() {
            return operands().get(1);
        }
        
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
        
        StringNormalizer(Value X, Optional<Integer> is_case_sensitive, Optional<String> locale, Optional<String[]> stopwords, Optional<String> case_change_action) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.is_case_sensitive.process(attrs, is_case_sensitive);
            Attribute.locale.process(attrs, locale);
            Attribute.stopwords.process(attrs, stopwords.map(String[]::clone));
            Attribute.case_change_action.process(attrs, case_change_action);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Sub(Value A, Value B) {
            super(NAME, List.of(A, B));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value A() {
            return operands().get(0);
        }
        
        public Value B() {
            return operands().get(1);
        }
        
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
        
        Tan(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        Tanh(Value input) {
            super(NAME, List.of(input));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
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
        
        ThresholdedRelu(Value X, Optional<Float> alpha) {
            super(NAME, List.of(X));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.alpha.process(attrs, alpha);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value X() {
            return operands().get(0);
        }
        
        public Optional<Float> alpha() {
            Float alpha = Attribute.alpha.access(Float.class, attributes);
            return Optional.ofNullable(alpha);
        }
        
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
        
        Tile(Value input, Value repeats) {
            super(NAME, List.of(input, repeats));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value input() {
            return operands().get(0);
        }
        
        public Value repeats() {
            return operands().get(1);
        }
        
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
        
        Transpose(Value data, Optional<int[]> perm) {
            super(NAME, List.of(data));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.perm.process(attrs, perm.map(int[]::clone));
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Optional<int[]> perm() {
            int[] perm = Attribute.perm.access(int[].class, attributes);
            return Optional.ofNullable(perm).map(int[]::clone);
        }
        
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
        
        TreeEnsemble(Value X, Optional<Integer> aggregate_function, Optional<Tensor> nodes_hitrates, int[] nodes_featureids, int[] nodes_falseleafs, Optional<Integer> post_transform, int[] nodes_trueleafs, Tensor nodes_modes, int[] nodes_falsenodeids, int[] nodes_truenodeids, Tensor leaf_weights, int[] leaf_targetids, int[] tree_roots, Optional<Integer> n_targets, Optional<int[]> nodes_missing_value_tracks_true, Optional<Tensor> membership_values, Tensor nodes_splits) {
            super(NAME, List.of(X));
            
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
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Unsqueeze(Value data, Value axes) {
            super(NAME, List.of(data, axes));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
        
        public Value data() {
            return operands().get(0);
        }
        
        public Value axes() {
            return operands().get(1);
        }
        
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
        
        Upsample(Value X, Value scales, Optional<String> mode) {
            super(NAME, List.of(X, scales));
            
            Map<String, Object> attrs = new HashMap<>();
            Attribute.mode.process(attrs, mode);
            this.attributes = Map.copyOf(attrs);
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
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
        
        Where(Value condition, Value X, Value Y) {
            super(NAME, List.of(condition, X, Y));
        }
        
        @Override
        public TypeElement resultType() {
            return operands().get(1).type();
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

}
