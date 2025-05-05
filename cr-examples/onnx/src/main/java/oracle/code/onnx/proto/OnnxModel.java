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

package oracle.code.onnx.proto;

import java.io.RandomAccessFile;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntSupplier;
import java.util.function.Supplier;

import oracle.code.onnx.proto.OnnxConstants.*;

// Generated from onnx.in.proto
public sealed interface OnnxModel {

    /// Attributes
    ///
    /// A named attribute containing either singular float, integer, string, graph,
    /// and tensor values, or repeated float, integer, string, graph, and tensor values.
    /// An AttributeProto MUST contain the name field, and *only one* of the
    /// following content fields, effectively enforcing a C/C++ union equivalent.
    public record AttributeProto (

        /// The name field MUST be present for this version of the IR.
        /// namespace Attribute
        @f(1) String name,

        /// if ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
        /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
        /// in parent scope.
        /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
        @f(21) String refAttrName,

        /// A human-readable documentation for this attribute. Markdown is allowed.
        @f(13) String docString,

        /// The type field MUST be present for this version of the IR.
        /// For 0.0.1 versions of the IR, this field was not defined, and
        /// implementations needed to use has_field heuristics to determine
        /// which value field was in use.  For IR_VERSION 0.0.2 or later, this
        /// field MUST be set and match the f|i|s|t|... field in use.  This
        /// change was made to accommodate proto3 implementations.
        /// discriminator that indicates which field below is in use
        @f(20) AttributeType type,

        /// Exactly ONE of the following fields must be present for this version of the IR
        /// float
        @f(2) Float f,

        /// int
        @f(3) Long i,

        /// UTF-8 string
        @f(4) byte[] s,

        /// tensor value
        @f(5) TensorProto t,

        /// graph
        @f(6) GraphProto g,

        /// sparse tensor value
        @f(22) SparseTensorProto sparseTensor,

        /// Do not use field below, it's deprecated.
        /// optional ValueProto v = 12;         // value - subsumes everything but graph
        /// type proto
        @f(14) TypeProto tp,

        /// list of floats
        @f(7) List<float[]> floats,

        /// list of ints
        @f(8) List<long[]> ints,

        /// list of UTF-8 strings
        @f(9) List<byte[]> strings,

        /// list of tensors
        @f(10) List<TensorProto> tensors,

        /// list of graph
        @f(11) List<GraphProto> graphs,

        /// list of sparse tensors
        @f(23) List<SparseTensorProto> sparseTensors,

        /// list of type protos
        @f(15) List<TypeProto> typeProtos) implements OnnxModel {
    }

    /// Defines information on value, including the name, the type, and
    /// the shape of the value.
    public record ValueInfoProto (

        /// This field MUST be present in this version of the IR.
        /// namespace Value
        @f(1) String name,

        /// This field MUST be present in this version of the IR for
        /// inputs and outputs of the top-level graph.
        @f(2) TypeProto type,

        /// A human-readable documentation for this value. Markdown is allowed.
        @f(3) String docString,

        /// Named metadata values; keys should be distinct.
        @f(4) List<StringStringEntryProto> metadataProps) implements OnnxModel {
    }

    /// Nodes
    ///
    /// Computation graphs are made up of a DAG of nodes, which represent what is
    /// commonly called a "layer" or "pipeline stage" in machine learning frameworks.
    ///
    /// For example, it can be a node of type "Conv" that takes in an image, a filter
    /// tensor and a bias tensor, and produces the convolved output.
    public record NodeProto (

        /// namespace Value
        @f(1) List<String> input,

        /// namespace Value
        @f(2) List<String> output,

        /// An optional identifier for this node in a graph.
        /// This field MAY be absent in this version of the IR.
        /// namespace Node
        @f(3) String name,

        /// The symbolic identifier of the Operator to execute.
        /// namespace Operator
        @f(4) String opType,

        /// The domain of the OperatorSet that specifies the operator named by op_type.
        /// namespace Domain
        @f(7) String domain,

        /// Overload identifier, used only to map this to a model-local function.
        @f(8) String overload,

        /// Additional named attributes.
        @f(5) List<AttributeProto> attribute,

        /// A human-readable documentation for this node. Markdown is allowed.
        @f(6) String docString,

        /// Named metadata values; keys should be distinct.
        @f(9) List<StringStringEntryProto> metadataProps) implements OnnxModel {
    }

    /// Training information
    /// TrainingInfoProto stores information for training a model.
    /// In particular, this defines two functionalities: an initialization-step
    /// and a training-algorithm-step. Initialization resets the model
    /// back to its original state as if no training has been performed.
    /// Training algorithm improves the model based on input data.
    ///
    /// The semantics of the initialization-step is that the initializers
    /// in ModelProto.graph and in TrainingInfoProto.algorithm are first
    /// initialized as specified by the initializers in the graph, and then
    /// updated by the "initialization_binding" in every instance in
    /// ModelProto.training_info.
    ///
    /// The field "algorithm" defines a computation graph which represents a
    /// training algorithm's step. After the execution of a
    /// TrainingInfoProto.algorithm, the initializers specified by "update_binding"
    /// may be immediately updated. If the targeted training algorithm contains
    /// consecutive update steps (such as block coordinate descent methods),
    /// the user needs to create a TrainingInfoProto for each step.
    public record TrainingInfoProto (

        /// This field describes a graph to compute the initial tensors
        /// upon starting the training process. Initialization graph has no input
        /// and can have multiple outputs. Usually, trainable tensors in neural
        /// networks are randomly initialized. To achieve that, for each tensor,
        /// the user can put a random number operator such as RandomNormal or
        /// RandomUniform in TrainingInfoProto.initialization.node and assign its
        /// random output to the specific tensor using "initialization_binding".
        /// This graph can also set the initializers in "algorithm" in the same
        /// TrainingInfoProto; a use case is resetting the number of training
        /// iteration to zero.
        ///
        /// By default, this field is an empty graph and its evaluation does not
        /// produce any output. Thus, no initializer would be changed by default.
        @f(1) GraphProto initialization,

        /// This field represents a training algorithm step. Given required inputs,
        /// it computes outputs to update initializers in its own or inference graph's
        /// initializer lists. In general, this field contains loss node, gradient node,
        /// optimizer node, increment of iteration count.
        ///
        /// An execution of the training algorithm step is performed by executing the
        /// graph obtained by combining the inference graph (namely "ModelProto.graph")
        /// and the "algorithm" graph. That is, the actual
        /// input/initializer/output/node/value_info/sparse_initializer list of
        /// the training graph is the concatenation of
        /// "ModelProto.graph.input/initializer/output/node/value_info/sparse_initializer"
        /// and "algorithm.input/initializer/output/node/value_info/sparse_initializer"
        /// in that order. This combined graph must satisfy the normal ONNX conditions.
        /// Now, let's provide a visualization of graph combination for clarity.
        /// Let the inference graph (i.e., "ModelProto.graph") be
        ///    tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d
        /// and the "algorithm" graph be
        ///    tensor_d -> Add -> tensor_e
        /// The combination process results
        ///    tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d -> Add -> tensor_e
        ///
        /// Notice that an input of a node in the "algorithm" graph may reference the
        /// output of a node in the inference graph (but not the other way round). Also, inference
        /// node cannot reference inputs of "algorithm". With these restrictions, inference graph
        /// can always be run independently without training information.
        ///
        /// By default, this field is an empty graph and its evaluation does not
        /// produce any output. Evaluating the default training step never
        /// update any initializers.
        @f(2) GraphProto algorithm,

        /// This field specifies the bindings from the outputs of "initialization" to
        /// some initializers in "ModelProto.graph.initializer" and
        /// the "algorithm.initializer" in the same TrainingInfoProto.
        /// See "update_binding" below for details.
        ///
        /// By default, this field is empty and no initializer would be changed
        /// by the execution of "initialization".
        @f(3) List<StringStringEntryProto> initializationBinding,

        /// Gradient-based training is usually an iterative procedure. In one gradient
        /// descent iteration, we apply
        ///
        /// x = x - r * g
        ///
        /// where "x" is the optimized tensor, "r" stands for learning rate, and "g" is
        /// gradient of "x" with respect to a chosen loss. To avoid adding assignments
        /// into the training graph, we split the update equation into
        ///
        /// y = x - r * g
        /// x = y
        ///
        /// The user needs to save "y = x - r * g" into TrainingInfoProto.algorithm. To
        /// tell that "y" should be assigned to "x", the field "update_binding" may
        /// contain a key-value pair of strings, "x" (key of StringStringEntryProto)
        /// and "y" (value of StringStringEntryProto).
        /// For a neural network with multiple trainable (mutable) tensors, there can
        /// be multiple key-value pairs in "update_binding".
        ///
        /// The initializers appears as keys in "update_binding" are considered
        /// mutable variables. This implies some behaviors
        /// as described below.
        ///
        ///  1. We have only unique keys in all "update_binding"s so that two
        ///     variables may not have the same name. This ensures that one
        ///     variable is assigned up to once.
        ///  2. The keys must appear in names of "ModelProto.graph.initializer" or
        ///     "TrainingInfoProto.algorithm.initializer".
        ///  3. The values must be output names of "algorithm" or "ModelProto.graph.output".
        ///  4. Mutable variables are initialized to the value specified by the
        ///     corresponding initializer, and then potentially updated by
        ///     "initializer_binding"s and "update_binding"s in "TrainingInfoProto"s.
        ///
        /// This field usually contains names of trainable tensors
        /// (in ModelProto.graph), optimizer states such as momentums in advanced
        /// stochastic gradient methods (in TrainingInfoProto.graph),
        /// and number of training iterations (in TrainingInfoProto.graph).
        ///
        /// By default, this field is empty and no initializer would be changed
        /// by the execution of "algorithm".
        @f(4) List<StringStringEntryProto> updateBinding) implements OnnxModel {
    }

    /// Models
    ///
    /// ModelProto is a top-level file/container format for bundling a ML model and
    /// associating its computation graph with metadata.
    ///
    /// The semantics of the model are described by the associated GraphProto's.
    public record ModelProto (

        /// The version of the IR this model targets. See Version enum above.
        /// This field MUST be present.
        @f(1) Long irVersion,

        /// The OperatorSets this model relies on.
        /// All ModelProtos MUST have at least one entry that
        /// specifies which version of the ONNX OperatorSet is
        /// being imported.
        ///
        /// All nodes in the ModelProto's graph will bind against the operator
        /// with the same-domain/same-op_type operator with the HIGHEST version
        /// in the referenced operator sets.
        @f(8) List<OperatorSetIdProto> opsetImport,

        /// The name of the framework or tool used to generate this model.
        /// This field SHOULD be present to indicate which implementation/tool/framework
        /// emitted the model.
        @f(2) String producerName,

        /// The version of the framework or tool used to generate this model.
        /// This field SHOULD be present to indicate which implementation/tool/framework
        /// emitted the model.
        @f(3) String producerVersion,

        /// Domain name of the model.
        /// We use reverse domain names as name space indicators. For example:
        /// `com.facebook.fair` or `com.microsoft.cognitiveservices`
        ///
        /// Together with `model_version` and GraphProto.name, this forms the unique identity of
        /// the graph.
        @f(4) String domain,

        /// The version of the graph encoded. See Version enum below.
        @f(5) Long modelVersion,

        /// A human-readable documentation for this model. Markdown is allowed.
        @f(6) String docString,

        /// The parameterized graph that is evaluated to execute the model.
        @f(7) GraphProto graph,

        /// Named metadata values; keys should be distinct.
        @f(14) List<StringStringEntryProto> metadataProps,

        /// Training-specific information. Sequentially executing all stored
        /// `TrainingInfoProto.algorithm`s and assigning their outputs following
        /// the corresponding `TrainingInfoProto.update_binding`s is one training
        /// iteration. Similarly, to initialize the model
        /// (as if training hasn't happened), the user should sequentially execute
        /// all stored `TrainingInfoProto.initialization`s and assigns their outputs
        /// using `TrainingInfoProto.initialization_binding`s.
        ///
        /// If this field is empty, the training behavior of the model is undefined.
        @f(20) List<TrainingInfoProto> trainingInfo,

        /// A list of function protos local to the model.
        ///
        /// The (domain, name, overload) tuple must be unique across the function protos in this list.
        /// In case of any conflicts the behavior (whether the model local functions are given higher priority,
        /// or standard operator sets are given higher priotity or this is treated as error) is defined by
        /// the runtimes.
        ///
        /// The operator sets imported by FunctionProto should be compatible with the ones
        /// imported by ModelProto and other model local FunctionProtos.
        /// Example, if same operator set say 'A' is imported by a FunctionProto and ModelProto
        /// or by 2 FunctionProtos then versions for the operator set may be different but,
        /// the operator schema returned for op_type, domain, version combination
        /// for both the versions should be same for every node in the function body.
        ///
        /// One FunctionProto can reference other FunctionProto in the model, however, recursive reference
        /// is not allowed.
        @f(25) List<FunctionProto> functions) implements OnnxModel {
    }

    /// StringStringEntryProto follows the pattern for cross-proto-version maps.
    /// See https://developers.google.com/protocol-buffers/docs/proto3#maps
    public record StringStringEntryProto (

        @f(1) String key,

        @f(2) String value) implements OnnxModel {
    }

    public record TensorAnnotation (

        @f(1) String tensorName,

        /// <key, value> pairs to annotate tensor specified by <tensor_name> above.
        /// The keys used in the mapping below must be pre-defined in ONNX spec.
        /// For example, for 8-bit linear quantization case, 'SCALE_TENSOR', 'ZERO_POINT_TENSOR' will be pre-defined as
        /// quantization parameter keys.
        @f(2) List<StringStringEntryProto> quantParameterTensorNames) implements OnnxModel {
    }

    /// Graphs
    ///
    /// A graph defines the computational logic of a model and is comprised of a parameterized
    /// list of nodes that form a directed acyclic graph based on their inputs and outputs.
    /// This is the equivalent of the "network" or "graph" in many deep learning
    /// frameworks.
    public record GraphProto (

        /// The nodes in the graph, sorted topologically.
        @f(1) List<NodeProto> node,

        /// The name of the graph.
        /// namespace Graph
        @f(2) String name,

        /// A list of named tensor values, used to specify constant inputs of the graph.
        /// Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
        /// The name MUST be unique across both initializer and sparse_initializer,
        /// but the name MAY also appear in the input list.
        @f(5) List<TensorProto> initializer,

        /// Initializers (see above) stored in sparse format.
        @f(15) List<SparseTensorProto> sparseInitializer,

        /// A human-readable documentation for this graph. Markdown is allowed.
        @f(10) String docString,

        /// The inputs and outputs of the graph.
        @f(11) List<ValueInfoProto> input,

        @f(12) List<ValueInfoProto> output,

        /// Information for the values in the graph. The ValueInfoProto.name's
        /// must be distinct. It is optional for a value to appear in value_info list.
        @f(13) List<ValueInfoProto> valueInfo,

        /// This field carries information to indicate the mapping among a tensor and its
        /// quantization parameter tensors. For example:
        /// For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
        /// which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
        @f(14) List<TensorAnnotation> quantizationAnnotation,

        /// Named metadata values; keys should be distinct.
        @f(16) List<StringStringEntryProto> metadataProps) implements OnnxModel {
    }

    /// Tensors
    ///
    /// A serialized tensor value.
    public record TensorProto (

        /// The shape of the tensor.
        @f(1) List<long[]> dims,

        /// The data type of the tensor.
        /// This field MUST have a valid TensorProto.DataType value
        @f(2) Integer dataType,

        @f(3) Segment segment,

        /// For float and complex64 values
        /// Complex64 tensors are encoded as a single array of floats,
        /// with the real components appearing in odd numbered positions,
        /// and the corresponding imaginary component appearing in the
        /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
        /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
        /// When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
        @f(4) List<float[]> floatData,

        /// For int32, uint8, int8, uint16, int16, uint4, int4, bool, (b)float16, float8, and float4:
        /// - (b)float16 and float8 values MUST be converted bit-wise into an unsigned integer
        ///   representation before being written to the buffer.
        /// - Each pair of uint4, int4, and float4 values MUST be packed as two 4-bit elements into a single byte.
        ///   The first element is stored in the 4 least significant bits (LSB),
        ///   and the second element is stored in the 4 most significant bits (MSB).
        ///
        /// Consequently:
        /// - For data types with a bit-width of 8 or greater, each `int32_data` stores one element.
        /// - For 4-bit data types, each `int32_data` stores two elements.
        ///
        /// When this field is present, the data_type field MUST be
        /// INT32, INT16, INT8, INT4, UINT16, UINT8, UINT4, BOOL, FLOAT16, BFLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, FLOAT4E2M1
        @f(5) List<int[]> int32Data,

        /// For strings.
        /// Each element of string_data is a UTF-8 encoded Unicode
        /// string. No trailing null, no leading BOM. The protobuf "string"
        /// scalar type is not used to match ML community conventions.
        /// When this field is present, the data_type field MUST be STRING
        @f(6) List<byte[]> stringData,

        /// For int64.
        /// When this field is present, the data_type field MUST be INT64
        @f(7) List<long[]> int64Data,

        /// Optionally, a name for the tensor.
        /// namespace Value
        @f(8) String name,

        /// A human-readable documentation for this tensor. Markdown is allowed.
        @f(12) String docString,

        /// Serializations can either use one of the fields above, or use this
        /// raw bytes field. The only exception is the string case, where one is
        /// required to store the content in the repeated bytes string_data field.
        ///
        /// When this raw_data field is used to store tensor value, elements MUST
        /// be stored in as fixed-width, little-endian order.
        /// Floating-point data types MUST be stored in IEEE 754 format.
        /// Complex64 elements must be written as two consecutive FLOAT values, real component first.
        /// Complex128 elements must be written as two consecutive DOUBLE values, real component first.
        /// Boolean type MUST be written one byte per tensor element (00000001 for true, 00000000 for false).
        /// uint4 and int4 values must be packed to 4bitx2, the first element is stored in the 4 LSB and the second element is stored in the 4 MSB.
        ///
        /// Note: the advantage of specific field rather than the raw_data field is
        /// that in some cases (e.g. int data), protobuf does a better packing via
        /// variable length storage, and may lead to smaller binary footprint.
        /// When this field is present, the data_type field MUST NOT be STRING or UNDEFINED
        @f(9) byte[] rawData,

        /// Data can be stored inside the protobuf file using type-specific fields or raw_data.
        /// Alternatively, raw bytes data can be stored in an external file, using the external_data field.
        /// external_data stores key-value pairs describing data location. Recognized keys are:
        /// - "location" (required) - POSIX filesystem path relative to the directory where the ONNX
        ///                           protobuf model was stored
        /// - "offset" (optional) - position of byte at which stored data begins. Integer stored as string.
        ///                         Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
        /// - "length" (optional) - number of bytes containing data. Integer stored as string.
        /// - "checksum" (optional) - SHA1 digest of file specified in under 'location' key.
        @f(13) List<StringStringEntryProto> externalData,

        /// If value not set, data is stored in raw_data (if set) otherwise in type-specified field.
        @f(14) DataLocation dataLocation,

        /// For double
        /// Complex128 tensors are encoded as a single array of doubles,
        /// with the real components appearing in odd numbered positions,
        /// and the corresponding imaginary component appearing in the
        /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
        /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
        /// When this field is present, the data_type field MUST be DOUBLE or COMPLEX128
        @f(10) List<double[]> doubleData,

        /// For uint64 and uint32 values
        /// When this field is present, the data_type field MUST be
        /// UINT32 or UINT64
        @f(11) List<long[]> uint64Data,

        /// Named metadata values; keys should be distinct.
        @f(16) List<StringStringEntryProto> metadataProps) implements OnnxModel {

        /// For very large tensors, we may want to store them in chunks, in which
        /// case the following fields will specify the segment that is stored in
        /// the current TensorProto.
        public record Segment (

            @f(1) Long begin,

            @f(2) Long end) implements OnnxModel {
        }
    }

    /// A serialized sparse-tensor value
    public record SparseTensorProto (

        /// The sequence of non-default values are encoded as a tensor of shape [NNZ].
        /// The default-value is zero for numeric tensors, and empty-string for string tensors.
        /// values must have a non-empty name present which serves as a name for SparseTensorProto
        /// when used in sparse_initializer list.
        @f(1) TensorProto values,

        /// The indices of the non-default values, which may be stored in one of two formats.
        /// (a) Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value
        /// corresponding to the j-th index of the i-th value (in the values tensor).
        /// (b) Indices can be a tensor of shape [NNZ], in which case the i-th value
        /// must be the linearized-index of the i-th value (in the values tensor).
        /// The linearized-index can be converted into an index tuple (k_1,...,k_rank)
        /// using the shape provided below.
        /// The indices must appear in ascending order without duplication.
        /// In the first format, the ordering is lexicographic-ordering:
        /// e.g., index-value [1,4] must appear before [2,1]
        @f(2) TensorProto indices,

        /// The shape of the underlying dense-tensor: [dim_1, dim_2, ... dim_rank]
        @f(3) List<long[]> dims) implements OnnxModel {
    }

    /// Defines a tensor shape. A dimension can be either an integer value
    /// or a symbolic variable. A symbolic variable represents an unknown
    /// dimension.
    public record TensorShapeProto (

        @f(1) List<Dimension> dim) implements OnnxModel {

        public record Dimension (

            @f(1) Long dimValue,

            /// namespace Shape
            @f(2) String dimParam,

            /// Standard denotation can optionally be used to denote tensor
            /// dimensions with standard semantic descriptions to ensure
            /// that operations are applied to the correct axis of a tensor.
            /// Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
            /// for pre-defined dimension denotations.
            @f(3) String denotation) implements OnnxModel {
        }
    }

    /// Types
    ///
    /// The standard ONNX data types.
    public record TypeProto (

        /// The type of a tensor.
        @f(1) Tensor tensorType,

        /// The type of a sequence.
        @f(4) Sequence sequenceType,

        /// The type of a map.
        @f(5) Map mapType,

        /// The type of an optional.
        @f(9) Optional optionalType,

        /// Type of the sparse tensor
        @f(8) SparseTensor sparseTensorType,

        @f(7) Opaque opaqueType,

        /// An optional denotation can be used to denote the whole
        /// type with a standard semantic description as to what is
        /// stored inside. Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition
        /// for pre-defined type denotations.
        @f(6) String denotation) implements OnnxModel {

        public record Tensor (

            /// This field MUST NOT have the value of UNDEFINED
            /// This field MUST have a valid TensorProto.DataType value
            /// This field MUST be present for this version of the IR.
            @f(1) Integer elemType,

            @f(2) TensorShapeProto shape) implements OnnxModel {
        }

        /// repeated T
        public record Sequence (

            /// The type and optional shape of each element of the sequence.
            /// This field MUST be present for this version of the IR.
            @f(1) TypeProto elemType) implements OnnxModel {
        }

        /// map<K,V>
        public record Map (

            /// This field MUST have a valid TensorProto.DataType value
            /// This field MUST be present for this version of the IR.
            /// This field MUST refer to an integral type ([U]INT{8|16|32|64}) or STRING
            @f(1) Integer keyType,

            /// This field MUST be present for this version of the IR.
            @f(2) TypeProto valueType) implements OnnxModel {
        }

        /// wrapper for Tensor, Sequence, or Map
        public record Optional (

            /// The type and optional shape of the element wrapped.
            /// This field MUST be present for this version of the IR.
            /// Possible values correspond to OptionalProto.DataType enum
            @f(1) TypeProto elemType) implements OnnxModel {
        }

        public record SparseTensor (

            /// This field MUST NOT have the value of UNDEFINED
            /// This field MUST have a valid TensorProto.DataType value
            /// This field MUST be present for this version of the IR.
            @f(1) Integer elemType,

            @f(2) TensorShapeProto shape) implements OnnxModel {
        }

        public record Opaque (

            /// When missing, the domain is the same as the model's.
            @f(1) String domain,

            /// The name is optional but significant when provided.
            @f(2) String name) implements OnnxModel {
        }
    }

    /// Operator Sets
    ///
    /// OperatorSets are uniquely identified by a (domain, opset_version) pair.
    public record OperatorSetIdProto (

        /// The domain of the operator set being identified.
        /// The empty string ("") or absence of this field implies the operator
        /// set that is defined as part of the ONNX specification.
        /// This field MUST be present in this version of the IR when referring to any other operator set.
        @f(1) String domain,

        /// The version of the operator set being identified.
        /// This field MUST be present in this version of the IR.
        @f(2) Long version) implements OnnxModel {
    }

    public record FunctionProto (

        /// The name of the function, similar to op_type in NodeProto.
        /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
        @f(1) String name,

        /// The inputs and outputs of the function.
        @f(4) List<String> input,

        @f(5) List<String> output,

        /// The attribute parameters of the function.
        /// It is for function parameters without default values.
        @f(6) List<String> attribute,

        /// The attribute protos of the function.
        /// It is for function attributes with default values.
        /// A function attribute shall be represented either as
        /// a string attribute or an AttributeProto, not both.
        @f(11) List<AttributeProto> attributeProto,

        /// The nodes in the function.
        @f(7) List<NodeProto> node,

        /// A human-readable documentation for this function. Markdown is allowed.
        @f(8) String docString,

        @f(9) List<OperatorSetIdProto> opsetImport,

        /// The domain which this function belongs to.
        /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
        @f(10) String domain,

        /// The overload identifier of the function.
        /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
        @f(13) String overload,

        /// Information for the values in the function. The ValueInfoProto.name's
        /// must be distinct and refer to names in the function (including inputs,
        /// outputs, and intermediate values). It is optional for a value to appear
        /// in value_info list.
        @f(12) List<ValueInfoProto> valueInfo,

        /// Named metadata values; keys should be distinct.
        @f(14) List<StringStringEntryProto> metadataProps) implements OnnxModel {
    }

    // Implementation


    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.RECORD_COMPONENT)
    @interface f {
        int value();
    }

    private static long decodeVarint(ByteBuffer data) {
        long i, shift = 0, value = 0;
        do {
            value |= ((i = data.get()) & 0x7f) << shift;
            shift += 7;
        } while ((i & 0x80) != 0);
        return value;
    }

    private static int countVarInts(ByteBuffer data) {
        long end  = decodeVarint(data);
        int start = data.position();
        end += start;
        int count = 0;
        while (data.position() < end) {
            if ((data.get() & 0x80) == 0) count++;
        }
        data.position(start);
        return count;
    }

    private static int[] readPackedInts(ByteBuffer data) {
        var ret = new int[countVarInts(data)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (int)decodeVarint(data);
        }
        return ret;
    }

    private static long[] readPackedLongs(ByteBuffer data) {
        var ret = new long[countVarInts(data)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = decodeVarint(data);
        }
        return ret;
    }

    private static float[] readPackedFloats(ByteBuffer data) {
        var ret = new float[(int)(decodeVarint(data)/4)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data.getFloat();
        }
        return ret;
    }

    private static double[] readPackedDoubles(ByteBuffer data) {
        var ret = new double[(int)(decodeVarint(data)/8)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data.getDouble();
        }
        return ret;
    }

    private static byte[] readBytes(ByteBuffer data) {
        var bytes = new byte[(int)decodeVarint(data)];
        data.get(bytes);
        return bytes;
    }

    private static Object readData(Class<?> baseType, boolean packed, ByteBuffer bb) {
        if (baseType == Integer.class) {
            return (int)decodeVarint(bb);
        } else if (baseType == int[].class) {
            return packed ? readPackedInts(bb) : new int[]{(int)decodeVarint(bb)};
        } else if (baseType == Long.class) {
            return decodeVarint(bb);
        } else if (baseType == long[].class) {
            return packed ? readPackedLongs(bb) : new long[]{decodeVarint(bb)};
        } else if (baseType == Float.class) {
            return bb.getFloat();
        } else if (baseType == float[].class) {
            return packed ? readPackedFloats(bb) : new float[] {bb.getFloat()};
        } else if (baseType == Double.class) {
            return bb.getDouble();
        } else if (baseType == double[].class) {
            return packed ? readPackedDoubles(bb) : new double[] {bb.getDouble()};
        } else if (baseType == byte[].class) {
            return readBytes(bb);
        } else if (baseType == String.class) {
            return new String(readBytes(bb));
        } else if (baseType.getEnclosingClass() == OnnxConstants.class) {
            int value = (int)decodeVarint(bb);
            for (Object cs : baseType.getEnumConstants()) {
                if (cs instanceof IntSupplier is && is.getAsInt() == value) {
                    return cs;
                }
            }
            throw new IllegalArgumentException(baseType.toString());
        } else {
            var size = decodeVarint(bb);
            int limit = bb.limit();
            var data = readFrom((Class<Record>)baseType, bb.limit(bb.position() + (int)size));
            bb.limit(limit);
            return data;
        }
    }

    private static int getRecordFieldIndex(RecordComponent[] rcs, int fieldIndex) {
        for (int i = 0; i < rcs.length; i++) {
            if (rcs[i].getAnnotation(f.class).value() == fieldIndex) {
                return i;
            }
        }
        throw new IllegalArgumentException("Field index " + fieldIndex + " not found in " + rcs[0].getDeclaringRecord());
    }

    private static <T> T readFrom(Class<T> type, ByteBuffer bb) {
        Object[] fieldsData = new Object[type.getRecordComponents().length];
        while (bb.remaining() > 0) {
            long tag = decodeVarint(bb);
            RecordComponent[] rcs = type.getRecordComponents();
            int rfi = getRecordFieldIndex(rcs, (int)tag >> 3);
            boolean packed = (tag & 7) == 2;
            RecordComponent rc = rcs[rfi];
            Class<?> rcType = rc.getType();
            if (rcType == List.class) {
                List list;
                if (fieldsData[rfi] instanceof List l) {
                    list = l;
                } else {
                    list = new ArrayList();
                    fieldsData[rfi] = list;
                }
                Class baseType = (Class)((ParameterizedType)rc.getGenericType()).getActualTypeArguments()[0];
                list.add(readData(baseType, packed, bb));
            } else {
                fieldsData[rfi] = readData(rcType, packed, bb);
            }
        }
        try {
            return (T)type.getDeclaredConstructors()[0].newInstance(fieldsData);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static void print(StringBuilder out, int indent, String name, Object value, boolean skipBigData) throws ReflectiveOperationException {
        if (value == null) return;
        out.append("  ".repeat(indent)).append(name);
        switch (value) {
            case List l -> {
                out.append(name.endsWith("s") ? ":" : "s:").append(System.lineSeparator());
                for (var el : l) print(out, indent + 1, "- " + (name.endsWith("s") ? name.substring(0, name.length() - 1) : name), el, skipBigData);
            }
            case Record r -> {
                out.append(':').append(System.lineSeparator());
                for (var rc : r.getClass().getRecordComponents()) {
                    print(out, indent + 2, rc.getName(), rc.getAccessor().invoke(r), skipBigData);
                }
            }
            case byte[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case long[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case float[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case double[] a ->
                out.append(checkSize(a.length, () -> Arrays.toString(a), skipBigData));
            case String s ->
                out.append(": \"").append(s).append('"').append(System.lineSeparator());
            default ->
                out.append(": ").append(value).append(System.lineSeparator());
        }
    }

    static final int SKIP_LIMIT = 1000;

    private static String checkSize(int size, Supplier<String> sup, boolean skipBigData) {
        return ": " + (skipBigData && size > SKIP_LIMIT ? "# skipped " + size + " values" : sup.get()) + System.lineSeparator();
    }

    default String toText() {
        return toText(true);
    }

    default String toText(boolean skipBigData) {
        try {
            var sb = new StringBuilder();
            print(sb, 0, "OnnxModel", this, skipBigData);
            return sb.toString();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static OnnxModel.ModelProto readFrom(byte[] onnxProtoModel) {
        return readFrom(ByteBuffer.wrap(onnxProtoModel));
    }

    public static OnnxModel.ModelProto readFrom(ByteBuffer onnxProtoModel) {
        return readFrom(OnnxModel.ModelProto.class, onnxProtoModel.order(ByteOrder.LITTLE_ENDIAN));
    }

    public static void main(String... args) throws Exception {
        for (var fName : args) {
            try (var in = new RandomAccessFile(fName, "r")) {
                OnnxModel.ModelProto model = readFrom(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()));
                System.out.println(model.toText());
            }
        }
    }
}
