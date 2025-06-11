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

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.function.BiConsumer;
import java.util.function.IntSupplier;

import oracle.code.onnx.proto.OnnxConstants.*;

// Generated from onnx.in.proto
public sealed class OnnxBuilder<T extends OnnxBuilder> {

    /// Attributes
    ///
    /// A named attribute containing either singular float, integer, string, graph,
    /// and tensor values, or repeated float, integer, string, graph, and tensor values.
    /// An AttributeProto MUST contain the name field, and *only one* of the
    /// following content fields, effectively enforcing a C/C++ union equivalent.
    public static final class AttributeProto extends OnnxBuilder<AttributeProto> {

        /// The name field MUST be present for this version of the IR.
        /// namespace Attribute
        public AttributeProto name(String name) {return _f(1, name);}

        /// if ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
        /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
        /// in parent scope.
        /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
        public AttributeProto refAttrName(String refAttrName) {return _f(21, refAttrName);}

        /// A human-readable documentation for this attribute. Markdown is allowed.
        public AttributeProto docString(String docString) {return _f(13, docString);}

        /// The type field MUST be present for this version of the IR.
        /// For 0.0.1 versions of the IR, this field was not defined, and
        /// implementations needed to use has_field heuristics to determine
        /// which value field was in use.  For IR_VERSION 0.0.2 or later, this
        /// field MUST be set and match the f|i|s|t|... field in use.  This
        /// change was made to accommodate proto3 implementations.
        /// discriminator that indicates which field below is in use
        public AttributeProto type(AttributeType type) {return _f(20, type);}

        /// Exactly ONE of the following fields must be present for this version of the IR
        /// float
        public AttributeProto f(float f) {return _f(2, f);}

        /// int
        public AttributeProto i(long i) {return _f(3, i);}

        /// UTF-8 string
        public AttributeProto s(byte[] s) {return _f(4, s);}

        /// tensor value
        public AttributeProto t(TensorProto t) {return _f(5, t);}

        /// graph
        public AttributeProto g(GraphProto g) {return _f(6, g);}

        /// sparse tensor value
        public AttributeProto sparseTensor(SparseTensorProto sparseTensor) {return _f(22, sparseTensor);}

        /// Do not use field below, it's deprecated.
        /// optional ValueProto v = 12;         // value - subsumes everything but graph
        /// type proto
        public AttributeProto tp(TypeProto tp) {return _f(14, tp);}

        /// list of floats
        public AttributeProto floats(float... floats) {return _f(7, floats);}

        /// list of ints
        public AttributeProto ints(long... ints) {return _f(8, ints);}

        /// list of UTF-8 strings
        public AttributeProto strings(byte[] strings) {return _f(9, strings);}

        /// list of tensors
        public AttributeProto tensors(TensorProto tensors) {return _f(10, tensors);}

        /// list of graph
        public AttributeProto graphs(GraphProto graphs) {return _f(11, graphs);}

        /// list of sparse tensors
        public AttributeProto sparseTensors(SparseTensorProto sparseTensors) {return _f(23, sparseTensors);}

        /// list of type protos
        public AttributeProto typeProtos(TypeProto typeProtos) {return _f(15, typeProtos);}
    }

    /// Defines information on value, including the name, the type, and
    /// the shape of the value.
    public static final class ValueInfoProto extends OnnxBuilder<ValueInfoProto> {

        /// This field MUST be present in this version of the IR.
        /// namespace Value
        public ValueInfoProto name(String name) {return _f(1, name);}

        /// This field MUST be present in this version of the IR for
        /// inputs and outputs of the top-level graph.
        public ValueInfoProto type(TypeProto type) {return _f(2, type);}

        /// A human-readable documentation for this value. Markdown is allowed.
        public ValueInfoProto docString(String docString) {return _f(3, docString);}

        /// Named metadata values; keys should be distinct.
        public ValueInfoProto metadataProps(StringStringEntryProto metadataProps) {return _f(4, metadataProps);}
    }

    /// Nodes
    ///
    /// Computation graphs are made up of a DAG of nodes, which represent what is
    /// commonly called a "layer" or "pipeline stage" in machine learning frameworks.
    ///
    /// For example, it can be a node of type "Conv" that takes in an image, a filter
    /// tensor and a bias tensor, and produces the convolved output.
    public static final class NodeProto extends OnnxBuilder<NodeProto> {

        /// namespace Value
        public NodeProto input(String input) {return _f(1, input);}

        /// namespace Value
        public NodeProto output(String output) {return _f(2, output);}

        /// An optional identifier for this node in a graph.
        /// This field MAY be absent in this version of the IR.
        /// namespace Node
        public NodeProto name(String name) {return _f(3, name);}

        /// The symbolic identifier of the Operator to execute.
        /// namespace Operator
        public NodeProto opType(String opType) {return _f(4, opType);}

        /// The domain of the OperatorSet that specifies the operator named by op_type.
        /// namespace Domain
        public NodeProto domain(String domain) {return _f(7, domain);}

        /// Overload identifier, used only to map this to a model-local function.
        public NodeProto overload(String overload) {return _f(8, overload);}

        /// Additional named attributes.
        public NodeProto attribute(AttributeProto attribute) {return _f(5, attribute);}

        /// A human-readable documentation for this node. Markdown is allowed.
        public NodeProto docString(String docString) {return _f(6, docString);}

        /// Named metadata values; keys should be distinct.
        public NodeProto metadataProps(StringStringEntryProto metadataProps) {return _f(9, metadataProps);}
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
    public static final class TrainingInfoProto extends OnnxBuilder<TrainingInfoProto> {

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
        public TrainingInfoProto initialization(GraphProto initialization) {return _f(1, initialization);}

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
        public TrainingInfoProto algorithm(GraphProto algorithm) {return _f(2, algorithm);}

        /// This field specifies the bindings from the outputs of "initialization" to
        /// some initializers in "ModelProto.graph.initializer" and
        /// the "algorithm.initializer" in the same TrainingInfoProto.
        /// See "update_binding" below for details.
        ///
        /// By default, this field is empty and no initializer would be changed
        /// by the execution of "initialization".
        public TrainingInfoProto initializationBinding(StringStringEntryProto initializationBinding) {return _f(3, initializationBinding);}

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
        public TrainingInfoProto updateBinding(StringStringEntryProto updateBinding) {return _f(4, updateBinding);}
    }

    /// Models
    ///
    /// ModelProto is a top-level file/container format for bundling a ML model and
    /// associating its computation graph with metadata.
    ///
    /// The semantics of the model are described by the associated GraphProto's.
    public static final class ModelProto extends OnnxBuilder<ModelProto> {

        /// The version of the IR this model targets. See Version enum above.
        /// This field MUST be present.
        public ModelProto irVersion(long irVersion) {return _f(1, irVersion);}

        /// The OperatorSets this model relies on.
        /// All ModelProtos MUST have at least one entry that
        /// specifies which version of the ONNX OperatorSet is
        /// being imported.
        ///
        /// All nodes in the ModelProto's graph will bind against the operator
        /// with the same-domain/same-op_type operator with the HIGHEST version
        /// in the referenced operator sets.
        public ModelProto opsetImport(OperatorSetIdProto opsetImport) {return _f(8, opsetImport);}

        /// The name of the framework or tool used to generate this model.
        /// This field SHOULD be present to indicate which implementation/tool/framework
        /// emitted the model.
        public ModelProto producerName(String producerName) {return _f(2, producerName);}

        /// The version of the framework or tool used to generate this model.
        /// This field SHOULD be present to indicate which implementation/tool/framework
        /// emitted the model.
        public ModelProto producerVersion(String producerVersion) {return _f(3, producerVersion);}

        /// Domain name of the model.
        /// We use reverse domain names as name space indicators. For example:
        /// `com.facebook.fair` or `com.microsoft.cognitiveservices`
        ///
        /// Together with `model_version` and GraphProto.name, this forms the unique identity of
        /// the graph.
        public ModelProto domain(String domain) {return _f(4, domain);}

        /// The version of the graph encoded. See Version enum below.
        public ModelProto modelVersion(long modelVersion) {return _f(5, modelVersion);}

        /// A human-readable documentation for this model. Markdown is allowed.
        public ModelProto docString(String docString) {return _f(6, docString);}

        /// The parameterized graph that is evaluated to execute the model.
        public ModelProto graph(GraphProto graph) {return _f(7, graph);}

        /// Named metadata values; keys should be distinct.
        public ModelProto metadataProps(StringStringEntryProto metadataProps) {return _f(14, metadataProps);}

        /// Training-specific information. Sequentially executing all stored
        /// `TrainingInfoProto.algorithm`s and assigning their outputs following
        /// the corresponding `TrainingInfoProto.update_binding`s is one training
        /// iteration. Similarly, to initialize the model
        /// (as if training hasn't happened), the user should sequentially execute
        /// all stored `TrainingInfoProto.initialization`s and assigns their outputs
        /// using `TrainingInfoProto.initialization_binding`s.
        ///
        /// If this field is empty, the training behavior of the model is undefined.
        public ModelProto trainingInfo(TrainingInfoProto trainingInfo) {return _f(20, trainingInfo);}

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
        public ModelProto functions(FunctionProto functions) {return _f(25, functions);}
    }

    /// StringStringEntryProto follows the pattern for cross-proto-version maps.
    /// See https://developers.google.com/protocol-buffers/docs/proto3#maps
    public static final class StringStringEntryProto extends OnnxBuilder<StringStringEntryProto> {

        public StringStringEntryProto key(String key) {return _f(1, key);}

        public StringStringEntryProto value(String value) {return _f(2, value);}
    }

    public static final class TensorAnnotation extends OnnxBuilder<TensorAnnotation> {

        public TensorAnnotation tensorName(String tensorName) {return _f(1, tensorName);}

        /// <key, value> pairs to annotate tensor specified by <tensor_name> above.
        /// The keys used in the mapping below must be pre-defined in ONNX spec.
        /// For example, for 8-bit linear quantization case, 'SCALE_TENSOR', 'ZERO_POINT_TENSOR' will be pre-defined as
        /// quantization parameter keys.
        public TensorAnnotation quantParameterTensorNames(StringStringEntryProto quantParameterTensorNames) {return _f(2, quantParameterTensorNames);}
    }

    /// Graphs
    ///
    /// A graph defines the computational logic of a model and is comprised of a parameterized
    /// list of nodes that form a directed acyclic graph based on their inputs and outputs.
    /// This is the equivalent of the "network" or "graph" in many deep learning
    /// frameworks.
    public static final class GraphProto extends OnnxBuilder<GraphProto> {

        /// The nodes in the graph, sorted topologically.
        public GraphProto node(NodeProto node) {return _f(1, node);}

        /// The name of the graph.
        /// namespace Graph
        public GraphProto name(String name) {return _f(2, name);}

        /// A list of named tensor values, used to specify constant inputs of the graph.
        /// Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
        /// The name MUST be unique across both initializer and sparse_initializer,
        /// but the name MAY also appear in the input list.
        public GraphProto initializer(TensorProto initializer) {return _f(5, initializer);}

        /// Initializers (see above) stored in sparse format.
        public GraphProto sparseInitializer(SparseTensorProto sparseInitializer) {return _f(15, sparseInitializer);}

        /// A human-readable documentation for this graph. Markdown is allowed.
        public GraphProto docString(String docString) {return _f(10, docString);}

        /// The inputs and outputs of the graph.
        public GraphProto input(ValueInfoProto input) {return _f(11, input);}

        public GraphProto output(ValueInfoProto output) {return _f(12, output);}

        /// Information for the values in the graph. The ValueInfoProto.name's
        /// must be distinct. It is optional for a value to appear in value_info list.
        public GraphProto valueInfo(ValueInfoProto valueInfo) {return _f(13, valueInfo);}

        /// This field carries information to indicate the mapping among a tensor and its
        /// quantization parameter tensors. For example:
        /// For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
        /// which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
        public GraphProto quantizationAnnotation(TensorAnnotation quantizationAnnotation) {return _f(14, quantizationAnnotation);}

        /// Named metadata values; keys should be distinct.
        public GraphProto metadataProps(StringStringEntryProto metadataProps) {return _f(16, metadataProps);}
    }

    /// Tensors
    ///
    /// A serialized tensor value.
    public static final class TensorProto extends OnnxBuilder<TensorProto> {

        /// The shape of the tensor.
        public TensorProto dims(long... dims) {return _f(1, dims);}

        /// The data type of the tensor.
        /// This field MUST have a valid TensorProto.DataType value
        public TensorProto dataType(int dataType) {return _f(2, dataType);}

        /// For very large tensors, we may want to store them in chunks, in which
        /// case the following fields will specify the segment that is stored in
        /// the current TensorProto.
        public static final class Segment extends OnnxBuilder<Segment> {

            public Segment begin(long begin) {return _f(1, begin);}

            public Segment end(long end) {return _f(2, end);}
        }

        public TensorProto segment(Segment segment) {return _f(3, segment);}

        /// For float and complex64 values
        /// Complex64 tensors are encoded as a single array of floats,
        /// with the real components appearing in odd numbered positions,
        /// and the corresponding imaginary component appearing in the
        /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
        /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
        /// When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
        public TensorProto floatData(float... floatData) {return _f(4, floatData);}

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
        public TensorProto int32Data(int... int32Data) {return _f(5, int32Data);}

        /// For strings.
        /// Each element of string_data is a UTF-8 encoded Unicode
        /// string. No trailing null, no leading BOM. The protobuf "string"
        /// scalar type is not used to match ML community conventions.
        /// When this field is present, the data_type field MUST be STRING
        public TensorProto stringData(byte[] stringData) {return _f(6, stringData);}

        /// For int64.
        /// When this field is present, the data_type field MUST be INT64
        public TensorProto int64Data(long... int64Data) {return _f(7, int64Data);}

        /// Optionally, a name for the tensor.
        /// namespace Value
        public TensorProto name(String name) {return _f(8, name);}

        /// A human-readable documentation for this tensor. Markdown is allowed.
        public TensorProto docString(String docString) {return _f(12, docString);}

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
        public TensorProto rawData(byte[] rawData) {return _f(9, rawData);}

        /// Data can be stored inside the protobuf file using type-specific fields or raw_data.
        /// Alternatively, raw bytes data can be stored in an external file, using the external_data field.
        /// external_data stores key-value pairs describing data location. Recognized keys are:
        /// - "location" (required) - POSIX filesystem path relative to the directory where the ONNX
        ///                           protobuf model was stored
        /// - "offset" (optional) - position of byte at which stored data begins. Integer stored as string.
        ///                         Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
        /// - "length" (optional) - number of bytes containing data. Integer stored as string.
        /// - "checksum" (optional) - SHA1 digest of file specified in under 'location' key.
        public TensorProto externalData(StringStringEntryProto externalData) {return _f(13, externalData);}

        /// If value not set, data is stored in raw_data (if set) otherwise in type-specified field.
        public TensorProto dataLocation(DataLocation dataLocation) {return _f(14, dataLocation);}

        /// For double
        /// Complex128 tensors are encoded as a single array of doubles,
        /// with the real components appearing in odd numbered positions,
        /// and the corresponding imaginary component appearing in the
        /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
        /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
        /// When this field is present, the data_type field MUST be DOUBLE or COMPLEX128
        public TensorProto doubleData(double... doubleData) {return _f(10, doubleData);}

        /// For uint64 and uint32 values
        /// When this field is present, the data_type field MUST be
        /// UINT32 or UINT64
        public TensorProto uint64Data(long... uint64Data) {return _f(11, uint64Data);}

        /// Named metadata values; keys should be distinct.
        public TensorProto metadataProps(StringStringEntryProto metadataProps) {return _f(16, metadataProps);}
    }

    /// A serialized sparse-tensor value
    public static final class SparseTensorProto extends OnnxBuilder<SparseTensorProto> {

        /// The sequence of non-default values are encoded as a tensor of shape \[NNZ].
        /// The default-value is zero for numeric tensors, and empty-string for string tensors.
        /// values must have a non-empty name present which serves as a name for SparseTensorProto
        /// when used in sparse_initializer list.
        public SparseTensorProto values(TensorProto values) {return _f(1, values);}

        /// The indices of the non-default values, which may be stored in one of two formats.
        /// (a) Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value
        /// corresponding to the j-th index of the i-th value (in the values tensor).
        /// (b) Indices can be a tensor of shape \[NNZ], in which case the i-th value
        /// must be the linearized-index of the i-th value (in the values tensor).
        /// The linearized-index can be converted into an index tuple (k_1,...,k_rank)
        /// using the shape provided below.
        /// The indices must appear in ascending order without duplication.
        /// In the first format, the ordering is lexicographic-ordering:
        /// e.g., index-value [1,4] must appear before [2,1]
        public SparseTensorProto indices(TensorProto indices) {return _f(2, indices);}

        /// The shape of the underlying dense-tensor: [dim_1, dim_2, ... dim_rank]
        public SparseTensorProto dims(long... dims) {return _f(3, dims);}
    }

    /// Defines a tensor shape. A dimension can be either an integer value
    /// or a symbolic variable. A symbolic variable represents an unknown
    /// dimension.
    public static final class TensorShapeProto extends OnnxBuilder<TensorShapeProto> {

        public static final class Dimension extends OnnxBuilder<Dimension> {

            public Dimension dimValue(long dimValue) {return _f(1, dimValue);}

            /// namespace Shape
            public Dimension dimParam(String dimParam) {return _f(2, dimParam);}

            /// Standard denotation can optionally be used to denote tensor
            /// dimensions with standard semantic descriptions to ensure
            /// that operations are applied to the correct axis of a tensor.
            /// Refer to https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
            /// for pre-defined dimension denotations.
            public Dimension denotation(String denotation) {return _f(3, denotation);}
        }

        public TensorShapeProto dim(Dimension dim) {return _f(1, dim);}
    }

    /// Types
    ///
    /// The standard ONNX data types.
    public static final class TypeProto extends OnnxBuilder<TypeProto> {

        public static final class Tensor extends OnnxBuilder<Tensor> {

            /// This field MUST NOT have the value of UNDEFINED
            /// This field MUST have a valid TensorProto.DataType value
            /// This field MUST be present for this version of the IR.
            public Tensor elemType(int elemType) {return _f(1, elemType);}

            public Tensor shape(TensorShapeProto shape) {return _f(2, shape);}
        }

        /// repeated T
        public static final class Sequence extends OnnxBuilder<Sequence> {

            /// The type and optional shape of each element of the sequence.
            /// This field MUST be present for this version of the IR.
            public Sequence elemType(TypeProto elemType) {return _f(1, elemType);}
        }

        /// map<K,V>
        public static final class Map extends OnnxBuilder<Map> {

            /// This field MUST have a valid TensorProto.DataType value
            /// This field MUST be present for this version of the IR.
            /// This field MUST refer to an integral type (\[U]INT{8|16|32|64}) or STRING
            public Map keyType(int keyType) {return _f(1, keyType);}

            /// This field MUST be present for this version of the IR.
            public Map valueType(TypeProto valueType) {return _f(2, valueType);}
        }

        /// wrapper for Tensor, Sequence, or Map
        public static final class Optional extends OnnxBuilder<Optional> {

            /// The type and optional shape of the element wrapped.
            /// This field MUST be present for this version of the IR.
            /// Possible values correspond to OptionalProto.DataType enum
            public Optional elemType(TypeProto elemType) {return _f(1, elemType);}
        }

        public static final class SparseTensor extends OnnxBuilder<SparseTensor> {

            /// This field MUST NOT have the value of UNDEFINED
            /// This field MUST have a valid TensorProto.DataType value
            /// This field MUST be present for this version of the IR.
            public SparseTensor elemType(int elemType) {return _f(1, elemType);}

            public SparseTensor shape(TensorShapeProto shape) {return _f(2, shape);}
        }

        public static final class Opaque extends OnnxBuilder<Opaque> {

            /// When missing, the domain is the same as the model's.
            public Opaque domain(String domain) {return _f(1, domain);}

            /// The name is optional but significant when provided.
            public Opaque name(String name) {return _f(2, name);}
        }

        /// The type of a tensor.
        public TypeProto tensorType(Tensor tensorType) {return _f(1, tensorType);}

        /// The type of a sequence.
        public TypeProto sequenceType(Sequence sequenceType) {return _f(4, sequenceType);}

        /// The type of a map.
        public TypeProto mapType(Map mapType) {return _f(5, mapType);}

        /// The type of an optional.
        public TypeProto optionalType(Optional optionalType) {return _f(9, optionalType);}

        /// Type of the sparse tensor
        public TypeProto sparseTensorType(SparseTensor sparseTensorType) {return _f(8, sparseTensorType);}

        public TypeProto opaqueType(Opaque opaqueType) {return _f(7, opaqueType);}

        /// An optional denotation can be used to denote the whole
        /// type with a standard semantic description as to what is
        /// stored inside. Refer to https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition
        /// for pre-defined type denotations.
        public TypeProto denotation(String denotation) {return _f(6, denotation);}
    }

    /// Operator Sets
    ///
    /// OperatorSets are uniquely identified by a (domain, opset_version) pair.
    public static final class OperatorSetIdProto extends OnnxBuilder<OperatorSetIdProto> {

        /// The domain of the operator set being identified.
        /// The empty string ("") or absence of this field implies the operator
        /// set that is defined as part of the ONNX specification.
        /// This field MUST be present in this version of the IR when referring to any other operator set.
        public OperatorSetIdProto domain(String domain) {return _f(1, domain);}

        /// The version of the operator set being identified.
        /// This field MUST be present in this version of the IR.
        public OperatorSetIdProto version(long version) {return _f(2, version);}
    }

    public static final class FunctionProto extends OnnxBuilder<FunctionProto> {

        /// The name of the function, similar to op_type in NodeProto.
        /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
        public FunctionProto name(String name) {return _f(1, name);}

        /// The inputs and outputs of the function.
        public FunctionProto input(String input) {return _f(4, input);}

        public FunctionProto output(String output) {return _f(5, output);}

        /// The attribute parameters of the function.
        /// It is for function parameters without default values.
        public FunctionProto attribute(String attribute) {return _f(6, attribute);}

        /// The attribute protos of the function.
        /// It is for function attributes with default values.
        /// A function attribute shall be represented either as
        /// a string attribute or an AttributeProto, not both.
        public FunctionProto attributeProto(AttributeProto attributeProto) {return _f(11, attributeProto);}

        /// The nodes in the function.
        public FunctionProto node(NodeProto node) {return _f(7, node);}

        /// A human-readable documentation for this function. Markdown is allowed.
        public FunctionProto docString(String docString) {return _f(8, docString);}

        public FunctionProto opsetImport(OperatorSetIdProto opsetImport) {return _f(9, opsetImport);}

        /// The domain which this function belongs to.
        /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
        public FunctionProto domain(String domain) {return _f(10, domain);}

        /// The overload identifier of the function.
        /// This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
        public FunctionProto overload(String overload) {return _f(13, overload);}

        /// Information for the values in the function. The ValueInfoProto.name's
        /// must be distinct and refer to names in the function (including inputs,
        /// outputs, and intermediate values). It is optional for a value to appear
        /// in value_info list.
        public FunctionProto valueInfo(ValueInfoProto valueInfo) {return _f(12, valueInfo);}

        /// Named metadata values; keys should be distinct.
        public FunctionProto metadataProps(StringStringEntryProto metadataProps) {return _f(14, metadataProps);}
    }

    // Implementation

    final ByteArrayOutputStream buf = new ByteArrayOutputStream();

    public byte[] getBytes() {
        return buf.toByteArray();
    }

    @SuppressWarnings("unchecked")
    public <P> T forEach(Iterable<P> sup, BiConsumer<T, ? super P> cons) {
        sup.forEach(p -> cons.accept((T)this, p));
        return (T)this;
    }

    void _encode(long number) {
        for (int i = 64 - Long.numberOfLeadingZeros(number); i > 7; i -= 7) {
            buf.write(0x80 | (int)number & 0x7f);
            number >>= 7;
        }
        buf.write((int)number & 0x7f);
    }

    void _encode(float value) {
        int bits =  Float.floatToRawIntBits(value);
        buf.write((byte)bits);
        buf.write((byte)(bits >> 8));
        buf.write((byte)(bits >> 16));
        buf.write((byte)(bits >> 24));
    }

    void _encode(double value) {
        long bits =  Double.doubleToRawLongBits(value);
        buf.write((byte)bits);
        buf.write((byte)(bits >> 8));
        buf.write((byte)(bits >> 16));
        buf.write((byte)(bits >> 24));
        buf.write((byte)(bits >> 32));
        buf.write((byte)(bits >> 40));
        buf.write((byte)(bits >> 48));
        buf.write((byte)(bits >> 56));
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, String value) {
        return value == null ? (T)this : _f(fieldIndex, value.getBytes(StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, byte[] bytes) {
        _encode(fieldIndex << 3 | 2);
        _encode(bytes.length);
        buf.writeBytes(bytes);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, float value) {
        _encode(fieldIndex << 3 | 5);
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, float... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, double value) {
        _encode(fieldIndex << 3 | 1);
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, double... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, long value) {
        _encode(fieldIndex << 3);
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, long... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, int... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, OnnxBuilder value) {
        return _f(fieldIndex, value.buf.toByteArray());
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, IntSupplier value) {
        return _f(fieldIndex, value.getAsInt());
    }
}
