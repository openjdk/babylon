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

import java.util.function.IntSupplier;

// Generated from onnx.in.proto
public final class OnnxConstants {

    /// Versioning
    ///
    /// ONNX versioning is specified in docs/IR.md and elaborated on in docs/Versioning.md
    ///
    /// To be compatible with both proto2 and proto3, we will use a version number
    /// that is not defined by the default value but an explicit enum number.
    public enum Version implements IntSupplier {

        /// proto3 requires the first enum value to be zero.
        /// We add this just to appease the compiler.
        _START_VERSION(0),

        /// The version field is always serialized and we will use it to store the
        /// version that the  graph is generated from. This helps us set up version
        /// control.
        /// For the IR, we are using simple numbers starting with 0x00000001,
        /// which was the version we published on Oct 10, 2017.
        IR_VERSION_2017_10_10(0x0000000000000001),

        /// IR_VERSION 2 published on Oct 30, 2017
        /// - Added type discriminator to AttributeProto to support proto3 users
        IR_VERSION_2017_10_30(0x0000000000000002),

        /// IR VERSION 3 published on Nov 3, 2017
        /// - For operator versioning:
        ///    - Added new message OperatorSetIdProto
        ///    - Added opset_import in ModelProto
        /// - For vendor extensions, added domain in NodeProto
        IR_VERSION_2017_11_3(0x0000000000000003),

        /// IR VERSION 4 published on Jan 22, 2019
        /// - Relax constraint that initializers should be a subset of graph inputs
        /// - Add type BFLOAT16
        IR_VERSION_2019_1_22(0x0000000000000004),

        /// IR VERSION 5 published on March 18, 2019
        /// - Add message TensorAnnotation.
        /// - Add quantization annotation in GraphProto to map tensor with its scale and zero point quantization parameters.
        IR_VERSION_2019_3_18(0x0000000000000005),

        /// IR VERSION 6 published on Sep 19, 2019
        /// - Add support for sparse tensor constants stored in model.
        ///   - Add message SparseTensorProto
        ///   - Add sparse initializers
        IR_VERSION_2019_9_19(0x0000000000000006),

        /// IR VERSION 7 published on May 8, 2020
        /// - Add support to allow function body graph to rely on multiple external operator sets.
        /// - Add a list to promote inference graph's initializers to global and
        ///   mutable variables. Global variables are visible in all graphs of the
        ///   stored models.
        /// - Add message TrainingInfoProto to store initialization
        ///   method and training algorithm. The execution of TrainingInfoProto
        ///   can modify the values of mutable variables.
        /// - Implicitly add inference graph into each TrainingInfoProto's algorithm.
        IR_VERSION_2020_5_8(0x0000000000000007),

        /// IR VERSION 8 published on July 30, 2021
        /// Introduce TypeProto.SparseTensor
        /// Introduce TypeProto.Optional
        /// Added a list of FunctionProtos local to the model
        /// Deprecated since_version and operator status from FunctionProto
        IR_VERSION_2021_7_30(0x0000000000000008),

        /// IR VERSION 9 published on May 5, 2023
        /// Added AttributeProto to FunctionProto so that default attribute values can be set.
        /// Added FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ.
        IR_VERSION_2023_5_5(0x0000000000000009),

        /// IR VERSION 10 published on March 25, 2024
        /// Added UINT4, INT4, overload field for functions and metadata_props on multiple proto definitions.
        IR_VERSION_2024_3_25(0x000000000000000A),

        /// IR VERSION 11 published on May 12, 2025
        /// Added FLOAT4E2M1, multi-device protobuf classes.
        IR_VERSION_2025_05_12(0x000000000000000B),

        /// IR VERSION 12 published on TBD
        /// Added FLOAT8E8M0.
        IR_VERSION(0x000000000000000C),
        ;

        final int value;

        Version(int value) {
            this.value = value;
        }

        @Override
        public int getAsInt() {
            return value;
        }
    }

    /// Note: this enum is structurally identical to the OpSchema::AttrType
    /// enum defined in schema.h.  If you rev one, you likely need to rev the other.
    public enum AttributeType implements IntSupplier {

        UNDEFINED(0),

        FLOAT(1),

        INT(2),

        STRING(3),

        TENSOR(4),

        GRAPH(5),

        SPARSE_TENSOR(11),

        TYPE_PROTO(13),

        FLOATS(6),

        INTS(7),

        STRINGS(8),

        TENSORS(9),

        GRAPHS(10),

        SPARSE_TENSORS(12),

        TYPE_PROTOS(14),
        ;

        final int value;

        AttributeType(int value) {
            this.value = value;
        }

        @Override
        public int getAsInt() {
            return value;
        }
    }

    public enum DataType implements IntSupplier {

        UNDEFINED(0),

        /// Basic types.
        /// float
        FLOAT(1),

        /// uint8_t
        UINT8(2),

        /// int8_t
        INT8(3),

        /// uint16_t
        UINT16(4),

        /// int16_t
        INT16(5),

        /// int32_t
        INT32(6),

        /// int64_t
        INT64(7),

        /// string
        STRING(8),

        /// bool
        BOOL(9),

        /// IEEE754 half-precision floating-point format (16 bits wide).
        /// This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
        FLOAT16(10),

        DOUBLE(11),

        UINT32(12),

        UINT64(13),

        /// complex with float32 real and imaginary components
        COMPLEX64(14),

        /// complex with float64 real and imaginary components
        COMPLEX128(15),

        /// Non-IEEE floating-point format based on IEEE754 single-precision
        /// floating-point number truncated to 16 bits.
        /// This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
        BFLOAT16(16),

        /// Non-IEEE floating-point format based on papers
        /// FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
        /// 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
        /// Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
        /// The computation usually happens inside a block quantize / dequantize
        /// fused by the runtime.
        /// float 8, mostly used for coefficients, supports nan, not inf
        FLOAT8E4M3FN(17),

        /// float 8, mostly used for coefficients, supports nan, not inf, no negative zero
        FLOAT8E4M3FNUZ(18),

        /// follows IEEE 754, supports nan, inf, mostly used for gradients
        FLOAT8E5M2(19),

        /// follows IEEE 754, supports nan, not inf, mostly used for gradients, no negative zero
        FLOAT8E5M2FNUZ(20),

        /// 4-bit integer data types
        /// Unsigned integer in range [0, 15]
        UINT4(21),

        /// Signed integer in range [-8, 7], using two's-complement representation
        INT4(22),

        /// 4-bit floating point data types
        FLOAT4E2M1(23),

        /// E8M0 type used as the scale for microscaling (MX) formats:
        /// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
        FLOAT8E8M0(24),
        ;

        final int value;

        DataType(int value) {
            this.value = value;
        }

        @Override
        public int getAsInt() {
            return value;
        }
    }

    /// Location of the data for this tensor. MUST be one of:
    /// - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
    /// - EXTERNAL - data stored in an external location as described by external_data field.
    public enum DataLocation implements IntSupplier {

        DEFAULT(0),

        EXTERNAL(1),
        ;

        final int value;

        DataLocation(int value) {
            this.value = value;
        }

        @Override
        public int getAsInt() {
            return value;
        }
    }

    /// Operator/function status.
    public enum OperatorStatus implements IntSupplier {

        EXPERIMENTAL(0),

        STABLE(1),
        ;

        final int value;

        OperatorStatus(int value) {
            this.value = value;
        }

        @Override
        public int getAsInt() {
            return value;
        }
    }
}
