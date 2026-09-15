/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package hat;

import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF16;
import hat.buffer.TensorF32;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.java.JavaType;

public class DType {

    private DType() {}

    // 1D
    public static final CodeType TENSOR_F32_TYPE = JavaType.type(TensorF32.class);
    public static final CodeType TENSOR_F16_TYPE = JavaType.type(TensorF16.class);

    // 2D
    public static final CodeType TENSOR_2D_F32_TYPE = JavaType.type(Tensor2DF32.class);
    public static final CodeType TENSOR_2D_F16_TYPE = JavaType.type(Tensor2DF16.class);

    // DType for primitives
    public static final CodeType Float = JavaType.FLOAT;
    public static final CodeType Double = JavaType.DOUBLE;
    public static final CodeType Boolean = JavaType.BOOLEAN;
    public static final CodeType Byte = JavaType.BYTE;
    public static final CodeType Char = JavaType.CHAR;
    public static final CodeType Short = JavaType.SHORT;
    public static final CodeType Int = JavaType.INT;
    public static final CodeType Long = JavaType.LONG;

}
