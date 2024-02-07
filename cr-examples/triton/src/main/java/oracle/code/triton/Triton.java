/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.triton;

import java.lang.reflect.Type;
import java.lang.runtime.CodeReflection;
import java.util.List;

public class Triton {
    private Triton() {
    }

    public static int programId(@Constant int axis) {
        throw new UnsupportedOperationException();
    }

    public static Tensor arange(@Constant int start, @Constant int end) {
        throw new UnsupportedOperationException();
    }

    public static Tensor load(Tensor ptr, Tensor mask) {
        throw new UnsupportedOperationException();
    }

    public static void store(Tensor ptr, Tensor value, Tensor mask) {
        throw new UnsupportedOperationException();
    }

    public static Tensor broadcast(Object o, TensorType type) {
        throw new UnsupportedOperationException();
    }

    public static Tensor expand(Tensor a, int axis) {
        throw new UnsupportedOperationException();
    }

    public static Tensor zeros(Class<?> eType, int... shape)  {
        throw new UnsupportedOperationException();
    }

    public static TensorType joinShape(TensorType a, TensorType b) {
        throw new UnsupportedOperationException();
    }

    public static Tensor dot(Tensor a, Tensor b) {
        throw new UnsupportedOperationException();
    }

    // Arithmetic

    public static Tensor add(Number a, Number b) {
        throw new UnsupportedOperationException();
    }

    public static Tensor sub(Number a, Number b) {
        throw new UnsupportedOperationException();
    }

    public static Tensor mul(Number a, Number b) {
        throw new UnsupportedOperationException();
    }

    public static Tensor div(Number a, Number b) {
        throw new UnsupportedOperationException();
    }

    public static Tensor mod(Number a, Number b) {
        throw new UnsupportedOperationException();
    }

    public static Tensor and(Number a, Number b) {
        throw new UnsupportedOperationException();
    }

    public enum CompareKind {
        Equal,
        LessThan,
        LessThanOrEqual,
        GreaterThan,
        GreaterThanOrEqual
    }

    public static Tensor compare(Number a, Number b, @Constant CompareKind ck) {
        throw new UnsupportedOperationException();
    }

    public static Tensor exp(Tensor a) {
        throw new UnsupportedOperationException();
    }

    public static int cdiv(Number x, Number div) {
        throw new UnsupportedOperationException();
    }

    // Conversions

    public static <T extends Number> T conv(Type t, T a) {
        throw new UnsupportedOperationException();
    }

    // Reductions

    public static Tensor max(Tensor a, @Constant int axis) {
        throw new UnsupportedOperationException();
    }

    public static Tensor sum(Tensor a, @Constant int axis) {
        throw new UnsupportedOperationException();
    }
}
