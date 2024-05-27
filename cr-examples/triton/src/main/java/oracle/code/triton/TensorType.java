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

import java.lang.reflect.code.TypeElement;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class TensorType extends TritonType {
    static final String NAME = "tensor";

    final TypeElement eType;
    final List<Integer> shape;
    final int size;

    public TensorType(TypeElement eType, List<Integer> shape) {
        this.eType = eType;
        this.shape = List.copyOf(shape);
        int s = 1;
        for (Integer i : shape) {
            s *= i;
        }
        this.size = s;
    }

    public TypeElement eType() {
        return eType;
    }

    public List<Integer> shape() {
        return shape;
    }

    public int size() {
        return size;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TensorType that = (TensorType) o;
        return Objects.equals(eType, that.eType) && Objects.equals(shape, that.shape);
    }

    @Override
    public int hashCode() {
        return Objects.hash(eType, shape);
    }

    @Override
    public ExternalizedTypeElement externalize() {
        List<ExternalizedTypeElement> args = new ArrayList<>();
        for (int i : shape) {
            args.add(new ExternalizedTypeElement("x" + i, List.of()));
        }
        args.add(eType.externalize());
        return new ExternalizedTypeElement(NAME, args);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
