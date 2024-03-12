/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.util.List;
import java.util.Map;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;

public abstract class SpirvOp extends Op {
    private final TypeElement type;

    SpirvOp(String opName) {
        super(opName, List.of());
        this.type = JavaType.VOID;
    }

    SpirvOp(String opName, TypeElement type, List<Value> operands) {
        super(opName, operands);
        this.type = type;
    }

    SpirvOp(String opName, TypeElement type, List<Value> operands, Map<String, Object> attributes) {
        super(opName, operands);
        this.type = type;
    }

    SpirvOp(SpirvOp that, CopyContext cc) {
        super(that, cc);
        this.type = that.type;
    }

    @Override
    public TypeElement resultType() {
        return type;
    }
}