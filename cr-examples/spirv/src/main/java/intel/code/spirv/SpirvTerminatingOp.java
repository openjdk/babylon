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

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedOp;

import java.util.List;
import java.util.Map;

public abstract class SpirvTerminatingOp extends AbstractOp.Terminating implements ExternalizedOp.Externalizable {
    private final String opName;
    private final CodeType type;

    SpirvTerminatingOp(String opName) {
        super(List.of());
        this.opName = opName;
        this.type = JavaType.VOID;
    }

    SpirvTerminatingOp(String opName, CodeType type, List<Value> operands, List<Block.Reference> successors) {
        super(operands, successors);
        this.opName = opName;
        this.type = type;
    }

    SpirvTerminatingOp(SpirvTerminatingOp that, CodeContext cc) {
        super(that, cc);
        this.opName = that.opName;
        this.type = that.type;
    }

    @Override
    public CodeType resultType() {
        return type;
    }

    @Override
    public String externalizeOpName() {
        return opName;
    }
}