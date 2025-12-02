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
package jdk.incubator.code.bytecode.impl;

import java.util.List;
import java.util.Map;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.JavaType;

/**
 * The terminating conditional multi-branch operation modeling {@code tableswitch} and {@code lookupswitch} instructions.
 * <p>
 * This operation accepts an int operand, variable number of distinct constant labels
 * and the same number of successors.
 * When the operand is matching one of the labels, the relevant successor is selected.
 * If none of the labels is matching, the default successor is selected.
 * Default is a successor with corresponds null label value.
 * The selected successor refers to the next block to branch to.
 */
public final class ConstantLabelSwitchOp extends Op implements Op.BlockTerminating {

    final List<Integer> labels;
    final List<Block.Reference> targets;

    public ConstantLabelSwitchOp(Value intSelector, List<Integer> labels, List<Block.Reference> targets) {
        super(List.of(intSelector));
        assert targets.size() == labels.size();
        this.labels = labels;
        this.targets = targets;
    }

    ConstantLabelSwitchOp(ConstantLabelSwitchOp that, CodeContext cc) {
        super(that, cc);
        this.labels = that.labels;
        this.targets = that.targets.stream().map(cc::getSuccessorOrCreate).toList();
    }

    @Override
    public ConstantLabelSwitchOp transform(CodeContext cc, CodeTransformer ot) {
        return new ConstantLabelSwitchOp(this, cc);
    }

    @Override
    public TypeElement resultType() {
        return JavaType.VOID;
    }

    public List<Integer> labels() {
        return labels;
    }

    @Override
    public List<Block.Reference> successors() {
        return targets;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("", labels);
    }

}
