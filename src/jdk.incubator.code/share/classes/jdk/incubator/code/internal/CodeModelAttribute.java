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

package jdk.incubator.code.internal;

import java.lang.classfile.AttributeMapper;
import java.lang.classfile.AttributedElement;
import java.lang.classfile.BufWriter;
import java.lang.classfile.ClassReader;
import java.lang.classfile.CustomAttribute;
import jdk.incubator.code.Op;

public class CodeModelAttribute extends CustomAttribute<CodeModelAttribute>{

    public enum Tag {
        LocationAttr,

        // CoreOp
        AddOp, AndOp, ArrayLoadOp, ArrayStoreOp, ArrayLengthOp, AshrOp, AssertOp, BranchOp, CastOp, ClosureCallOp,
        ClosureOp, ComplOp, ConcatOp, ConditionalBranchOp, ConstantOp, ConvOp, DivOp, EqOp, ExceptionRegionEnter,
        ExceptionRegionExit, FieldLoadOp, FieldStoreOp, FuncCallOp, FuncOp, GeOp, GtOp, InstanceOfOp, InvokeOp,
        LambdaOp, LeOp, LshlOp, LshrOp, LtOp, ModOp, ModuleOp, MonitorEnterOp, MonitorExitOp, MulOp, NegOp, NeqOp,
        NewOp, NotOp, OrOp, QuotedOp, ReturnOp, SubOp, ThrowOp, TupleLoadOp, TupleOp, TupleWithOp, UnreachableOp,
        VarLoadOp, VarStoreOp, VarOp, XorOp, YieldOp,

        // ExtendedOp
        JavaBlockOp, JavaBreakOp, JavaConditionalAndOp, JavaConditionalExpressionOp, JavaConditionalOrOp,
        JavaContinueOp, JavaDoWhileOp, JavaEnhancedForOp, JavaForOp, JavaIfOp, JavaLabeledOp, JavaSwitchExpressionOp,
        JavaSwitchFallthroughOp, JavaSwitchStatementOp, JavaSynchronizedOp, JavaTryOp, JavaYieldOp, JavaWhileOp,
        MatchAllPatternOp, MatchOp, RecordPatternOp, TypePatternOp;
    }

    public static final String NAME = "CodeModel";

    public static final AttributeMapper<CodeModelAttribute> MAPPER = new AttributeMapper<>() {

        @Override
        public String name() {
            return NAME;
        }

        @Override
        public CodeModelAttribute readAttribute(AttributedElement enclosing, ClassReader cr, int pos) {
            return new CodeModelAttribute(new OpReader(cr, pos).readOp(null, null));
        }

        @Override
        public void writeAttribute(BufWriter buf, CodeModelAttribute attr) {
            buf.writeIndex(buf.constantPool().utf8Entry(NAME));
            int lengthIndex = buf.size();
            buf.writeInt(0);
            new OpWriter(buf).writeOp(attr.op);
            int written = buf.size() - lengthIndex - 4;
            buf.patchInt(lengthIndex, 4, written);
        }

        @Override
        public AttributeMapper.AttributeStability stability() {
            return AttributeMapper.AttributeStability.CP_REFS;
        }
    };

    public static CodeModelAttribute of(Op op) {
        return new CodeModelAttribute(op);
    }

    private final Op op;

    private CodeModelAttribute(Op op) {
        super(MAPPER);
        this.op = op;
    }

    public Op op() {
        return op;
    }
}
