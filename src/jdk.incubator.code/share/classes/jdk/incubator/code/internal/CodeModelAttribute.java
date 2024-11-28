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
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExtendedOp;

/**
 * <pre>
 * CodeModel_attribute {
 *     u2 attribute_name_index;
 *     u4 attribute_length;
 *     op_info;
 * }
 *
 * op_info {
 *     u2 op_name_index;
 *     u2 op_operands_length;
 *     u2 op_operands[op_operands_length];
 *     u2 op_result_type_index;
 *     u2 op_attributes_length;
 *     op_attribute_info op_attributes_table[op_attributes_length];
 *     u2 nested_bodies_length;
 *     {   u2 body_func_type_index;
 *         block_content_info; // entry block
 *         u2 blocks_length;
 *         {   u2 block_parameters_length;
 *             u2 block_parameter_type_index[block_parameters_length];
 *             block_content_info;
 *         } blocks_table[blocks_length];
 *     } nested_bodies_table[nested_bodies_length];
 * }
 *
 * union op_attribute_info {
 *     value_attribute_info;
 *     location_attribute_info;
 * }
 *
 * value_attribute_info {
 *     u2 attribute_name_index;
 *     u2 attribute_value_index;
 * }
 *
 * location_attribute_info {
 *     u2 location_attribute_name_index;
 *     u2 source_index;
 *     u2 line_number;
 *     u2 column_number;
 * }
 *
 * block_content_info {
 *     u2 ops_length;
 *     op_info ops_table[ops_length];
 *     terminal_op_info;
 * } blocks_table[blocks_length];
 *
 * terminal_op_info {
 *     op_info;
 *     u2 successors_length;
 *     {   u2 successor_block_index;
 *         u2 block_arguments_length;
 *         u2 block_arguments[block_arguments_length];
 *     } successors_table[successors_length]
 * }
 */
public class CodeModelAttribute extends CustomAttribute<CodeModelAttribute>{

    public enum OpTag {
        AddOp(CoreOp.AddOp.NAME),
        AndOp(CoreOp.AndOp.NAME),
        ArrayLoadOp(CoreOp.ArrayAccessOp.ArrayLoadOp.NAME),
        ArrayStoreOp(CoreOp.ArrayAccessOp.ArrayStoreOp.NAME),
        ArrayLengthOp(CoreOp.ArrayLengthOp.NAME),
        AshrOp(CoreOp.AshrOp.NAME),
        AssertOp(CoreOp.AssertOp.NAME),
        BranchOp(CoreOp.BranchOp.NAME),
        CastOp(CoreOp.CastOp.NAME),
        ClosureCallOp(CoreOp.ClosureCallOp.NAME),
        ClosureOp(CoreOp.ClosureOp.NAME),
        ComplOp(CoreOp.ComplOp.NAME),
        ConcatOp(CoreOp.ConcatOp.NAME),
        ConditionalBranchOp(CoreOp.ConditionalBranchOp.NAME),
        ConstantOp(CoreOp.ConstantOp.NAME),
        ConvOp(CoreOp.ConvOp.NAME),
        DivOp(CoreOp.DivOp.NAME),
        EqOp(CoreOp.EqOp.NAME),
        ExceptionRegionEnter(CoreOp.ExceptionRegionEnter.NAME),
        ExceptionRegionExit(CoreOp.ExceptionRegionExit.NAME),
        FieldLoadOp(CoreOp.FieldAccessOp.FieldLoadOp.NAME),
        FieldStoreOp(CoreOp.FieldAccessOp.FieldStoreOp.NAME),
        FuncCallOp(CoreOp.FuncCallOp.NAME),
        FuncOp(CoreOp.FuncOp.NAME),
        GeOp(CoreOp.GeOp.NAME),
        GtOp(CoreOp.GtOp.NAME),
        InstanceOfOp(CoreOp.InstanceOfOp.NAME),
        InvokeOp(CoreOp.InvokeOp.NAME),
        LambdaOp(CoreOp.LambdaOp.NAME),
        LeOp(CoreOp.LeOp.NAME),
        LshlOp(CoreOp.LshlOp.NAME),
        LshrOp(CoreOp.LshrOp.NAME),
        LtOp(CoreOp.LtOp.NAME),
        ModOp(CoreOp.ModOp.NAME),
        ModuleOp(CoreOp.ModuleOp.NAME),
        MonitorEnterOp(CoreOp.MonitorOp.MonitorEnterOp.NAME),
        MonitorExitOp(CoreOp.MonitorOp.MonitorExitOp.NAME),
        MulOp(CoreOp.MulOp.NAME),
        NegOp(CoreOp.NegOp.NAME),
        NeqOp(CoreOp.NeqOp.NAME),
        NewOp(CoreOp.NewOp.NAME),
        NotOp(CoreOp.NotOp.NAME),
        OrOp(CoreOp.OrOp.NAME),
        QuotedOp(CoreOp.QuotedOp.NAME),
        ReturnOp(CoreOp.ReturnOp.NAME),
        SubOp(CoreOp.SubOp.NAME),
        ThrowOp(CoreOp.ThrowOp.NAME),
        TupleLoadOp(CoreOp.TupleLoadOp.NAME),
        TupleOp(CoreOp.TupleOp.NAME),
        TupleWithOp(CoreOp.TupleWithOp.NAME),
        UnreachableOp(CoreOp.UnreachableOp.NAME),
        VarLoadOp(CoreOp.VarAccessOp.VarLoadOp.NAME),
        VarStoreOp(CoreOp.VarAccessOp.VarStoreOp.NAME),
        VarOp(CoreOp.VarOp.NAME),
        XorOp(CoreOp.XorOp.NAME),
        YieldOp(CoreOp.YieldOp.NAME),
        JavaBlockOp(ExtendedOp.JavaBlockOp.NAME),
        JavaBreakOp(ExtendedOp.JavaBreakOp.NAME),
        JavaConditionalAndOp(ExtendedOp.JavaConditionalAndOp.NAME),
        JavaConditionalExpressionOp(ExtendedOp.JavaConditionalExpressionOp.NAME),
        JavaConditionalOrOp(ExtendedOp.JavaConditionalOrOp.NAME),
        JavaContinueOp(ExtendedOp.JavaContinueOp.NAME),
        JavaDoWhileOp(ExtendedOp.JavaDoWhileOp.NAME),
        JavaEnhancedForOp(ExtendedOp.JavaEnhancedForOp.NAME),
        JavaForOp(ExtendedOp.JavaForOp.NAME),
        JavaIfOp(ExtendedOp.JavaIfOp.NAME),
        JavaLabeledOp(ExtendedOp.JavaLabeledOp.NAME),
        JavaSwitchExpressionOp(ExtendedOp.JavaSwitchExpressionOp.NAME),
        JavaSwitchFallthroughOp(ExtendedOp.JavaSwitchFallthroughOp.NAME),
        JavaSwitchStatementOp(ExtendedOp.JavaSwitchStatementOp.NAME),
        MatchAllPatternOp(ExtendedOp.PatternOps.MatchAllPatternOp.NAME),
        MatchOp(ExtendedOp.PatternOps.MatchOp.NAME),
        RecordPatternOp(ExtendedOp.PatternOps.RecordPatternOp.NAME),
        TypePatternOp(ExtendedOp.PatternOps.TypePatternOp.NAME);

        final String opName;
        OpTag(String opName) {
            this.opName = opName;
        }
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
