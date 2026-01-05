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
package optkl.util.ops;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

public interface Precedence {
    interface LoadOrConv extends Precedence {}
    interface Multiplicative extends Precedence{};
    interface Additive extends Precedence{}
    interface Store extends Precedence{}

    static boolean needsParenthesis(Op parent, Op child) {
        return precedenceOf(parent) <= precedenceOf(child);
    }

    static int precedenceOf(Op op) {
        return switch (op) {
            case Precedence.LoadOrConv _,
                    CoreOp.YieldOp _,
                 JavaOp.InvokeOp _,
                 CoreOp.FuncCallOp _ ,
                 JavaOp.FieldAccessOp _,
                 CoreOp.VarAccessOp.VarLoadOp _,
                 CoreOp.ConstantOp _,
                 JavaOp.LambdaOp _,
                 CoreOp.TupleOp _,
                 JavaOp.WhileOp _
                    -> 0;   //  ()[ ] .
            case JavaOp.ConvOp _,
                 JavaOp.NegOp  _
                    -> 1; //  ++ --+ -! ~ (type) *(deref) &(addressof) sizeof
            case Precedence.Multiplicative _,
                    JavaOp.ModOp _,
                 JavaOp.MulOp _,
                 JavaOp.DivOp _,
                 JavaOp.NotOp _
                    -> 2; //  * / %
            case Precedence.Additive _,
                 JavaOp.AddOp _,
                 JavaOp.SubOp _
                    -> 3; //  = + -
            case JavaOp.AshrOp _,
                 JavaOp.LshlOp _,
                 JavaOp.LshrOp _
                    -> 4; // 4 = << >>
            case JavaOp.LtOp _,
                 JavaOp.GtOp _,
                 JavaOp.LeOp _,
                 JavaOp.GeOp _
                    -> 5; //  < <= > >=
            case JavaOp.EqOp _,
                 JavaOp.NeqOp _
                    -> 6;  // == !=
            case JavaOp.AndOp _
                    -> 7; //  &
            case JavaOp.XorOp _
                    -> 8; // ^
            case JavaOp.OrOp _
                    -> 9; // |
            case JavaOp.ConditionalAndOp _
                    -> 10;//&&
            case JavaOp.ConditionalOrOp _
                    -> 11;// ||
            case JavaOp.ConditionalExpressionOp _
                    -> 12;// ()?:
            case Precedence.Store _,
                 CoreOp.VarOp _,
                 CoreOp.VarAccessOp.VarStoreOp _
                    -> 13;  // = += -= *= /= %= &= ^= |= <<= >>=
            case CoreOp.ReturnOp _-> 14;
            default -> throw new IllegalStateException("[Illegal] Precedence Op not registered: " + op.getClass().getSimpleName());

        };
    }

}
