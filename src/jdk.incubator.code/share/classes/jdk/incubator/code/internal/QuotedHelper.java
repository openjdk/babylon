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

import jdk.incubator.code.*;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.CoreOp.FuncOp;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.SequencedMap;

public class QuotedHelper {

    public static Quoted makeQuoted(FuncOp funcOp, Object[] args) {

        CoreOp.OpAndValues opAndValues = CoreOp.quotedOp(funcOp);

        // map captured values to their corresponding runtime values
        // captured value can be:
        // 1- block param
        // 2- result of VarOp whose initial value is constant
        // 3- result of VarOp whose initial value is block param
        List<Block.Parameter> params = funcOp.parameters();
        SequencedMap<Value, Object> m = new LinkedHashMap<>();
        for (Value v : opAndValues.operandsAndCaptures()) {
            if (v instanceof Block.Parameter p) {
                Object rv = args[params.indexOf(p)];
                m.put(v, rv);
            } else if (v instanceof Op.Result opr && opr.op() instanceof CoreOp.VarOp varOp) {
                if (varOp.initOperand() instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp cop) {
                    m.put(v, CoreOp.Var.of(cop.value()));
                } else if (varOp.initOperand() instanceof Block.Parameter p) {
                    Object rv = args[params.indexOf(p)];
                    m.put(v, CoreOp.Var.of(rv));
                }
            }
        }

        return new Quoted(opAndValues.op(), m);
    }
}
