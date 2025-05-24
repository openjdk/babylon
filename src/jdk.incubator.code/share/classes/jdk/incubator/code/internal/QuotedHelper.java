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
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.CoreOp.FuncOp;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;

public class QuotedHelper {

    public static Quoted makeQuoted(FuncOp op, Object[] args) {

        CoreOp.QuotedOp qop = (CoreOp.QuotedOp) op.body().entryBlock().ops().stream()
                .filter(o -> o instanceof CoreOp.QuotedOp).findFirst().orElseThrow();

        Iterator<Object> argsIterator = Arrays.stream(args).iterator();
        LinkedHashMap<Value, Object> m = new LinkedHashMap<>();
        for (Value capturedValue : qop.capturedValues()) {
            if (capturedValue instanceof Block.Parameter) {
                m.put(capturedValue, argsIterator.next());
            } else if (capturedValue instanceof Op.Result opr && opr.op() instanceof CoreOp.VarOp varOp) {
                if (varOp.initOperand() instanceof Block.Parameter) {
                    m.put(capturedValue, new Interpreter.VarBox(argsIterator.next()));
                } else if (varOp.initOperand() instanceof Op.Result opr2 && opr2.op() instanceof CoreOp.ConstantOp cop) {
                    m.put(capturedValue, new Interpreter.VarBox(cop.value()));
                }
            }
        }

        return new Quoted(qop.quotedOp(), m);
    }
}
