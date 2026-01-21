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
package experiments;

import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;

import static optkl.OpHelper.Invoke.invoke;
import optkl.OpHelper.Invoke.Virtual;

public class UnpackingFluentBinaryOps {


    @Reflect
    public static S32x2 center(S32x2 min, S32x2 max) {
        var two = S32x2.of(2, 2);
        return  min.add(max).div(two);
    }


    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();

         Trxfmr.of(lookup, UnpackingFluentBinaryOps.class, "center", S32x2.class, S32x2.class)
                 .toText("// (Code Model) before transform", "//-------")
                 .toJava("// (Java) before mapping", "//-------")

                .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                    if (invoke(lookup,c.op()) instanceof Virtual v && v.named(Regex.of("(add|mul|div|mod|sub)"))) {
                        var lhs = c.mappedOperand(0);
                        var rhs = c.mappedOperand(1);
                        c.replace(switch (v.name()) { // we replace the call with one of ....
                            case "add" -> JavaOp.add(lhs, rhs);
                            case "sub" -> JavaOp.sub(lhs, rhs);
                            case "mul" -> JavaOp.mul(lhs, rhs);
                            case "div" -> JavaOp.div(lhs, rhs);
                            case "mod" -> JavaOp.mod(lhs, rhs);
                            default -> throw new IllegalStateException("how");
                        });
                    }
                })
                 .toText("// (Code Model) after transform", "// -------")
                 .toJava("// (Java) after transform ", "// -------");
    }

}
