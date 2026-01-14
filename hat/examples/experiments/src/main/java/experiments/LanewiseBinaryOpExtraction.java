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
import optkl.InvokeQuery;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;

import static experiments.LanewiseUtils.createBinaryOp;


public class LanewiseBinaryOpExtraction {

    @Reflect
    public static S32x2 center(S32x2 min, S32x2 max) {
        return min.add(max).div(S32x2.of(2,2));
    }

    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();
        var binaryOpQuery = InvokeQuery.create(lookup);
        Trxfmr.of(lookup, LanewiseBinaryOpExtraction.class, "center", S32x2.class, S32x2.class)
                .toJava("// (Java) before mapping", "//-------")
                .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                    if (binaryOpQuery.matches(c, $ ->
                            // does it look like a fluent binary op we don't care about the name
                            $.returns(S32x2.class) && $.isInstance() && $.receives( S32x2.class)
                    ) instanceof InvokeQuery.Match match) {
                        c.replace(createBinaryOp(match.helper(), c.mappedOperand(0), c.mappedOperand(1)));
                    }
                })
                .toJava("// (Java) after transform ", "// -------");
    }

}
