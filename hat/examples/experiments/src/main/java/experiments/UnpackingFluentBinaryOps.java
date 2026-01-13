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
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;

public class UnpackingFluentBinaryOps {
    public static class XY {
        private XY(int x, int y) {
            this.x = x;
            this.y = y;
        }

        private int x, y;

        static XY of(int x, int y) {
            return new XY(x, y);
        }

        static XY mul(XY lhs, XY rhs) {
            return new XY(lhs.x * rhs.x, lhs.y * rhs.y);
        }

        static XY add(XY lhs, XY rhs) {
            return new XY(lhs.x + rhs.x, lhs.y + rhs.y);
        }

        static XY sub(XY lhs, XY rhs) {
            return new XY(lhs.x - rhs.x, lhs.y - rhs.y);
        }

        static XY div(XY lhs, XY rhs) {
            return new XY(lhs.x / rhs.x, lhs.y / rhs.y);
        }

        static XY mod(XY lhs, XY rhs) {
            return new XY(lhs.x % rhs.x, lhs.y % rhs.y);
        }

        public XY mul(XY xy) {
            return mul(this, xy);
        }

        public XY add(XY xy) {
            return add(this, xy);
        }

        public XY sub(XY xy) {
            return sub(this, xy);
        }

        public XY div(XY xy) {
            return div(this, xy);
        }

        public XY mod(XY xy) {
            return mod(this, xy);
        }

    }

    @Reflect
    public static XY center(XY min, XY max) {
        var two = XY.of(2, 2);
        return  min.add(max).div(two);
    }


    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();
         var mathOperatorQuery = InvokeQuery.create(lookup);
         Trxfmr.of(lookup,
                        UnpackingFluentBinaryOps.class, "center", XY.class, XY.class)
                 .toText("// (Code Model) before transform", "//-------")
                 .toJava("// (Java) before mapping", "//-------")

                .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                    if (mathOperatorQuery.matches(c, // $ here is an Invoke helper...
                            $ -> $.named(Regex.of("(add|mul|div|mod|sub)"))) instanceof InvokeQuery.Match match) {
                        var lhs = c.mappedOperand(0);
                        var rhs = c.mappedOperand(1);
                        c.replace(switch (match.helper().name()) { // we replace the call with one of ....
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
