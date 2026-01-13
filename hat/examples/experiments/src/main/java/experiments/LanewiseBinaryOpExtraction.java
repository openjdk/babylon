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
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.InvokeQuery;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;

import java.util.Arrays;
import java.util.Optional;

public class LanewiseBinaryOpExtraction {

    public record XY(int x, int y) {
        @Reflect
        static XY of(int x, int y) {
            return new XY(x, y);
        }

        @Reflect
        static XY addEm(XY lhs, XY rhs) { // We don't have to call this add.
            return new XY(lhs.x + rhs.x, lhs.y + rhs.y);
        }

        @Reflect
        static XY sub(XY lhs, XY rhs) {
            return new XY(lhs.x - rhs.x, lhs.y - rhs.y);
        }

        @Reflect
        static XY div(XY lhs, XY rhs) {
            return new XY(lhs.x / rhs.x, lhs.y / rhs.y);
        }

        @Reflect
        static XY mod(XY lhs, XY rhs) {
            return new XY(lhs.x % rhs.x, lhs.y % rhs.y);
        }

        @Reflect
        static XY mul(XY lhs, XY rhs) {
            return new XY(lhs.x * rhs.x, lhs.y * rhs.y);
        }

        @Reflect
        public XY mul(XY xy) {
            return mul(this, xy);
        }

        @Reflect
        public XY mul(int scalar) {
            return mul(this, XY.of(scalar, scalar));
        }

        @Reflect
        public XY addEm(XY xy) { // we don't have to call this add either
            return addEm(this, xy);
        }

        @Reflect
        public XY sub(XY xy) {
            return sub(this, xy);
        }

        @Reflect
        public XY div(XY xy) {
            return div(this, xy);
        }

        @Reflect
        public XY mod(XY xy) {
            return mod(this, xy);
        }
    }


    /**
     * Look for first real BinaryOp by recursively decending through nested invokes until we find a BinaryOp
     *
     * @param lookup
     * @param funcOp
     * @return The binaryOp from one of the reachable methods
     */

    static JavaOp.BinaryOp getLaneWiseOp(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        if (funcOp.elements().filter(o -> o instanceof JavaOp.BinaryOp).map(o -> (JavaOp.BinaryOp) o).findFirst()
                instanceof Optional<JavaOp.BinaryOp> optionalBinaryOp && optionalBinaryOp.isPresent()) {
            return optionalBinaryOp.get();
        } else {
           return  Invoke.stream(lookup,funcOp).map(invoke ->getLaneWiseOp(lookup,invoke.targetMethodModelOrThrow())).findFirst().get();
        }
    }

    /*
     Create a binary Op by reflecting over code model of the reftype of the invoke method and determining the lanewise op.
    */

    /**
     * Reflectively a Replacement for
     * static JavaOp.BinaryOp createBinaryOp(String name, Value lhs, Value rhs){
     * return switch (name) {
     * case "add" -> JavaOp.add(lhs, rhs);
     * case "sub" -> JavaOp.sub(lhs, rhs);
     * case "mul" -> JavaOp.mul(lhs, rhs);
     * case "div" -> JavaOp.div(lhs, rhs);
     * case "mod" -> JavaOp.mod(lhs, rhs);
     * default -> throw new IllegalStateException("missed one");
     * }
     * }
     */

    static JavaOp.BinaryOp createBinaryOp(String name, Value lhs, Value rhs) {
        var opMethod = Arrays.stream(JavaOp.class.getDeclaredMethods()).filter(m -> m.getName().equals(name)).findFirst().get();
        try {
            return (JavaOp.BinaryOp) opMethod.invoke(null, lhs, rhs);
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }


    static JavaOp.BinaryOp createBinaryOpFromCodeModel(Invoke invoke, Value lhs, Value rhs) {
        //in our case invoke is on we now have XY.class
        JavaOp.BinaryOp laneWiseBinaryOp =getLaneWiseOp(invoke.lookup(), invoke.targetMethodModelOrThrow());// getLaneWiseOp(invoke);  // begin the search for lanewise Op.. So if invoke.name()=="add" we would expect "AddOp"
        String simpleNameWithOpSuffix = laneWiseBinaryOp.getClass().getSimpleName();                                          // CoreOp.AddOp -> AddOp
        String simpleNameSansOp = simpleNameWithOpSuffix.substring(0, simpleNameWithOpSuffix.length() - "Op".length());  // AddOp -> Add
        String simpleName = simpleNameSansOp.substring(0, 1).toLowerCase() + simpleNameSansOp.substring(1);     // Add->add
        return createBinaryOp(simpleName, lhs, rhs);  // now we can reflectifly create a new AddOp.
    }

    @Reflect
    public static XY center(XY min, XY max) {
        var two = XY.of(2, 2);
        return min.addEm(max).div(two).addEm(XY.of(1,1)).mul(XY.of(15,15));
    }

    public static void main(String[] args) throws NoSuchMethodException {
        var lookup = MethodHandles.lookup();
        var binaryOpQuery = InvokeQuery.create(lookup);
        Trxfmr.of(lookup, LanewiseBinaryOpExtraction.class, "center", XY.class, XY.class)
                .toJava("// (Java) before mapping", "//-------")
                .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                    if (binaryOpQuery.matches(c, $ -> // does it look like a fluent binary op we don't care about the name
                            $.returns(XY.class) && $.isInstance() && $.receives( XY.class)
                    ) instanceof InvokeQuery.Match match) {
                        c.replace(createBinaryOpFromCodeModel(match.helper(), c.mappedOperand(0), c.mappedOperand(1)));
                    }
                })
                .toJava("// (Java) after transform ", "// -------");
    }

}
