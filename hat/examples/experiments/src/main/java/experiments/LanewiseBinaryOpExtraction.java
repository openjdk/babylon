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
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.InvokeQuery;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Optional;



public class LanewiseBinaryOpExtraction {

    /**
     * Look for first real BinaryOp by recursively descending through nested invokes until we find a BinaryOp
     *
     * @param invoke An Invoke call which either includes a BinaryOp or calls a reflectable method that does.
     * @return The binaryOp from either this method or one reachable from it
     * @throws RuntimeException if we fail to locate a Binary op
     *
     * This method is recursive
     *
     * The assumption is that only one BinaryOp type will be found.
     *
     * We first trivially test if the target of the invoke has a binary op (stream test on code elements) , if it does we return it
     * If not then we find all invokes from this call and scan (recursively) until we find an invoke that contains a Binary Op.
     *
     * So if we had
     *   @Reflect
     *   static S32x2 mul(S32x2 lhs, S32x2 rhs) { #1
     *        return new S32x2(lhs.x * rhs.x, lhs.y * rhs.y);
     *   }
     *
     *   @Reflect
     *   public S32x2 mul(S32x2 rhs) { // #2
     *       return mul(this, rhs);
     *   }
     *
     *   And our invoke was called on the static op form  (#1) we would 'trivially' return MulOp
     *
     *   If the invoke was on the fluent form  (#2) we would not find one trivially so would
     *   recurse inside the fluent op on all methods, until we find an invoke that does.
     *

     */
    static JavaOp.BinaryOp getLaneWiseOp(OpHelper.Named.NamedStaticOrInstance.Invoke invoke) {
        if (invoke.targetMethodModelOrThrow().elements().filter(o -> o instanceof JavaOp.BinaryOp).map(o -> (JavaOp.BinaryOp) o).findFirst()
                instanceof Optional<JavaOp.BinaryOp> optionalBinaryOp && optionalBinaryOp.isPresent()) {
            return optionalBinaryOp.get();
        } else {
            return  OpHelper.Named.NamedStaticOrInstance.Invoke
                    .stream(invoke.lookup(),invoke.targetMethodModelOrThrow())
                    .map(LanewiseBinaryOpExtraction::getLaneWiseOp)
                    .findFirst()
                    .get();
        }
    }

    /**
     * Here we determine the lanewise BinaryOp type (see #getLineWiseOP) and reflectively scan JavaOp.class for a suitable factory and invoke.
     * @param invoke An invoke helper representing the top level fluent or static call we assume is a binary op
     * @param lhs the mapped lhs value
     * @param rhs the mapped rhs value
     * @return a new binary Op
     *
     * @throws RuntimeException if we can't find an Op.
     */

    static JavaOp.BinaryOp createBinaryOp(OpHelper.Named.NamedStaticOrInstance.Invoke invoke, Value lhs, Value rhs) {
        JavaOp.BinaryOp laneWiseBinaryOp =getLaneWiseOp(invoke);
        Class<JavaOp.BinaryOp> clazz = (Class<JavaOp.BinaryOp>) laneWiseBinaryOp.getClass();
        var optionalMethod = Arrays.stream(JavaOp.class.getDeclaredMethods()).filter(m ->
                m.getParameterCount()==2 && clazz.isAssignableFrom(m.getReturnType())
        ).findFirst();
        if (optionalMethod.isPresent()) {
            try {
                return  (JavaOp.BinaryOp) optionalMethod.get().invoke(null, lhs, rhs);
            } catch (IllegalAccessException | InvocationTargetException e) {
                throw new RuntimeException(e.getMessage());
            }
        }else{
            throw new RuntimeException("Failed to find binary op factory for "+clazz);
        }
    }



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
                    if (binaryOpQuery.matches(c, $ ->// trivially look for a fluent style binary Op such as  S32x2.add(S32x2 rhs)
                            $.isInstance() && $.returns(S32x2.class) &&  $.receives( S32x2.class)
                    ) instanceof InvokeQuery.Match match) {
                        c.replace(
                                createBinaryOp(match.helper(), c.mappedOperand(0), c.mappedOperand(1))
                        );
                    }
                })
                .toJava("// (Java) after transform ", "// -------");
    }

}
