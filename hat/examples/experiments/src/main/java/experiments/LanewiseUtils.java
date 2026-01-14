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
import optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Optional;

public class LanewiseUtils {
    /**
     * Look for first real BinaryOp by recursively decending through nested invokes until we find a BinaryOp
     *
     * We first test if the target of the invoke has a binary op, if it does we return it
     * If not then we find the first invoke in the target of this invoke that returns a binary Op and return that one
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
     *   And our invoke was #1 we would return MulOp
     *
     *   If the invoke was #2 we would recurse inside and then end up at #1 and return Mul Op.
     * @return The binaryOp from one of the reachable methods
     */

    static JavaOp.BinaryOp getLaneWiseOp(Invoke invoke) {
        if (invoke.targetMethodModelOrThrow().elements().filter(o -> o instanceof JavaOp.BinaryOp).map(o -> (JavaOp.BinaryOp) o).findFirst()
                instanceof Optional<JavaOp.BinaryOp> optionalBinaryOp && optionalBinaryOp.isPresent()) {
            return optionalBinaryOp.get();
        } else {
           return  Invoke.stream(invoke.lookup(),invoke.targetMethodModelOrThrow()).map(LanewiseUtils::getLaneWiseOp).findFirst().get();
        }
    }

    static JavaOp.BinaryOp createBinaryOpViaNameSwitch(String name, Value lhs, Value rhs) {
        return switch (name) {
           case "add" -> JavaOp.add(lhs, rhs);
           case "sub" -> JavaOp.sub(lhs, rhs);
           case "div" -> JavaOp.div(lhs, rhs);
            case "mul" -> JavaOp.mul(lhs, rhs);
           case "mod" -> JavaOp.mod(lhs, rhs);
           default -> throw new IllegalStateException("missed one");
      };
    }

    static JavaOp.BinaryOp createBinaryOpFromNameViaReflection(String name, Value lhs, Value rhs) {
        var opMethod = Arrays.stream(JavaOp.class.getDeclaredMethods()).filter(m -> m.getName().equals(name)).findFirst().get();
        try {
            return (JavaOp.BinaryOp) opMethod.invoke(null, lhs, rhs);
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }
    static <T extends JavaOp.BinaryOp> T createBinaryOpViaClass(Class<T> clazz, Value lhs, Value rhs) {
        String name = clazz.getName();

        var opMethod = Arrays.stream(JavaOp.class.getDeclaredMethods())
                .filter(m -> {
                    Class<?> returnType =  m.getReturnType();
                    String returnTypeName = returnType.getName();
                    boolean sameName = returnTypeName.equals(name);
                    if (sameName){
                        sameName=sameName;
                    }
                    return returnType.isAssignableFrom(clazz) && sameName;
                }).findFirst().get();
        try {
            return (T) opMethod.invoke(null, lhs, rhs);
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }
    static JavaOp.BinaryOp createBinaryOpFromCodeModelViaClass(Invoke invoke, Value lhs, Value rhs) {
        JavaOp.BinaryOp laneWiseBinaryOp =getLaneWiseOp(invoke);
        Class<JavaOp.BinaryOp> clazz = (Class<JavaOp.BinaryOp>) laneWiseBinaryOp.getClass();
        return createBinaryOpViaClass(clazz, lhs, rhs);
    }

    static JavaOp.BinaryOp createBinaryOpFromCodeModelFromNameViaReflectedName(Invoke invoke, Value lhs, Value rhs) {
        JavaOp.BinaryOp laneWiseBinaryOp =getLaneWiseOp(invoke);                                          // search for lanewise Op..
        String nameWithOpSuffix = laneWiseBinaryOp.getClass().getSimpleName();                            // CoreOp.AddOp -> AddOp
        String nameSansOp = nameWithOpSuffix.substring(0, nameWithOpSuffix.length() - "Op".length());     // AddOp -> Add
        String simpleName = nameSansOp.substring(0, 1).toLowerCase() + nameSansOp.substring(1); // Add->add
        return createBinaryOpFromNameViaReflection(simpleName, lhs, rhs);                                                     // now we can reflectifly create a new AddOp.
    }
    static JavaOp.BinaryOp createBinaryOpFromCodeModelFromNameViaNameSwitch(Invoke invoke, Value lhs, Value rhs) {
        JavaOp.BinaryOp laneWiseBinaryOp =getLaneWiseOp(invoke);                                          // search for lanewise Op..
        String nameWithOpSuffix = laneWiseBinaryOp.getClass().getSimpleName();                            // CoreOp.AddOp -> AddOp
        String nameSansOp = nameWithOpSuffix.substring(0, nameWithOpSuffix.length() - "Op".length());     // AddOp -> Add
        String simpleName = nameSansOp.substring(0, 1).toLowerCase() + nameSansOp.substring(1); // Add->add
        return createBinaryOpViaNameSwitch(simpleName, lhs, rhs);                                                     // now we can reflectifly create a new AddOp.
    }
    static JavaOp.BinaryOp createBinaryOp(Invoke invoke, Value lhs, Value rhs) {
        JavaOp.BinaryOp laneWiseBinaryOp =getLaneWiseOp(invoke);                                          // search for lanewise Op..
        String nameWithOpSuffix = laneWiseBinaryOp.getClass().getSimpleName();                            // CoreOp.AddOp -> AddOp
        String nameSansOp = nameWithOpSuffix.substring(0, nameWithOpSuffix.length() - "Op".length());     // AddOp -> Add
        String simpleName = nameSansOp.substring(0, 1).toLowerCase() + nameSansOp.substring(1); // Add->add
        return createBinaryOpViaNameSwitch(simpleName, lhs, rhs);                                                     // now we can reflectifly create a new AddOp.
    }

}
