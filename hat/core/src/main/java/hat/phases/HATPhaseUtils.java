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
package hat.phases;

import hat.types.S16ImplOfF16;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.OpHelper;

import java.util.Optional;
import java.util.Set;
import static optkl.OpHelper.resultFromFirstOperandOrNull;

public class HATPhaseUtils {

    public static Op findOpInResultFromFirstOperandsOrNull(Op op, Class<?> ...classes) {
        Set<Class<?>> set =Set.of(classes);
        while (!set.contains(op.getClass())) {
            if (resultFromFirstOperandOrNull(op) instanceof Op.Result result) {
                op = result.op();
            } else {
                return null;
            }
        }
        return op;
    }

    public static Class<?> reduceFloatType(Optional<OpHelper.Invoke> invoke) {
        if (invoke.isPresent() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().refType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }

    public static Class<?> reduceFloatTypeFromReturnType(Optional<OpHelper.Invoke> invoke) {
        if (invoke.isPresent() &&  S16ImplOfF16.codeTypeToFloatClassOrNull(invoke.orElse(null), (ClassType) invoke.get().returnType()) instanceof Class<? extends S16ImplOfF16> category) {
            return category;
        }
        return null;
    }


    private HATPhaseUtils() {
        /* This utility class should not be instantiated */
    }
}
