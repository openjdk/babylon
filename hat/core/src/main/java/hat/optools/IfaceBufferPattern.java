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
package hat.optools;

import hat.device.DeviceType;
import hat.types.HAType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Invoke;
import optkl.ifacemapper.MappableIface;

import java.lang.invoke.MethodHandles;
import java.util.function.Predicate;

import static optkl.Invoke.invokeOpHelper;

public interface IfaceBufferPattern extends CodeModelPattern {

    static boolean isInvokeOp(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return invokeOpHelper(lookup,invokeOp) instanceof Invoke invoke && invoke.refIs(MappableIface.class);//;isAssignable(lookup, javaRefType(invokeOp), MappableIface.class));
    }

    static boolean isIfaceBufferInvokeOpWithName(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, Predicate<String> namePredicate) {

        return invokeOpHelper(lookup,invokeOp) instanceof Invoke  invoke
                && invoke.refIs( DeviceType.class, MappableIface.class, HAType.class)
                && namePredicate.test(invoke.name());
    }
}
