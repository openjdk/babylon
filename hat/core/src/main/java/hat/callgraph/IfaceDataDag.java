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
package hat.callgraph;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.util.Dag;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.Stream;

public class IfaceDataDag extends Dag<IfaceDataDag.IfaceInfo> {
    interface IfaceInfo {
        ClassType classType();
        Class<IfaceValue> clazz();
           record Impl(ClassType classType, Class<IfaceValue> clazz) implements IfaceInfo {
        }
        default String dotName() {
            if (IfaceValue.Struct.class.isAssignableFrom(clazz())) {
                return clazz().getSimpleName()+"_s";
            } else if (IfaceValue.Union.class.isAssignableFrom(clazz())) {
                return clazz().getSimpleName()+"_u";
            }
            return clazz().getSimpleName();
        }
    }
    static Stream<IfaceInfo> declaredMethodIfaceReturnTypes(IfaceInfo iface) {
        return Arrays.stream(iface.clazz().getDeclaredMethods())
                .map(Method::getReturnType)
                .filter(IfaceValue.class::isAssignableFrom)
                .map(clazz -> new IfaceInfo.Impl((ClassType) JavaType.type(clazz.describeConstable().get()), (Class<IfaceValue>)clazz));
    }

    // recursive
    void addEdge(IfaceInfo from, IfaceInfo to) {
        if (!from.equals(to)) {
            declaredMethodIfaceReturnTypes(to).forEach(retType ->
                    addEdge(to, retType)
            );
            add(from, to, _ -> {
            });
        }
    }

    public IfaceDataDag(MethodHandles.Lookup lookup, CoreOp.FuncOp inlinedEntrypointFuncOp) {
        inlinedEntrypointFuncOp.elements()
                .filter(ce -> ce instanceof Op)
                .map(ce -> ((Op) ce).resultType())
                .filter(typeElement -> typeElement instanceof ClassType)
                .map(classType -> new IfaceInfo.Impl((ClassType) classType, (Class<IfaceValue>) OpHelper.classTypeToTypeOrThrow(lookup, (ClassType) classType)))
                .filter(impl -> IfaceValue.class.isAssignableFrom(impl.clazz)).forEach(iface ->
                        add(iface, _ ->
                                declaredMethodIfaceReturnTypes(iface).forEach(retType ->
                                        addEdge(iface, retType)
                                )
                        )
                );
        closeRanks();
    }
}
