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

import hat.device.NonMappableIface;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.Buffer;
import optkl.util.Dag;

import java.lang.constant.ClassDesc;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public class IfaceDataDag extends Dag<IfaceDataDag.IfaceInfo> {
    static public class IfaceInfo{
        public final ClassType classType;
        public final Class clazz;
        IfaceInfo(ClassType classType, Class<?> clazz){
            if (classType==null || clazz == null){
                throw new RuntimeException("no nulls here ");
            }
            this.classType = classType;
            this.clazz = clazz;
        }

        @Override
        public int hashCode() {
            return Objects.hash(classType,clazz);
        }

        @Override
        public boolean equals(Object o) {
            return (this == o)
                    || (o instanceof IfaceInfo that && Objects.equals(classType, that.classType) && Objects.equals(clazz, that.clazz));
        }

        static IfaceInfo of(Class<?> clazz) {
            return new IfaceInfo((ClassType) JavaType.type(clazz.describeConstable().get()), clazz);
        }
    }


    IfaceDataDag(MethodHandles.Lookup lookup) {
        super(lookup);
    }

    public static Predicate<Class<?>> ifacePredicate = IfaceValue.class::isAssignableFrom; // both mappable (for mem segments) and non mappable (for private/shared mem)

    static Stream<IfaceInfo> declaredMethodIfaceReturnTypes(IfaceInfo iface){
        return Arrays.stream(iface.clazz.getDeclaredMethods())
                .map(Method::getReturnType)
                .filter(ifacePredicate)
                .map(IfaceInfo::of);
    }

    // recursive
     void addEdge(IfaceInfo from, IfaceInfo to) {
        if (!from.equals(to)){
            declaredMethodIfaceReturnTypes(to).forEach(retType ->
                addEdge(to,retType)
            );
            add(from, to, _->{});
       }
    }

    static public IfaceDataDag of(MethodHandles.Lookup lookup, CoreOp.FuncOp inlinedEntrypointFuncOp) {
        var dag = new IfaceDataDag(lookup);

        inlinedEntrypointFuncOp.elements().filter(ce -> ce instanceof Op).map(ce -> ((Op) ce).resultType())
                .filter(typeElement -> typeElement instanceof ClassType)
                .map(typeElement -> (ClassType)typeElement)
                .map(classType -> new IfaceInfo(classType,(Class<?>)OpHelper.classTypeToTypeOrThrow(lookup, classType)))
                .filter(ifaceInfo->ifacePredicate.test(ifaceInfo.clazz)).forEach(iface->
                            dag.add(iface, _->
                                declaredMethodIfaceReturnTypes(iface).forEach(retType ->
                                        dag.addEdge(iface, retType)
                                )
                            )
                );
        dag.closeRanks();
        return dag;
    }
}
