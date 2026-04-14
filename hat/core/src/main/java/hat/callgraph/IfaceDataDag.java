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

import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;
import optkl.util.Dag;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class IfaceDataDag<I extends IfaceValue> extends Dag<IfaceDataDag.IfaceInfo<I>> {
     public IfaceDataDag.IfaceInfo<I>  getNode(Class<I> clazz) {
        return new IfaceDataDag.IfaceInfo.Impl<>((ClassType) JavaType.type(clazz.describeConstable().get()), clazz);
    }

      public IfaceInfo<I>  getNode(MethodHandles.Lookup lookup, ClassType classType) {
        return getNode((Class<I>) OpHelper.classTypeToTypeOrThrow(lookup, classType));
    }

    public Stream<IfaceInfo<I>> methodsWithIfaceReturnTypes(Class<I>clazz) {
        return Arrays.stream(clazz.getDeclaredMethods())
                .map(Method::getReturnType)
                .filter(IfaceValue.class::isAssignableFrom)
                .map(ifaceClass->getNode(((Class<I>)ifaceClass)));
    }

    public interface IfaceInfo<I extends IfaceValue> {
        ClassType classType();
        Class<I> clazz();
        record Impl<I extends IfaceValue>(ClassType classType, Class<I> clazz) implements IfaceInfo<I> {
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

    // recursive
    public void addEdge(IfaceInfo<I> from, IfaceInfo<I> to) {
        if (!from.equals(to)) {
            methodsWithIfaceReturnTypes(to.clazz()).forEach(retType ->
                    addEdge(to, retType)
            );
            add(from, to);
        }
    }

    public IfaceDataDag(Consumer<IfaceDataDag<I>> init){
       init.accept(this);
       closeRanks();
    }
}
