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
package hat.device;

import hat.callgraph.IfaceDataDag;
import hat.types.F16;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;
import optkl.codebuilders.C99CodeBuilder;
import optkl.codebuilders.CodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

public class DeviceSchema<T extends NonMappableIface> {
    private final IfaceDataDag ifaceDataDag = new IfaceDataDag();
    private final Class<T> clazz;
    private final List<List<String>> members = new ArrayList<>();
    private final Map<String, Integer> arraySize = new HashMap<>();
    private final C99CodeBuilder<?> representationBuilder;
    private final Set<Class<IfaceValue>> visited = new HashSet<>();

    private static final Map<Class<?>, String> specialTypes = new HashMap<>();

    static {
        specialTypes.put(F16.class, "half");
    }

    public DeviceSchema(Class<T> clazz) {
        this.representationBuilder = new C99CodeBuilder<>(new ScopedCodeBuilderContext(MethodHandles.lookup(), null));
        this.clazz = clazz;
        Object o = clazz;
        var ifaceClazz = (Class<IfaceValue>) o;
        var root = new IfaceDataDag.IfaceInfo.Impl((ClassType) JavaType.type(clazz.describeConstable().get()), ifaceClazz);
        this.ifaceDataDag.add(root);
        members.add(new ArrayList<>());
    }

    int currentLevel = 0;

    public static <T extends NonMappableIface> DeviceSchema<T> of(Class<T> klass, Consumer<DeviceSchema<T>> schemaBuilder) {
        DeviceSchema<T> deviceSchema = new DeviceSchema<>(klass);
        schemaBuilder.accept(deviceSchema);
        deviceSchema.materialize();
        deviceSchema.ifaceDataDag.closeRanks();
        return deviceSchema;
    }


    public DeviceSchema<T> withFields(String... fields) {
        this.members.get(currentLevel).addAll(List.of(fields));
        return this;
    }

    public DeviceSchema<T> withField(String fieldName) {
        return withFields(fieldName);
    }

    public DeviceSchema<T> withArray(String fieldName, int size) {
        withField(fieldName);//this.members.get(currentLevel).add(fieldName);
        arraySize.put(fieldName, size);
        return this;
    }

    public DeviceSchema<T> withDeps(Class<?> klass, Consumer<DeviceSchema<T>> depConsumer) {
        // increment the level
        this.currentLevel++; //  currentLevel== (this.members.size()-1))
        this.members.add(new ArrayList<>());
        depConsumer.accept(this);
        materialize(representationBuilder, klass);
        return this;
    }

    private boolean isInterfaceType(Class<?> type) {
        return type.isInterface();
    }

    // Materialize methods are only reachable within this class.
    private void materialize() {
        materialize(representationBuilder, clazz);
    }

    // The following method generates an intermediate representation in text form for each level
    // of the hierarchy.
    // It inspects each type and its members. If a member is also a non-primitive type
    // then it recursively inspect its inner members.
    // We keep track of all generated data structured by maintaining a visited set. Thus,
    // we avoid duplicates in the text form.
    // recursive
    private void materialize(C99CodeBuilder<?> builder, Class<?> clazz) {
        builder.lt();
        builder.id(clazz.getName()).colon();
        visited.add((Class<IfaceValue>) clazz);
        for (String fieldName : members.get(currentLevel)) {
            boolean wasProcessed = false;
            for (Method method : clazz.getDeclaredMethods()) {
                if (method.getName().equals(fieldName) && method.getReturnType() instanceof Class<?> returnType && !returnType.equals(void.class)) {
                    if (isInterfaceType(returnType) && !visited.contains(returnType)) {
                        // inspect the dependency and add it at the front of the string builder
                        C99CodeBuilder<?> depsBuilder = new C99CodeBuilder<>(
                                new ScopedCodeBuilderContext(
                                        builder.scopedCodeBuilderContext().lookup(),
                                        builder.scopedCodeBuilderContext().funcOp()
                                )
                        );
                        depsBuilder.preformatted(builder.getText());
                        materialize(depsBuilder, returnType); // recurses here
                        builder = depsBuilder;
                    }
                    boolean isArray = arraySize.containsKey(method.getName());
                    builder
                            .either(isArray,
                                    CodeBuilder::osbrace,                                           // [==array
                                    $ -> $.id("s")                                                // s==scalar
                            )
                            .colon().type(specialTypes.getOrDefault(clazz, returnType.getName()))          // type
                            .colon().id(method.getName())                                            // name
                            .when(isArray, $ -> $
                                    .colon().id(Integer.toString(arraySize.get(method.getName())))   // Array size
                            )
                            .semicolon();                   // member separator
                    wasProcessed = true;
                }
            }
            if (!wasProcessed) {
                throw new RuntimeException("could not find method " + fieldName + " in class " + clazz.getName());
            }
        }
        currentLevel--;
        builder.gt();
    }

    public String toText() {
        return this.representationBuilder.getText();
    }
}
