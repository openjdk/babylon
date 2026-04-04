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

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

public class DeviceSchema<T extends NonMappableIface> {
    private final IfaceDataDag<T> ifaceDataDag = new IfaceDataDag<>();
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
        this.representationBuilder = new C99CodeBuilder<>();
        this.clazz = clazz;
        this.ifaceDataDag.add(new IfaceDataDag.IfaceInfo.Impl<>((ClassType) JavaType.type(clazz.describeConstable().get()), clazz));
        members.add(new ArrayList<>());
    }

    int currentLevel = 0;

    public static <T extends NonMappableIface> DeviceSchema<T> of(Class<T> klass, Consumer<DeviceSchema<T>> schemaBuilder) {
        DeviceSchema<T> deviceSchema = new DeviceSchema<>(klass);
        schemaBuilder.accept(deviceSchema);
        deviceSchema.materialize(deviceSchema.representationBuilder,deviceSchema.clazz);
        deviceSchema.ifaceDataDag.closeRanks();
        return deviceSchema;
    }


    public DeviceSchema<T> fields(String... fields) {
        this.members.get(currentLevel).addAll(List.of(fields));
        return this;
    }

    public DeviceSchema<T> field(String fieldName) {
        return fields(fieldName);
    }

    public DeviceSchema<T> array(String fieldName, int size) {
        field(fieldName);
        arraySize.put(fieldName, size);
        return this;
    }
    public DeviceSchema<T> array(String fieldName, int size, Class<?> klass, Consumer<DeviceSchema<T>> depConsumer) {
        array(fieldName,size);
        this.currentLevel++; //  currentLevel== (this.members.size()-1))  // increment the level
        this.members.add(new ArrayList<>());
        depConsumer.accept(this);
        materialize(representationBuilder, klass);
        return this;
    }


    // The following recursive method generates an intermediate representation in text form for each level
    // of the hierarchy.
    // It inspects each type and its members. If a member is also a non-primitive type
    // then it recursively inspect its inner members.
    // We keep track of all generated data structured by maintaining a visited set. Thus,
    // we avoid duplicates in the text form.
    private C99CodeBuilder<?> materialize(C99CodeBuilder<?> builder, Class<?> clazz) {
        builder.lt();
        builder.id(clazz.getName()).colon();
        visited.add((Class<IfaceValue>) clazz);
        for (String fieldName : members.get(currentLevel)) {
            boolean wasProcessed = false;
            for (Method method : clazz.getDeclaredMethods()) {
                if (method.getName().equals(fieldName) && method.getReturnType() instanceof Class<?> returnType && !returnType.equals(void.class)) {
                    if (returnType.isInterface() && !visited.contains(returnType)) {
                        builder = materialize(new C99CodeBuilder<>(builder),returnType);// recurses here
                    }
                    boolean isArray = arraySize.containsKey(method.getName());
                    builder
                            .either(isArray,
                                    CodeBuilder::osbrace,                                             // [==array
                                    $ -> $.id("s")                                               // s==scalar
                            )
                            .colon().type(specialTypes.getOrDefault(clazz, returnType.getName()))     // type
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
        return builder.gt();
    }

    public String toText() {
        return this.representationBuilder.getText();
    }
}
