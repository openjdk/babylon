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

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

public class DeviceSchema<T extends DeviceType> {

    private final Class<T> klass;
    private final List<List<String>> members = new ArrayList<>();
    private final Map<String, Integer> arraySize = new HashMap<>();
    private final Map<Class<?>, Consumer<DeviceSchema<T>>> deps = new HashMap<>();
    private final StringBuilder representationBuilder = new StringBuilder();
    private final Set<String> visited = new HashSet<>();

    public DeviceSchema(Class<T> klass) {
        this.klass = klass;
    }
    int currentLevel = 0;

    public static <T extends DeviceType> DeviceSchema<T> of(Class<T> klass, Consumer<DeviceSchema<T>> schemaBuilder) {
        DeviceSchema<T> deviceSchema =  new DeviceSchema<>(klass);
        schemaBuilder.accept(deviceSchema);
        deviceSchema.materialize();
        return deviceSchema;
    }

    public DeviceSchema<T> withField(String fieldName) {
        if (members.isEmpty()) {
            members.add(new ArrayList<>());
        }
        this.members.get(currentLevel).add(fieldName);
        return this;
    }

    public DeviceSchema<T> withArray(String fieldName, int size) {
        if (members.isEmpty()) {
            members.add(new LinkedList<>());
        }
        this.members.get(currentLevel).add(fieldName);
        arraySize.put(fieldName, size);
        return this;
    }

    public DeviceSchema<T> withDeps(Class<?> klass, Consumer<DeviceSchema<T>> depConsumer) {
        // increment the level
        this.currentLevel++;
        this.members.add(new LinkedList<>());
        deps.put(klass, depConsumer);
        depConsumer.accept(this);
        materialize(representationBuilder, klass);
        return this;
    }

    private boolean isInterfaceType(Class<?> type) {
        return type.isInterface();
    }

    // Materialize methods are only reachable within this class.
    private DeviceSchema<T> materialize() {
        materialize(representationBuilder, klass);
        return this;
    }

    // The following method generates an intermediate representation in text form for each level
    // of the hierarchy.
    // It inspects each type and its members. If a member is also a non-primitive type
    // then it recursively inspect its inner members.
    // We keep track of all generated data structured by maintaining a visited set. Thus,
    // we avoid duplicates in the text form.
    private DeviceSchema<T> materialize(StringBuilder sb, Class<?> klass) {
        try {
            Class<?> aClass = Class.forName(klass.getName());
            Method[] declaredMethods = aClass.getDeclaredMethods();
            sb.append("<");
            sb.append(klass.getName());
            sb.append(':');
            visited.add(klass.getName());

            for (String fieldName : members.get(currentLevel)) {
                boolean wasProcessed = false;
                for (Method method : declaredMethods) {
                    method.setAccessible(true);
                    if (method.getName().equals(fieldName)) {
                        Class<?> returnType = method.getReturnType();
                        if (returnType.equals(void.class)) {
                            continue;
                        }

                        if (isInterfaceType(returnType) && !visited.contains(returnType.getName())) {
                            // inspect the dependency and add it at the front of the string builder
                            StringBuilder depsBuilder = new StringBuilder();
                            materialize(depsBuilder, returnType);
                            sb = depsBuilder.append(sb);
                        }

                        if (arraySize.containsKey(method.getName())) {
                            sb.append("[");                        // Array indicator
                            sb.append(":");                        // separator
                            sb.append(returnType.getName());       // type
                            sb.append(":");                        // separator
                            sb.append(method.getName());           // variableName
                            sb.append(":");                        // separator
                            sb.append(arraySize.get(method.getName()));  // Array size
                            sb.append(";");                        // member separator
                        } else {
                            sb.append("s");                         // scalar indicator
                            sb.append(":");                         // separator
                            sb.append(method.getReturnType());      // type
                            sb.append(":");                         // separator
                            sb.append(method.getName());            // var name
                            sb.append(";");                         // member separator
                        }
                        wasProcessed = true;
                    }
                }
                if (!wasProcessed) {
                    throw new RuntimeException("could not find method " + fieldName + " in class " + klass.getName());
                }
                currentLevel--;
            }

        } catch (ClassNotFoundException e) {
            IO.println("Error during materialization of DeviceType: " + e.getMessage());
        }

        sb.append(">");
        return this;
    }

    public String toText() {
        return this.representationBuilder.toString();
    }
}
