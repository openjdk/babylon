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

import hat.types.F16;
import hat.codebuilders.C99HATCodeBuilder;

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
    private final C99HATCodeBuilder<?> representationBuilder = new C99HATCodeBuilder<>();
    private final Set<String> visited = new HashSet<>();

    private static final Map<Class<?>, String> specialTypes = new HashMap<>();

    static {
        specialTypes.put(F16.class, "half");
    }

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
        depConsumer.accept(this);
        materialize(representationBuilder, klass);
        return this;
    }

    private boolean isInterfaceType(Class<?> type) {
        return type.isInterface();
    }

    // Materialize methods are only reachable within this class.
    private void materialize() {
        materialize(representationBuilder, klass);
    }

    // The following method generates an intermediate representation in text form for each level
    // of the hierarchy.
    // It inspects each type and its members. If a member is also a non-primitive type
    // then it recursively inspect its inner members.
    // We keep track of all generated data structured by maintaining a visited set. Thus,
    // we avoid duplicates in the text form.
    private void materialize(C99HATCodeBuilder<?> builder, Class<?> klass) {
            Method[] declaredMethods = klass.getDeclaredMethods();
            builder.lt().identifier(klass.getName()).colon();
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
                            C99HATCodeBuilder<?> depsBuilder = new C99HATCodeBuilder<>();
                            depsBuilder.preformatted(builder.getText());
                            materialize(depsBuilder, returnType);
                            builder = depsBuilder;
                        }

                        String type = returnType.getName();
                        if (specialTypes.containsKey(klass)) {
                            type = specialTypes.get(klass);
                        }

                        if (arraySize.containsKey(method.getName())) {
                            builder.osbrace()                       // Array indicator
                                    .colon()                        // separator
                                    .typeName(type)                 // type
                                    .colon()                        // separator
                                    .identifier(method.getName())   // variableName
                                    .colon()                        // separator
                                    .identifier(Integer.toString(arraySize.get(method.getName()))) // Array size
                                    .semicolon();                   // member separator
                        } else {
                            builder.identifier("s")            // scalar indicator
                                    .colon()                        // separator
                                    .typeName(type)                 // type
                                    .colon()                        // separator
                                    .identifier(method.getName())   // var name
                                    .semicolon();                   // member separator
                        }
                        wasProcessed = true;
                    }
                }
                if (!wasProcessed) {
                    throw new RuntimeException("could not find method " + fieldName + " in class " + klass.getName());
                }
                currentLevel--;
            }
        builder.gt();
    }

    public String toText() {
        return this.representationBuilder.getText();
    }
}
