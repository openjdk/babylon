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

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

public class DeviceSchema<T extends DeviceType> {

    private Class<T> klass;
    private Set<String> fields = new HashSet<>();
    private Map<String, Integer> arraySize = new HashMap<>();
    private Map<Class<?>, Consumer<DeviceSchema<T>>> deps = new HashMap<>();
    private StringBuilder representationBuilder = new StringBuilder();

    public DeviceSchema(Class<T> klass) {
        this.klass = klass;
    }

    public static <T extends DeviceType> DeviceSchema<T> of(Class<T> klass, Consumer<DeviceSchema<T>> schemaBuilder) {
        DeviceSchema<T> deviceSchema =  new DeviceSchema<>(klass);
        schemaBuilder.accept(deviceSchema);
        deviceSchema.materialize();
        return deviceSchema;
    }

    public DeviceSchema<T> withField(String fieldName) {
        this.fields.add(fieldName);
        return this;
    }

    public DeviceSchema<T> withArray(String fieldName, int size) {
        this.fields.add(fieldName);
        arraySize.put(fieldName, size);
        return this;
    }

    public DeviceSchema<T> withDeps(Class<?> f16Class, Consumer<DeviceSchema<T>> depConsumer) {
        deps.put(f16Class, depConsumer);
        depConsumer.accept(this);
        return this;
    }

    private boolean isPrimitiveType(Class<?> type) {
        return type.isPrimitive();
    }

    private boolean isInterfaceType(Class<?> type) {
        return type.isInterface();
    }

    private DeviceSchema<T> materialize() {
        StringBuilder sb = new StringBuilder();
        materialize(sb, klass);
        return this;
    }

    private DeviceSchema<T> materialize(StringBuilder sb, Class<?> klass) {
        try {
            Class<?> aClass = Class.forName(klass.getName());
            sb.append("<");
            sb.append(klass.getName());
            sb.append(':');
            Field[] declaredFields = aClass.getDeclaredFields();
            Arrays.stream(declaredFields).forEach(field -> field.setAccessible(true));
            Method[] declaredMethods = aClass.getDeclaredMethods();
            for (String fieldName : fields) {
                for (Method method : declaredMethods) {
                    method.setAccessible(true);
                    if (method.getName().contains(fieldName)) {
                        if (arraySize.containsKey(method.getName())) {
                            Class<?> returnType = method.getReturnType();
                            if (!returnType.equals(void.class)) {
                                if (isInterfaceType(returnType)) {
                                    // inspect the dependency and add it on top of the builder
                                    StringBuilder depsBuilder = new StringBuilder();
                                    materialize(depsBuilder, returnType);
                                    sb = depsBuilder.append(sb);
                                }
                                sb.append("[");
                                sb.append(":");
                                sb.append(returnType.getName());
                                sb.append(":");
                                sb.append(method.getName());
                                sb.append(":");
                                sb.append(arraySize.get(method.getName()));
                                sb.append(";");
                                break;
                            }
                        } else {
                            // it is an scalar value
                            if (!method.getReturnType().equals(void.class)) {
                                sb.append("s");
                                sb.append(":");
                                sb.append(method.getReturnType());
                                sb.append(":");
                                sb.append(method.getName());
                                sb.append(";");
                                break;
                            }
                        }
                    }
                }
            }
        } catch (ClassNotFoundException e) {
            IO.println("Error during materialization of DeviceType: " + e.getMessage());
        }
        sb.append(">");
        this.representationBuilder = sb;
        return this;
    }

    public String toText() {
        return this.representationBuilder.toString();
    }
}
