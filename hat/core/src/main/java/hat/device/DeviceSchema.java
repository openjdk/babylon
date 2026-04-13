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

import hat.codebuilders.C99HATKernelBuilder;
import optkl.IfaceValue;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class DeviceSchema<T extends NonMappableIface> {

    public interface Builder<T extends NonMappableIface>{
        DeviceSchema<T> deviceSchema();
        Class<?> clazz();
        List<NamedMember<T>> members();
        default Class<?> getReturnType(String fieldName){
            var opt = Arrays.stream(clazz().getMethods())
                    .filter(m->m.getName().equals(fieldName) && !m.getReturnType().equals(void.class))
                    .map(Method::getReturnType).findFirst();
            if (opt.isPresent()){
                return opt.get();
            }else {
                throw new RuntimeException("No return value for "+clazz()+" via name "+fieldName);
            }
        }

        default Builder<T> fields(String... fieldNames) {
            Stream.of(fieldNames).map(fieldName->new Field<>(this, getReturnType(fieldName), fieldName, new ArrayList<>()))
                    .forEach(field->members().add(field));
            return this;
        }

        default Builder<T> field(String fieldName) {
            return fields(fieldName);
        }

        default Builder<T> array(String fieldName, int size) {
            members().add(new Array<>(this, getReturnType(fieldName), fieldName,size, new ArrayList<>()));
            return this;
        }

        default Builder<T> array(String fieldName, int size, Consumer<Builder<T>> depConsumer) {
            depConsumer.accept(array(fieldName,size).members().getLast());
            return this;
        }
    }
    public interface NamedMember<T extends NonMappableIface> extends Builder<T> {
        Builder<T> parent();
        String name();
        @Override default DeviceSchema<T> deviceSchema() {
            return parent().deviceSchema();
        }
    }

    public record Root<T extends NonMappableIface>(DeviceSchema<T> deviceSchema, Class<T> clazz, List<NamedMember<T>> members)implements Builder<T>{};
    public record Field<T extends NonMappableIface>(Builder<T> parent, Class<?> clazz, String name, List<NamedMember<T>> members)implements NamedMember<T> {}
    public record Array<T extends NonMappableIface>(Builder<T> parent, Class<?> clazz, String name, int size, List<NamedMember<T>> members) implements NamedMember<T> {}
    public final Root<T> root;

    private DeviceSchema(Class<T> clazz,Consumer<Builder<T>> schemaBuilder) {
        this.root = new Root<>(this, clazz, new ArrayList<>());
        schemaBuilder.accept(this.root);
    }

    public static <T extends NonMappableIface> DeviceSchema<T> of(Class<T> clazz, Consumer<Builder<T>> schemaBuilder) {
        return  new DeviceSchema<>(clazz,schemaBuilder);
    }

    public <B extends C99HATKernelBuilder<B>> Root<T> typedef(B builder) {
        builder.typedefStruct(root.clazz(), _ ->
                root.members().forEach(m -> builder
                        .either(IfaceValue.class.isAssignableFrom(m.clazz()),
                                _ -> builder.suffix_t(m.clazz().getSimpleName()),
                                _ -> builder.type(m.clazz().getSimpleName())
                        )
                        .sp().id(m.name())
                        .when(m instanceof DeviceSchema.Array,
                                _ ->builder.sbrace(_ -> builder.intConst(((DeviceSchema.Array<?>) m).size()))
                        ).semicolon().nl()
                )
        );
        return root;
    }

    public static  <T extends NonMappableIface> DeviceSchema<T> getDeviceSchemaOrThrow(Class<NonMappableIface> clazz) {
        try {
            var schemaField = clazz.getDeclaredField("deviceSchema");
            schemaField.setAccessible(true);
            if (schemaField.get(schemaField) instanceof DeviceSchema<?> deviceSchema) {
                return (DeviceSchema<T>) deviceSchema;
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        throw new RuntimeException("No DeviceSchema in "+clazz);
    }

}
