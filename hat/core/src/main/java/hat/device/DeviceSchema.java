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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

public class DeviceSchema<T extends NonMappableIface> {
    public final Class<T> clazz;
    public interface Builder<T extends NonMappableIface>{
        DeviceSchema<T> deviceSchema();
        Class<?> clazz();
        List<Builder<T>> members();

        default Class<?> getReturnType(String fieldName){
            return Arrays.stream(clazz().getDeclaredMethods())
                    .filter(m->m.getName().equals(fieldName) && !m.getReturnType().equals(void.class))
                    .map(m->m.getReturnType()).findFirst().get();
        }

        default Builder<T> fields(String... fieldNames) {
            var fieldNameList = List.of(fieldNames);
            members().addAll(fieldNameList.stream().map(fieldName->new Field<>(this,getReturnType(fieldName),fieldName,new ArrayList<>())).toList());
            return this;
        }

        default Builder<T> field(String fieldName) {
            return fields(fieldName);
        }

      //  default Builder<T> array(String fieldName, int size) {
        //    members().add(new Array<>(this,getReturnType(fieldName),fieldName,size,new ArrayList<>()));
          //  return this;
       // }
        default Builder<T> array(String fieldName, int size) {
            var retType = getReturnType(fieldName);
            //if (!retType.equals(clazz)){
              //  throw new RuntimeException("Oh my");
            //}
            members().add(new Array<>(this,retType,fieldName,size,new ArrayList<>()));
            return this;
        }

        default Builder<T> array(String fieldName, int size, Consumer<Builder<T>> depConsumer) {
            array(fieldName,size);
            depConsumer.accept(members().getLast());
            return this;
        }
    }
    public interface NamedMember<T extends NonMappableIface> extends Builder<T> {
        Builder<T> parent();
        String name();
        @Override
        default DeviceSchema<T> deviceSchema() {
            return parent().deviceSchema();
        }
    }
    public interface NamedArrayMember<T extends NonMappableIface> extends NamedMember<T>{
        int size();
    }
    public record Root<T extends NonMappableIface>(DeviceSchema<T> deviceSchema, Class<T> clazz, List<Builder<T>> members)implements Builder<T>{};
    public record Field<T extends NonMappableIface>(Builder<T> parent, Class<?> clazz, String name, List<Builder<T>> members)implements NamedMember<T> {}
    public record Array<T extends NonMappableIface>(Builder<T> parent, Class<?> clazz, String name, int size, List<Builder<T>> members) implements NamedArrayMember<T> {}
    public final Root<T> root;
    public DeviceSchema(Class<T> clazz) {
        this.clazz = clazz;
        this.root = new Root<>(this,clazz, new ArrayList<>());
    }


    public static <T extends NonMappableIface> DeviceSchema<T> of(Class<T> clazz, Consumer<Builder<T>> schemaBuilder) {
        DeviceSchema<T> deviceSchema = new DeviceSchema<>(clazz);
        schemaBuilder.accept(deviceSchema.root);
        return deviceSchema;
    }
}
