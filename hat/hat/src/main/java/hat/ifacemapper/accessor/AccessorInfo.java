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

package hat.ifacemapper.accessor;

import hat.ifacemapper.Schema;
import hat.util.Result;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Set;

import static hat.ifacemapper.accessor.AccessorInfo.AccessorType.GETTER;
import static hat.ifacemapper.accessor.AccessorInfo.AccessorType.GETTER_AND_SETTER;
import static hat.ifacemapper.accessor.AccessorInfo.AccessorType.SETTER;
import static hat.ifacemapper.accessor.Cardinality.ARRAY;
import static hat.ifacemapper.accessor.Cardinality.SCALAR;
import static hat.ifacemapper.accessor.ValueType.INTERFACE;
import static hat.ifacemapper.accessor.ValueType.VALUE;


public record AccessorInfo(Key key,
                           Method method,
                           Class<?> type,
                           LayoutInfo layoutInfo,
                           long offset) {

    public AccessorInfo ofMethod(Key key, Method method, Class<?> type, LayoutInfo layoutInfo, long offset) {
        return new AccessorInfo(key, method, type, layoutInfo, offset);
    }


    public GroupLayout targetLayout() {
        return (GroupLayout) layoutInfo().arrayInfo()
                .map(ArrayInfo::elementLayout)
                .orElse(layoutInfo().layout());
    }

    public enum AccessorType {GETTER, SETTER,GETTER_AND_SETTER}

    /**
     * These are the various combinations that exists. Not all of them are
     * supported even though they can sometimes be expressed in interfaces and records.
     */
    public enum Key {
        //                                                 Mapping supported for
        SCALAR_VALUE_GETTER(SCALAR, VALUE, GETTER, EnumSet.of(INTERFACE)),
        SCALAR_VALUE_SETTER(SCALAR, VALUE, SETTER, EnumSet.of(INTERFACE)),
        SCALAR_VALUE_GETTER_AND_SETTER(SCALAR, VALUE, GETTER_AND_SETTER, EnumSet.of(INTERFACE)),
        SCALAR_INTERFACE_GETTER(SCALAR, INTERFACE, GETTER, EnumSet.of(INTERFACE)),
        ARRAY_VALUE_GETTER(ARRAY, VALUE, GETTER, EnumSet.of(INTERFACE)),
        ARRAY_VALUE_SETTER(ARRAY, VALUE, SETTER, EnumSet.of(INTERFACE)),
        ARRAY_VALUE_GETTER_AND_SETTER(ARRAY, VALUE, GETTER_AND_SETTER, EnumSet.of(INTERFACE)),
        ARRAY_INTERFACE_GETTER(ARRAY, INTERFACE, GETTER, EnumSet.of(INTERFACE));

        private final Cardinality cardinality;
        private final ValueType valueType;
        private final AccessorType accessorType;
        private final Set<ValueType> supportedFor;

        Key(Cardinality cardinality,
            ValueType valueType,
            AccessorType accessorType,
            Set<ValueType> supportedFor) {
            this.cardinality = cardinality;
            this.valueType = valueType;
            this.accessorType = accessorType;
            this.supportedFor = supportedFor;
        }

        public Cardinality cardinality() {
            return cardinality;
        }

        public ValueType valueType() {
            return valueType;
        }

        public AccessorType accessorType() {
            return accessorType;
        }

        public boolean isSupportedFor(ValueType type) {
            return supportedFor.contains(type);
        }

        public static Key of(Cardinality cardinality,
                             ValueType valueType,
                             AccessorType accessorType) {

            for (Key k : Key.values()) {
                if (k.cardinality == cardinality && valueType == k.valueType && accessorType == k.accessorType) {
                    return k;
                }
            }
            throw new InternalError("Should not reach here");
        }

        /**
         * From the iface mapper we get these mappings
         * <p>
         * T foo()             getter iface|primitive  0 args                  , return T     returnType T
         * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
         * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
         * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
         *
         * @param m The reflected method
         * @return Class represeting the type this method is mapped to
         */
        public static Key of(Method m) {
            Class<?> returnType = m.getReturnType();
            Class<?>[] paramTypes = m.getParameterTypes();
            if (paramTypes.length == 0 && returnType.isInterface()) {
                return SCALAR_INTERFACE_GETTER;
            } else if (paramTypes.length == 0 && returnType.isPrimitive()) {
                return SCALAR_VALUE_GETTER;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                return SCALAR_VALUE_SETTER;
            } else if (paramTypes.length == 1 && MemorySegment.class.isAssignableFrom(paramTypes[0]) && returnType == Void.TYPE) {
                return SCALAR_VALUE_SETTER;
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isInterface()) {
                return ARRAY_INTERFACE_GETTER;
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isPrimitive()) {
                return ARRAY_VALUE_GETTER;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                return ARRAY_VALUE_SETTER;
            } else {
                throw new IllegalStateException("no possible key for " + m);
            }
        }

        public static Key of(Class<?> iface, String name) {
            var methods = iface.getDeclaredMethods();
            Result<Key> modeResult = new Result<>();
            Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
                var thisMode = Key.of(matchingMethod);
                if (!modeResult.isPresent()) {
                    modeResult.of(thisMode);
                } else if ((modeResult.get().equals(ARRAY_VALUE_GETTER) && thisMode.equals(ARRAY_VALUE_SETTER))
                        || (modeResult.get().equals(ARRAY_VALUE_SETTER) && thisMode.equals(ARRAY_VALUE_GETTER))) {
                    modeResult.of(ARRAY_VALUE_GETTER_AND_SETTER);
                } else if ((modeResult.get().equals(SCALAR_VALUE_GETTER) && thisMode.equals(SCALAR_VALUE_SETTER))
                        || (modeResult.get().equals(SCALAR_VALUE_SETTER) && thisMode.equals(SCALAR_VALUE_GETTER))) {
                    modeResult.of(SCALAR_VALUE_GETTER_AND_SETTER);
                }
            });
            if (!modeResult.isPresent()) {
                throw new IllegalStateException("no possible key for " + iface + " " + name);
                // modeResult.of(Mode.PRIMITIVE_GETTER_AND_SETTER);
            }
            return modeResult.get();
        }
    }

}
