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
package optkl.ifacemapper;


import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.carriers.CommonCarrier;
import optkl.ifacemapper.accessor.AccessorInfo;
import optkl.ifacemapper.accessor.ValueType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class Schema<T extends MappableIface> {
    final public IfaceType rootIfaceType;
    public Class<T> iface;

    public static abstract class SchemaNode {
        public static final class Padding extends FieldNode {
            final public long len;

            Padding(IfaceType parent, long len) {
                super(parent, AccessorInfo.Key.NONE, "pad" + len);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "padding " + len + " bytes");
            }
        }
    }

    Schema(Class<T> iface, IfaceType rootIfaceType) {
        this.iface = iface;
        this.rootIfaceType = rootIfaceType;
    }

    public T allocate(CommonCarrier commonCarrier, int... boundLengths) {
        BoundSchema<?> boundSchema = new BoundSchema<>(this, boundLengths);
        T instance = (T) boundSchema.allocate(commonCarrier.lookup(), commonCarrier);
        MemorySegment memorySegment = MappableIface.getMemorySegment(instance);
        int[] count = new int[]{0};


        boundSchema.boundArrayFields().forEach(boundArrayFieldLayout -> {
            boundArrayFieldLayout.dimFields.forEach(dimLayout -> {
                long dimOffset = dimLayout.offset();
                int dim = boundLengths[count[0]++];
                if (dimLayout.field instanceof FieldNode.ArrayLen arrayLen) {
                    if (arrayLen.key.accessorType.equals(AccessorInfo.AccessorType.GETTER_AND_SETTER)) {
                        throw new IllegalStateException("You have a bound array dim field " + dimLayout.field.name + " controlling size of " + boundArrayFieldLayout.field.name + "[] which has a setter ");
                    }
                    if (arrayLen.type == Long.TYPE) {
                        memorySegment.set(ValueLayout.JAVA_LONG, dimOffset, dim);
                    } else if (arrayLen.type == Integer.TYPE) {
                        memorySegment.set(ValueLayout.JAVA_INT, dimOffset, dim);
                    } else {
                        throw new IllegalArgumentException("Unsupported array length type: " + arrayLen.type);
                    }
                }
            });
        });


        return instance;
    }

    public static <T extends Buffer> Schema<T> of(Class<T> iface, Consumer<IfaceType> parentFieldConsumer) {
        var struct = new IfaceType.Struct(null, (Class<MappableIface>) (Object) iface); // why the need for this?
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface, struct);
    }

    interface SchemaConstructor<T extends Buffer> {
        Class<T> iface();

        Schema<T> create();

    }

    record SchemaFromDSLMethod<T extends Buffer>(Class<T> iface, Method method,
                                                 CoreOp.FuncOp funcOp) implements SchemaConstructor<T> {
        static <T extends Buffer> SchemaFromDSLMethod<T> of(Class<T> iface) {
            try {
                Method schemaMethod = iface.getDeclaredMethod("schema");
                var possibleFuncOp = Op.ofMethod(schemaMethod);
                if (possibleFuncOp.isPresent()) {
                    return new SchemaFromDSLMethod<>(iface, schemaMethod, possibleFuncOp.get());
                } else {
                    System.out.println("Schema DSL method found for " + iface.getSimpleName() + " but no code model\n     Did you forget to annotate with @" + Reflect.class.getSimpleName() + "?");
                }
            } catch (NoSuchMethodException nsme) {
                // throw new RuntimeException(nsme);
            }
            return null;
        }

        static <T extends Buffer> Method methodOrNull(Class<T> iface, String name, Class<?>... types) {
            try {
                return iface.getDeclaredMethod(name, types);
            } catch (NoSuchMethodException noSuchMethodException) {
                return null;
            }
        }

        record Receiver(JavaOp.InvokeOp array, List<JavaOp.InvokeOp> args){
            public String arrayName() {
                return array.invokeDescriptor().name();
            }

        }

        //recurses
        private static Receiver consumedInvoke(JavaOp.InvokeOp firstArg, Op.Result result){// we have a call which is possibly being passed to another method  " array(length)"
            if (result.op() instanceof JavaOp.InvokeOp invokeOp){
                return new Receiver(invokeOp, List.of(firstArg));
            }else if (result.op() instanceof JavaOp.ConvOp convOp) {
                return ( consumedInvoke(firstArg, convOp.result().uses().iterator().next()));
            }else if (result.op() instanceof JavaOp.MulOp mul) {
                //So this works for single mul say array(width()*height);
                var maybeInvoke = mul.result().uses().iterator().next();
                var invoke = (JavaOp.InvokeOp)(maybeInvoke.op() instanceof JavaOp.ConvOp invokeOp?maybeInvoke.uses().iterator().next().op():maybeInvoke.op());
                var lhs = (JavaOp.InvokeOp)(((Op.Result)mul.operands().get(0)).op());
                var rhs = (JavaOp.InvokeOp)(((Op.Result)mul.operands().get(1)).op());
                var list = List.of(lhs,rhs );
                return new Receiver(invoke, list);
            }else{
                return null;
            }
        }
        /*
        Consider this strawman iface
        public interface S32Array2D extends Buffer {
          @Reflect default void schema(){array(width()*height());};
          Schema<S32Array2D> schema = Schema.of(S32Array2D.class);
          int width();
          int height();
          int array(long idx);
          void array(long idx, int i);
         }

         We are trying to replicate the manual

         Schema<S32Array2D> schema = Schema.of(S32Array2D.class, s32Array->s32Array
              .arrayLen("width","height").array("array"));
         */

        @Override
        public Schema<T> create() {
            // Just collect the unique names of all declared methods. In our straw man case we expect ("width","height","array" ..... a bunch of others )
            var declared = Arrays.stream(iface.getDeclaredMethods()).map(m -> m.getName()).collect(Collectors.toSet());
            // We need to track ones we have handled
            var handled = new HashSet<>();
            return Schema.of(iface, (schemaBuilder) -> {
                funcOp.elements()
                        .filter(ce -> ce instanceof JavaOp.InvokeOp)
                        .map(ce -> (JavaOp.InvokeOp) ce)
                        .forEach(invokeOp -> {
                            String name = invokeOp.invokeDescriptor().name();
                            if (name.equals("schema")){
                               //System.out.println("This could get recursive very quickly");
                            }else if (name.equals("pad")) {
                                if (invokeOp.operands().get(1) instanceof Op.Result result && result.op() instanceof CoreOp.ConstantOp constOp) {
                                    int padLength = switch (constOp.value()) {
                                        case Integer i -> i.intValue();
                                        case Long l -> l.intValue();
                                        default -> throw new RuntimeException("long or int const expected in pad()");
                                    };
                                   // System.out.println("...pad("+padLength+")");
                                    schemaBuilder.pad(padLength);
                                } else {
                                    throw new RuntimeException("pad(x) long or int const expected as operand 1");
                                }
                            } else if (declared.contains(name) && !handled.contains(name)){
                                var uses = invokeOp.result().uses();
                                if (uses.isEmpty()){
                                   // System.out.println("..."+name+"()");
                                    schemaBuilder.field(name);
                                    handled.add(name);
                                }else if (uses.size()==1){
                                    // assuming   @Reflect default void schema(){array(width()*height());};
                                    // The nature of the model is that we will find  methods 'width' and 'height' first and which are used to bind the dimensions of 'array'
                                    // So given that invokeOp -> width() and name == "width" we are interested in finding a method that consumes this (in our case  array(width()....))
                                    if (uses.iterator().next() instanceof Op.Result result) {// is might it a constant like we have only one, so probably a constant like length
                                        // we have a call which is possibly being passed to another method say we have width and we want to find invokeOp -> "array(width())"
                                        if (consumedInvoke(invokeOp, result) instanceof Receiver  receiver
                                              && receiver.args.stream().map(i->i.invokeDescriptor().name()).filter(declared::contains).toList() instanceof List<String> containedConsumers
                                               && receiver.args.size() == containedConsumers.size()){
                                            // in our case we expect Reciever (array,[width, height])
                                            schemaBuilder.arrayLen(containedConsumers).array(receiver.arrayName());
                                            handled.add(receiver.arrayName());
                                            containedConsumers.forEach(c->handled.add(c));
                                            //handled.add(consumerMethodName);
                                        } else {
                                            throw new IllegalStateException("Wait  a minute! the schema order is corrupt ");
                                        }
                                    } else {
                                        throw new IllegalStateException("how did we get here?");
                                    }
                                }else {
                                    throw new IllegalStateException("Schema order seems to use "+name+" in more than one binding!?");
                                }
                            }else{
                               // System.out.println("skipping "+name);
                            }

                        });
            });
        }
    }

    public static <T extends Buffer> Schema<T> of(Class<T> iface) {
        if (SchemaFromDSLMethod.of(iface) instanceof Schema.SchemaFromDSLMethod<T> schemaFromDSLMethod) {
            return schemaFromDSLMethod.create();
        }
        throw new RuntimeException("No schemaDSL or annotated fields, you will need to pass a builder for " + iface.getCanonicalName());
    }

    public void toText(Consumer<String> stringConsumer) {
        rootIfaceType.toText("", stringConsumer);
    }

    public static abstract sealed class IfaceType
            permits IfaceType.Union, IfaceType.Struct {
        public final IfaceType parent;
        public List<FieldNode> fields = new ArrayList<>();
        public List<IfaceType> ifaceTypes = new ArrayList<>();
        public Class<MappableIface> iface;

        <T extends FieldNode> T addField(T child) {
            fields.add(child);
            return child;
        }

        <T extends IfaceType> T addIfaceTypeNode(T child) {
            ifaceTypes.add(child);
            return child;
        }

        IfaceType(IfaceType parent, Class<MappableIface> iface) {
            this.parent = parent;
            this.iface = iface;
        }

        IfaceType getChild(Class<?> iface) {
            Optional<IfaceType> ifaceTypeNodeOptional = ifaceTypes
                    .stream()
                    .filter(n -> n.iface.equals(iface))
                    .findFirst();
            if (ifaceTypeNodeOptional.isPresent()) {
                return ifaceTypeNodeOptional.get();
            } else {
                throw new IllegalStateException("no supported iface type");
            }
        }

        public void visitTypes(int depth, Consumer<IfaceType> ifaceTypeNodeConsumer) {
            ifaceTypes.forEach(t -> t.visitTypes(depth + 1, ifaceTypeNodeConsumer));
            ifaceTypeNodeConsumer.accept(this);
        }


        public IfaceType struct(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
            parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Struct(this, (Class<MappableIface>) MapperUtil.typeOf(iface, name))));
            return this;
        }

        public IfaceType union(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
            parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Union(this, (Class<MappableIface>) MapperUtil.typeOf(iface, name))));
            return this;
        }

        public IfaceType field(String name) {
            var key = AccessorInfo.Key.of(iface, name);
            var typeOf = MapperUtil.typeOf(iface, name);
            addField(MapperUtil.isMemorySegment(typeOf)
                    ? new FieldNode.AddressField(this, key, (Class<MemorySegment>) typeOf, name)
                    : MapperUtil.isMappableIface(typeOf)
                    ? new FieldNode.IfaceField(this, key, this.getChild(typeOf), name)
                    : new FieldNode.PrimitiveField(this, key, typeOf, name));
            return this;
        }

        public IfaceType atomic(String name) {
            addField(new FieldNode.AtomicField(this, AccessorInfo.Key.of(iface, name), MapperUtil.typeOf(iface, name), name));
            return this;
        }

        public IfaceType pad(int len) {
            addField(new SchemaNode.Padding(this, len));
            return this;
        }

        public IfaceType field(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
            AccessorInfo.Key fieldKey = AccessorInfo.Key.of(iface, name);
            Class<MappableIface> fieldType = (Class<MappableIface>) MapperUtil.typeOf(iface, name);
            IfaceType structOrUnion = MapperUtil.isStruct(fieldType) ? new Struct(this, fieldType) : new Union(this, fieldType);
            addIfaceTypeNode(structOrUnion);
            addField(new FieldNode.IfaceField(this, fieldKey, structOrUnion, name));
            parentSchemaNodeConsumer.accept(structOrUnion);
            return this;
        }

        public IfaceType fields(String name1, String name2, Consumer<IfaceType> parentSchemaNodeConsumer) {
            AccessorInfo.Key fieldKey1 = AccessorInfo.Key.of(iface, name1);
            AccessorInfo.Key fieldKey2 = AccessorInfo.Key.of(iface, name2);
            if (!fieldKey1.equals(fieldKey2)) {
                throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different keys");
            }
            Class<MappableIface> structOrUnionType = (Class<MappableIface>) MapperUtil.typeOf(iface, name1);
            Class<?> fieldTypeCheck = MapperUtil.typeOf(iface, name2);
            if (!structOrUnionType.equals(fieldTypeCheck)) {
                throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different types");
            }
            IfaceType ifaceType = MapperUtil.isStruct(iface)
                    ? new Struct(this, structOrUnionType)
                    : new Union(this, structOrUnionType);
            addIfaceTypeNode(ifaceType);
            addField(new FieldNode.IfaceField(this, fieldKey1, ifaceType, name1));
            addField(new FieldNode.IfaceField(this, fieldKey2, ifaceType, name2));

            parentSchemaNodeConsumer.accept(ifaceType);
            return this;
        }

        public IfaceType fields(String... names) {
            for (var name : names) {
                field(name);
            }
            return this;
        }

        public IfaceType array(String name, int len) {
            AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
            var typeof = MapperUtil.typeOf(iface, name);
            addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                    ? new FieldNode.IfaceFixedArray(this, arrayKey, this.getChild(typeof), name, len)
                    : new FieldNode.PrimitiveFixedArray(this, arrayKey, typeof, name, len));
            return this;
        }

        public IfaceType array(String name, int len, Consumer<IfaceType> parentFieldConsumer) {
            AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
            Class<MappableIface> structOrUnionType = (Class<MappableIface>) MapperUtil.typeOf(iface, name);
            IfaceType ifaceType = MapperUtil.isStruct(iface)
                    ? new Struct(this, structOrUnionType)
                    : new Union(this, structOrUnionType);
            parentFieldConsumer.accept(ifaceType);
            addIfaceTypeNode(ifaceType);
            addField(new FieldNode.IfaceFixedArray(this, arrayKey, ifaceType, name, len));
            return this;
        }

        private IfaceType fieldControlledArray(String name, List<FieldNode.ArrayLen> arrayLenFields, int stride) {
            AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
            var typeOf = MapperUtil.typeOf(iface, name);
            addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                    ? new FieldNode.IfaceFieldControlledArray(this, arrayKey, this.getChild(typeOf), name, arrayLenFields, stride)
                    : new FieldNode.PrimitiveFieldControlledArray(this, arrayKey, typeOf, name, arrayLenFields, stride));
            return this;
        }

        public static class ArrayBuildState {
            IfaceType ifaceType;
            List<FieldNode.ArrayLen> arrayLenFields;
            long padding = 0;
            int stride = 1;

            public IfaceType array(String name) {
                return ifaceType.fieldControlledArray(name, arrayLenFields, stride);
            }

            public ArrayBuildState stride(int stride) {
                this.stride = stride;
                return this;
            }

            public ArrayBuildState pad(long padding) {
                this.padding = padding;
                var paddingField = new SchemaNode.Padding(ifaceType, padding);
                ifaceType.addField(paddingField);
                return this;
            }

            public IfaceType array(String name, Consumer<IfaceType> parentFieldConsumer) {
                Class<MappableIface> arrayType = (Class<MappableIface>) MapperUtil.typeOf(this.ifaceType.iface, name);
                IfaceType ifaceType = MapperUtil.isStruct(arrayType)
                        ? new Struct(this.ifaceType, arrayType)
                        : new Union(this.ifaceType, arrayType);
                parentFieldConsumer.accept(ifaceType);
                this.ifaceType.addIfaceTypeNode(ifaceType);
                this.ifaceType.fieldControlledArray(name, arrayLenFields, stride);


                return this.ifaceType;
            }

            ArrayBuildState(IfaceType ifaceType, List<FieldNode.ArrayLen> arrayLenFields) {
                this.ifaceType = ifaceType;
                this.arrayLenFields = arrayLenFields;
            }
        }

        public ArrayBuildState buildArray() {
            return new ArrayBuildState(this, null);
        }

        public ArrayBuildState arrayLen(List<String> arrayLenFieldNames) {
            List<FieldNode.ArrayLen> arrayLenFields = new ArrayList<>();
            arrayLenFieldNames.forEach(arrayLenFieldName -> {
                var arrayLenField = new FieldNode.ArrayLen(this, AccessorInfo.Key.of(iface, arrayLenFieldName), MapperUtil.typeOf(iface, arrayLenFieldName), arrayLenFieldName);
                addField(arrayLenField);
                arrayLenFields.add(arrayLenField);
            });
            return new ArrayBuildState(this, arrayLenFields);
        }

        public ArrayBuildState arrayLen(String... arrayLenFieldNames) {
            return arrayLen(List.of(arrayLenFieldNames));
        }


        public void toText(String indent, Consumer<String> stringConsumer) {
            stringConsumer.accept(indent);
            if (MapperUtil.isUnion(iface)) {
                stringConsumer.accept("union");
            } else if (MapperUtil.isStructOrBuffer(iface)) {
                stringConsumer.accept("struct");
            } else {
                throw new IllegalStateException("Expecting a union or a struct  ");
            }
            stringConsumer.accept(" " + iface + "{");
            stringConsumer.accept("\n");
            ifaceTypes.forEach(ifaceType -> {
                ifaceType.toText(indent + " TYPE: ", stringConsumer);
                stringConsumer.accept("\n");
            });
            fields.forEach(field -> {
                field.toText(indent + " FIELD: "+field.name+" ", stringConsumer);

                stringConsumer.accept("\n");
            });

            stringConsumer.accept(indent);
            stringConsumer.accept("}");
        }

        public static final class Struct extends IfaceType {
            Struct(IfaceType parent, Class<MappableIface> type) {
                super(parent, type);
            }
        }

        public static final class Union extends IfaceType {
            Union(IfaceType parent, Class<MappableIface> type) {
                super(parent, type);
            }
        }
    }

    public static abstract sealed class FieldNode
            permits FieldNode.AddressField, FieldNode.AbstractIfaceField, SchemaNode.Padding, FieldNode.AbstractPrimitiveField {
        public IfaceType parent;
        public final AccessorInfo.Key key;
        public final String name;

        FieldNode(IfaceType parent, AccessorInfo.Key key, String name) {
            this.parent = parent;
            this.key = key;
            this.name = name;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);


        public static final class AddressField extends FieldNode {
            Class<MemorySegment> type;

            AddressField(IfaceType parent, AccessorInfo.Key key, Class<MemorySegment> type, String name) {
                super(parent, key, name);
                this.type = type;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "address " + key + ":" + type);
            }
        }

        public static abstract sealed class AbstractPrimitiveField extends FieldNode
                permits ArrayLen, AtomicField, PrimitiveArray, PrimitiveField {
            public Class<?> type;

            AbstractPrimitiveField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, name);
                this.type = type;
            }
        }

        public static final class ArrayLen extends AbstractPrimitiveField {
            ArrayLen(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + key + ":" + type);
            }
        }

        public static final class AtomicField extends AbstractPrimitiveField {
            AtomicField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + key + ":" + type);
            }
        }


        public static final class PrimitiveField extends AbstractPrimitiveField {
            PrimitiveField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "primitive field " + key + ":" + type);
            }
        }


        public abstract static sealed class PrimitiveArray extends AbstractPrimitiveField permits PrimitiveFieldControlledArray, PrimitiveFixedArray {
            PrimitiveArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }
        }


        public static final class PrimitiveFixedArray extends PrimitiveArray {
            public int len;

            PrimitiveFixedArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name, int len) {
                super(parent, key, type, name);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "primitive array [" + len + "]");
            }
        }

        public static final class PrimitiveFieldControlledArray extends PrimitiveArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            PrimitiveFieldControlledArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, type, name);
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + type + "] whose dims are bound by (" );
                boolean first=true;
                for (ArrayLen arrayLenField : arrayLenFields) {
                    if (first){
                        first=false;
                    }else{
                        stringConsumer.accept(", ");
                    }
                    stringConsumer.accept(arrayLenField.name);
                }
                stringConsumer.accept(")");
            }
        }

        public static abstract sealed class AbstractIfaceField extends FieldNode
                permits FieldNode.IfaceArray, FieldNode.IfaceField {
            public IfaceType ifaceType;

            AbstractIfaceField(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, name);
                this.ifaceType = ifaceType;
            }
        }


        public static final class IfaceField extends AbstractIfaceField {

            IfaceField(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, ifaceType, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "mappable field " + key + ":" + ifaceType.iface);
            }
        }


        public abstract static sealed class IfaceArray extends AbstractIfaceField permits IfaceFieldControlledArray, IfaceFixedArray {
            IfaceArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, ifaceType, name);
            }
        }

        public static final class IfaceFixedArray extends IfaceArray {
            public int len;

            IfaceFixedArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name, int len) {
                super(parent, key, ifaceType, name);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [" + len + "]");
            }
        }

        public static final class IfaceFieldControlledArray extends IfaceArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            IfaceFieldControlledArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, ifaceType, name);
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + ifaceType.iface + "] where len defined by " + arrayLenFields);
            }
        }


    }
}
