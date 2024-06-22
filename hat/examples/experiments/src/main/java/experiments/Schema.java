package experiments;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class Schema<T extends Buffer>  {
    Field.ParentField field;
    Class<T> iface;

    public GroupLayout layout(int ... arrayLengths) {

        LinkedList<Integer> lengthsToBind = new LinkedList<>();
        for (var i:arrayLengths){
            lengthsToBind.add(i);
        }
        List<MemoryLayout> memoryLayouts = new ArrayList<>();
        field.addLayout(memoryLayouts,lengthsToBind);
        return (GroupLayout) memoryLayouts.getFirst().withName(iface.getSimpleName());
    }



    static class AccessStyle {
        enum Style {ROOT,
            PRIMITIVE_GETTER_AND_SETTER, PRIMITIVE_GETTER, PRIMITIVE_SETTER,
            IFACE_GETTER,
            PRIMITIVE_ARRAY_SETTER, PRIMITIVE_ARRAY_GETTER,PRIMITIVE_ARRAY_GETTER_AND_SETTER,
            IFACE_ARRAY_GETTER;
        };
        Style style;
        Class<?> type;
        String name;
        AccessStyle(Style style, Class<?> type, String name) {
            this.style = style;
            this.type = type;
            this.name = name;
        }
        @Override
        public String toString() {
            return style.name()+":"+type.getSimpleName()+":"+name;
        }
        static AccessStyle of(Class<?> iface, String name) {
            AccessStyle[] accessStyle= new AccessStyle[]{null};
            Arrays.stream(iface.getDeclaredMethods()).filter(m -> m.getName().equals(name)).forEach(m -> {
                Class<?> returnType = m.getReturnType();
                Class<?>[] parameTypes = m.getParameterTypes();
                if (m.getParameterCount() == 0 && returnType.isInterface()) {
                    if (accessStyle[0]!=null){
                        throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].style);
                    }
                    accessStyle[0] = new AccessStyle(Style.IFACE_GETTER, returnType, name);
                }else    if (m.getParameterCount() == 0 &&returnType.isPrimitive()) {
                    if (accessStyle[0]!=null){
                       if (accessStyle[0].style == Style.PRIMITIVE_SETTER){
                           accessStyle[0] = new AccessStyle(Style.PRIMITIVE_GETTER_AND_SETTER, returnType, name);
                       }else{
                           throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].style);
                       }
                    }else {
                        accessStyle[0] = new AccessStyle(Style.PRIMITIVE_GETTER, returnType, name);
                    }
                } else if (m.getParameterCount() == 1 && parameTypes[0].isPrimitive() && returnType == Void.TYPE) {
                    if (accessStyle[0]!=null){
                        if (accessStyle[0].style == Style.PRIMITIVE_GETTER){
                            accessStyle[0] = new AccessStyle(Style.PRIMITIVE_GETTER_AND_SETTER, returnType, name);
                        }else{
                            throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].style);
                        }
                    }else {
                        accessStyle[0] = new AccessStyle(Style.PRIMITIVE_SETTER, returnType, name);
                    }
                } else if (m.getParameterCount() == 1 && parameTypes[0] == Long.TYPE && returnType.isInterface()) {
                    if (accessStyle[0]!=null){
                        throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].style);
                    }
                    accessStyle[0] = new AccessStyle(Style.IFACE_ARRAY_GETTER, returnType, name);
                } else if (m.getParameterCount() == 1 && parameTypes[0] == Long.TYPE && returnType.isPrimitive()) {
                    if (accessStyle[0]!=null) {
                        if (accessStyle[0].style == Style.PRIMITIVE_ARRAY_SETTER) {
                            accessStyle[0] = new AccessStyle(Style.PRIMITIVE_ARRAY_GETTER_AND_SETTER, returnType, name);
                        } else {
                            throw new IllegalStateException(name + " already dermined to to be " + accessStyle[0].style);
                        }
                    }else {
                        accessStyle[0] = new AccessStyle(Style.PRIMITIVE_ARRAY_GETTER, returnType, name);
                    }
                } else if (returnType == Void.TYPE && parameTypes.length == 2 &&
                        parameTypes[0] == Long.TYPE && parameTypes[1].isPrimitive()){
                    if (accessStyle[0]!=null) {
                        if (accessStyle[0].style == Style.PRIMITIVE_ARRAY_GETTER) {
                            accessStyle[0] = new AccessStyle(Style.PRIMITIVE_ARRAY_GETTER_AND_SETTER, parameTypes[1], name);
                        } else {
                            throw new IllegalStateException(name + " already dermined to to be " + accessStyle[0].style);
                        }
                    }else {
                        accessStyle[0] = new AccessStyle(Style.PRIMITIVE_ARRAY_SETTER, parameTypes[1], name);
                    }
                } else {
                    System.out.println("skiping " + m);
                }
            });
            if (accessStyle[0] == null) {
                accessStyle[0] = new AccessStyle(Style.ROOT,iface, "root");
            }
            return accessStyle[0];
        }
    }
    private static final Map<Class<?>, MemoryLayout> typeToLayout = new HashMap<>();
    static {
        typeToLayout.put(Integer.TYPE, JAVA_INT);
        typeToLayout.put(Float.TYPE, ValueLayout.JAVA_FLOAT);
        typeToLayout.put(Long.TYPE, ValueLayout.JAVA_LONG);
        typeToLayout.put(Double.TYPE, ValueLayout.JAVA_DOUBLE);
        typeToLayout.put(Character.TYPE, ValueLayout.JAVA_CHAR);
        typeToLayout.put(Short.TYPE, ValueLayout.JAVA_SHORT);
        typeToLayout.put(Byte.TYPE, ValueLayout.JAVA_BYTE);
        typeToLayout.put(Boolean.TYPE, ValueLayout.JAVA_BOOLEAN);
    }
    static boolean isBuffer(Class<?> clazz){
        return clazz.isInterface() && Buffer.class.isAssignableFrom(clazz);
    }
    static boolean isStruct(Class<?> clazz){
        return clazz.isInterface() && Buffer.StructChild.class.isAssignableFrom(clazz);
    }
    static boolean isStructOrBuffer(Class<?> clazz){
        return clazz.isInterface() && (Buffer.class.isAssignableFrom(clazz) || Buffer.StructChild.class.isAssignableFrom(clazz));
    }
    static boolean isUnion(Class<?> clazz){
        return  clazz.isInterface() && Buffer.UnionChild.class.isAssignableFrom(clazz);
    }
    static boolean isMappable(Class<?> clazz){
        return  isStruct(clazz)||isBuffer(clazz)||isUnion(clazz);
    }

    private static MemoryLayout typeToLayout(Class<?> clazz) {
        if (typeToLayout.containsKey(clazz)) {
            return typeToLayout.get(clazz);
        } else if (!isMappable(clazz)) {
            throw new IllegalStateException("What to do with mappable "+clazz);
        } else {
            throw new IllegalStateException("What to do with UNmappable "+clazz);
        }

    }

    public static abstract class Field {

        ParentField parent;

        Field(ParentField parent) {
            this.parent = parent;
        }

        public abstract void toText(int depth, Consumer<String> stringConsumer);

        public static class Padding extends Field {
            int len;
            Padding(ParentField parent, String name, int len) {
                super(parent);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("padding ");
            }
            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToAdd){
                memoryLayouts.add(MemoryLayout.paddingLayout(len));
                return memoryLayouts;
            }
        }
        public static class ArrayLen extends Field {
            AccessStyle accessStyle;
            ArrayLen(ParentField parent,   AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("arrayLen " +accessStyle);
            }
            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToAdd){
                memoryLayouts.add(typeToLayout(accessStyle.type).withName(accessStyle.name));
                return memoryLayouts;
            }
        }
        public static class Primitive extends Field {
            AccessStyle accessStyle;
            Primitive(ParentField parent,   AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("primitive " +accessStyle);
            }

            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToAdd){
                memoryLayouts.add(typeToLayout(accessStyle.type).withName(accessStyle.name));
                return memoryLayouts;
            }
        }

        public static abstract class ParentField extends Field  {
            List<Field> children = new ArrayList<>();
            AccessStyle accessStyle;

            ParentField(ParentField parent, AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                stringConsumer.accept("{\n");
                children.forEach(c -> {
                    c.toText(depth + 1, stringConsumer);
                    stringConsumer.accept("\n");
                });
                stringConsumer.accept("}");
            }


            public ParentField struct(String name, Consumer<ParentField> fb) {
                var struct = new Struct(this, name, AccessStyle.of(accessStyle.type, name));
                children.add(struct);
                fb.accept(struct);
                return this;
            }

            public ParentField union(String name, Consumer<ParentField> fb) {
                var union = new Union(this, name, AccessStyle.of(accessStyle.type, name));
                children.add(union);
                fb.accept(union);
                return this;
            }

            public ParentField field(String name) {
                children.add(new Primitive(this, AccessStyle.of(accessStyle.type,name)));
                return this;
            }

            public ParentField field(String name, Consumer<ParentField>parentFieldConsumer) {
                AccessStyle newAccessStyle = AccessStyle.of(accessStyle.type,name);
                children.add(new Primitive(this, newAccessStyle));


                ParentField field;
                if (isStruct(newAccessStyle.type)){
                    field = new Field.Struct(this, newAccessStyle.type.getSimpleName(),newAccessStyle);
                }else if (isUnion(newAccessStyle.type)) {
                    field = new Field.Union(this, newAccessStyle.type.getSimpleName(), newAccessStyle);
                }else{
                    throw new IllegalArgumentException("Unsupported array type: " + newAccessStyle.type);
                }
                parentFieldConsumer.accept(field);
                children.add(field);
                return this;
            }
            public ParentField fields(String name1,String name2, Consumer<ParentField>parentFieldConsumer) {
                AccessStyle newAccessStyle1 = AccessStyle.of(accessStyle.type,name1);
                AccessStyle newAccessStyle2 = AccessStyle.of(accessStyle.type,name2);
                children.add(new Primitive(this, newAccessStyle1));
                children.add(new Primitive(this, newAccessStyle2));

                ParentField field;
                if (isStruct(newAccessStyle1.type)){
                    field = new Field.Struct(this, newAccessStyle1.type.getSimpleName(),newAccessStyle1);
                }else if (isUnion(newAccessStyle1.type)) {
                    field = new Field.Union(this, newAccessStyle2.type.getSimpleName(), newAccessStyle2);
                }else{
                    throw new IllegalArgumentException("Unsupported array type: " + newAccessStyle2.type);
                }
                parentFieldConsumer.accept(field);
                children.add(field);
                return this;
            }
            public ParentField fields(String ...names) {
                for (var name:names){
                    field(name);
                }
                return this;
            }

            public ParentField array(String name, int len) {
                children.add(new FixedArray(this, name, AccessStyle.of(accessStyle.type,name), len));
                return this;
            }
            public ParentField array(String name, int len, Consumer<ParentField> parentFieldConsumer) {
                 AccessStyle newAccessStyle = AccessStyle.of(accessStyle.type,name);

                ParentField field;
                if (isStruct(newAccessStyle.type)){
                    field = new Field.Struct(this, newAccessStyle.type.getSimpleName(),newAccessStyle);
                }else if (isUnion(newAccessStyle.type)) {
                    field = new Field.Union(this, newAccessStyle.type.getSimpleName(), newAccessStyle);
                }else{
                    throw new IllegalArgumentException("Unsupported array type: " + newAccessStyle.type);
                }
                parentFieldConsumer.accept(field);
                children.add(field);
                children.add(new FixedArray(this, name, AccessStyle.of(accessStyle.type,name), len));
                return this;
                // builder.children.add(field);
            }


            private ParentField fieldControlledArray(String name, ArrayLen arrayLen) {
                children.add(new FieldControlledArray(this, name,  AccessStyle.of(accessStyle.type,name), arrayLen));
                return this;
            }

            public ParentField fieldControlledArray(String name, String arrayLenFieldName) {
                var arrayLen = new ArrayLen(this,  AccessStyle.of(accessStyle.type,arrayLenFieldName));
                children.add(arrayLen);
                return fieldControlledArray(name, arrayLen);
            }


            public static class ArrayBuildState {
                ParentField builder;
                ArrayLen arrayLenField;

                ParentField array(String name) {
                    return builder.fieldControlledArray(name, arrayLenField);
                }

                ParentField array(String name, Consumer<ParentField> parentFieldConsumer) {
                   AccessStyle newAccessStyle = AccessStyle.of(builder.accessStyle.type,name);
                    builder.fieldControlledArray(name, arrayLenField);
                    ParentField field;
                    if (isStruct(newAccessStyle.type)){
                        field = new Field.Struct(builder, builder.accessStyle.type.getSimpleName(),newAccessStyle);
                    }else if (isUnion(newAccessStyle.type)) {
                        field = new Field.Union(builder, builder.accessStyle.type.getSimpleName(), newAccessStyle);
                    }else{
                        throw new IllegalArgumentException("Unsupported array type: " + builder.accessStyle.type);
                    }

                    parentFieldConsumer.accept(field);
                    builder.children.add(field);
                    return builder;
                }

                ArrayBuildState(ParentField builder, ArrayLen arrayLenField) {
                    this.builder = builder;
                    this.arrayLenField = arrayLenField;
                }
            }

            public ArrayBuildState arrayLen(String arrayLenFieldName) {
                var arrayLenField = new ArrayLen(this,  AccessStyle.of(accessStyle.type,arrayLenFieldName));
                children.add(arrayLenField);
                return new ArrayBuildState(this, arrayLenField);
            }

            public void flexArray(String name) {
                children.add(new FlexArray(this, name, null));
            }


        }
        public static class Struct extends ParentField {
            Struct(ParentField parent, String name, AccessStyle accessStyle) {
                super(parent, accessStyle);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("struct " + accessStyle);
                super.toText(depth + 1, stringConsumer);
            }
            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                List<MemoryLayout> structLayouts = new ArrayList<>();
                children.forEach(c->{
                    c.addLayout(structLayouts, lengthsToBind);
                });
                memoryLayouts.add(MemoryLayout.structLayout(structLayouts.toArray(new MemoryLayout[0])));
                return memoryLayouts;
            }
        }

        public static class Union extends ParentField {
            Union(ParentField parent, String name, AccessStyle accessStyle) {
                super(parent, accessStyle);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("union name");
                super.toText(depth + 1, stringConsumer);
            }
            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                List<MemoryLayout> unionLayouts = new ArrayList<>();
                children.forEach(c->{
                    c.addLayout(unionLayouts, lengthsToBind);
                });
                memoryLayouts.add(MemoryLayout.unionLayout(unionLayouts.toArray(new MemoryLayout[0])));
                return memoryLayouts;
            }
        }

        public abstract static class Array extends Field {
          String name;
           AccessStyle elementAccessStyle;

            Array(ParentField parent, String name,  AccessStyle elementAccessStyle) {
                super(parent);
                this.name = name;
                this.elementAccessStyle = elementAccessStyle;
            }
        }

        public static class FixedArray extends Array {
            int len;

            FixedArray(ParentField parent, String name, AccessStyle elementAccessStyle, int len) {
                super(parent, name, elementAccessStyle);
                this.len = len;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("array [" + len + "]");
            }

            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                memoryLayouts.add(MemoryLayout.sequenceLayout(len, typeToLayout(elementAccessStyle.type)).withName(elementAccessStyle.name));
                return memoryLayouts;
            }
        }

        public static class FlexArray extends Array {

            FlexArray(ParentField parent, String name, AccessStyle elementAccessStyle) {
                super(parent, name, elementAccessStyle);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("array [?] ");
            }

            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                memoryLayouts.add(MemoryLayout.sequenceLayout(0, typeToLayout(elementAccessStyle.type)).withName(elementAccessStyle.name));
                return memoryLayouts;
            }
        }

        public static class FieldControlledArray extends Array {
            ArrayLen arrayLen;

            FieldControlledArray(ParentField parent, String name, AccessStyle elementAccessStyle, ArrayLen arrayLen) {
                super(parent, name, elementAccessStyle);
                this.arrayLen = arrayLen;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept(elementAccessStyle.name+"["+elementAccessStyle+"] where len defined by " + arrayLen.accessStyle);
            }
            @Override
            List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                memoryLayouts.add(MemoryLayout.sequenceLayout(lengthsToBind.removeFirst(), typeToLayout(elementAccessStyle.type)).withName(elementAccessStyle.name));
                return memoryLayouts;
            }

        }


        abstract List<MemoryLayout>  addLayout(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind);
    }


    Schema(Class<T> iface, Field.ParentField field) {
        this.iface = iface;
        this.field = field;
    }
    static  class BoundSchema<T extends Buffer>{
        Schema<T> schema;
        MemoryLayout memoryLayout;
        int [] boundLengths;
        T instance;
        BoundSchema(T instance, Schema<T> schema, MemoryLayout memoryLayout, int[] boundLengths) {
            this.instance = instance;
            this.schema = schema;
            this.memoryLayout = memoryLayout;
            this.boundLengths = boundLengths;
        }
    }

    BoundSchema<T> allocate(BufferAllocator bufferAllocator, int ...boundLengths){
        var layout = layout(boundLengths);
        System.out.println(layout);
        var segmentMapper = SegmentMapper.of(MethodHandles.lookup(), iface, layout);
        return new BoundSchema<T>(bufferAllocator.allocate(segmentMapper), this,layout,boundLengths);
    }


    public static <T extends Buffer>Schema<T> of(Class<T> iface, Consumer<Field.ParentField> fb) {

        Field.ParentField field = null;

        if (isBuffer(iface)){
            field = new Field.Struct(null, iface.getSimpleName(),AccessStyle.of(iface,iface.getSimpleName()));
        }else {
            throw new IllegalStateException("must be a Buffer");
        }
        fb.accept(field);
        return new Schema<T>(iface, field);
    }

    void toText(Consumer<String> stringConsumer){
        field.toText(0,stringConsumer);
    }

}
