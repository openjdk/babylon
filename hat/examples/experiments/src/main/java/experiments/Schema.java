package experiments;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class Schema<T extends Buffer>  {
    AbstractField.ParentField field;
    Class<T> iface;

    public GroupLayout layout(int ... arrayLengths) {
        LinkedList<Integer> lengthsToBind = new LinkedList<>();
        for (var i:arrayLengths){
            lengthsToBind.add(i);
        }
        List<MemoryLayout> memoryLayouts = new ArrayList<>();
        field.collectLayouts(memoryLayouts,lengthsToBind);
        return (GroupLayout) memoryLayouts.getFirst().withName(iface.getSimpleName());
    }

    static class AccessStyle {
        enum Mode {
            ROOT(false,false,false,false,false),
            PRIMITIVE_GETTER_AND_SETTER(false,true,false,true,true),
            PRIMITIVE_GETTER(false,true,false,false,true),
            PRIMITIVE_SETTER(false,true,false,true,false),
            IFACE_GETTER(false,false,true,false,true),
            PRIMITIVE_ARRAY_SETTER(true,true,false,true,false),
            PRIMITIVE_ARRAY_GETTER(true,true,false,false,true),
            PRIMITIVE_ARRAY_GETTER_AND_SETTER(true,true,false,true,true),
            IFACE_ARRAY_GETTER(true, false,true,false,true);
            boolean array;
            boolean primitive;
            boolean iface;
            boolean setter;
            boolean getter;
            Mode(boolean array, boolean primitive, boolean iface, boolean setter, boolean getter) {
                this.array=array;
                this.primitive=primitive;
                this.iface = iface;
                this.getter = getter;
                this.setter = setter;
            }
            static Mode of(Method m) {
                    Class<?> returnType = m.getReturnType();
                    Class<?>[] paramTypes = m.getParameterTypes();
                    if (paramTypes.length == 0 && returnType.isInterface()) {
                        return IFACE_GETTER;
                    }else if (paramTypes.length == 0 &&returnType.isPrimitive()) {
                        return PRIMITIVE_GETTER;
                    } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                        return PRIMITIVE_SETTER;
                    } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isInterface()) {
                        return IFACE_ARRAY_GETTER;
                    } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isPrimitive()) {
                        return PRIMITIVE_ARRAY_GETTER;
                    } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                            paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()){
                        return PRIMITIVE_ARRAY_SETTER;
                    } else {
                        System.out.println("skiping " + m);
                        return  null;
                    }
            }
        };
        Mode mode;
        Class<?> type;
        String name;
        AccessStyle(Mode mode, Class<?> type, String name) {
            this.mode = mode;
            this.type = type;
            this.name = name;
        }
        @Override
        public String toString() {
            return mode.name()+":"+type.getSimpleName()+":"+name;
        }
        static Class<?> methodToType(Method m){
            Class<?> returnType = m.getReturnType();
            Class<?>[] paramTypes = m.getParameterTypes();
            if (paramTypes.length == 0 && (returnType.isInterface() || returnType.isPrimitive())) {
               return returnType;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
               return paramTypes[0];
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && (returnType.isInterface()|| returnType.isPrimitive())) {
               return returnType;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()){
              return  paramTypes[1];
            } else {
                System.out.println("skipping " + m);
                return null;
            }
        }

        static AccessStyle of(Class<?> iface, String name) {
            AccessStyle accessStyle= new AccessStyle(null,null,name);
            Arrays.stream(iface.getDeclaredMethods()).filter(m -> m.getName().equals(name)).forEach(m -> {
                AccessStyle.Mode mode = AccessStyle.Mode.of(m);
                Class<?> type = methodToType(m);
                if (accessStyle.type == null){
                    accessStyle.type = type;
                }else if (!accessStyle.type.equals(type)){
                    throw new IllegalStateException("type mismatch for "+name);
                }
                if (accessStyle.mode == null){
                    accessStyle.mode = mode;
                }else  if (
                        (accessStyle.mode == Mode.PRIMITIVE_ARRAY_GETTER && mode == Mode.PRIMITIVE_ARRAY_SETTER)
                       || (accessStyle.mode == Mode.PRIMITIVE_ARRAY_SETTER && mode == Mode.PRIMITIVE_ARRAY_GETTER)
                ){
                    accessStyle.mode = Mode.PRIMITIVE_ARRAY_GETTER_AND_SETTER;
                } else  if (
                    (accessStyle.mode == Mode.PRIMITIVE_GETTER && mode == Mode.PRIMITIVE_SETTER)
                            || (accessStyle.mode == Mode.PRIMITIVE_SETTER && mode == Mode.PRIMITIVE_GETTER)
                ) {
                    accessStyle.mode = Mode.PRIMITIVE_GETTER_AND_SETTER;
                }else {
                    throw new IllegalStateException("mode mismatch for "+name);
                }
                /*
                Class<?> returnType = m.getReturnType();
                Class<?>[] paramTypes = m.getParameterTypes();
                if (m.getParameterCount() == 0 && returnType.isInterface()) {
                    if (accessStyle[0]!=null){
                        throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].mode);
                    }
                    accessStyle[0] = new AccessStyle(Mode.IFACE_GETTER, returnType, name);
                }else if (m.getParameterCount() == 0 &&returnType.isPrimitive()) {
                    if (accessStyle[0]!=null){
                       if (accessStyle[0].mode == Mode.PRIMITIVE_SETTER){
                           accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_GETTER_AND_SETTER, returnType, name);
                       }else{
                           throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].mode);
                       }
                    }else {
                        accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_GETTER, returnType, name);
                    }
                } else if (m.getParameterCount() == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                    if (accessStyle[0]!=null){
                        if (accessStyle[0].mode == Mode.PRIMITIVE_GETTER){
                            accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_GETTER_AND_SETTER, paramTypes[0], name);
                        }else{
                            throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].mode);
                        }
                    }else {
                        accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_SETTER, paramTypes[0], name);
                    }
                } else if (m.getParameterCount() == 1 && paramTypes[0] == Long.TYPE && returnType.isInterface()) {
                    if (accessStyle[0]!=null){
                        throw new IllegalStateException(name+" already dermined to to be "+accessStyle[0].mode);
                    }
                    accessStyle[0] = new AccessStyle(Mode.IFACE_ARRAY_GETTER, returnType, name);
                } else if (m.getParameterCount() == 1 && paramTypes[0] == Long.TYPE && returnType.isPrimitive()) {
                    if (accessStyle[0]!=null) {
                        if (accessStyle[0].mode == Mode.PRIMITIVE_ARRAY_SETTER) {
                            accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_ARRAY_GETTER_AND_SETTER, returnType, name);
                        } else {
                            throw new IllegalStateException(name + " already dermined to to be " + accessStyle[0].mode);
                        }
                    }else {
                        accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_ARRAY_GETTER, returnType, name);
                    }
                } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                        paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()){
                    if (accessStyle[0]!=null) {
                        if (accessStyle[0].mode == Mode.PRIMITIVE_ARRAY_GETTER) {
                            accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_ARRAY_GETTER_AND_SETTER, paramTypes[1], name);
                        } else {
                            throw new IllegalStateException(name + " already dermined to to be " + accessStyle[0].mode);
                        }
                    }else {
                        accessStyle[0] = new AccessStyle(Mode.PRIMITIVE_ARRAY_SETTER, paramTypes[1], name);
                    }
                } else {
                    System.out.println("skiping " + m);
                } */
            });
            if (accessStyle.type == null && accessStyle.mode==null) {
                accessStyle.type = iface;
                accessStyle.name="root";
                accessStyle.mode=Mode.ROOT;
            }
            return accessStyle;
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

    public static abstract class AbstractField {
        ParentField parent;

        AbstractField(ParentField parent) {
            this.parent = parent;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);

        public static class Padding extends AbstractField {
            int len;
            Padding(ParentField parent,  int len) {
                super(parent);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent+"padding "+len+" bytes");
            }
            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToAdd){
                memoryLayouts.add(MemoryLayout.paddingLayout(len));
            }
        }
        public static class ArrayLen extends AbstractField {
            AccessStyle accessStyle;
            ArrayLen(ParentField parent,   AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            @Override
            public void toText(String indent,  Consumer<String> stringConsumer) {
                stringConsumer.accept(indent+"arrayLen " +accessStyle);
            }
            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToAdd){
                if (accessStyle.type.isPrimitive()){
                    memoryLayouts.add(typeToLayout.get(accessStyle.type).withName(accessStyle.name));
                }else {
                    // the type is mapped in the parent.
                    throw new IllegalStateException("type of arraylen should be int or long");
                }
            }
        }
        public static class Field extends AbstractField {
            AccessStyle accessStyle;
            Field(ParentField parent, AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }
            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent+"field " +accessStyle);
            }
            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToAdd){
                if (accessStyle.type.isPrimitive()) {
                    memoryLayouts.add(typeToLayout.get(accessStyle.type).withName(accessStyle.name));
                }else{
                    ParentField layoutContainer = parent.children.stream().filter(c->c instanceof ParentField).map(c->(ParentField)c)
                            .filter(p->p.accessStyle.type.equals(accessStyle.type)).findFirst().get();

                    layoutContainer.collectLayouts(memoryLayouts,lengthsToAdd, accessStyle.name);
                 //  throw new IllegalStateException("handle case where type of field is not primitive");
                }
            }
        }
        public static abstract class ParentField extends AbstractField {
            List<AbstractField> children = new ArrayList<>();
            AccessStyle accessStyle;
            ParentField(ParentField parent, AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }
            public ParentField struct(String name, Consumer<ParentField> fb) {
                var struct = new Struct(this,  AccessStyle.of(accessStyle.type, name));
                children.add(struct);
                fb.accept(struct);
                return this;
            }

            public ParentField union(String name, Consumer<ParentField> fb) {
                var union = new Union(this,  AccessStyle.of(accessStyle.type, name));
                children.add(union);
                fb.accept(union);
                return this;
            }

            public ParentField field(String name) {
                children.add(new Field(this, AccessStyle.of(accessStyle.type,name)));
                return this;
            }
            public ParentField pad(int len) {
                children.add(new Padding(this, len));
                return this;
            }
            public ParentField field(String name, Consumer<ParentField>parentFieldConsumer) {
                AccessStyle newAccessStyle = AccessStyle.of(accessStyle.type,name);
                children.add(new Field(this, newAccessStyle));
                ParentField field;
                if (isStruct(newAccessStyle.type)){
                    field = new AbstractField.Struct(this,newAccessStyle);
                }else if (isUnion(newAccessStyle.type)) {
                    field = new AbstractField.Union(this, newAccessStyle);
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
                children.add(new Field(this, newAccessStyle1));
                children.add(new Field(this, newAccessStyle2));

                ParentField field;
                if (isStruct(newAccessStyle1.type)){
                    field = new AbstractField.Struct(this, newAccessStyle1);
                }else if (isUnion(newAccessStyle1.type)) {
                    field = new AbstractField.Union(this, newAccessStyle2);
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

            public static ParentField  createStructOrUnion(ParentField parent, AccessStyle accessStyle){
                if (isStruct(accessStyle.type)){
                    return new AbstractField.Struct(parent, accessStyle);
                }else if (isUnion(accessStyle.type)) {
                    return  new AbstractField.Union(parent, accessStyle);
                }
                    throw new IllegalArgumentException("Unsupported array type: " + accessStyle.type);

            }
            public ParentField array(String name, int len, Consumer<ParentField> parentFieldConsumer) {
                AccessStyle newAccessStyle = AccessStyle.of(accessStyle.type,name);
                ParentField field = createStructOrUnion(this, newAccessStyle);
                parentFieldConsumer.accept(field);
                children.add(field);
                children.add(new FixedArray(this, name, AccessStyle.of(accessStyle.type,name), len));
                return this;
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
                ParentField parentField;
                ArrayLen arrayLenField;

                ParentField array(String name) {
                    return parentField.fieldControlledArray(name, arrayLenField);
                }

                ParentField array(String name, Consumer<ParentField> parentFieldConsumer) {
                   AccessStyle newAccessStyle = AccessStyle.of(parentField.accessStyle.type,name);
                    parentField.fieldControlledArray(name, arrayLenField);
                    ParentField field = createStructOrUnion(parentField, newAccessStyle);
                    parentFieldConsumer.accept(field);
                    parentField.children.add(field);
                    return parentField;
                }

                ArrayBuildState(ParentField parentField, ArrayLen arrayLenField) {
                    this.parentField = parentField;
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

            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind) {

                collectLayouts(memoryLayouts,lengthsToBind,null);
            }


            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind, String name) {
                List<MemoryLayout> layouts = new ArrayList<>();
                children.forEach(c->{
                    if (!(c instanceof ParentField)) {
                        c.collectLayouts(layouts, lengthsToBind);
                    }
                });
                MemoryLayout memoryLayout = null;
                if (isUnion(accessStyle.type)) {
                    memoryLayout =(name != null && !name.isEmpty())
                            ? MemoryLayout.unionLayout(layouts.toArray(new MemoryLayout[0])).withName(name)
                            :MemoryLayout.unionLayout(layouts.toArray(new MemoryLayout[0]));
                }else if (isStructOrBuffer(accessStyle.type)){
                    memoryLayout =(name != null && !name.isEmpty())
                            ? MemoryLayout.structLayout(layouts.toArray(new MemoryLayout[0])).withName(name)
                            :MemoryLayout.structLayout(layouts.toArray(new MemoryLayout[0]));
                }else{
                    throw new IllegalStateException("Oh my ");
                }
                memoryLayouts.add(memoryLayout);
            }
            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (isUnion(accessStyle.type)) {
                    stringConsumer.accept("union");
                }else if(isStructOrBuffer(accessStyle.type)){
                    stringConsumer.accept("struct");
                }else{
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + accessStyle + "{");
                stringConsumer.accept("\n");
                children.forEach(c -> {
                    c.toText(indent+" ", stringConsumer);
                    stringConsumer.accept("\n");
                });
                 stringConsumer.accept(indent);
                stringConsumer.accept("}");
            }
        }
        public static class Struct extends ParentField {
            Struct(ParentField parent,  AccessStyle accessStyle) {
                super(parent, accessStyle);
            }
        }

        public static class Union extends ParentField {
            Union(ParentField parent,  AccessStyle accessStyle) {
                super(parent, accessStyle);
            }
        }

        public abstract static class Array extends AbstractField {
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
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent+"array [" + len + "]");
            }

            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                if (elementAccessStyle.type.isPrimitive()) {
                    memoryLayouts.add(MemoryLayout.sequenceLayout(len, typeToLayout.get(elementAccessStyle.type)).withName(elementAccessStyle.name));
                }else{
                    ParentField layoutContainer = parent.children.stream().filter(c->c instanceof ParentField).map(c->(ParentField)c)
                            .filter(p->p.accessStyle.type.equals(elementAccessStyle.type)).findFirst().get();

                    layoutContainer.collectLayouts(memoryLayouts,lengthsToBind, elementAccessStyle.name);
                   // throw new IllegalStateException("handle case where fixed array element type is not primitive");
                }
            }
        }

        public static class FlexArray extends Array {
            FlexArray(ParentField parent, String name, AccessStyle elementAccessStyle) {
                super(parent, name, elementAccessStyle);
            }
            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent+"array [?] ");
            }

            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                if (elementAccessStyle.type.isPrimitive()) {
                    memoryLayouts.add(MemoryLayout.sequenceLayout(0, typeToLayout.get(elementAccessStyle.type)).withName(elementAccessStyle.name));
                }else{
                    throw new IllegalStateException("handle case where flex array element type is not primitive");
                }
            }
        }

        public static class FieldControlledArray extends Array {
            ArrayLen arrayLen;

            FieldControlledArray(ParentField parent, String name, AccessStyle elementAccessStyle, ArrayLen arrayLen) {
                super(parent, name, elementAccessStyle);
                this.arrayLen = arrayLen;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent+elementAccessStyle.name+"["+elementAccessStyle+"] where len defined by " + arrayLen.accessStyle);
            }
            @Override
            void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind){
                if (elementAccessStyle.type.isPrimitive()) {
                    memoryLayouts.add(MemoryLayout.sequenceLayout(lengthsToBind.removeFirst(), typeToLayout.get(elementAccessStyle.type)).withName(elementAccessStyle.name));
                }else{
                    // We should find a Struct or Union matching the type in the parent
                    ParentField layoutContainer = parent.children.stream().filter(c->c instanceof ParentField).map(c->(ParentField)c)
                            .filter(p->p.accessStyle.type.equals(elementAccessStyle.type)).findFirst().get();

                    layoutContainer.collectLayouts(memoryLayouts,lengthsToBind, elementAccessStyle.name);
                  //  GroupLayout last =  (GroupLayout) memoryLayouts.getLast();
                   // System.out.println(last);
                   // last.withName(elementAccessStyle.name);
                   // System.out.println(last);
                }
            }
        }

        abstract void collectLayouts(List<MemoryLayout> memoryLayouts, LinkedList<Integer> lengthsToBind);
    }


    Schema(Class<T> iface, AbstractField.ParentField field) {
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


    public static <T extends Buffer>Schema<T> of(Class<T> iface, Consumer<AbstractField.ParentField> fb) {
        AbstractField.ParentField field = null;
        if (isBuffer(iface)){
            field = new AbstractField.Struct(null, AccessStyle.of(iface,iface.getSimpleName()));
        }else {
            throw new IllegalStateException("must be a Buffer");
        }
        fb.accept(field);
        return new Schema<T>(iface, field);
    }

    void toText(Consumer<String> stringConsumer){
        field.toText("",stringConsumer);
    }

}
