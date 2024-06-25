package hat;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.HatData;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_CHAR;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public class Schema<T extends Buffer> {
    AbstractField.ParentField field;
    public Class<T> iface;

    static class LayoutCollector{
        private Stack<List<MemoryLayout>> stack = new Stack<>();
        private int[] arrayLengths;
        private List<AbstractField.FieldControlledArray> bound = new ArrayList<>();
        private int idx;
        LayoutCollector(int[] arrayLength){
            this.arrayLengths = arrayLength;
        }
        GroupLayout groupLayout = null;
        void pop(){
            if (stack.size()==1){
                groupLayout =  (GroupLayout) stack.peek().getFirst();;
            }
            stack.pop();
        }
        void push(){
            stack.push(new ArrayList<>());
        }
        void add(MemoryLayout memoryLayout){
            stack.peek().add(memoryLayout);
        }
        int getIdx(){
            return arrayLengths[idx++];
        }
        void add(AbstractField.FieldControlledArray boundArray){
            bound.add(boundArray);
        }
        public MemoryLayout[] array() {
            return stack.peek().toArray(new MemoryLayout[0]);
        }

    }
    public LayoutCollector collectLayouts(int... arrayLengths) {
        LayoutCollector layoutCollector = new LayoutCollector(arrayLengths);
        layoutCollector.push();
        field.collectLayouts(layoutCollector);
        layoutCollector.pop();
        return layoutCollector;
    }
    public GroupLayout layout(int... arrayLengths) {
        return collectLayouts(arrayLengths).groupLayout.withName(iface.getSimpleName());
    }

    static class AccessStyle {
        enum Mode {
            ROOT(false, false, false, false, false),
            PRIMITIVE_GETTER_AND_SETTER(false, true, false, true, true),
            PRIMITIVE_GETTER(false, true, false, false, true),
            PRIMITIVE_SETTER(false, true, false, true, false),
            IFACE_GETTER(false, false, true, false, true),
            PRIMITIVE_ARRAY_SETTER(true, true, false, true, false),
            PRIMITIVE_ARRAY_GETTER(true, true, false, false, true),
            PRIMITIVE_ARRAY_GETTER_AND_SETTER(true, true, false, true, true),
            IFACE_ARRAY_GETTER(true, false, true, false, true);
            boolean array;
            boolean primitive;
            boolean iface;
            boolean setter;
            boolean getter;

            Mode(boolean array, boolean primitive, boolean iface, boolean setter, boolean getter) {
                this.array = array;
                this.primitive = primitive;
                this.iface = iface;
                this.getter = getter;
                this.setter = setter;
            }

            /**
             * From the iface mapper
             * T foo()             getter iface|primitive  0 args                  , return T     returnType T
             * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
             * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
             * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
             *
             * @param m The reflected method
             * @return Class represeting the type this method is mapped to
             */
            static Mode of(Method m) {
                Class<?> returnType = m.getReturnType();
                Class<?>[] paramTypes = m.getParameterTypes();
                if (paramTypes.length == 0 && returnType.isInterface()) {
                    return IFACE_GETTER;
                } else if (paramTypes.length == 0 && returnType.isPrimitive()) {
                    return PRIMITIVE_GETTER;
                } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                    return PRIMITIVE_SETTER;
                } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isInterface()) {
                    return IFACE_ARRAY_GETTER;
                } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isPrimitive()) {
                    return PRIMITIVE_ARRAY_GETTER;
                } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                        paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                    return PRIMITIVE_ARRAY_SETTER;
                } else {
                    System.out.println("skiping " + m);
                    return null;
                }
            }

            Mode possiblyPromote(Mode mode) {
                if ((this.equals(PRIMITIVE_ARRAY_GETTER) && mode.equals(PRIMITIVE_ARRAY_SETTER))
                        || (this.equals(PRIMITIVE_ARRAY_SETTER) && mode.equals(PRIMITIVE_ARRAY_GETTER))) {
                    return Mode.PRIMITIVE_ARRAY_GETTER_AND_SETTER;
                } else if ((this.equals(PRIMITIVE_GETTER) && mode.equals(Mode.PRIMITIVE_SETTER))
                        || (this.equals(PRIMITIVE_SETTER) && mode.equals(Mode.PRIMITIVE_GETTER))) {
                    return Mode.PRIMITIVE_GETTER_AND_SETTER;
                } else {
                    return this;
                }
            }
        }

        Mode mode;
        Class<?> type;
        String name;
        List<Method> methods = new ArrayList<>();
        AccessStyle(Mode mode, Class<?> type, String name) {
            this.mode = mode;
            this.type = type;
            this.name = name;
        }

        @Override
        public String toString() {
            return mode.name() + ":" + type.getSimpleName() + ":" + name;
        }

        /**
         * From the iface mapper
         * T foo()             getter iface|primitive  0 args                  , return T     returnType T
         * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
         * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
         * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
         *
         * @param m The reflected method
         * @return Class represeting the type this method is mapped to
         */
        static Class<?> methodToType(Method m) {
            Class<?> returnType = m.getReturnType();
            Class<?>[] paramTypes = m.getParameterTypes();
            if (paramTypes.length == 0 && (returnType.isInterface() || returnType.isPrimitive())) {
                return returnType;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                return paramTypes[0];
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && (returnType.isInterface() || returnType.isPrimitive())) {
                return returnType;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                return paramTypes[1];
            } else {
                System.out.println("skipping " + m);
                return null;
            }
        }

        static AccessStyle of(Class<?> iface, String name) {
            AccessStyle accessStyle = new AccessStyle(null, null, name);
            var methods = iface.getDeclaredMethods();
            Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
                AccessStyle.Mode mode = AccessStyle.Mode.of(matchingMethod);
                Class<?> type = methodToType(matchingMethod);
                accessStyle.methods.add(matchingMethod);
                accessStyle.type = type;
                if (accessStyle.type == null) {
                    accessStyle.type = type;
                } else if (!accessStyle.type.equals(type)) {
                    throw new IllegalStateException("type mismatch for " + name);
                }
                //  The enum knows how to promote GETTER to GETTER_AND_SETTER if prev mode was GETTER and this SETTER and vice versa
                accessStyle.mode = (accessStyle.mode == null) ? mode : accessStyle.mode.possiblyPromote(mode);
            });
            if (accessStyle.type == null && accessStyle.mode == null) {
                accessStyle.type = iface;
                accessStyle.name = "root";
                accessStyle.mode = Mode.ROOT;
            }
            return accessStyle;
        }
    }

    static boolean isBuffer(Class<?> clazz) {
        return clazz.isInterface() && Buffer.class.isAssignableFrom(clazz);
    }

    static boolean isStruct(Class<?> clazz) {
        return clazz.isInterface() && Buffer.StructChild.class.isAssignableFrom(clazz);
    }

    static boolean isStructOrBuffer(Class<?> clazz) {
        return clazz.isInterface() && (Buffer.class.isAssignableFrom(clazz) || Buffer.StructChild.class.isAssignableFrom(clazz));
    }

    static boolean isUnion(Class<?> clazz) {
        return clazz.isInterface() && Buffer.UnionChild.class.isAssignableFrom(clazz);
    }

    static boolean isMappable(Class<?> clazz) {
        return isStruct(clazz) || isBuffer(clazz) || isUnion(clazz);
    }

    public static abstract class AbstractField {
        ParentField parent;

        AbstractField(ParentField parent) {
            this.parent = parent;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);

        public static class Padding extends AbstractField {
            int len;

            Padding(ParentField parent, int len) {
                super(parent);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "padding " + len + " bytes");
            }

            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                layoutCollector.add(MemoryLayout.paddingLayout(len));
            }
        }

        /**
         * Get a layout which describes the accessStyle.
         *
         * If accessStyle holds a primitive (int, float) then just map to JAVA_INT, JAVA_FLOAT value layouts
         * Otherwise we look through the parent's children.  Which should include a struct/union matching the type.
         * @param accessStyle
         * @param layoutCollector
         * @return
         */
      //  MemoryLayout getLayout(AccessStyle accessStyle, LinkedList<Integer> lengthsToBind, List<FieldControlledArray> boundArrays) {
        MemoryLayout getLayout(AccessStyle accessStyle, LayoutCollector layoutCollector) {
            MemoryLayout memoryLayout = null;
            if (accessStyle.type == Integer.TYPE) {
                memoryLayout = JAVA_INT;
            } else if (accessStyle.type == Float.TYPE) {
                memoryLayout = JAVA_FLOAT;
            } else if (accessStyle.type == Long.TYPE) {
                memoryLayout = JAVA_LONG;
            } else if (accessStyle.type == Double.TYPE) {
                memoryLayout = JAVA_DOUBLE;
            } else if (accessStyle.type == Short.TYPE) {
                memoryLayout = JAVA_SHORT;
            } else if (accessStyle.type == Character.TYPE) {
                memoryLayout = JAVA_CHAR;
            } else if (accessStyle.type == Byte.TYPE) {
                memoryLayout = JAVA_BYTE;
            } else if (accessStyle.type == Boolean.TYPE) {
                memoryLayout = JAVA_BOOLEAN;
            } else {
                ParentField o = parent.childFields.stream().filter(c -> c instanceof ParentField).map(c -> (ParentField) c)
                        .filter(p -> p.accessStyle.type.equals(accessStyle.type)).findFirst().get();
                layoutCollector.push();

                o.childFields.forEach(c -> {
                    if (!(c instanceof AbstractField.ParentField)) {

                        c.collectLayouts(layoutCollector);
                    }
                });

                MemoryLayout[] childLayoutsAsArray = layoutCollector.array();
                layoutCollector.pop();
                if (isUnion(o.accessStyle.type)) {
                    memoryLayout = MemoryLayout.unionLayout(childLayoutsAsArray);
                } else if (isStructOrBuffer(o.accessStyle.type)) {
                    memoryLayout = MemoryLayout.structLayout(childLayoutsAsArray);
                } else {
                    throw new IllegalStateException("Recursing through layout collections and came across  "+o.accessStyle.type);
                }
            }
            return memoryLayout;
        }

        public static class ArrayLen extends AbstractField {
            AccessStyle accessStyle;

            ArrayLen(ParentField parent, AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + accessStyle);
            }

            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                 layoutCollector.add(getLayout(accessStyle, layoutCollector).withName(accessStyle.name));
            }
        }

        public static class AtomicField extends AbstractField {
            AccessStyle accessStyle;

            AtomicField(ParentField parent, AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + accessStyle);
            }
            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                 layoutCollector.add(getLayout(accessStyle, layoutCollector).withName(accessStyle.name));
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
                stringConsumer.accept(indent + "field " + accessStyle);
            }
            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                layoutCollector.add(getLayout(accessStyle, layoutCollector).withName(accessStyle.name));
            }

        }

        public static abstract class ParentField extends AbstractField {
            private List<AbstractField> childFields = new ArrayList<>();
            Map<Class<?>,AbstractField> typeMap = new HashMap<>();
            AccessStyle accessStyle;
            <T extends AbstractField> T addChildField(T child) {
                childFields.add(child);
                return child;
            }
            ParentField(ParentField parent, AccessStyle accessStyle) {
                super(parent);
                this.accessStyle = accessStyle;
            }

            public ParentField struct(String name, Consumer<ParentField> fb) {
                var struct = new Struct(this, AccessStyle.of(accessStyle.type, name));
                addChildField(struct);
                typeMap.put(accessStyle.type,struct);
                fb.accept(struct);
                return this;
            }

            public ParentField union(String name, Consumer<ParentField> fb) {
                var union = new Union(this, AccessStyle.of(accessStyle.type, name));
                addChildField(union);
                typeMap.put(accessStyle.type,union);
                fb.accept(union);
                return this;
            }

            public ParentField field(String name) {
                addChildField(new Field(this, AccessStyle.of(accessStyle.type, name)));
                return this;
            }

            public ParentField atomic(String name) {
                addChildField(new AtomicField(this, AccessStyle.of(accessStyle.type, name)));
                return this;
            }

            public ParentField pad(int len) {
                addChildField(new Padding(this, len));
                return this;
            }

            public ParentField field(String name, Consumer<ParentField> parentFieldConsumer) {
                AccessStyle newAccessStyle = AccessStyle.of(accessStyle.type, name);
                addChildField(new Field(this, newAccessStyle));
                ParentField field;
                if (isStruct(newAccessStyle.type)) {
                    field = new AbstractField.Struct(this, newAccessStyle);
                } else if (isUnion(newAccessStyle.type)) {
                    field = new AbstractField.Union(this, newAccessStyle);
                } else {
                    throw new IllegalArgumentException("Unsupported field type: " + newAccessStyle.type);
                }
                parentFieldConsumer.accept(field);
                addChildField(field);
                typeMap.put(newAccessStyle.type,field);
                return this;
            }

            public ParentField fields(String name1, String name2, Consumer<ParentField> parentFieldConsumer) {
                AccessStyle newAccessStyle1 = AccessStyle.of(accessStyle.type, name1);
                AccessStyle newAccessStyle2 = AccessStyle.of(accessStyle.type, name2);
                addChildField(new Field(this, newAccessStyle1));
                addChildField(new Field(this, newAccessStyle2));

                ParentField field;
                if (isStruct(newAccessStyle1.type)) {
                    field = new AbstractField.Struct(this, newAccessStyle1);
                } else if (isUnion(newAccessStyle1.type)) {
                    field = new AbstractField.Union(this, newAccessStyle2);
                } else {
                    throw new IllegalArgumentException("Unsupported array type: " + newAccessStyle2.type);
                }
                parentFieldConsumer.accept(field);
                addChildField(field);
                return this;
            }

            public ParentField fields(String... names) {
                for (var name : names) {
                    field(name);
                }
                return this;
            }

            public ParentField array(String name, int len) {
                addChildField(new FixedArray(this, name, AccessStyle.of(accessStyle.type, name), len));
                return this;
            }

            public static ParentField createStructOrUnion(ParentField parent, AccessStyle accessStyle) {
                if (isStruct(accessStyle.type)) {
                    return new AbstractField.Struct(parent, accessStyle);
                } else if (isUnion(accessStyle.type)) {
                    return new AbstractField.Union(parent, accessStyle);
                }
                throw new IllegalArgumentException("Unsupported array type: " + accessStyle.type);

            }

            public ParentField array(String name, int len, Consumer<ParentField> parentFieldConsumer) {
                AccessStyle newAccessStyle = AccessStyle.of(accessStyle.type, name);
                ParentField field = createStructOrUnion(this, newAccessStyle);
                parentFieldConsumer.accept(field);
                addChildField(field);
                typeMap.put(accessStyle.type,field);
                addChildField(new FixedArray(this, name, AccessStyle.of(accessStyle.type, name), len));
                return this;
            }

            private ParentField fieldControlledArray(String name, ArrayLen arrayLen) {
                addChildField(new FieldControlledArray(this, name, AccessStyle.of(accessStyle.type, name), arrayLen));
                return this;
            }


            public static class ArrayBuildState {
                ParentField parentField;
                ArrayLen arrayLenField;

                public ParentField array(String name) {
                    return parentField.fieldControlledArray(name, arrayLenField);
                }

                public ParentField array(String name, Consumer<ParentField> parentFieldConsumer) {
                    AccessStyle newAccessStyle = AccessStyle.of(parentField.accessStyle.type, name);
                    parentField.fieldControlledArray(name, arrayLenField);
                    ParentField field = createStructOrUnion(parentField, newAccessStyle);
                    parentFieldConsumer.accept(field);
                    parentField.addChildField(field);
                    parentField.typeMap.put(parentField.accessStyle.type,field);
                    return parentField;
                }

                ArrayBuildState(ParentField parentField, ArrayLen arrayLenField) {
                    this.parentField = parentField;
                    this.arrayLenField = arrayLenField;
                }
            }

            public ArrayBuildState arrayLen(String arrayLenFieldName) {
                var arrayLenField = new ArrayLen(this, AccessStyle.of(accessStyle.type, arrayLenFieldName));
                addChildField(arrayLenField);
                return new ArrayBuildState(this, arrayLenField);
            }

            public void flexArray(String name) {
                addChildField(new FlexArray(this, name, null));
            }

            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                layoutCollector.push();
                childFields.forEach(c -> {
                    if (!(c instanceof ParentField)) {
                        c.collectLayouts(layoutCollector);
                    }
                });
                MemoryLayout memoryLayout = null;
                if (isUnion(accessStyle.type)) {
                    memoryLayout =MemoryLayout.unionLayout(layoutCollector.array());
                } else if (isStructOrBuffer(accessStyle.type)) {
                    memoryLayout = MemoryLayout.structLayout(layoutCollector.array());
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                layoutCollector.pop();
                layoutCollector.add(memoryLayout);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (isUnion(accessStyle.type)) {
                    stringConsumer.accept("union");
                } else if (isStructOrBuffer(accessStyle.type)) {
                    stringConsumer.accept("struct");
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + accessStyle + "{");
                stringConsumer.accept("\n");
                childFields.forEach(c -> {
                    c.toText(indent + " ", stringConsumer);
                    stringConsumer.accept("\n");
                });
                stringConsumer.accept(indent);
                stringConsumer.accept("}");
            }
        }

        public static class Struct extends ParentField {
            Struct(ParentField parent, AccessStyle accessStyle) {
                super(parent, accessStyle);
            }
        }

        public static class Union extends ParentField {
            Union(ParentField parent, AccessStyle accessStyle) {
                super(parent, accessStyle);
            }
        }

        public abstract static class Array extends AbstractField {
            String name;
            AccessStyle elementAccessStyle;

            Array(ParentField parent, String name, AccessStyle elementAccessStyle) {
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
                stringConsumer.accept(indent + "array [" + len + "]");
            }

            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                MemoryLayout elementLayout = getLayout(elementAccessStyle, layoutCollector).withName(elementAccessStyle.type.getSimpleName());;
                SequenceLayout sequenceLayout = MemoryLayout.sequenceLayout(len, elementLayout).withName(elementAccessStyle.name);
                layoutCollector.add(sequenceLayout);
            }
        }

        public static class FlexArray extends Array {
            FlexArray(ParentField parent, String name, AccessStyle elementAccessStyle) {
                super(parent, name, elementAccessStyle);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [?] ");
            }

            void collectLayouts(LayoutCollector layoutCollector) {
                MemoryLayout elementLayout = getLayout(elementAccessStyle, layoutCollector).withName(elementAccessStyle.type.getSimpleName());;
                SequenceLayout sequenceLayout = MemoryLayout.sequenceLayout(0, elementLayout).withName(elementAccessStyle.name);
                layoutCollector.add(sequenceLayout);
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
                stringConsumer.accept(indent + elementAccessStyle.name + "[" + elementAccessStyle + "] where len defined by " + arrayLen.accessStyle);
            }

            @Override
            void collectLayouts(LayoutCollector layoutCollector) {
                MemoryLayout elementLayout = getLayout(elementAccessStyle, layoutCollector).withName(elementAccessStyle.type.getSimpleName());;
                SequenceLayout sequenceLayout = MemoryLayout.sequenceLayout(layoutCollector.getIdx(), elementLayout).withName(elementAccessStyle.name);
                layoutCollector.add(sequenceLayout);
                layoutCollector.add(this);
            }
        }
      abstract void collectLayouts(LayoutCollector layoutCollector);
    }


    Schema(Class<T> iface, AbstractField.ParentField field) {
        this.iface = iface;
        this.field = field;
    }

    public static class BoundSchema<T extends Buffer> implements HatData {
        public Schema<T> schema;
        public GroupLayout layout;
        int[] boundLengths;


        BoundSchema(Schema<T> schema, GroupLayout layout, int[] boundLengths) {

            this.schema = schema;
            this.layout = layout;
            this.boundLengths = boundLengths;
        }
    }

    public T allocate(BufferAllocator bufferAllocator, int... boundLengths) {
        var layout = layout(boundLengths);

        var boundSchema = new BoundSchema<T>( this, layout, boundLengths);
        var segmentMapper = SegmentMapper.of(MethodHandles.lookup(),iface, layout, boundSchema);
        return bufferAllocator.allocate(segmentMapper);
    }


    public static <T extends Buffer> Schema<T> of(Class<T> iface, Consumer<AbstractField.ParentField> parentFieldConsumer) {
        AccessStyle accessStyle = AccessStyle.of(iface, iface.getSimpleName());
        var struct = new AbstractField.Struct(null, accessStyle);
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface, struct);
    }

    public void toText(Consumer<String> stringConsumer) {
        field.toText("", stringConsumer);
    }

}
