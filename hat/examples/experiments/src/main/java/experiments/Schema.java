package experiments;

import hat.buffer.Buffer;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.UnionLayout;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class Schema  {

    public static abstract class Field<T extends MemoryLayout> {
        private static final Map<Class<?>, MemoryLayout> typeToLayout = new HashMap<>();
        static {
            typeToLayout.put(Integer.TYPE, ValueLayout.JAVA_INT);
            typeToLayout.put(Float.TYPE, ValueLayout.JAVA_FLOAT);
            typeToLayout.put(Long.TYPE, ValueLayout.JAVA_LONG);
            typeToLayout.put(Double.TYPE, ValueLayout.JAVA_DOUBLE);
            typeToLayout.put(Character.TYPE, ValueLayout.JAVA_CHAR);
            typeToLayout.put(Short.TYPE, ValueLayout.JAVA_SHORT);
            typeToLayout.put(Byte.TYPE, ValueLayout.JAVA_BYTE);
            typeToLayout.put(Boolean.TYPE, ValueLayout.JAVA_BOOLEAN);
        }

        private static MemoryLayout typeToLayout(Class<?> clazz) {
            if (typeToLayout.containsKey(clazz)) {
                return typeToLayout.get(clazz);
            } else if (clazz.isInterface()) {
                if (clazz.isAssignableFrom(Buffer.StructChild.class) || clazz.isAssignableFrom(Buffer.UnionChild.class)) {
                    try {
                        if (clazz.getDeclaredField("schema") instanceof java.lang.reflect.Field field) {
                           throw new IllegalStateException("schema ref");
                          //  return ((Schema) field.get(null)).field;
                        } else {
                            throw new RuntimeException("no Schema field found");
                        }
                    } catch (NoSuchFieldException e) {
                        throw new RuntimeException(e);
                  //  } catch (IllegalAccessException e) {
                   //     throw new RuntimeException(e);
                    }
                } else {
                    throw new IllegalStateException("no Struct or Union found for " + clazz);
                }
            } else {
                throw new IllegalStateException("wtft");
            }
        }

        FieldHolder parent;
        String name;
        T layout;

        Field(FieldHolder parent, String name) {
            this.parent = parent;
            this.name = name;
        }

        public abstract void toText(int depth, Consumer<String> stringConsumer);

        public static class Builder implements FieldHolder {
            Class<?> iface;
            List<Field<?>> children = new LinkedList<>();
            FieldHolder parent;


            public Class<?> iface() {
                return iface;
            }

            public List<Field<?>> children() {
                return children;
            }

            public FieldHolder parent() {
                return parent;
            }

            Class<?> getTypeForName(String name) {
                Class<?>[] clazz = new Class[]{null};
                Arrays.stream(iface().getDeclaredMethods()).filter(m -> m.getName().equals(name)).forEach(m -> {
                    if (m.getParameterCount() == 0 && m.getReturnType().isInterface()) {
                        clazz[0] = m.getReturnType();
                    }
                });
                return clazz[0];
            }

            MemoryLayout layout() {
                return null;
            }
            /*
                List<MemoryLayout> memoryLayouts = new ArrayList<>();
                Map<String, Integer> orderMap = new LinkedHashMap<>();
                Arrays.stream(order).forEach(s -> orderMap.put(s, orderMap.size())); // order[0] -> 0, order[1] -> r
                Set<String> done = new HashSet<>();
                Arrays.stream(bufferClass.getDeclaredMethods())
                        .filter(m -> orderMap.containsKey(m.getName()))                        //only methods named in array
                        .sorted(Comparator.comparingInt(lhs -> orderMap.get(lhs.getName()))) // sort by order in the array
                        .forEach(m -> {
                            String name = m.getName();
                            if (!done.contains(name)) {
                                MemoryLayout layout = null;
                                var rt = m.getReturnType();
                                if (rt == Void.TYPE) {
                                    if (m.getParameterCount() == 1) {
                                        layout = typeToLayout(m.getParameterTypes()[0]);
                                    } else if (m.getParameterCount() == 2) {
                                        throw new IllegalStateException("never");
                                    }
                                } else {
                                    layout = typeToLayout(rt);
                                }
                                if (layout instanceof ValueLayout) {
                                    memoryLayouts.add(layout.withName(name));
                                } else if (layout instanceof StructLayout) {
                                    memoryLayouts.add(layout.withName(name + "::struct"));
                                }
                                done.add(name);
                            }

                        });

                MemoryLayout.structLayout(memoryLayouts.toArray(new MemoryLayout[0])).withName(bufferClass.getName());
            } */

            public Builder struct(String name, Class<?> clazz, Consumer<Builder> fb) {
                Builder builder = new Builder(clazz);
                fb.accept(builder);
                children.add(new Struct(this, name, clazz, builder.children));
                return this;
            }

            public Builder struct(String name, Consumer<Builder> fb) {
                return struct(name, getTypeForName(name), fb);
            }

            public Builder struct(String name, FieldHolder schemaFieldHolder) {
                children.add(new Struct(this, name, getTypeForName(name), schemaFieldHolder.children()));
                return this;
            }

            public Builder struct(String name, Schema schema) {
                children.add(new Struct(this, name, getTypeForName(name), schema.field.children()));
                return this;
            }

            public Builder union(String name, Class<?> clazz, Consumer<Builder> fb) {
                Builder builder = new Builder(clazz);
                fb.accept(builder);
                children.add(new Struct(this, name, clazz, builder.children));
                return this;
            }

            public Builder union(String name, Consumer<Builder> fb) {
                return union(name, getTypeForName(name), fb);
            }

            public Builder union(String name, FieldHolder fieldHolder) {
                children.add(new Struct(this, name, getTypeForName(name), fieldHolder.children()));
                return this;
            }

            public Builder union(String name, Schema schema) {
                children.add(new Struct(this, name, getTypeForName(name), schema.field.children()));
                return this;
            }

            public Builder primitive(String name) {
                children.add(new Primitive(this, name));
                return this;
            }

            public Builder array(String name, int len) {
                children.add(new FixedArray(this, name, getTypeForName(name), len));
                return this;
            }

            public Builder fieldControlledArray(String name, Primitive primitive) {
                children.add(new FieldControlledArray(this, name, getTypeForName(name), primitive));
                return this;
            }

            public Builder fieldControlledArray(String name, String controllingFieldName) {
                var primitiveField = new Primitive(this, controllingFieldName);
                children.add(primitiveField);
                return fieldControlledArray(name, primitiveField);
            }

            public class ArrayLen {
                Builder builder;
                Primitive controllingField;

                Builder array(String name) {
                    return builder.fieldControlledArray(name, controllingField);
                }

                ArrayLen(Builder builder, Primitive controllingField) {
                    this.builder = builder;
                    this.controllingField = controllingField;
                }
            }

            public ArrayLen arrayLen(String controllingFieldName) {

                var primitiveField = new Primitive(this, controllingFieldName);
                children.add(primitiveField);
                return new ArrayLen(this, primitiveField);
            }

            public void flexArray(String name) {
                children.add(new FlexArray(this, name, null));
            }


            Builder(Class<?> iface) {
                this.iface = iface;
            }

            void toText(int depth, Consumer<String> stringConsumer) {
                children.stream().forEach(c -> c.toText(depth + 1, stringConsumer));
            }
        }

        public static interface FieldHolder {
            List<Field<?>> children();
        }

        public static class Padding<T extends PaddingLayout> extends Field<T> {
            Padding(FieldHolder parent, String name) {
                super(parent, name);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("padding ");
            }
        }

        public static class Primitive extends Field<ValueLayout> {

            Primitive(FieldHolder parent, String name) {
                super(parent, name);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("primitive " + name);
            }
        }

        public static abstract class StructOrUnion<T extends GroupLayout> extends Field<T> implements Field.FieldHolder {
            Class<?> iface;
            List<Field<?>> children;
            FieldHolder parent;

            public Class<?> iface() {
                return iface;
            }

            public List<Field<?>> children() {
                return children;
            }

            public FieldHolder parent() {
                return parent;
            }

            StructOrUnion(FieldHolder parent, String name, Class<?> iface, List<Field<?>> children) {
                super(parent, name);
                this.iface = iface;
                this.children = children;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                children.forEach(c -> {
                    c.toText(depth + 1, stringConsumer);
                    stringConsumer.accept("\n");
                });
            }

        }

        public static class Struct extends StructOrUnion<StructLayout> {
            Struct(FieldHolder parent, String name, Class<?> iface, List<Field<?>> schemaFields) {
                super(parent, name, iface, schemaFields);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("struct " + name);
                super.toText(depth + 1, stringConsumer);
            }
        }

        public static class Union extends StructOrUnion<UnionLayout> {
            Union(FieldHolder parent, String name, Class<?> iface, List<Field<?>> schemaFields) {
                super(parent, name, iface, schemaFields);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("union name");
                super.toText(depth + 1, stringConsumer);
            }
        }

        public abstract static class Array extends Field<SequenceLayout> {
            Class<?> elementClass;

            Array(FieldHolder parent, String name, Class<?> elementClass) {
                super(parent, name);
                this.elementClass = elementClass;
            }
        }

        public static class FixedArray extends Array {
            int len;

            FixedArray(FieldHolder parent, String name, Class<?> elementClass, int len) {
                super(parent, name, elementClass);
                this.len = len;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("array [" + len + "]");
            }
        }

        public static class FlexArray extends Array {

            FlexArray(FieldHolder parent, String name, Class<?> elementClass) {
                super(parent, name, elementClass);
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("array [?] ");
            }
        }

        public static class FieldControlledArray extends Array {
            Primitive primitive;

            FieldControlledArray(FieldHolder parent, String name, Class<?> elementClass, Primitive primitive) {
                super(parent, name, elementClass);
                this.primitive = primitive;
            }

            @Override
            public void toText(int depth, Consumer<String> stringConsumer) {
                IntStream.range(0, depth).forEach(_ -> stringConsumer.accept(" "));
                stringConsumer.accept("field controlled " + primitive.name + " " + name);
            }
        }
    }

    Field.StructOrUnion field;
    Schema(Field.StructOrUnion field) {
        this.field = field;
    }

    public static <T extends Buffer> Schema of(Class<?> iface, Consumer<Field.Builder> fb) {
        Field.Builder builder = new Field.Builder(iface);
        fb.accept(builder);
        if (Buffer.class.isAssignableFrom(iface) || Buffer.StructChild.class.isAssignableFrom(iface)){
            return new Schema(new Field.Struct(null, iface.getSimpleName(),iface,builder.children));
        }else if (Buffer.StructChild.class.isAssignableFrom(iface)){
            return new Schema(new Field.Union(null, iface.getSimpleName(),iface,builder.children));
        }
        throw new IllegalStateException("must be a Sturct, Union or Buffer");
    }

}
