/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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


package experiments;


import hat.buffer.Buffer;

import java.lang.constant.ClassDesc;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.code.*;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.runtime.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

public class LayoutExample {

    /*
       struct {
          StructTwo struct;
          int i;
       }
     */

        public interface Outer extends Buffer {


            interface Inner extends Buffer.StructChild  {
                int i();

                void i(int v);

                float f();

                void f(float v);

              //  Schema schema = Schema.of(Inner.class, b->b.primitive("i").primitive("f"));
            }

            Inner right();
            Inner left();
            int i();
            void i(int v);


            Schema schema = Schema.of(Outer.class, b->b
                            .struct("left", left->left
                                    .field("i")
                                    .field("f")
                            )
                           // .struct("right", Inner.schema)
                            .field("i")
            );
        }


    @CodeReflection
    static float m(Outer s1) {
        // StructOne* s1
        // s1 -> i
        int i = s1.i();
        // s1 -> *s2
        Outer.Inner s2 = s1.left();
        // s2 -> i
        i += s2.i();
        // s2 -> f
        float f = s2.f();
        return i + f;
    }


    public static void main(String[] args) {
        Optional<Method> om = Stream.of(LayoutExample.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("m"))
                .findFirst();

        Method m = om.orElseThrow();
        CoreOp.FuncOp f= m.getCodeModel().orElseThrow();
        f = SSA.transform(f);
        System.out.println(f.toText());
        FunctionType functionType = transformStructClassToPtr(MethodHandles.lookup(), f);
        System.out.println(f.toText());
        CoreOp.FuncOp pm = transformInvokesToPtrs(MethodHandles.lookup(), f, functionType);
        System.out.println(pm.toText());
    }
    static FunctionType transformStructClassToPtr(MethodHandles.Lookup l,
                                                CoreOp.FuncOp f) {
        List<TypeElement> pTypes = new ArrayList<>();
        for (Block.Parameter p : f.parameters()) {
            pTypes.add(transformStructClassToPtr(l, p.type()));
        }
        return FunctionType.functionType(
                transformStructClassToPtr(l, f.invokableType().returnType()), pTypes);
    }

    static CoreOp.FuncOp transformInvokesToPtrs(MethodHandles.Lookup l,
                                                CoreOp.FuncOp f, FunctionType functionType) {

        var builder= CoreOp.func(f.funcName(), functionType);

        var funcOp = builder.body(funcBlock -> {
            funcBlock.transformBody(f.body(), funcBlock.parameters(), (b, op) -> {
                if (op instanceof CoreOp.InvokeOp invokeOp
                        && invokeOp.hasReceiver()
                        && invokeOp.operands().getFirst() instanceof Value receiver) {
                    if (bufferOrBufferChildClass(l, receiver.type()) != null) {
                        Value ptr = b.context().getValue(receiver);
                        PtrToMember ptrToMemberOp = new PtrToMember(ptr, invokeOp.invokeDescriptor().name());
                        Op.Result memberPtr = b.op(ptrToMemberOp);

                        if (invokeOp.operands().size() == 1) {
                            // Pointer access and (possibly) value load
                            if (ptrToMemberOp.resultType().layout() instanceof ValueLayout) {
                                Op.Result v = b.op(new PtrLoadValue(memberPtr));
                                b.context().mapValue(invokeOp.result(), v);
                            } else {
                                b.context().mapValue(invokeOp.result(), memberPtr);
                            }
                        } else {
                            // @@@
                            // Value store
                            throw new UnsupportedOperationException();
                        }
                    } else {
                        b.op(op);
                    }
                } else {
                    b.op(op);
                }
                return b;
            });
        });
        return funcOp;
    }



    static boolean isBufferOrBufferChild(Class<?> maybeIface) {
        return  maybeIface.isInterface() && (
                Buffer.class.isAssignableFrom(maybeIface)
                        || Buffer.Child.class.isAssignableFrom(maybeIface)
        );

    }
    static Schema bufferOrBufferChildSchema(MethodHandles.Lookup l, Class<?> maybeBufferOrBufferChild) {
        if (isBufferOrBufferChild(maybeBufferOrBufferChild)) {
            throw new IllegalArgumentException();
        }
        Field schemaField;
        try {
            schemaField = maybeBufferOrBufferChild.getField("schema");
           return  (Schema)schemaField.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
    static Class<?> bufferOrBufferChildClass(MethodHandles.Lookup l, TypeElement t) {
        try {
            if (!(t instanceof JavaType jt) || !(jt.resolve(l) instanceof Class<?> c)) {
                return null;
            }
            return isBufferOrBufferChild(c) ? c : null;
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
    static TypeElement transformStructClassToPtr(MethodHandles.Lookup l, TypeElement type) {
        if (bufferOrBufferChildClass(l, type) instanceof Class<?> sc) {
            return new PtrType(bufferOrBufferChildSchema(l, sc));
        } else {
            return type;
        }
    }

    public static final class PtrType implements TypeElement {
        static final String NAME = "ptr";
        MemoryLayout layout;
        Schema schema;
        final JavaType returnType;

        public PtrType(MemoryLayout layout) {
            this.layout = layout;
            this.returnType = switch (layout) {
                case StructLayout _ -> JavaType.type(ClassDesc.of(layout.name().orElseThrow()));
                case AddressLayout _ -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
                case ValueLayout valueLayout -> JavaType.type(valueLayout.carrier());
                default -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
            };
        }
        public PtrType(Schema schema) {
            this.schema = schema;
            this.layout= null;//schema.layout();
            this.returnType = switch (layout) {
                case StructLayout _ -> JavaType.type(ClassDesc.of(layout.name().orElseThrow()));
                case AddressLayout _ -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
                case ValueLayout valueLayout -> JavaType.type(valueLayout.carrier());
                default -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
            };
        }

        public JavaType returnType() {
            return returnType;
        }

        public MemoryLayout layout() {
            return layout;
        }
        public Schema schema() {
            return schema;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            PtrType ptrType = (PtrType) o;
            return Objects.equals(layout, ptrType.layout);
        }

        @Override
        public int hashCode() {
            return Objects.hash(layout);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(returnType.externalize()));
        }

        @Override
        public String toString() {
            return externalize().toString();
        }
    }

    @OpFactory.OpDeclaration(PtrToMember.NAME)
    public static final class PtrToMember extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        public static final String ATTRIBUTE_NAME = "name";

        final String simpleMemberName;
        final long memberOffset;
        final PtrType resultType;

        PtrToMember(PtrToMember that, CopyContext cc) {
            super(that, cc);
            this.simpleMemberName = that.simpleMemberName;
            this.memberOffset = that.memberOffset;
            this.resultType = that.resultType;
        }

        @Override
        public PtrToMember transform(CopyContext cc, OpTransformer ot) {
            return new PtrToMember(this, cc);
        }

        public PtrToMember(Value ptr, String simpleMemberName) {
            super(NAME, List.of(ptr));
            this.simpleMemberName = simpleMemberName;

            if (!(ptr.type() instanceof PtrType ptrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            // @@@ Support group layout
            if (!(ptrType.layout() instanceof StructLayout structLayout)) {
                throw new IllegalArgumentException("Pointer type layout is not a struct layout: " + ptrType.layout());
            }

            // Find the actual member name from the simple member name
            String memberName = findMemberName(structLayout, simpleMemberName);
            MemoryLayout.PathElement p = MemoryLayout.PathElement.groupElement(memberName);
            this.memberOffset = structLayout.byteOffset(p);
            MemoryLayout memberLayout = structLayout.select(p);
            // Remove any simple member name from the layout
            MemoryLayout ptrLayout = memberLayout instanceof StructLayout
                    ? memberLayout.withName(className(memberName))
                    : memberLayout.withoutName();
            this.resultType = new PtrType(ptrLayout);
        }

        // @@@ Change to return member index
        static String findMemberName(StructLayout sl, String simpleMemberName) {
            for (MemoryLayout layout : sl.memberLayouts()) {
                String memberName = layout.name().orElseThrow();
                if (simpleMemberName(memberName).equals(simpleMemberName)) {
                    return memberName;
                }
            }
            throw new NoSuchElementException("No member found: " + simpleMemberName + " " + sl);
        }

        static String simpleMemberName(String memberName) {
            int i = memberName.indexOf("::");
            return i != -1
                    ? memberName.substring(i + 2)
                    : memberName;
        }

        static String className(String memberName) {
            int i = memberName.indexOf("::");
            return i != -1
                    ? memberName.substring(0, i)
                    : null;
        }

        @Override
        public PtrType resultType() {
            return resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", simpleMemberName);
            attrs.put(ATTRIBUTE_OFFSET, memberOffset);
            return attrs;
        }

        public String simpleMemberName() {
            return simpleMemberName;
        }

        public long memberOffset() {
            return memberOffset;
        }

        public Value ptrValue() {
            return operands().get(0);
        }
    }


    @OpFactory.OpDeclaration(PtrToMember.NAME)
    public static final class PtrAddOffset extends Op {
        public static final String NAME = "ptr.add.offset";

        PtrAddOffset(PtrAddOffset that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public PtrAddOffset transform(CopyContext cc, OpTransformer ot) {
            return new PtrAddOffset(this, cc);
        }

        public PtrAddOffset(Value ptr, Value offset) {
            super(NAME, List.of(ptr, offset));

            if (!(ptr.type() instanceof PtrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            if (!(offset.type() instanceof PrimitiveType pt && pt.equals(JavaType.LONG))) {
                throw new IllegalArgumentException("Offset value is not of primitve long type: " + offset.type());
            }
        }

        @Override
        public TypeElement resultType() {
            return ptrValue().type();
        }

        public Value ptrValue() {
            return operands().get(0);
        }

        public Value offsetValue() {
            return operands().get(1);
        }
    }

    @OpFactory.OpDeclaration(PtrToMember.NAME)
    public static final class PtrLoadValue extends Op {
        public static final String NAME = "ptr.load.value";

        final JavaType resultType;

        PtrLoadValue(PtrLoadValue that, CopyContext cc) {
            super(that, cc);
            this.resultType = that.resultType;
        }

        @Override
        public PtrLoadValue transform(CopyContext cc, OpTransformer ot) {
            return new PtrLoadValue(this, cc);
        }

        public PtrLoadValue(Value ptr) {
            super(NAME, List.of(ptr));

            if (!(ptr.type() instanceof PtrType ptrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            if (!(ptrType.layout() instanceof ValueLayout)) {
                throw new IllegalArgumentException("Pointer type layout is not a value layout: " + ptrType.layout());
            }
            this.resultType = ptrType.returnType();
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }

        public Value ptrValue() {
            return operands().get(0);
        }
    }

    @OpFactory.OpDeclaration(PtrToMember.NAME)
    public static final class PtrStoreValue extends Op {
        public static final String NAME = "ptr.store.value";

        PtrStoreValue(PtrStoreValue that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public PtrStoreValue transform(CopyContext cc, OpTransformer ot) {
            return new PtrStoreValue(this, cc);
        }

        public PtrStoreValue(Value ptr, Value v) {
            super(NAME, List.of(ptr));

            if (!(ptr.type() instanceof PtrType ptrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            if (!(ptrType.layout() instanceof ValueLayout)) {
                throw new IllegalArgumentException("Pointer type layout is not a value layout: " + ptrType.layout());
            }
            if (!(ptrType.returnType().equals(v.type()))) {
                throw new IllegalArgumentException("Pointer reference type is not same as value to store type: "
                        + ptrType.returnType() + " " + v.type());
            }
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        public Value ptrValue() {
            return operands().get(0);
        }
    }
}

