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
package hat;

import hat.buffer.Buffer;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.OpWrapper;

import java.lang.constant.ClassDesc;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.Block;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.op.OpFactory;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.PrimitiveType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static hat.ifacemapper.MapperUtil.isMappableIface;

public class OpsAndTypes {

    public static <T extends MappableIface> TypeElement convertToPtrTypeIfPossible(MethodHandles.Lookup lookup, TypeElement typeElement, BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        try {
            if (typeElement instanceof JavaType javaType
                    && javaType.resolve(lookup) instanceof Class<?> possiblyMappableIface
                    && isMappableIface(possiblyMappableIface)) {
                return new OpsAndTypes.HatPtrType<>((Class<T>) possiblyMappableIface, boundSchema.rootBoundSchemaNode());
            } else {
                return typeElement;
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }


    }

    public static <T extends MappableIface> MemoryLayout getLayout(Class<T> mappableIface) {
        try {
            return (MemoryLayout) mappableIface.getDeclaredField("LAYOUT").get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static <T extends MappableIface> Class<T> getMappableClassOrNull(MethodHandles.Lookup lookup, TypeElement typeElement) {
        try {
            return (typeElement instanceof JavaType javaType
                    && javaType.resolve(lookup) instanceof Class<?> possiblyMappableIface
                    && MappableIface.class.isAssignableFrom(possiblyMappableIface)) ? (Class<T>) possiblyMappableIface : null;
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static FunctionType transformTypes(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, Object... args) {
        List<TypeElement> transformedTypeElements = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            Block.Parameter parameter = funcOp.parameters().get(i);
            TypeElement parameterTypeElement = null;
            if (args[i] instanceof Buffer buffer) {
                var boundSchema = Buffer.getBoundSchema(buffer);
                parameterTypeElement = convertToPtrTypeIfPossible(lookup, parameter.type(), boundSchema, boundSchema.schema().rootIfaceType);
            } else {
                parameterTypeElement = parameter.type();
            }
            transformedTypeElements.add(parameterTypeElement);
        }
        TypeElement returnTypeElement = convertToPtrTypeIfPossible(lookup, funcOp.invokableType().returnType(), null, null);
        return FunctionType.functionType(returnTypeElement, transformedTypeElements);
    }

    public static <T extends MappableIface> CoreOp.FuncOp transformInvokesToPtrs(MethodHandles.Lookup lookup,
                                                                                 CoreOp.FuncOp ssaForm, FunctionType functionType) {
        return CoreOp.func(ssaForm.funcName(), functionType).body(funcBlock -> {
            funcBlock.transformBody(ssaForm.body(), funcBlock.parameters(), (builder, op) -> {
                /*
                   We are looking for
                      interface Iface extends Buffer // or Buffer.StructChild
                         T foo();
                         void foo(T foo);
                      }
                   Were T is either a primitive or a nested iface mapping and foo matches the field name
                 */

                if (op instanceof CoreOp.InvokeOp invokeOp
                        && OpWrapper.wrap(lookup, invokeOp) instanceof InvokeOpWrapper invokeOpWrapper
                        && invokeOpWrapper.hasOperands()
                        && invokeOpWrapper.isIfaceBufferMethod()
                        && invokeOpWrapper.getReceiver() instanceof Value iface // Is there a containing iface type Iface
                        && getMappableClassOrNull(lookup, iface.type()) != null
                ) {
                    Value hatPtrTypeValue = builder.context().getValue(iface);// ? Ensure we have an output value for the iface
                    // HatPtr.HatPtrType existingPtr = (HatPtr.HatPtrType)ifaceValue.type();
                    String fieldName = invokeOpWrapper.name();
                    //   BoundSchema.BoundSchemaNode boundSchemaNode = existingPtr.boundSchemaNode;
                    //    BoundSchema.FieldLayout layout= boundSchemaNode.getChild(fieldName);
                    OpsAndTypes.HatPtrOp<T> hatPtrOp = new OpsAndTypes.HatPtrOp<>(hatPtrTypeValue, fieldName);         // Create ptrOp to replace invokeOp
                    Op.Result ptrResult = builder.op(hatPtrOp);// replace and capture the result of the invoke
                    if (invokeOpWrapper.operandCount() == 1) {                  // No args (operand(0)==containing iface))
                        /*
                          this turns into a load
                          interface Iface extends Buffer // or Buffer.StructChild
                              T foo();
                          }
                         */
                        if (hatPtrOp.hatPtrType == null) { // are we pointing to a primitive
                            OpsAndTypes.HatPtrLoadValue primitiveLoad = new OpsAndTypes.HatPtrLoadValue(iface.type(), ptrResult);
                            Op.Result replacedReturnValue = builder.op(primitiveLoad);
                            builder.context().mapValue(invokeOp.result(), replacedReturnValue);
                        } else {                                                 // pointing to another iface mappable
                            builder.context().mapValue(invokeOp.result(), ptrResult);
                        }
                    } else if (invokeOpWrapper.operandCount() == 2) {
                         /*
                          This turns into a store
                          interface Iface extends Buffer // or Buffer.StructChild
                              void foo(T);
                          }
                         */
                        if (hatPtrOp.hatPtrType == null) { // are we pointing to a primitive
                            Value valueToStore = builder.context().getValue(invokeOpWrapper.operandNAsValue(1));
                            OpsAndTypes.HatPtrStoreValue primitiveStore = new OpsAndTypes.HatPtrStoreValue(iface.type(), ptrResult, valueToStore);
                            Op.Result replacedReturnValue = builder.op(primitiveStore);
                            builder.context().mapValue(invokeOp.result(), replacedReturnValue);
                        } else {                                                 // pointing to another iface mappable
                            builder.context().mapValue(invokeOp.result(), ptrResult);
                        }
                    } else {
                        builder.op(op);
                    }
                } else {
                    builder.op(op);
                }
                return builder; // why? oh why?
            });
        });
    }


    public abstract static class HatOp extends Op {
        private final TypeElement type;

        HatOp(String opName, TypeElement type, List<Value> operands) {
            super(opName, operands);
            this.type = type;
        }

        HatOp(HatOp that, CopyContext cc) {
            super(that, cc);
            this.type = that.type;
        }

        @Override
        public TypeElement resultType() {
            return type;
        }
    }


    public final static class HatKernelContextOp extends HatOp {
        public final static String NAME = "hat.kc.op";
        public final String fieldName;

        public HatKernelContextOp(String fieldName, TypeElement typeElement, List<Value> operands) {
            super(NAME + "." + fieldName, typeElement, operands);
            this.fieldName = fieldName;
        }

        public HatKernelContextOp(String fieldName, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME + "." + fieldName, replacer.currentResultType(), replacer.currentOperandValues());
            this.fieldName = fieldName;

        }

        public HatKernelContextOp(String fieldName, TypeElement typeElement, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME + "." + fieldName, typeElement, replacer.currentOperandValues());
            this.fieldName = fieldName;
        }

        public HatKernelContextOp(HatKernelContextOp that, CopyContext cc) {
            super(that, cc);
            this.fieldName = that.fieldName;
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatKernelContextOp(this, cc);
        }
    }


    public abstract sealed static class HatType implements TypeElement permits HatPtrType {
        String name;

        HatType(String name) {
            this.name = name;
        }
    }

    public static final class HatPtrType<T extends MappableIface> extends HatType {
        static final String NAME = "ptrType";

        public final BoundSchema.BoundSchemaNode<?> boundSchemaNode;
        public final Class<T> mappableIface;
        final MemoryLayout layout;
        final JavaType referringType;

        public HatPtrType(Class<T> mappableIface, BoundSchema.BoundSchemaNode<?> boundSchemaNode) {
            super(NAME);
            this.mappableIface = mappableIface;
            this.boundSchemaNode = boundSchemaNode;
            this.layout = boundSchemaNode.memoryLayouts.getFirst();
            if (layout instanceof StructLayout structLayout) {
                this.referringType = JavaType.type(boundSchemaNode.ifaceType.iface);
            } else if (layout instanceof ValueLayout valueLayout) {
                var referringTypeClassDesc = valueLayout.carrier();
                this.referringType = JavaType.type(referringTypeClassDesc);
            } else if (layout instanceof SequenceLayout sequenceLayout) {
                var referringTypeClassDesc = ClassDesc.of(sequenceLayout.name().orElseThrow());
                this.referringType = JavaType.type(referringTypeClassDesc);
            } else {
                throw new UnsupportedOperationException("Unsupported member layout: " + layout);
            }
        }

        public JavaType referringType() {
            return referringType;
        }

        public MemoryLayout layout() {
            return layout;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            HatPtrType<T> hatPtrType = (HatPtrType<T>) o;
            return Objects.equals(layout, hatPtrType.layout);
        }

        @Override
        public int hashCode() {
            return Objects.hash(layout);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(referringType.externalize()));
        }

        @Override
        public String toString() {
            return externalize().toString();
        }
    }


    @OpFactory.OpDeclaration(HatPtrOp.NAME)
    public static final class HatPtrOp<T extends MappableIface> extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        public final OpsAndTypes.HatPtrType<T> hatPtrType;
        public final TypeElement resultType;
        public final String simpleMemberName;
        public final long memberOffset;


        HatPtrOp(HatPtrOp<T> that, CopyContext cc) {
            super(that, cc);
            this.hatPtrType = that.hatPtrType;
            this.resultType = that.resultType;
            this.simpleMemberName = that.simpleMemberName;
            this.memberOffset = that.memberOffset;

        }

        @Override
        public HatPtrOp<T> transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrOp<T>(this, cc);
        }

        public HatPtrOp(Value ptr, String simpleMemberName) {
            super(NAME, List.of(ptr));
            this.simpleMemberName = simpleMemberName;
            if (ptr.type() instanceof OpsAndTypes.HatPtrType<?> hatPtrType) {
                this.hatPtrType = (OpsAndTypes.HatPtrType<T>) hatPtrType;
                var boundSchemaNode = hatPtrType.boundSchemaNode;
                var boundIfaceType = boundSchemaNode.ifaceType;
                if (boundIfaceType instanceof Schema.IfaceType.Struct || boundIfaceType instanceof Schema.IfaceType.Union) {
                    MemoryLayout.PathElement memberPathElement = MemoryLayout.PathElement.groupElement(simpleMemberName);
                    var layout = hatPtrType.layout();
                    if (layout instanceof GroupLayout) {
                        this.memberOffset = hatPtrType.layout().byteOffset(memberPathElement);
                        MemoryLayout memberLayout = hatPtrType.layout().select(memberPathElement);
                        System.out.println(memberLayout);
                        this.resultType = new OpsAndTypes.HatPtrType<>((Class<T>) boundIfaceType.iface, boundSchemaNode.children.get(0));
                    } else {
                        throw new IllegalStateException("Where did this layout come from");
                    }
                } else {
                    throw new IllegalArgumentException("Pointer type layout is not struct  union struct  " + hatPtrType.layout());
                }
            } else {
                this.resultType = ptr.type();
                this.memberOffset = 0;
                this.hatPtrType = null;
                // this.resultType = null;//new HatPtr.HatPtrType<>((Class<T>) hatPtrType.mappableIface,boundSchemaNode.children.get(0));
                //  throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
        }


        @Override
        public TypeElement resultType() {
            return resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", simpleMemberName);
            attrs.put(ATTRIBUTE_OFFSET, memberOffset);
            return attrs;
        }
    }


    public static abstract class HatPtrAccessValue extends Op {
        final String name;
        final TypeElement resultType;
        final TypeElement typeElement;

        HatPtrAccessValue(String name, TypeElement typeElement, HatPtrAccessValue that, CopyContext cc) {
            super(that, cc);
            this.name = name;
            this.typeElement = typeElement;
            this.resultType = that.resultType;
        }

        public HatPtrAccessValue(String name, TypeElement typeElement, TypeElement resultType, List<Value> values) {
            super(name, values);
            this.name = name;
            this.typeElement = typeElement;
            this.resultType = resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    @OpFactory.OpDeclaration(HatPtrLoadValue.NAME)
    public static final class HatPtrLoadValue extends HatPtrAccessValue {
        public static final String NAME = "ptr.load.value";

        HatPtrLoadValue(TypeElement typeElement, HatPtrLoadValue that, CopyContext cc) {
            super(NAME, typeElement, that, cc);
        }

        public HatPtrLoadValue(TypeElement typeElement, Value ptr) {
            super(NAME, typeElement, ((OpsAndTypes.HatPtrType<?>) ptr.type()).referringType(), List.of(ptr));
        }
        public HatPtrLoadValue(TypeElement typeElement, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME, typeElement, replacer.currentResultType(), replacer.currentOperandValues());
        }

        @Override
        public HatPtrLoadValue transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrLoadValue(typeElement, this, cc);
        }
    }

    @OpFactory.OpDeclaration(HatPtrStoreValue.NAME)
    public static final class HatPtrStoreValue extends HatPtrAccessValue {
        public static final String NAME = "ptr.store.value";

        public HatPtrStoreValue(TypeElement typeElement, HatPtrStoreValue that, CopyContext cc) {
            super(NAME, typeElement, that, cc);
        }

        public HatPtrStoreValue(TypeElement typeElement, PrimitiveType resultType, List<Value> operandValues) {
            super(NAME, typeElement, resultType, operandValues);
        }

        public HatPtrStoreValue(TypeElement typeElement, Value ptr, Value arg1) {
            super(NAME, typeElement, JavaType.VOID, List.of(ptr, arg1));
        }

        public HatPtrStoreValue(TypeElement typeElement, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME, typeElement, replacer.currentResultType(), replacer.currentOperandValues());
        }


        @Override
        public HatPtrStoreValue transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrStoreValue(typeElement, this, cc);
        }
    }

}
