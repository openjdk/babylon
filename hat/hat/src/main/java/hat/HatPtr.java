package hat;

import hat.buffer.Buffer;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.OpWrapper;

import java.lang.constant.ClassDesc;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class HatPtr {

        public static <T extends MappableIface> TypeElement convertToPtrTypeIfPossible(MethodHandles.Lookup lookup, TypeElement typeElement, BoundSchema<?> boundSchema, Schema.SchemaNode.IfaceTypeNode ifaceTypeNode) {
            if (getMappableClassOrNull(lookup, typeElement) instanceof Class<?> clazz){
               // MemoryLayout layout = boundSchema.getLayout(clazz);
                return new HatPtr.HatPtrType<>((Class<T>) clazz, getLayout((Class<T>) clazz));
            }else{
                return typeElement;
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
                return (typeElement instanceof JavaType jt
                        && jt.resolve(lookup) instanceof Class<?> possiblyMappableIface
                        && MappableIface.class.isAssignableFrom(possiblyMappableIface)) ? (Class<T>) possiblyMappableIface : null;
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }

        public static FunctionType transformTypes(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, Object ...args) {
            List<TypeElement> transformedTypeElements = new ArrayList<>();
            for (int i = 0; i < args.length; i++) {
                Block.Parameter parameter = funcOp.parameters().get(i);
                TypeElement parameterTypeElement=null;
                if (args[i] instanceof Buffer buffer) {
                    var boundSchema = Buffer.getBoundSchema(buffer);
                    parameterTypeElement=convertToPtrTypeIfPossible(lookup, parameter.type(), boundSchema,boundSchema.schema().rootIfaceTypeNode);
                }else{
                    parameterTypeElement =parameter.type();
                }
                transformedTypeElements.add(parameterTypeElement);
            }
            TypeElement returnTypeElement = convertToPtrTypeIfPossible(lookup, funcOp.invokableType().returnType(),null, null);
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
                            && OpWrapper.wrap(invokeOp) instanceof InvokeOpWrapper invokeOpWrapper
                            && invokeOpWrapper.hasOperands()
                            && invokeOpWrapper.isIfaceBufferMethod()
                            && invokeOpWrapper.getReceiver() instanceof Value iface // Is there a containing iface type Iface
                            && getMappableClassOrNull(lookup, iface.type()) != null
                    ) {
                        Value ifaceValue = builder.context().getValue(iface);     // ? Ensure we have an output value for the iface
                        HatPtr.HatPtrOp<T> hatPtrOp = new HatPtr.HatPtrOp<>(ifaceValue, invokeOpWrapper.name());         // Create ptrOp to replace invokeOp
                        Op.Result ptrResult = builder.op(hatPtrOp);// replace and capture the result of the invoke
                        if (invokeOpWrapper.operandCount() == 1) {                  // No args (operand(0)==containing iface))
                        /*
                          this turns into a load
                          interface Iface extends Buffer // or Buffer.StructChild
                              T foo();
                          }
                         */
                            if (hatPtrOp.resultType().layout() instanceof ValueLayout) { // are we pointing to a primitive
                                HatPtr.HatPtrLoadValue primitiveLoad = new HatPtr.HatPtrLoadValue(iface.type(), ptrResult);
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
                            if (hatPtrOp.resultType().layout() instanceof ValueLayout) { // are we pointing to a primitive
                                Value valueToStore = builder.context().getValue(invokeOpWrapper.operandNAsValue(1));
                                HatPtr.HatPtrStoreValue primitiveStore = new HatPtr.HatPtrStoreValue(iface.type(), ptrResult, valueToStore);
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


    public abstract  static class HatOp extends Op {
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
        public final static String NAME="hat.kc.op";
        public final String fieldName;
        public HatKernelContextOp(String fieldName, TypeElement typeElement, List<Value> operands) {
            super(NAME+"."+fieldName, typeElement, operands);
            this.fieldName=fieldName;
        }
        public HatKernelContextOp(String fieldName, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME+"."+fieldName, replacer.currentResultType(), replacer.currentOperandValues());
            this.fieldName=fieldName;

        }
        public HatKernelContextOp(String fieldName,TypeElement typeElement,FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME+"."+fieldName, typeElement, replacer.currentOperandValues());
            this.fieldName=fieldName;
        }
        public HatKernelContextOp(HatKernelContextOp that, CopyContext cc) {
            super(that, cc); this.fieldName = that.fieldName;
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatKernelContextOp(this, cc);
        }
    }

/*
    public abstract static  class HatPtrOp extends HatOp {

        public HatPtrOp(String name, TypeElement typeElement, List<Value> operands) {
            super(name, typeElement, operands);
        }
        public HatPtrOp(String name, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(name, replacer.currentResultType(), replacer.currentOperandValues());
        }
        public HatPtrOp(String name, TypeElement typeElement,FuncOpWrapper.WrappedOpReplacer replacer) {
            super(name, typeElement, replacer.currentOperandValues());
        }

        public HatPtrOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }
    }
    public final static class HatPtrStoreOp extends HatPtrOp {
        public final static String NAME="hat.ptr.store";
        public HatPtrStoreOp(TypeElement typeElement, List<Value> operands) {
            super(NAME, typeElement, operands);
        }
        public HatPtrStoreOp(FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME, replacer);
        }
        public HatPtrStoreOp(TypeElement typeElement,FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME, typeElement,replacer);
        }
        public HatPtrStoreOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrStoreOp(this, cc);
        }
    }
    public final static class HatPtrLoadOp extends HatPtrOp {
        public final static String NAME="hat.ptr.load";
        public HatPtrLoadOp(TypeElement typeElement, List<Value> operands) {
            super(NAME, typeElement, operands);
        }
        public HatPtrLoadOp(TypeElement typeElement, FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME, typeElement, replacer);
        }
        public HatPtrLoadOp( FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME,  replacer);
        }
        public HatPtrLoadOp(HatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrStoreOp(this, cc);
        }
    }

*/



    public abstract sealed static class HatType implements TypeElement permits HatPtrType {
        String name;
        HatType(String name){
            this.name = name;
        }
    }

    public static final class HatPtrType<T extends MappableIface> extends HatType {
        static final String NAME = "ptrType";
        public final Class<T> mappableIface;
        final MemoryLayout layout;
        final JavaType referringType;

        public HatPtrType(Class<T> mappableIface, MemoryLayout layout) {
            super(NAME);
            this.mappableIface = mappableIface;
            this.layout = layout;
            if (layout instanceof StructLayout structLayout){
                this.referringType =  JavaType.type(ClassDesc.of(structLayout.name().orElseThrow()));
            }else if  (layout instanceof ValueLayout valueLayout) {
                this.referringType = JavaType.type(valueLayout.carrier());
            }else    if (layout instanceof SequenceLayout sequenceLayout){
                this.referringType =  JavaType.type(ClassDesc.of(sequenceLayout.name().orElseThrow()));
            }else {
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


   /* public static final class HatPtrType extends HatType {
        static final String NAME = "hat.ptr";
        final TypeElement type;

        public HatPtrType(TypeElement type) {
            super(NAME);
            this.type = type;

        }

        public TypeElement type() {
            return type;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            HatPtrType hatPtrType = (HatPtrType) o;
            return Objects.equals(type, hatPtrType.type);
        }

        @Override
        public int hashCode() {
            return Objects.hash(type);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(type.externalize()));
        }

        @Override
        public String toString() {
            return externalize().toString();
        }
    } */



    @OpFactory.OpDeclaration(HatPtrOp.NAME)
    public static final class HatPtrOp<T extends MappableIface> extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        final HatPtr.HatPtrType<T> hatPtrType;
        final HatPtr.HatPtrType<T> resultType;
        final String simpleMemberName;
        final long memberOffset;


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
            if (ptr.type() instanceof HatPtr.HatPtrType<?> hatPtrType) {

                if (hatPtrType.layout() instanceof StructLayout structLayout) {
                    this.hatPtrType = (HatPtr.HatPtrType<T>) hatPtrType;
                    MemoryLayout.PathElement memberPathElement = MemoryLayout.PathElement.groupElement(simpleMemberName);
                    this.memberOffset = structLayout.byteOffset(memberPathElement);
                    MemoryLayout memberLayout = structLayout.select(memberPathElement);
                    this.resultType = new HatPtr.HatPtrType<>((Class<T>) hatPtrType.mappableIface, memberLayout);
                }else   if (hatPtrType.layout() instanceof SequenceLayout sequenceLayout) {
                    this.hatPtrType = (HatPtr.HatPtrType<T>) hatPtrType;
                    MemoryLayout.PathElement memberPathElement = MemoryLayout.PathElement.groupElement(simpleMemberName);
                    this.memberOffset = sequenceLayout.byteOffset(memberPathElement);
                    MemoryLayout memberLayout = sequenceLayout.select(memberPathElement);
                    this.resultType = new HatPtr.HatPtrType<>((Class<T>) hatPtrType.mappableIface, memberLayout);
                } else {
                    throw new IllegalArgumentException("Pointer type layout is not a struct layout: " + hatPtrType.layout());
                }
            } else {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
        }


        @Override
        public HatPtr.HatPtrType<T> resultType() {
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
            super(NAME, typeElement, ((HatPtr.HatPtrType<?>) ptr.type()).referringType(), List.of(ptr));
        }
        //  public HatPtrLoadValue(String name,TypeElement typeElement, FuncOpWrapper.WrappedOpReplacer replacer) {
        //    super(name, typeElement, replacer.currentResultType(), replacer.currentOperandValues());
        //  }
        public HatPtrLoadValue(TypeElement typeElement,FuncOpWrapper.WrappedOpReplacer replacer) {
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
      //  public HatPtrStoreValue(String name,TypeElement typeElement, FuncOpWrapper.WrappedOpReplacer replacer) {
        //    super(name, typeElement, replacer.currentResultType(), replacer.currentOperandValues());
      //  }
        public HatPtrStoreValue(TypeElement typeElement,FuncOpWrapper.WrappedOpReplacer replacer) {
            super(NAME, typeElement, replacer.currentResultType(), replacer.currentOperandValues());
        }


        @Override
        public HatPtrStoreValue transform(CopyContext cc, OpTransformer ot) {
            return new HatPtrStoreValue(typeElement, this, cc);
        }


    }






}
