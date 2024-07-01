package experiments;

import experiments.ifaceinvoketoptr.InvokeToPtr;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.constant.ClassDesc;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.SSA;
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
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class PtrDebugBackend extends DebugBackend {
    static boolean isBufferOrStruct(Class<?> possibleBufferOrStruct) {
        return Buffer.class.isAssignableFrom(possibleBufferOrStruct)
                || Buffer.StructChild.class.isAssignableFrom(possibleBufferOrStruct);
    }

    static FunctionType transformTypes(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        List<TypeElement> transformedTypeElements = new ArrayList<>();
        Optional<Class<?>> optionalMappableClass;
        TypeElement typeElement=null;
        for (Block.Parameter parameter : funcOp.parameters()) {
            typeElement = parameter.type();
            optionalMappableClass = mappableClass(lookup,typeElement);
            typeElement = (optionalMappableClass.isPresent())?new PtrType(getLayout(optionalMappableClass.get())): typeElement;
            transformedTypeElements.add(typeElement);
        }
        typeElement = funcOp.invokableType().returnType();
        optionalMappableClass = mappableClass(lookup,  typeElement);
        typeElement = (optionalMappableClass.isPresent())?new PtrType(getLayout(optionalMappableClass.get())): typeElement;
        return FunctionType.functionType(typeElement, transformedTypeElements);
    }

    static CoreOp.FuncOp transformInvokesToPtrs(MethodHandles.Lookup lookup,
                                                CoreOp.FuncOp ssaForm, FunctionType functionType) {
        return CoreOp.func(ssaForm.funcName(), functionType).body(funcBlock -> {
            funcBlock.transformBody(ssaForm.body(), funcBlock.parameters(), (builder, op) -> {
                if (op instanceof CoreOp.InvokeOp invokeOp && invokeOp.hasReceiver() && invokeOp.operands().size() == 1) {
                    Value receiver = invokeOp.operands().getFirst();
                    TypeElement receiverTypeElement = receiver.type();
                    if (mappableClass(lookup, receiverTypeElement).isPresent()) {
                        PtrToMemberOp ptrToMemberOp = new PtrToMemberOp( builder.context().getValue(receiver), invokeOp.invokeDescriptor().name());
                        Op.Result memberPtr = builder.op(ptrToMemberOp);
                        MemoryLayout memoryLayout = ptrToMemberOp.resultType().layout();
                        if (memoryLayout instanceof ValueLayout) {
                            builder.context().mapValue(invokeOp.result(), builder.op(new PtrLoadValue(memberPtr)));
                        } else {
                            builder.context().mapValue(invokeOp.result(), memberPtr);
                        }
                    } else {
                        builder.op(op);
                    }
                } else {
                    builder.op(op);
                }
                return builder;
            });
        });
    }

    static MemoryLayout getLayout(Class<?> clazz){
        try {
            return (MemoryLayout) clazz.getDeclaredField("LAYOUT").get(null);
        } catch (NoSuchFieldException e) {
            throw new RuntimeException(e);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    static Optional<Class<?>> mappableClass(MethodHandles.Lookup lookup, TypeElement typeElement) {
        try {
            return (typeElement instanceof JavaType jt
                    && jt.resolve(lookup) instanceof Class<?> c
                    && isBufferOrStruct(c)) ? Optional.of(c) : Optional.empty();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static final class PtrType implements TypeElement {
        static final String NAME = "ptr";
        final MemoryLayout layout;
        final JavaType referringType;

        public PtrType(MemoryLayout layout) {
            this.layout = layout;
            this.referringType = switch (layout) {
                case StructLayout _ -> JavaType.type(ClassDesc.of(layout.name().orElseThrow()));
                case AddressLayout _ -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
                case ValueLayout valueLayout -> JavaType.type(valueLayout.carrier());
                default -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
            };
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
            PtrType ptrType = (PtrType) o;
            return Objects.equals(layout, ptrType.layout);
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

    @OpFactory.OpDeclaration(PtrToMemberOp.NAME)
    public static final class PtrToMemberOp extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        final String simpleMemberName;
        final long memberOffset;
        final PtrType resultType;

        PtrToMemberOp(PtrToMemberOp that, CopyContext cc) {
            super(that, cc);
            this.simpleMemberName = that.simpleMemberName;
            this.memberOffset = that.memberOffset;
            this.resultType = that.resultType;
        }

        @Override
        public PtrToMemberOp transform(CopyContext cc, OpTransformer ot) {
            return new PtrToMemberOp(this, cc);
        }

        public PtrToMemberOp(Value ptr, String simpleMemberName) {
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
            MemoryLayout.PathElement memberPathElement = MemoryLayout.PathElement.groupElement(memberName);
            this.memberOffset = structLayout.byteOffset(memberPathElement);
            MemoryLayout memberLayout = structLayout.select(memberPathElement);
            // Remove any simple member name from the layout
            MemoryLayout ptrLayout = memberLayout instanceof StructLayout
                    ? memberLayout.withName(className(memberName))
                    : memberLayout.withoutName();
            this.resultType = new PtrType(ptrLayout);
        }

        // @@@ Change to return member index
        static String findMemberName(StructLayout sl, String simpleMemberName) {
            return sl.memberLayouts().stream()
                    .map(layout -> layout.name().orElseThrow())
                    .filter(name -> simpleMemberName(name).equals(simpleMemberName))
                    .findFirst().orElseThrow();
        }

        static Pattern regex = Pattern.compile("(.*)::(.*)");

        static String simpleMemberName(String memberName) {
            return regex.matcher(memberName) instanceof Matcher matcher && matcher.matches()
                    ? matcher.group(2) : memberName;
        }

        static String className(String memberName) {
            return regex.matcher(memberName) instanceof Matcher matcher && matcher.matches()
                    ? matcher.group(1) : null;
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
    }


    @OpFactory.OpDeclaration(PtrToMemberOp.NAME)
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
            this.resultType = ptrType.referringType();
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }

    }

    @OpFactory.OpDeclaration(PtrToMemberOp.NAME)
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
            if (!(ptrType.referringType().equals(v.type()))) {
                throw new IllegalArgumentException("Pointer reference type is not same as value to store type: "
                        + ptrType.referringType() + " " + v.type());
            }
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {

        var highLevelForm = kernelCallGraph.entrypoint.method.getCodeModel().orElseThrow();
        System.out.println("Initial code model");
        System.out.println(highLevelForm.toText());
        System.out.println("------------------");

        // highLevelForm.lower();
        CoreOp.FuncOp ssaInvokeForm = SSA.transform(highLevelForm);
        System.out.println("SSA form which maintains original invokes and args");
        System.out.println(ssaInvokeForm.toText());
        System.out.println("------------------");

        FunctionType functionType = transformTypes(MethodHandles.lookup(), ssaInvokeForm);
        System.out.println("SSA form with types transformed args");
        System.out.println(ssaInvokeForm.toText());
        System.out.println("------------------");

        CoreOp.FuncOp ssaPtrForm = transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
        System.out.println("SSA form with invokes replaced by ptrs");
        System.out.println(ssaPtrForm.toText());
    }
}
