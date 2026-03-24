package jdk.incubator.code.runtime;

import java.lang.classfile.ClassBuilder;
import java.lang.classfile.ClassElement;
import static java.lang.classfile.ClassFile.ACC_PRIVATE;
import static java.lang.classfile.ClassFile.ACC_STATIC;
import static java.lang.classfile.ClassFile.ACC_SYNCHRONIZED;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.constantpool.ConstantPoolBuilder;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.classfile.constantpool.MethodRefEntry;
import java.lang.classfile.constantpool.NameAndTypeEntry;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.internal.access.JavaLangInvokeAccess;
import jdk.internal.access.SharedSecrets;

import java.lang.constant.ClassDesc;
import static java.lang.constant.ConstantDescs.*;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.CallSite;
import java.lang.invoke.LambdaConversionException;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleInfo;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;
import java.util.List;
import java.util.Objects;

/**
 * Provides runtime support for creating reflectable lambdas. A reflectable lambda is a lambda whose
 * code model can be inspected using {@link Op#ofLambda(Object)}.
 * @see LambdaMetafactory
 * @see Op#ofLambda(Object)
 */
public class ReflectableLambdaMetafactory {

    private static final String NAME_METHOD_QUOTED = "__internal_quoted";
    private static final String QUOTED_FIELD_NAME = "quoted";
    private static final String MODEL_FIELD_NAME = "model";

    private ReflectableLambdaMetafactory() {
        // nope
    }

    /**
     * Metafactory used to create a reflectable lambda.
     * <p>
     * The functionality provided by this metafactory is identical to that in
     * {@link LambdaMetafactory#metafactory(Lookup, String, MethodType, MethodType, MethodHandle, MethodType)}
     * with one important difference: this metafactory expects the provided name to be encoded in the following form:
     * <code>
     *     lambdaName=opMethodName
     * </code>
     * The {@code lambdaName} part of the name is passed to the regular metafactory method, along with all the other
     * parameters unchanged. The {@code opMethod} part of the name is used to locate a method in the
     * {@linkplain Lookup#lookupClass() lookup class} associated with the provided lookup. This method is expected
     * to accept no parameters and return a {@link Op}, namely the model of the reflectable lambda.
     * <p>
     * This means the clients can pass the lambda returned by this factory to the {@link Op#ofLambda(Object)} method,
     * to access the code model of the lambda expression dynamically.
     *
     * @param caller The lookup
     * @param interfaceMethodName The name of the method to implement.
     *                            This is encoded in the format described above.
     * @param factoryType The expected signature of the {@code CallSite}.
     * @param interfaceMethodType Signature and return type of method to be
     *                            implemented by the function object.
     * @param implementation A direct method handle describing the implementation
     *                       method which should be called at invocation time.
     * @param dynamicMethodType The signature and return type that should
     *                          be enforced dynamically at invocation time.
     * @return a CallSite whose target can be used to perform capture, generating
     *         a reflectable lambda instance implementing the interface named by {@code factoryType}.
     *         The code model for such instance can be inspected using {@link Op#ofLambda(Object)}.
     *
     * @throws LambdaConversionException If, after the lambda name is decoded,
     *         the parameters of the call are invalid for
     *         {@link LambdaMetafactory#metafactory(Lookup, String, MethodType, MethodType, MethodHandle, MethodType)}
     * @throws NullPointerException If any argument is {@code null}.
     *
     * @see LambdaMetafactory#metafactory(Lookup, String, MethodType, MethodType, MethodHandle, MethodType)
     * @see Op#ofLambda(Object)
     */
    public static CallSite metafactory(MethodHandles.Lookup caller,
                                       String interfaceMethodName,
                                       MethodType factoryType,
                                       MethodType interfaceMethodType,
                                       MethodHandle implementation,
                                       MethodType dynamicMethodType)
            throws LambdaConversionException {
        DecodedName decodedName = findReflectableOpGetter(caller, interfaceMethodName);
        ReflectableLambdaInfo reflectableLambdaInfo = decodedName.reflectableLambdaInfo;
        LambdaTransform transform = new LambdaTransform(caller, factoryType, implementation, reflectableLambdaInfo);
        return JLI_ACCESS.metafactoryInternal(caller, decodedName.name, factoryType, interfaceMethodType,
                implementation, dynamicMethodType, transform,
                List.of(implementation, reflectableLambdaInfo.opHandle(), reflectableLambdaInfo.extractOpHandle()));
    }

    /**
     * Metafactory used to create a reflectable lambda.
     * <p>
     * The functionality provided by this metafactory is identical to that in
     * {@link LambdaMetafactory#altMetafactory(Lookup, String, MethodType, Object...)}
     * with one important difference: this metafactory expects the provided name to be encoded in the following form:
     * <code>
     *     lambdaName=opMethodName
     * </code>
     * The {@code lambdaName} part of the name is passed to the regular metafactory method, along with all the other
     * parameters unchanged. The {@code opMethod} part of the name is used to locate a method in the
     * {@linkplain Lookup#lookupClass() lookup class} associated with the provided lookup. This method is expected
     * to accept no parameters and return a {@link Op}, namely the model of the reflectable lambda.
     * <p>
     * This means the clients can pass the lambda returned by this factory to the {@link Op#ofLambda(Object)} method,
     * to access the code model of the lambda expression dynamically.
     *
     * @param caller The lookup
     * @param interfaceMethodName The name of the method to implement.
     *                            This is encoded in the format described above.
     * @param factoryType The expected signature of the {@code CallSite}.
     * @param args An array of {@code Object} containing the required
     *              arguments {@code interfaceMethodType}, {@code implementation},
     *              {@code dynamicMethodType}, {@code flags}, and any
     *              optional arguments, as required by {@link LambdaMetafactory#altMetafactory(Lookup, String, MethodType, Object...)}
     * @return a CallSite whose target can be used to perform capture, generating
     *         a reflectable lambda instance implementing the interface named by {@code factoryType}.
     *         The code model for such instance can be inspected using {@link Op#ofLambda(Object)}.
     *
     * @throws LambdaConversionException If, after the lambda name is decoded,
     *         the parameters of the call are invalid for
     *         {@link LambdaMetafactory#altMetafactory(Lookup, String, MethodType, Object...)}
     * @throws NullPointerException If any argument, or any component of {@code args},
     *         is {@code null}.
     * @throws IllegalArgumentException If {@code args} are invalid for
     *         {@link LambdaMetafactory#altMetafactory(Lookup, String, MethodType, Object...)}
     *
     * @see LambdaMetafactory#altMetafactory(Lookup, String, MethodType, Object...)
     * @see Op#ofLambda(Object)
     */
    public static CallSite altMetafactory(MethodHandles.Lookup caller,
                                          String interfaceMethodName,
                                          MethodType factoryType,
                                          Object... args)
            throws LambdaConversionException {
        DecodedName decodedName = findReflectableOpGetter(caller, interfaceMethodName);
        MethodHandle implementation = extractArg(args, 1, MethodHandle.class);
        ReflectableLambdaInfo reflectableLambdaInfo = decodedName.reflectableLambdaInfo;
        LambdaTransform transform = new LambdaTransform(caller, factoryType, implementation, reflectableLambdaInfo);
        return JLI_ACCESS.altMetafactoryInternal(caller, decodedName.name, factoryType, transform,
                List.of(implementation, reflectableLambdaInfo.opHandle(), reflectableLambdaInfo.extractOpHandle()),
                args);
    }

    private static <T> T extractArg(Object[] args, int index, Class<T> type) {
        if (index >= args.length) {
            throw new IllegalArgumentException("missing argument");
        }
        Object result = Objects.requireNonNull(args[index]);
        if (!type.isInstance(result)) {
            throw new IllegalArgumentException("argument has wrong type");
        }
        return type.cast(result);
    }

    static final JavaLangInvokeAccess JLI_ACCESS = SharedSecrets.getJavaLangInvokeAccess();

    record ReflectableLambdaInfo(ClassDesc quotedClass, ClassDesc funcOpClass,
                                 MethodHandle extractOpHandle, MethodHandle opHandle) { }

    record DecodedName(String name, ReflectableLambdaInfo reflectableLambdaInfo) { }

    private static DecodedName findReflectableOpGetter(MethodHandles.Lookup lookup, String interfaceMethodName) throws LambdaConversionException {
        String[] implNameParts = interfaceMethodName.split("=");
        if (implNameParts.length != 2) {
            throw new LambdaConversionException("Bad method name: " + interfaceMethodName);
        }
        try {
            return new DecodedName(
                    implNameParts[0],
                    newReflectableLambdaInfo(lookup.findStatic(lookup.lookupClass(), implNameParts[1], MethodType.methodType(Op.class))));
        } catch (ReflectiveOperationException ex) {
            throw new LambdaConversionException(ex);
        }
    }

    private static ReflectableLambdaInfo newReflectableLambdaInfo(MethodHandle handle) {
        class Holder {
            static final ClassDesc QUOTED_CLASS_DESC = Quoted.class.describeConstable().get();
            static final ClassDesc FUNC_OP_CLASS_DESC = FuncOp.class.describeConstable().get();
            static final MethodHandle QUOTED_EXTRACT_OP_HANDLE;

            static {
                try {
                    QUOTED_EXTRACT_OP_HANDLE = MethodHandles.lookup()
                            .findStatic(Quoted.class, "extractOp",
                                    MethodType.methodType(Quoted.class, FuncOp.class, Object[].class));
                } catch (Throwable ex) {
                    throw new ExceptionInInitializerError(ex);
                }
            }
        }
        return new ReflectableLambdaInfo(Holder.QUOTED_CLASS_DESC, Holder.FUNC_OP_CLASS_DESC,
                Holder.QUOTED_EXTRACT_OP_HANDLE, handle);
    }

    static class LambdaTransform implements ClassTransform {

        final ReflectableLambdaInfo reflectableLambdaInfo;
        final ClassDesc lambdaClassSymbol;
        final MethodHandleInfo quotableOpGetterInfo;
        final ClassDesc[] argDescs;

        public LambdaTransform(MethodHandles.Lookup caller, MethodType factoryType, MethodHandle implementation, ReflectableLambdaInfo reflectableLambdaInfo) throws LambdaConversionException {
            this.reflectableLambdaInfo = reflectableLambdaInfo;
            this.lambdaClassSymbol = ClassDesc.ofInternalName(sanitizedTargetClassName(caller.lookupClass()).concat("$$Lambda"));
            argDescs = factoryType.parameterList().stream().map(cls -> cls.describeConstable().get()).toArray(ClassDesc[]::new);
            try {
                quotableOpGetterInfo = caller.revealDirect(reflectableLambdaInfo.opHandle()); // may throw SecurityException
            } catch (IllegalArgumentException e) {
                throw new LambdaConversionException(implementation + " is not direct or cannot be cracked");
            }
            if (quotableOpGetterInfo.getReferenceKind() != MethodHandleInfo.REF_invokeStatic) {
                throw new LambdaConversionException(String.format("Unsupported MethodHandle kind: %s", quotableOpGetterInfo));
            }

        }


        @Override
        public void accept(ClassBuilder clb, ClassElement cle) {
            clb.with(cle);
        }

        @Override
        public void atEnd(ClassBuilder clb) {
            // the field that will hold the quoted instance
            clb.withField(QUOTED_FIELD_NAME, reflectableLambdaInfo.quotedClass(), ACC_PRIVATE);
            // the field that will hold the model
            clb.withField(MODEL_FIELD_NAME, reflectableLambdaInfo.funcOpClass(),
                    ACC_PRIVATE | ACC_STATIC);
            generateQuotedMethod(clb);
        }

        /**
        * Generate method #__internal_quoted()
         */
        private void generateQuotedMethod(ClassBuilder clb) {
            clb.withMethodBody(NAME_METHOD_QUOTED, MethodTypeDesc.of(reflectableLambdaInfo.quotedClass()), ACC_PRIVATE, (cob) ->
                cob.aload(0)
                        .invokevirtual(lambdaClassSymbol, "getQuoted", MethodTypeDesc.of(reflectableLambdaInfo.quotedClass()))
                        .areturn());
            // generate helper methods
            /*
            synchronized Quoted getQuoted() {
                Quoted v = quoted;
                if (v == null) {
                    v = quoted = Quoted.extractOp(getModel(), captures);
                }
                return v;
            }
            * */
            clb.withMethodBody("getQuoted", MethodTypeDesc.of(reflectableLambdaInfo.quotedClass()),
                    ACC_PRIVATE + ACC_SYNCHRONIZED, cob ->
                        cob.aload(0)
                            .getfield(lambdaClassSymbol, QUOTED_FIELD_NAME, reflectableLambdaInfo.quotedClass())
                            .astore(1)
                            .aload(1)
                            .ifThen(Opcode.IFNULL, bcb -> {
                                bcb.aload(0); // will be used by putfield

                                // load class data: MH to Quoted.extractOp
                                ConstantPoolBuilder cp = bcb.constantPool();
                                MethodHandleEntry bsmDataAt = cp.methodHandleEntry(BSM_CLASS_DATA_AT);
                                NameAndTypeEntry natMH = cp.nameAndTypeEntry(DEFAULT_NAME, CD_MethodHandle);
                                bcb.ldc(cp.constantDynamicEntry(cp.bsmEntry(bsmDataAt, List.of(cp.intEntry(2))), natMH));

                                bcb.invokestatic(lambdaClassSymbol, "getModel", MethodTypeDesc.of(reflectableLambdaInfo.funcOpClass()));

                                // load captured args in array
                                int capturedArity = argDescs.length;
                                bcb.loadConstant(capturedArity)
                                        .anewarray(CD_Object);
                                for (int i = 0; i < capturedArity; i++) {
                                    bcb.dup()
                                            .loadConstant(i)
                                            .aload(0)
                                            .getfield(lambdaClassSymbol, "arg$" + (i + 1), argDescs[i]);
                                    boxIfTypePrimitive(bcb, TypeKind.from(argDescs[i]));
                                    bcb.aastore();
                                }

                                // invoke Quoted.extractOp
                                bcb.invokevirtual(CD_MethodHandle, "invokeExact", reflectableLambdaInfo.extractOpHandle().type().describeConstable().get())
                                        .dup_x1()
                                        .putfield(lambdaClassSymbol, QUOTED_FIELD_NAME, reflectableLambdaInfo.quotedClass())
                                        .astore(1);

                            })
                            .aload(1)
                            .areturn());

            /*
            private static synchronized CoreOp.FuncOp getModel() {
                FuncOp v = model;
                if (v == null) {
                    v = model = ...invoke lambda op building method...
                }
                return v;
            }
            * */
            ClassDesc funcOpClassDesc = reflectableLambdaInfo.funcOpClass();
            clb.withMethodBody("getModel", MethodTypeDesc.of(reflectableLambdaInfo.funcOpClass()),
                    ACC_PRIVATE + ACC_STATIC + ACC_SYNCHRONIZED, cob ->
                        cob.getstatic(lambdaClassSymbol, MODEL_FIELD_NAME, funcOpClassDesc)
                            .astore(0)
                            .aload(0)
                            .ifThen(Opcode.IFNULL, bcb -> {
                                // load class data: MH to op building method
                                ConstantPoolBuilder cp = clb.constantPool();
                                MethodHandleEntry bsmDataAt = cp.methodHandleEntry(BSM_CLASS_DATA_AT);
                                NameAndTypeEntry natMH = cp.nameAndTypeEntry(DEFAULT_NAME, CD_MethodHandle);
                                cob.ldc(cp.constantDynamicEntry(cp.bsmEntry(bsmDataAt, List.of(cp.intEntry(1))), natMH));
                                MethodType mtype = quotableOpGetterInfo.getMethodType();
                                cob.invokevirtual(CD_MethodHandle, "invokeExact", mtype.describeConstable().get())
                                        .checkcast(funcOpClassDesc)
                                        .dup()
                                        .putstatic(lambdaClassSymbol, MODEL_FIELD_NAME, funcOpClassDesc)
                                        .astore(0);
                            })
                            .aload(0)
                            .areturn());
        }


        static void boxIfTypePrimitive(CodeBuilder cob, TypeKind tk) {
            var cp = cob.constantPool();
            switch (tk) {
                case BOOLEAN -> cob.invokestatic(box(cp, CD_boolean, CD_Boolean));
                case BYTE -> cob.invokestatic(box(cp, CD_byte, CD_Byte));
                case CHAR -> cob.invokestatic(box(cp, CD_char, CD_Character));
                case DOUBLE -> cob.invokestatic(box(cp, CD_double, CD_Double));
                case FLOAT -> cob.invokestatic(box(cp, CD_float, CD_Float));
                case INT -> cob.invokestatic(box(cp, CD_int, CD_Integer));
                case LONG -> cob.invokestatic(box(cp, CD_long, CD_Long));
                case SHORT -> cob.invokestatic(box(cp, CD_short, CD_Short));
            }
        }

        private static MethodRefEntry box(ConstantPoolBuilder cp, ClassDesc primitive, ClassDesc target) {
            return cp.methodRefEntry(target, "valueOf", MethodTypeDesc.of(target, primitive));
        }

        private static String sanitizedTargetClassName(Class<?> targetClass) {
            String name = targetClass.getName();
            if (targetClass.isHidden()) {
                // use the original class name
                name = name.replace('/', '_');
            }
            return name.replace('.', '/');
        }

    }
}
