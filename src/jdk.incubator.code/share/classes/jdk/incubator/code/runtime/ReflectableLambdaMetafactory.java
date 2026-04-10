package jdk.incubator.code.runtime;

import java.lang.classfile.ClassBuilder;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.constantpool.ConstantPoolBuilder;
import java.lang.classfile.constantpool.MethodRefEntry;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.CallSite;
import java.lang.invoke.LambdaConversionException;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;
import java.util.List;

import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.internal.access.JavaLangInvokeAccess;
import jdk.internal.access.SharedSecrets;

import static java.lang.classfile.ClassFile.ACC_PRIVATE;
import static java.lang.classfile.ClassFile.ACC_STATIC;
import static java.lang.classfile.ClassFile.ACC_SYNCHRONIZED;
import static java.lang.constant.ConstantDescs.*;
import java.lang.constant.DynamicConstantDesc;
import java.util.function.Function;

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

    static final ClassDesc CD_Quoted = Quoted.class.describeConstable().get();
    static final ClassDesc CD_FuncOp = FuncOp.class.describeConstable().get();
    static final ClassDesc CD_Op = Op.class.describeConstable().get();
    static final MethodTypeDesc MTD_extractOp = MethodTypeDesc.of(CD_Quoted, CD_FuncOp, CD_Object.arrayType());
    static final DynamicConstantDesc<?> DCD_CLASS_DATA = DynamicConstantDesc.ofNamed(BSM_CLASS_DATA, DEFAULT_NAME, CD_List);

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
        LambdaFinisher finisher = new LambdaFinisher(caller.lookupClass(), factoryType.parameterList(), decodedName.opHandle);
        return JLI_ACCESS.metafactoryInternal(caller, decodedName.name, factoryType, interfaceMethodType,
                implementation, dynamicMethodType, finisher);
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
        LambdaFinisher finisher = new LambdaFinisher(caller.lookupClass(), factoryType.parameterList(), decodedName.opHandle);
        return JLI_ACCESS.altMetafactoryInternal(caller, decodedName.name, factoryType, finisher, args);
    }

    static final JavaLangInvokeAccess JLI_ACCESS = SharedSecrets.getJavaLangInvokeAccess();

    record DecodedName(String name, MethodHandle opHandle) { }

    private static DecodedName findReflectableOpGetter(MethodHandles.Lookup lookup, String interfaceMethodName) throws LambdaConversionException {
        String[] implNameParts = interfaceMethodName.split("=");
        if (implNameParts.length != 2) {
            throw new LambdaConversionException("Bad method name: " + interfaceMethodName);
        }
        try {
            return new DecodedName(
                    implNameParts[0],
                    lookup.findStatic(lookup.lookupClass(), implNameParts[1], MethodType.methodType(Op.class)));
        } catch (ReflectiveOperationException ex) {
            throw new LambdaConversionException(ex);
        }
    }

    static class LambdaFinisher implements Function<ClassBuilder, Object> {

        final ClassDesc lambdaClassSymbol;
        final ClassDesc[] argDescs;
        final MethodHandle opHandle;

        public LambdaFinisher(Class<?> callerClass, List<Class<?>> parameterTypes, MethodHandle opHandle) throws LambdaConversionException {
            this.lambdaClassSymbol = ClassDesc.ofInternalName(sanitizedTargetClassName(callerClass).concat("$$Lambda"));
            this.argDescs = parameterTypes.stream().map(cls -> cls.describeConstable().get()).toArray(ClassDesc[]::new);
            this.opHandle = opHandle;
        }

        @Override
        public Object apply(ClassBuilder clb) {
            // the field that will hold the quoted instance
            clb.withField(QUOTED_FIELD_NAME, CD_Quoted, ACC_PRIVATE);
            // the field that will hold the model
            clb.withField(MODEL_FIELD_NAME, CD_FuncOp,
                    ACC_PRIVATE | ACC_STATIC);
            // Generate method #__internal_quoted()
            clb.withMethodBody(NAME_METHOD_QUOTED, MethodTypeDesc.of(CD_Quoted), ACC_PRIVATE, (cob) ->
                cob.aload(0)
                   .invokevirtual(lambdaClassSymbol, "getQuoted", MethodTypeDesc.of(CD_Quoted))
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
            */
            clb.withMethodBody("getQuoted", MethodTypeDesc.of(CD_Quoted),
                    ACC_PRIVATE + ACC_SYNCHRONIZED, cob ->
                        cob.aload(0)
                           .getfield(lambdaClassSymbol, QUOTED_FIELD_NAME, CD_Quoted)
                           .astore(1)
                           .aload(1)
                           .ifThen(Opcode.IFNULL, bcb -> {
                               bcb.aload(0) // will be used by putfield
                                  .invokestatic(lambdaClassSymbol, "getModel", MethodTypeDesc.of(CD_FuncOp))
                               // load captured args in array
                                  .loadConstant(argDescs.length)
                                  .anewarray(CD_Object);
                               for (int i = 0; i < argDescs.length; i++) {
                                   bcb.dup()
                                      .loadConstant(i)
                                      .aload(0)
                                      .getfield(lambdaClassSymbol, "arg$" + (i + 1), argDescs[i]);
                                   boxIfTypePrimitive(bcb, TypeKind.from(argDescs[i]));
                                   bcb.aastore();
                               }
                               // invoke Quoted.extractOp
                               bcb.invokestatic(CD_Quoted, "extractOp", MTD_extractOp)
                                  .dup_x1()
                                  .putfield(lambdaClassSymbol, QUOTED_FIELD_NAME, CD_Quoted)
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
            */
            clb.withMethodBody("getModel", MethodTypeDesc.of(CD_FuncOp),
                    ACC_PRIVATE + ACC_STATIC + ACC_SYNCHRONIZED, cob ->
                        cob.getstatic(lambdaClassSymbol, MODEL_FIELD_NAME, CD_FuncOp)
                           .astore(0)
                           .aload(0)
                           .ifThen(Opcode.IFNULL, bcb ->
                               // last item in the class data list is a method handle to get the op
                               bcb.ldc(DCD_CLASS_DATA)
                                  .invokeinterface(CD_List, "getLast", MethodTypeDesc.of(CD_Object))
                                  .checkcast(CD_MethodHandle)
                                  .invokevirtual(CD_MethodHandle, "invokeExact", MethodTypeDesc.of(CD_Op))
                                  .checkcast(CD_FuncOp)
                                  .dup()
                                  .putstatic(lambdaClassSymbol, MODEL_FIELD_NAME, CD_FuncOp)
                                  .astore(0))
                           .aload(0)
                           .areturn());
            // return opHandle as additional class data
            return opHandle;
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
