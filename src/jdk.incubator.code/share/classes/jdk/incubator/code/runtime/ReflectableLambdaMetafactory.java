package jdk.incubator.code.runtime;

import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.internal.access.JavaLangInvokeAccess;
import jdk.internal.access.JavaLangInvokeAccess.ReflectableLambdaInfo;
import jdk.internal.access.SharedSecrets;

import java.lang.constant.ClassDesc;
import java.lang.invoke.CallSite;
import java.lang.invoke.LambdaConversionException;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;

/**
 * Provides runtime support for creating reflectable lambdas. A reflectable lambda is a lambda whose
 * code model can be inspected using {@link Op#ofLambda(Object)}.
 * @see LambdaMetafactory
 * @see Op#ofLambda(Object)
 */
public class ReflectableLambdaMetafactory {

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
        return JLI_ACCESS.metafactoryInternal(caller, decodedName.name, factoryType, interfaceMethodType,
                implementation, dynamicMethodType, decodedName.reflectableLambdaInfo);
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
        return JLI_ACCESS.altMetafactoryInternal(caller, decodedName.name, factoryType, decodedName.reflectableLambdaInfo, args);
    }

    static final JavaLangInvokeAccess JLI_ACCESS = SharedSecrets.getJavaLangInvokeAccess();

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
}
