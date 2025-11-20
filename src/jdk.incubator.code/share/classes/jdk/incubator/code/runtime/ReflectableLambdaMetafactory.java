package jdk.incubator.code.runtime;

import jdk.incubator.code.Op;
import jdk.internal.access.JavaLangInvokeAccess;
import jdk.internal.access.SharedSecrets;

import java.lang.invoke.CallSite;
import java.lang.invoke.LambdaConversionException;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.SerializedLambda;

public class ReflectableLambdaMetafactory {

    private ReflectableLambdaMetafactory() {
        // nope
    }

    /**
     * Facilitates the creation of simple "function objects" that implement one
     * or more interfaces by delegation to a provided {@link MethodHandle},
     * after appropriate type adaptation and partial evaluation of arguments.
     * Typically used as a <em>bootstrap method</em> for {@code invokedynamic}
     * call sites, to support the <em>lambda expression</em> and <em>method
     * reference expression</em> features of the Java Programming Language.
     *
     * <p>This is the standard, streamlined metafactory; additional flexibility
     * is provided by {@link #altMetafactory(MethodHandles.Lookup, String, MethodType, Object...)}.
     * A general description of the behavior of this method is provided
     * {@link LambdaMetafactory above}.
     *
     * <p>When the target of the {@code CallSite} returned from this method is
     * invoked, the resulting function objects are instances of a class which
     * implements the interface named by the return type of {@code factoryType},
     * declares a method with the name given by {@code interfaceMethodName} and the
     * signature given by {@code interfaceMethodType}.  It may also override additional
     * methods from {@code Object}.
     *
     * @param caller Represents a lookup context with the accessibility
     *               privileges of the caller.  Specifically, the lookup context
     *               must have {@linkplain MethodHandles.Lookup#hasFullPrivilegeAccess()
     *               full privilege access}.
     *               When used with {@code invokedynamic}, this is stacked
     *               automatically by the VM.
     * @param interfaceMethodName The name of the method to implement.  When used with
     *                            {@code invokedynamic}, this is provided by the
     *                            {@code NameAndType} of the {@code InvokeDynamic}
     *                            structure and is stacked automatically by the VM.
     * @param factoryType The expected signature of the {@code CallSite}.  The
     *                    parameter types represent the types of capture variables;
     *                    the return type is the interface to implement.   When
     *                    used with {@code invokedynamic}, this is provided by
     *                    the {@code NameAndType} of the {@code InvokeDynamic}
     *                    structure and is stacked automatically by the VM.
     * @param interfaceMethodType Signature and return type of method to be
     *                            implemented by the function object.
     * @param implementation A direct method handle describing the implementation
     *                       method which should be called (with suitable adaptation
     *                       of argument types and return types, and with captured
     *                       arguments prepended to the invocation arguments) at
     *                       invocation time.
     * @param dynamicMethodType The signature and return type that should
     *                          be enforced dynamically at invocation time.
     *                          In simple use cases this is the same as
     *                          {@code interfaceMethodType}.
     * @return a CallSite whose target can be used to perform capture, generating
     *         instances of the interface named by {@code factoryType}
     * @throws LambdaConversionException If {@code caller} does not have full privilege
     *         access, or if {@code interfaceMethodName} is not a valid JVM
     *         method name, or if the return type of {@code factoryType} is not
     *         an interface, or if {@code implementation} is not a direct method
     *         handle referencing a method or constructor, or if the linkage
     *         invariants are violated, as defined {@link LambdaMetafactory above}.
     * @throws NullPointerException If any argument is {@code null}.
     */
    public static CallSite metafactory(MethodHandles.Lookup caller,
                                       String interfaceMethodName,
                                       MethodType factoryType,
                                       MethodType interfaceMethodType,
                                       MethodHandle implementation,
                                       MethodType dynamicMethodType)
            throws LambdaConversionException {
        DecodedName decodedName = findQuotableOpGetter(caller, interfaceMethodName);
        return JLI_ACCESS.metafactoryInternal(caller, decodedName.name, factoryType, interfaceMethodType,
                implementation, dynamicMethodType, decodedName.opGetter);
    }

    /**
     * Facilitates the creation of simple "function objects" that implement one
     * or more interfaces by delegation to a provided {@link MethodHandle},
     * after appropriate type adaptation and partial evaluation of arguments.
     * Typically used as a <em>bootstrap method</em> for {@code invokedynamic}
     * call sites, to support the <em>lambda expression</em> and <em>method
     * reference expression</em> features of the Java Programming Language.
     *
     * <p>This is the general, more flexible metafactory; a streamlined version
     * is provided by {@link #metafactory(java.lang.invoke.MethodHandles.Lookup,
     * String, MethodType, MethodType, MethodHandle, MethodType)}.
     * A general description of the behavior of this method is provided
     * {@link LambdaMetafactory above}.
     *
     * <p>The argument list for this method includes three fixed parameters,
     * corresponding to the parameters automatically stacked by the VM for the
     * bootstrap method in an {@code invokedynamic} invocation, and an {@code Object[]}
     * parameter that contains additional parameters.  The declared argument
     * list for this method is:
     *
     * <pre>{@code
     *  CallSite altMetafactory(MethodHandles.Lookup caller,
     *                          String interfaceMethodName,
     *                          MethodType factoryType,
     *                          Object... args)
     * }</pre>
     *
     * <p>but it behaves as if the argument list is as follows:
     *
     * <pre>{@code
     *  CallSite altMetafactory(MethodHandles.Lookup caller,
     *                          String interfaceMethodName,
     *                          MethodType factoryType,
     *                          MethodType interfaceMethodType,
     *                          MethodHandle implementation,
     *                          MethodType dynamicMethodType,
     *                          int flags,
     *                          int altInterfaceCount,        // IF flags has MARKERS set
     *                          Class... altInterfaces,       // IF flags has MARKERS set
     *                          int altMethodCount,           // IF flags has BRIDGES set
     *                          MethodType... altMethods      // IF flags has BRIDGES set
     *                          )
     * }</pre>
     *
     * <p>Arguments that appear in the argument list for
     * {@link #metafactory(MethodHandles.Lookup, String, MethodType, MethodType, MethodHandle, MethodType)}
     * have the same specification as in that method.  The additional arguments
     * are interpreted as follows:
     * <ul>
     *     <li>{@code flags} indicates additional options; this is a bitwise
     *     OR of desired flags.  Defined flags are {@link LambdaMetafactory#FLAG_BRIDGES},
     *     {@link LambdaMetafactory#FLAG_MARKERS}, and {@link LambdaMetafactory#FLAG_SERIALIZABLE}.</li>
     *     <li>{@code altInterfaceCount} is the number of additional interfaces
     *     the function object should implement, and is present if and only if the
     *     {@code FLAG_MARKERS} flag is set.</li>
     *     <li>{@code altInterfaces} is a variable-length list of additional
     *     interfaces to implement, whose length equals {@code altInterfaceCount},
     *     and is present if and only if the {@code FLAG_MARKERS} flag is set.</li>
     *     <li>{@code altMethodCount} is the number of additional method signatures
     *     the function object should implement, and is present if and only if
     *     the {@code FLAG_BRIDGES} flag is set.</li>
     *     <li>{@code altMethods} is a variable-length list of additional
     *     methods signatures to implement, whose length equals {@code altMethodCount},
     *     and is present if and only if the {@code FLAG_BRIDGES} flag is set.</li>
     * </ul>
     *
     * <p>Each class named by {@code altInterfaces} is subject to the same
     * restrictions as {@code Rd}, the return type of {@code factoryType},
     * as described {@link LambdaMetafactory above}.  Each {@code MethodType}
     * named by {@code altMethods} is subject to the same restrictions as
     * {@code interfaceMethodType}, as described {@link LambdaMetafactory above}.
     *
     * <p>When FLAG_SERIALIZABLE is set in {@code flags}, the function objects
     * will implement {@code Serializable}, and will have a {@code writeReplace}
     * method that returns an appropriate {@link SerializedLambda}.  The
     * {@code caller} class must have an appropriate {@code $deserializeLambda$}
     * method, as described in {@link SerializedLambda}.
     *
     * <p>When the target of the {@code CallSite} returned from this method is
     * invoked, the resulting function objects are instances of a class with
     * the following properties:
     * <ul>
     *     <li>The class implements the interface named by the return type
     *     of {@code factoryType} and any interfaces named by {@code altInterfaces}</li>
     *     <li>The class declares methods with the name given by {@code interfaceMethodName},
     *     and the signature given by {@code interfaceMethodType} and additional signatures
     *     given by {@code altMethods}</li>
     *     <li>The class may override methods from {@code Object}, and may
     *     implement methods related to serialization.</li>
     * </ul>
     *
     * @param caller Represents a lookup context with the accessibility
     *               privileges of the caller.  Specifically, the lookup context
     *               must have {@linkplain MethodHandles.Lookup#hasFullPrivilegeAccess()
     *               full privilege access}.
     *               When used with {@code invokedynamic}, this is stacked
     *               automatically by the VM.
     * @param interfaceMethodName The name of the method to implement.  When used with
     *                            {@code invokedynamic}, this is provided by the
     *                            {@code NameAndType} of the {@code InvokeDynamic}
     *                            structure and is stacked automatically by the VM.
     * @param factoryType The expected signature of the {@code CallSite}.  The
     *                    parameter types represent the types of capture variables;
     *                    the return type is the interface to implement.   When
     *                    used with {@code invokedynamic}, this is provided by
     *                    the {@code NameAndType} of the {@code InvokeDynamic}
     *                    structure and is stacked automatically by the VM.
     * @param  args An array of {@code Object} containing the required
     *              arguments {@code interfaceMethodType}, {@code implementation},
     *              {@code dynamicMethodType}, {@code flags}, and any
     *              optional arguments, as described above
     * @return a CallSite whose target can be used to perform capture, generating
     *         instances of the interface named by {@code factoryType}
     * @throws LambdaConversionException If {@code caller} does not have full privilege
     *         access, or if {@code interfaceMethodName} is not a valid JVM
     *         method name, or if the return type of {@code factoryType} is not
     *         an interface, or if any of {@code altInterfaces} is not an
     *         interface, or if {@code implementation} is not a direct method
     *         handle referencing a method or constructor, or if the linkage
     *         invariants are violated, as defined {@link LambdaMetafactory above}.
     * @throws NullPointerException If any argument, or any component of {@code args},
     *         is {@code null}.
     * @throws IllegalArgumentException If the number or types of the components
     *         of {@code args} do not follow the above rules, or if
     *         {@code altInterfaceCount} or {@code altMethodCount} are negative
     *         integers.
     */
    public static CallSite altMetafactory(MethodHandles.Lookup caller,
                                          String interfaceMethodName,
                                          MethodType factoryType,
                                          Object... args)
            throws LambdaConversionException {
        DecodedName decodedName = findQuotableOpGetter(caller, interfaceMethodName);
        return JLI_ACCESS.altMetafactoryInternal(caller, decodedName.name, factoryType, decodedName.opGetter, args);
    }

    static final JavaLangInvokeAccess JLI_ACCESS = SharedSecrets.getJavaLangInvokeAccess();

    record DecodedName(String name, MethodHandle opGetter) { }

    private static DecodedName findQuotableOpGetter(MethodHandles.Lookup lookup, String interfaceMethodName) throws LambdaConversionException {
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
}
