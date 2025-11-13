package jdk.incubator.code.dialect.java.impl;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.constant.Constable;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;
import java.lang.invoke.TypeDescriptor;
import java.lang.invoke.VarHandle;
import java.util.NoSuchElementException;
import java.util.function.Supplier;

public class ResolutionHelper {
    interface HandleResolver<X extends Constable, T extends TypeDescriptor> {
        X resolve(Lookup lookup, Class<?> refc, String name, T type) throws ReflectiveOperationException;

        HandleResolver<MethodHandle, MethodType> FIND_STATIC = MethodHandles.Lookup::findStatic;
        HandleResolver<MethodHandle, MethodType> FIND_VIRTUAL = MethodHandles.Lookup::findVirtual;
        HandleResolver<MethodHandle, MethodType> FIND_SPECIAL = (l, refc, name, type) -> l.findSpecial(refc, name, type, l.lookupClass());
        HandleResolver<MethodHandle, MethodType> FIND_CONSTRUCTOR = (l, refc, name, type) -> l.findConstructor(refc, type);
        HandleResolver<MethodHandle, Class<?>> FIND_STATIC_GETTER = MethodHandles.Lookup::findStaticGetter;
        HandleResolver<MethodHandle, Class<?>> FIND_GETTER = MethodHandles.Lookup::findGetter;
        HandleResolver<VarHandle, Class<?>> FIND_STATIC_VARHANDLE = MethodHandles.Lookup::findStaticVarHandle;
        HandleResolver<VarHandle, Class<?>> FIND_VARHANDLE = MethodHandles.Lookup::findVarHandle;
    }

    sealed interface Result<H extends Constable> {
        H handle() throws ReflectiveOperationException;

        default Result<H> orElse(Supplier<Result<H>> resultSupplier) {
            if (this instanceof Success<?>) return this;
            else return resultSupplier.get();
        }
    }
    record Success<H extends Constable>(H handle) implements Result<H> { }
    record Failure<H extends Constable>(ReflectiveOperationException error) implements Result<H> {
        @Override
        public H handle() throws ReflectiveOperationException {
            throw error;
        }
    }

    static <Z extends Constable, T extends TypeDescriptor> Result<Z> resolveHandle(HandleResolver<Z, T> resolver, MethodHandles.Lookup l, Class<?> refc, String name, T type) {
        try {
            Z res = resolver.resolve(l, refc, name, type);
            return new Success<>(res);
        } catch (ReflectiveOperationException ex) {
            return new Failure<>(ex);
        }
    }

    // public API

    public static Class<?> resolveClass(MethodHandles.Lookup l, TypeElement t) throws ReflectiveOperationException {
        if (t instanceof JavaType jt) {
            return (Class<?>)jt.erasure().resolve(l);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    public static MethodType resolveMethodType(MethodHandles.Lookup l, FunctionType t) throws ReflectiveOperationException {
        if (t instanceof FunctionType ft) {
            return MethodRef.toNominalDescriptor(ft)
                    .resolveConstantDesc(l);
        } else {
            throw new UnsupportedOperationException();
        }
    }

    public static MethodHandle resolveMethod(MethodHandles.Lookup l, MethodRef methodRef, InvokeKind kind) throws ReflectiveOperationException {
        Class<?> refC = resolveClass(l, methodRef.refType());
        MethodType mt = resolveMethodType(l, methodRef.type());
        HandleResolver<MethodHandle, MethodType> resolver = switch (kind) {
            case INSTANCE -> HandleResolver.FIND_VIRTUAL;
            case STATIC -> HandleResolver.FIND_STATIC;
            case SUPER -> HandleResolver.FIND_SPECIAL;
        };
        return resolveHandle(resolver, l, refC, methodRef.name(), mt).handle();
    }

    public static MethodHandle resolveMethod(MethodHandles.Lookup l, MethodRef methodRef) throws ReflectiveOperationException {
        Class<?> refC = resolveClass(l, methodRef.refType());
        MethodType mt = resolveMethodType(l, methodRef.type());
        return resolveHandle(HandleResolver.FIND_STATIC, l, refC, methodRef.name(), mt)
                .orElse(() -> resolveHandle(HandleResolver.FIND_VIRTUAL, l, refC, methodRef.name(), mt))
                .handle();
    }

    public static MethodHandle resolveFieldGetter(MethodHandles.Lookup l, FieldRef fieldRef) throws ReflectiveOperationException {
        Class<?> refC = resolveClass(l, fieldRef.refType());
        Class<?> ft = resolveClass(l, fieldRef.type());
        return resolveHandle(HandleResolver.FIND_STATIC_GETTER, l, refC, fieldRef.name(), ft)
                .orElse(() -> resolveHandle(HandleResolver.FIND_GETTER, l, refC, fieldRef.name(), ft))
                .handle();
    }

    public static VarHandle resolveFieldHandle(MethodHandles.Lookup l, FieldRef fieldRef) throws ReflectiveOperationException {
        Class<?> refC = resolveClass(l, fieldRef.refType());
        Class<?> ft = resolveClass(l, fieldRef.type());
        return resolveHandle(HandleResolver.FIND_STATIC_VARHANDLE, l, refC, fieldRef.name(), ft)
                .orElse(() -> resolveHandle(HandleResolver.FIND_VARHANDLE, l, refC, fieldRef.name(), ft))
                .handle();
    }

    public static MethodHandle resolveConstructor(MethodHandles.Lookup l, Class<?> refc, MethodType type) throws ReflectiveOperationException {
        return resolveHandle(HandleResolver.FIND_CONSTRUCTOR, l, refc, null, type).handle();
    }
}
