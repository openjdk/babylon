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
package optkl;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.ifacemapper.MappableIface;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.function.Predicate;
import java.util.stream.Stream;


public interface Invoke extends OpHelper<JavaOp.InvokeOp>{

    static Stream<Invoke> stream(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
       return  funcOp.elements().filter(ce->ce instanceof JavaOp.InvokeOp).map(ce->invokeOpHelper(lookup,ce));
    }

    default  boolean isStatic(){
        return op().invokeKind().equals(JavaOp.InvokeOp.InvokeKind.STATIC);
    }
     default  boolean isInstance(){
        return op().invokeKind().equals(JavaOp.InvokeOp.InvokeKind.INSTANCE);
    }
    @Override default String name(){
        return op().invokeDescriptor().name();
    }
    default <T>boolean returns(Class<T> clazz){
        return isAssignable((JavaType)op().resultType(),clazz);
    }
    default boolean receives(Class<?>... classes){
        boolean  assignable = true;
        for (int i=isStatic()?1:0; assignable && i< classes.length; i++) {
            var operand = op().operands().get(i);
            TypeElement resultType = operand.type() instanceof VarType varType?varType.valueType():null;
            assignable &= isAssignable((JavaType) resultType,classes[i-(isStatic()?1:0)]);
        }
        return assignable;
    }

    default Method resolvedMethodOrNull(){
        try {
            return op().invokeDescriptor().resolveToMethod(lookup()) instanceof Method method ? method : null;
        }catch (ReflectiveOperationException rope){
            return null;
        }
    }

     static Invoke invokeOpHelper(MethodHandles.Lookup lookup, CodeElement<?,?> codeElement){
        record Impl(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) implements Invoke {}

        return codeElement instanceof JavaOp.InvokeOp invokeOp? new Impl(lookup,invokeOp): null;
    }

     default boolean refIs(Class<?> ...classes) {
        return OpTkl.isAssignable(lookup(), op().invokeDescriptor().refType(), classes);
    }

     default boolean returnsArray() {
        return op().resultType() instanceof ArrayType;
    }

     default boolean returnsVoid() {
        return op().invokeDescriptor().type().returnType().equals(JavaType.VOID);
    }

     default   TypeElement returnType() {
        return op().invokeDescriptor().type().returnType();
    }

    default boolean returnsInt(){
        return returnType().equals(JavaType.INT);
    }



    default boolean returnsClassType(){
        return returnType() instanceof ClassType;
    }


    default TypeElement refType(){
        return op().invokeDescriptor().refType();
    }

    default boolean returnsPrimitive(){
        return returnType() instanceof PrimitiveType ;
    }
    default boolean returnsFloat(){
        return returnType() == JavaType.FLOAT;
    }
    default boolean returnsChar(){
       return returnType() ==   JavaType.CHAR;
    }
    default boolean returnsShort(){
        return returnType() ==   JavaType.SHORT ;
    }
    default boolean returns16BitValue(){
        return returnsChar()||returnsShort();
    }
    default Method resolveMethodOrNull() {
        try {
            return op().invokeDescriptor().resolveToMethod(lookup());
        } catch (ReflectiveOperationException e) {
           return null;
        }
    }
    default Method resolveMethodOrThrow() {
        try {
            return op().invokeDescriptor().resolveToMethod(lookup());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    default Class<?> classOrThrow() {
        if (refType() instanceof ClassType classType) {
            return (Class<?>) OpTkl.classTypeToTypeOrThrow(lookup(), classType);
        } else {
            throw new IllegalStateException(" javaRef class is null");
        }
    }

    default boolean isMappableIface() {
        return refIs(MappableIface.class);
    }
    static Invoke getTargetInvoke(MethodHandles.Lookup lookup, JavaOp.LambdaOp lambdaOp, Class<?>... classes) {
        return lambdaOp.body().entryBlock().ops().stream()
                .filter(ce -> ce instanceof JavaOp.InvokeOp)
                .map(ce -> invokeOpHelper(lookup,ce))
                .filter(Invoke::isStatic)
                .filter(invoke -> OpTkl.isAssignable(lookup, invoke.op().operands().getFirst().type(), classes))
                .findFirst()
                .orElseThrow();
    }


    default Op.Result returnResult(){
        return op().result();
    }
}
