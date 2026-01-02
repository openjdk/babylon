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

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.Regex;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.function.Predicate;

import static optkl.Invoke.invokeOpHelper;

public interface OpHelper<T extends Op> extends LookupCarrier {
    T op();

    String name();

    default boolean named(Regex regex){
        return regex.matches(name());
    }
    default boolean named(String name){
        return name().equals(name);
    }
    default boolean named(Predicate<String> predicate){
        return predicate.test(name());
    }
    default <C>boolean isAssignable(JavaType javaType, Class<C> clazz){
        try {
            boolean isit1=  OpTkl.isAssignable(lookup(),javaType,clazz);
            var basicType = javaType.toBasicType();
            var resolveType = basicType.resolve(lookup());
            var isit2 = resolveType.getTypeName().equals(clazz.getName());
            var resolveTypeClass = resolveType.getClass();
            var isit3 = clazz.isAssignableFrom(resolveTypeClass);
            var isit4 = resolveTypeClass.isAssignableFrom(clazz);
            // System.out.println("isit1="+isit1+",isit2="+isit2+",isit3="+isit3+",isit4="+isit4);
            return  isit2;

        } catch (ReflectiveOperationException e) {
            System.out.println("Hmm");
            throw new RuntimeException(e);
        }
    }
    default  int operandCount(){
        return op().operands().size();
    }
    default Op.Result operandNAsResultOrNull(int i){
        return OpTkl.operandAsResult(op(),i) instanceof Op.Result result?result:null;
    }
    default Op.Result  operandNAsResultOrThrow(int i){
        if (operandNAsResultOrNull(i) instanceof Op.Result result){
            return result;
        }else {
            throw new IllegalStateException("Expecting operand "+i+" to be a result");
        }
    }

    default Op opFromOperandNAsResultOrNull(int i){
        return operandNAsResultOrNull(i) instanceof Op.Result result && result.op() instanceof Op op ?op:null;
    }
    default Op opFromOperandNAsResultOrThrow(int i){
        if ( opFromOperandNAsResultOrNull(i)  instanceof Op op){
            return op;
        }else {
            throw new IllegalStateException("Expecting operand "+i+" to be a result which yields an Op ");
        }
    }





}
