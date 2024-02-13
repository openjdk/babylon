/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import java.lang.reflect.code.Op;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.type.JavaType;
import java.util.stream.Stream;

import static java.lang.reflect.code.descriptor.MethodDesc.method;
import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;
import static java.lang.reflect.code.op.CoreOps.*;
import static java.lang.reflect.code.type.JavaType.type;

public interface Queryable<T> {
    JavaType TYPE = type(Queryable.class);

    QueryProvider provider();

    // T
    JavaType elementType();

    // Queryable<T> -> Queryable<U>
    FuncOp expression();

    @SuppressWarnings("unchecked")
    default Queryable<T> where(QuotablePredicate<T> f) {
        LambdaOp l = (LambdaOp) f.quoted().op();
        return (Queryable<T>) insertQuery(elementType(), "where", l);
    }

    @SuppressWarnings("unchecked")
    default <R> Queryable<R> select(QuotableFunction<T, R> f) {
        LambdaOp l = (LambdaOp) f.quoted().op();
        return (Queryable<R>) insertQuery((JavaType) l.funcDescriptor().returnType(), "select", l);
    }

    private Queryable<?> insertQuery(JavaType elementType, String methodName, LambdaOp lambdaOp) {
        // Copy function expression, replacing return operation
        FuncOp queryExpression = expression();
        JavaType queryableType = type(Queryable.TYPE, elementType);
        FuncOp nextQueryExpression = func("query",
                methodType(queryableType, queryExpression.funcDescriptor().parameters()))
                .body(b -> b.inline(queryExpression, b.parameters(), (block, query) -> {
                    Op.Result fi = block.op(lambdaOp);

                    MethodDesc md = method(Queryable.TYPE, methodName,
                            methodType(Queryable.TYPE, ((JavaType) lambdaOp.functionalInterface()).rawType()));
                    Op.Result queryable = block.op(invoke(queryableType, md, query, fi));

                    block.op(_return(queryable));
                }));

        return provider().createQuery(elementType, nextQueryExpression);
    }

    @SuppressWarnings("unchecked")
    default QueryResult<Stream<T>> elements() {
        JavaType resultType = type(type(Stream.class), elementType());
        return (QueryResult<Stream<T>>) insertQueryResult(resultType, "elements");
    }

    @SuppressWarnings("unchecked")
    default QueryResult<Long> count() {
        JavaType resultType = JavaType.LONG;
        return (QueryResult<Long>) insertQueryResult(resultType, "count");
    }

    private QueryResult<?> insertQueryResult(JavaType resultType, String methodName) {
        // Copy function expression, replacing return operation
        FuncOp queryExpression = expression();
        JavaType queryResultType = JavaType.type(QueryResult.TYPE, resultType);
        FuncOp queryResultExpression = func("queryResult",
                methodType(queryResultType, queryExpression.funcDescriptor().parameters()))
                .body(b -> b.inline(queryExpression, b.parameters(), (block, query) -> {
                    MethodDesc md = method(Queryable.TYPE, methodName,
                            methodType(QueryResult.TYPE));
                    Op.Result queryResult = block.op(invoke(queryResultType, md, query));

                    block.op(_return(queryResult));
                }));

        return provider().createQueryResult(resultType, queryResultExpression);
    }
}
