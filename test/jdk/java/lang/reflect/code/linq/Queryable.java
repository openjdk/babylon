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
import java.lang.reflect.code.descriptor.TypeDesc;
import java.util.stream.Stream;

import static java.lang.reflect.code.descriptor.MethodDesc.method;
import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;
import static java.lang.reflect.code.descriptor.TypeDesc.type;
import static java.lang.reflect.code.op.CoreOps.*;

public interface Queryable<T> {
    TypeDesc DESC = TypeDesc.type(Queryable.class);

    QueryProvider provider();

    // T
    TypeDesc elementDesc();

    // Queryable<T> -> Queryable<U>
    FuncOp expression();

    @SuppressWarnings("unchecked")
    default Queryable<T> where(QuotablePredicate<T> f) {
        LambdaOp l = (LambdaOp) f.quoted().op();
        return (Queryable<T>) insertQuery(elementDesc(), "where", l);
    }

    @SuppressWarnings("unchecked")
    default <R> Queryable<R> select(QuotableFunction<T, R> f) {
        LambdaOp l = (LambdaOp) f.quoted().op();
        return (Queryable<R>) insertQuery(l.funcDescriptor().returnType(), "select", l);
    }

    private Queryable<?> insertQuery(TypeDesc elementDesc, String methodName, LambdaOp lambdaOp) {
        // Copy function expression, replacing return operation
        FuncOp queryExpression = expression();
        TypeDesc queryableDesc = TypeDesc.type(Queryable.DESC, elementDesc);
        FuncOp nextQueryExpression = func("query",
                methodType(queryableDesc, queryExpression.funcDescriptor().parameters()))
                .body(b -> b.inline(queryExpression, b.parameters(), (block, query) -> {
                    Op.Result fi = block.op(lambdaOp);

                    MethodDesc md = method(Queryable.DESC, methodName,
                            methodType(Queryable.DESC, lambdaOp.functionalInterface().rawType()));
                    Op.Result queryable = block.op(invoke(queryableDesc, md, query, fi));

                    block.op(_return(queryable));
                }));

        return provider().createQuery(elementDesc, nextQueryExpression);
    }

    @SuppressWarnings("unchecked")
    default QueryResult<Stream<T>> elements() {
        TypeDesc resultDesc = type(type(Stream.class), elementDesc());
        return (QueryResult<Stream<T>>) insertQueryResult(resultDesc, "elements");
    }

    @SuppressWarnings("unchecked")
    default QueryResult<Long> count() {
        TypeDesc resultDesc = TypeDesc.LONG;
        return (QueryResult<Long>) insertQueryResult(resultDesc, "count");
    }

    private QueryResult<?> insertQueryResult(TypeDesc resultDesc, String methodName) {
        // Copy function expression, replacing return operation
        FuncOp queryExpression = expression();
        TypeDesc queryResultDesc = TypeDesc.type(QueryResult.DESC, resultDesc);
        FuncOp queryResultExpression = func("queryResult",
                methodType(queryResultDesc, queryExpression.funcDescriptor().parameters()))
                .body(b -> b.inline(queryExpression, b.parameters(), (block, query) -> {
                    MethodDesc md = method(Queryable.DESC, methodName,
                            methodType(QueryResult.DESC));
                    Op.Result queryResult = block.op(invoke(queryResultDesc, md, query));

                    block.op(_return(queryResult));
                }));

        return provider().createQueryResult(resultDesc, queryResultExpression);
    }
}
