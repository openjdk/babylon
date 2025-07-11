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

import jdk.incubator.code.Op;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.java.JavaType.parameterized;
import static jdk.incubator.code.dialect.java.MethodRef.method;
import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaType.type;

public interface Queryable<T> {
    JavaType TYPE = type(Queryable.class);

    QueryProvider provider();

    // T
    JavaType elementType();

    // Queryable<T> -> Queryable<U>
    FuncOp expression();

    @SuppressWarnings("unchecked")
    default Queryable<T> where(QuotablePredicate<T> f) {
        JavaOp.LambdaOp l = (JavaOp.LambdaOp) Op.ofQuotable(f).get().op();
        return (Queryable<T>) insertQuery(elementType(), "where", l);
    }

    @SuppressWarnings("unchecked")
    default <R> Queryable<R> select(QuotableFunction<T, R> f) {
        JavaOp.LambdaOp l = (JavaOp.LambdaOp) Op.ofQuotable(f).get().op();
        return (Queryable<R>) insertQuery((JavaType) l.invokableType().returnType(), "select", l);
    }

    private Queryable<?> insertQuery(JavaType elementType, String methodName, JavaOp.LambdaOp lambdaOp) {
        // Copy function expression, replacing return operation
        FuncOp queryExpression = expression();
        JavaType queryableType = parameterized(Queryable.TYPE, elementType);
        FuncOp nextQueryExpression = func("query",
                functionType(queryableType, queryExpression.invokableType().parameterTypes()))
                .body(b -> Inliner.inline(b, queryExpression, b.parameters(), (block, query) -> {
                    Op.Result fi = block.op(lambdaOp);

                    MethodRef md = method(Queryable.TYPE, methodName,
                            functionType(Queryable.TYPE, ((ClassType) lambdaOp.functionalInterface()).rawType()));
                    Op.Result queryable = block.op(JavaOp.invoke(queryableType, md, query, fi));

                    block.op(return_(queryable));
                }));

        return provider().createQuery(elementType, nextQueryExpression);
    }

    @SuppressWarnings("unchecked")
    default QueryResult<Stream<T>> elements() {
        JavaType resultType = parameterized(type(Stream.class), elementType());
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
        JavaType queryResultType = parameterized(QueryResult.TYPE, resultType);
        FuncOp queryResultExpression = func("queryResult",
                functionType(queryResultType, queryExpression.invokableType().parameterTypes()))
                .body(b -> Inliner.inline(b, queryExpression, b.parameters(), (block, query) -> {
                    MethodRef md = method(Queryable.TYPE, methodName,
                            functionType(QueryResult.TYPE));
                    Op.Result queryResult = block.op(JavaOp.invoke(queryResultType, md, query));

                    block.op(return_(queryResult));
                }));

        return provider().createQueryResult(resultType, queryResultExpression);
    }
}
