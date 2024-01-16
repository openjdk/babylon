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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static java.lang.reflect.code.descriptor.MethodDesc.method;
import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;
import static java.lang.reflect.code.descriptor.TypeDesc.type;
import static java.lang.reflect.code.op.CoreOps.*;

/*
 * @test
 * @run testng TestLinqUsingQuotable
 */

public class TestLinqUsingQuotable {

    // Quotable functional interfaces

    public interface QuotablePredicate<T> extends Quotable, Predicate<T> {
    }

    public interface QuotableFunction<T, R> extends Quotable, Function<T, R> {
    }


    // Query interfaces

    public interface Queryable<T> {
        TypeDesc elementType();

        // Queryable<T> -> Queryable<U>
        FuncOp expression();

        QueryProvider provider();

        @SuppressWarnings("unchecked")
        default Queryable<T> where(QuotablePredicate<T> f) {
            LambdaOp l = (LambdaOp) f.quoted().op();
            return (Queryable<T>) insertQuery(elementType(), "where", l);
        }

        @SuppressWarnings("unchecked")
        default <R> Queryable<R> select(QuotableFunction<T, R> f) {
            LambdaOp l = (LambdaOp) f.quoted().op();
            return (Queryable<R>) insertQuery(l.funcDescriptor().returnType(), "select", l);
        }

        private Queryable<?> insertQuery(TypeDesc elementType, String methodName, LambdaOp lambdaOp) {
            QueryProvider qp = provider();

            // Copy function expression, replacing return operation
            FuncOp currentQueryExpression = expression();
            TypeDesc queryableType = TypeDesc.type(qp.queryableType(), elementType);
            FuncOp nextQueryExpression = func("query",
                    methodType(queryableType, currentQueryExpression.funcDescriptor().parameters()))
                    .body(b -> b.inline(currentQueryExpression, b.parameters(), (block, query) -> {
                        Op.Result fi = block.op(lambdaOp);

                        MethodDesc md = method(qp.queryableType(), methodName,
                                methodType(qp.queryableType(), lambdaOp.functionalInterface()).erase());
                        Op.Result queryable = block.op(invoke(queryableType, md, query, fi));

                        block.op(_return(queryable));
                    }));

            return qp.createQuery(elementType, nextQueryExpression);
        }

        @SuppressWarnings("unchecked")
        default QueryResult<Stream<T>> elements() {
            TypeDesc resultType = type(type(Stream.class), elementType());
            return (QueryResult<Stream<T>>) insertQueryResult(resultType, "elements");
        }

        @SuppressWarnings("unchecked")
        default QueryResult<Long> count() {
            return (QueryResult<Long>) insertQueryResult(TypeDesc.LONG, "count");
        }

        private QueryResult<?> insertQueryResult(TypeDesc resultType, String methodName) {
            QueryProvider qp = provider();

            // Copy function expression, replacing return operation
            FuncOp currentQueryExpression = expression();
            TypeDesc queryResultType = TypeDesc.type(qp.queryResultType(), resultType);
            FuncOp queryResultExpression = func("queryResult",
                    methodType(queryResultType, currentQueryExpression.funcDescriptor().parameters()))
                    .body(b -> b.inline(currentQueryExpression, b.parameters(), (block, query) -> {
                        MethodDesc md = method(qp.queryableType(), methodName, methodType(qp.queryResultType()));
                        Op.Result queryResult = block.op(invoke(queryResultType, md, query));

                        block.op(_return(queryResult));
                    }));
            return qp.createQueryResult(resultType, queryResultExpression);
        }
    }

    public interface QueryResult<T> {
        TypeDesc resultType();

        // Queryable<T> -> QueryResult<T>
        FuncOp expression();

        Object execute();
    }

    public interface QueryProvider {
        TypeDesc queryableType();

        TypeDesc queryResultType();

        Queryable<?> createQuery(TypeDesc elementType, FuncOp queryExpression);

        QueryResult<?> createQueryResult(TypeDesc resultType, FuncOp expression);

        <T> Queryable<T> newQuery(Class<T> elementType);
    }


    // Query implementation

    public static final class TestQueryable<T> implements Queryable<T> {
        final TypeDesc elementType;
        final TestQueryProvider provider;
        final FuncOp expression;

        TestQueryable(Class<T> tableClass, TestQueryProvider qp) {
            this.elementType = type(tableClass);
            this.provider = qp;

            TypeDesc queryableType = TypeDesc.type(qp.queryableType(), elementType);
            // Initial expression is an identity function
            var funDescriptor = methodType(queryableType, queryableType);
            this.expression = func("query", funDescriptor)
                    .body(b -> b.op(_return(b.parameters().get(0))));
        }

        TestQueryable(TypeDesc elementType, TestQueryProvider provider, FuncOp expression) {
            this.elementType = elementType;
            this.provider = provider;
            this.expression = expression;
        }

        @Override
        public TypeDesc elementType() {
            return elementType;
        }

        @Override
        public FuncOp expression() {
            return expression;
        }

        @Override
        public QueryProvider provider() {
            return provider;
        }
    }

    public record TestQueryResult<T>(TypeDesc resultType, FuncOp expression) implements QueryResult<T> {
        @Override
        public Object execute() {
            // @@@ Compile/translate the expression and execute it
            throw new UnsupportedOperationException();
        }
    }

    public static final class TestQueryProvider implements QueryProvider {
        final TypeDesc queryableType;
        final TypeDesc queryResultType;

        TestQueryProvider() {
            this.queryableType = TypeDesc.type(Queryable.class);
            this.queryResultType = TypeDesc.type(QueryResult.class);
        }

        @Override
        public TypeDesc queryableType() {
            return queryableType;
        }

        @Override
        public TypeDesc queryResultType() {
            return queryResultType;
        }

        @Override
        public Queryable<?> createQuery(TypeDesc elementType, FuncOp expression) {
            return new TestQueryable<>(elementType, this, expression);
        }

        @Override
        public QueryResult<?> createQueryResult(TypeDesc resultType, FuncOp expression) {
            return new TestQueryResult<>(resultType, expression);
        }

        @Override
        public <T> Queryable<T> newQuery(Class<T> elementType) {
            return new TestQueryable<>(elementType, this);
        }
    }


    static class Customer {
        // 1st column
        String contactName;
        // 2nd column
        String phone;
        // 3rd column
        String city;
    }

    @Test
    public void testSimpleQuery() {
        QueryProvider qp = new TestQueryProvider();

        QueryResult<Stream<String>> qr = qp.newQuery(Customer.class)
                .where(c -> c.city.equals("London"))
                .select(c -> c.contactName).elements();

        qr.expression().writeTo(System.out);

        @SuppressWarnings("unchecked")
        QueryResult<Stream<String>> qr2 = (QueryResult<Stream<String>>) Interpreter.invoke(MethodHandles.lookup(),
                qr.expression(), qp.newQuery(Customer.class));

        qr2.expression().writeTo(System.out);

        Assert.assertEquals(qr.expression().toText(), qr2.expression().toText());
    }
}
