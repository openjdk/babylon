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
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;
import java.util.stream.Stream;

import static java.lang.reflect.code.type.MethodRef.method;
import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.type.FunctionType.functionType;

/*
 * @test
 * @run testng TestLinqUsingQuoted
 */

public class TestLinqUsingQuoted {

    // Query interfaces

    public interface Queryable {
        TypeElement elementType();

        // Queryable<T> -> Queryable<U>
        FuncOp expression();

        QueryProvider provider();

        // T -> boolean
        // Predicate<T>
        default Queryable where(Quoted f) {
            // @@@@ validate
            ClosureOp c = (ClosureOp) f.op();
            return insertQuery(elementType(), "where", c);
        }

        // T -> R
        // Function<T, R>
        default Queryable select(Quoted f) {
            // @@@@ validate
            ClosureOp c = (ClosureOp) f.op();
            return insertQuery(c.invokableType().returnType(), "select", c);
        }

        private Queryable insertQuery(TypeElement et, String name, ClosureOp c) {
            QueryProvider qp = provider();

            FuncOp currentQueryExpression = expression();
            FuncOp nextQueryExpression = currentQueryExpression.transform((block, op) -> {
                if (op instanceof ReturnOp rop && rop.ancestorBody() == currentQueryExpression.body()) {
                    Value query = block.context().getValue(rop.returnValue());

                    Op.Result quotedLambda = block.op(quoted(block.parentBody(), qblock -> c));

                    MethodRef md = method(qp.queryableType(), name,
                            functionType(qp.queryableType(), QuotedOp.QUOTED_TYPE));
                    Op.Result queryable = block.op(invoke(md, query, quotedLambda));

                    block.op(_return(queryable));
                } else {
                    block.op(op);
                }
                return block;
            });

            return qp.createQuery(et, nextQueryExpression);
        }

        // Iterate
        // Queryable -> Stream
        default QueryResult elements() {
            TypeElement resultType = JavaType.type(JavaType.type(Stream.class), (JavaType) elementType());
            return insertQueryResult("elements", resultType);
        }

        // Count
        // Queryable -> Long
        default QueryResult count() {
            return insertQueryResult("count", JavaType.LONG);
        }

        private QueryResult insertQueryResult(String name, TypeElement resultType) {
            QueryProvider qp = provider();

            // Copy function expression, replacing return type
            FuncOp currentQueryExpression = expression();
            FuncOp nextQueryExpression = func("queryresult",
                    functionType(qp.queryResultType(), currentQueryExpression.invokableType().parameterTypes()))
                    .body(b -> b.inline(currentQueryExpression, b.parameters(), (block, query) -> {
                        MethodRef md = method(qp.queryableType(), name, functionType(qp.queryResultType()));
                        Op.Result queryResult = block.op(invoke(md, query));

                        block.op(_return(queryResult));
                    }));
            return qp.createQueryResult(resultType, nextQueryExpression);
        }
    }

    public interface QueryResult {
        TypeElement resultType();

        // Queryable -> QueryResult
        FuncOp expression();

        Object execute();
    }

    public interface QueryProvider {
        TypeElement queryableType();

        TypeElement queryResultType();

        Queryable createQuery(TypeElement elementType, FuncOp expression);

        QueryResult createQueryResult(TypeElement resultType, FuncOp expression);

        Queryable newQuery(TypeElement elementType);
    }


    // Query implementation

    public static final class TestQueryable implements Queryable {
        final TypeElement elementType;
        final TestQueryProvider provider;
        final FuncOp expression;

        TestQueryable(TypeElement elementType, TestQueryProvider provider) {
            this.elementType = elementType;
            this.provider = provider;

            // Initial expression is an identity function
            var funType = functionType(provider().queryableType(), provider().queryableType());
            this.expression = func("query", funType)
                    .body(b -> b.op(_return(b.parameters().get(0))));
        }

        TestQueryable(TypeElement elementType, TestQueryProvider provider, FuncOp expression) {
            this.elementType = elementType;
            this.provider = provider;
            this.expression = expression;
        }

        @Override
        public TypeElement elementType() {
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

    public record TestQueryResult(TypeElement resultType, FuncOp expression) implements QueryResult {
        @Override
        public Object execute() {
            // @@@ Compile/translate the expression and execute it
            throw new UnsupportedOperationException();
        }
    }

    public static final class TestQueryProvider implements QueryProvider {
        final TypeElement queryableType;
        final TypeElement queryResultType;

        TestQueryProvider() {
            this.queryableType = JavaType.type(Queryable.class);
            this.queryResultType = JavaType.type(QueryResult.class);
        }

        @Override
        public TypeElement queryableType() {
            return queryableType;
        }

        @Override
        public TypeElement queryResultType() {
            return queryResultType;
        }

        @Override
        public TestQueryable createQuery(TypeElement elementType, FuncOp expression) {
            return new TestQueryable(elementType, this, expression);
        }

        @Override
        public QueryResult createQueryResult(TypeElement resultType, FuncOp expression) {
            return new TestQueryResult(resultType, expression);
        }

        @Override
        public Queryable newQuery(TypeElement elementType) {
            return new TestQueryable(elementType, this);
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

        QueryResult qr = qp.newQuery(JavaType.type(Customer.class))
                // c -> c.city.equals("London")
                .where((Customer c) -> c.city.equals("London"))
                // c -> c.contactName
                .select((Customer c) -> c.contactName).elements();

        qr.expression().writeTo(System.out);

        QueryResult qr2 = (QueryResult) Interpreter.invoke(MethodHandles.lookup(),
                qr.expression(), qp.newQuery(JavaType.type(Customer.class)));

        qr2.expression().writeTo(System.out);

        Assert.assertEquals(qr.expression().toText(), qr2.expression().toText());
    }
}
